Before you can profile a kernel you have to be able to *write* one, and before you can write one you have to internalize a mental model so simple it feels like it can't possibly be enough: **one thread, one element.** You launch thousands of threads, each one computes its own index, does a tiny amount of work at that index, and stops. There is no loop over the array in your kernel — the loop *is* the grid of threads. Sasha Rush's [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) is the fastest way I know to burn that model into your hands, because it strips a CUDA kernel down to a single line and forces you to get the indexing exactly right. This worklog walks the first six puzzles. They look trivial — the "solution" to each is one line — but every one of them installs a specific reflex you will use in every kernel on this site, including the [naive GEMM](gemm-kernel-1-naive.html) where `blockIdx.y * blockDim.y + threadIdx.y` shows up for real.[[sn: The puzzles are written in Python against a tiny NUMBA-style CUDA simulator, not real `.cu` files, so you can run them in a notebook with no GPU. The index arithmetic is identical to CUDA C; `cuda.threadIdx.x` is exactly `threadIdx.x`. Everything transfers.]]

The harness gives each puzzle a function like `call(out, a)` that runs *once per thread*. Inside it you get `cuda.threadIdx.x`, `cuda.blockIdx.x`, and `cuda.blockDim.x`, and your job is to figure out which element of `out` this particular thread owns and write it. That's the whole game.

## Puzzle 1 — Map: the identity between thread and index

The first puzzle asks for `out[i] = a[i] + 10` over an array of `SIZE = 4`, launched with exactly four threads in one block. The entire kernel is one line:

```python
def call(out, a) -> None:
    local_i = cuda.threadIdx.x
    out[local_i] = a[local_i] + 10
```

That's it. Thread 0 handles element 0, thread 3 handles element 3, and they all run at the same time. The insight this teaches is the load-bearing one for everything that follows: **`threadIdx.x` is not a loop counter you advance — it is a coordinate you are handed.** In a serial `for i in range(4)` the runtime gives you `i = 0`, then `1`, then `2`. Here the hardware gives *four different threads* four different values of `threadIdx.x` simultaneously, and each thread only ever sees its own. You are not iterating over the array; you are stamping one thread onto each cell of it.

[[fig: A hand-drawn figure titled "Map: one thread per cell". On the left, four small rounded boxes stacked vertically labeled in black "thread 0", "thread 1", "thread 2", "thread 3", each with a purple code note "local_i = threadIdx.x". In the middle an input array `a` drawn as 4 cells with blue diagonal hatch, values 0,1,2,3, red dimension label "SIZE = 4". On the right an output array `out` drawn as 4 pale-yellow-hatch cells. Four blue dashed arrows connect thread k straight across to a[k] and out[k], each arrow labeled "+10" in orange. A green handwritten spec note top-right: "1 block · 4 threads · blockDim.x = 4". Bottom dashed takeaway box: "the grid of threads IS the loop — no for-loop in the kernel". || Puzzle 1. Each thread is handed one coordinate and touches exactly one cell.]]

## Puzzle 2 — Zip: nothing new, and that's the point

Zip adds two arrays: `out[i] = a[i] + b[i]`, still `SIZE = 4`, still four threads. The solution is what you'd guess:

```python
def call(out, a, b) -> None:
    local_i = cuda.threadIdx.x
    out[local_i] = a[local_i] + b[local_i]
```

The reason this is a separate puzzle is subtle and worth saying out loud: a second input buffer changes *nothing* about the indexing. Each thread still owns one index `local_i` and reads that same index from *both* inputs. This is the seed of the whole idea of an **element-wise** (or *pointwise*) operation — `add`, `mul`, `relu`, `x + 1` — where the arithmetic intensity is fixed no matter how many operands you have, because the work per element is constant. It's a preview of why these ops are so brutally [memory-bound](the-three-regimes.html): here we do one add and touch three memory locations. Two reads, one write, one flop. That ratio is the reason element-wise kernels never get near the tensor cores, and Zip is where you first feel it in your fingers.

## Puzzle 3 — Guards: threads outnumber the work

Now the harness changes the launch: `SIZE` is still 4, but it spins up **8 threads**. Threads 4 through 7 have no element to write. If they run the Puzzle 1 code they will index `a[4]`, `a[5]`, … straight off the end of the buffer — an out-of-bounds read, which on a real GPU is undefined behavior at best and a memory fault at worst. The fix is the single most repeated pattern in all of CUDA, the **guard**:

```python
def call(out, a, size) -> None:
    local_i = cuda.threadIdx.x
    if local_i < size:
        out[local_i] = a[local_i] + 10
```

You almost never get to choose a launch that exactly matches your data. Grids are launched in whole blocks, block sizes are fixed at 32, 64, 256, and your array size is whatever the model handed you, so the last block virtually always has threads that fall off the end.[[sn: You round up: `gridDim = ceil(N / blockDim)`. That deliberately over-launches, which is *why* the tail threads exist. The guard `if i < N` is the price of rounding up, and it is essentially free — a single predicated compare that the extra threads resolve to a no-op.]] The guard is how you launch *too many* threads on purpose and let the surplus ones quietly do nothing. Every kernel in the GEMM ladder has a version of `if (m < N && n < N)` guarding the write; you can see it in the naive kernel's body. This puzzle is where that reflex is born.

There's a second, quieter lesson hiding here about how the surplus threads actually behave. They don't crash and they don't stall the block — the ones that fail the `if` simply sit out the write while their neighbors proceed. That divergence within the block is cheap here because the two paths are "do the write" and "do nothing," but the general mechanism (a warp splitting on a branch) is [SIMT divergence](simt-and-divergence.html), and it is not always this cheap.

## Puzzle 4 — Map 2D: a second axis, same idea

The next puzzle moves to a `2 × 2` array and launches a `3 × 3` block of threads, using both `threadIdx.x` and `threadIdx.y`:

```python
def call(out, a, size) -> None:
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, j] + 10
```

Two things get installed at once. First, **threads are laid out in up to three dimensions** — `threadIdx` has `.x`, `.y`, `.z` — and you address a 2D grid of data with a 2D grid of threads, which is exactly the right shape for images and matrices. Second, the guard now has to check *both* axes (`i < size and j < size`), because the `3 × 3` launch over-covers a `2 × 2` array on both edges. The guard scales with the dimensionality of your problem; a 2D kernel needs a 2D guard. Hold onto this — when we assign one thread per output element of a matrix in the [naive GEMM](gemm-kernel-1-naive.html), this `(i, j)` layout is precisely what we use, with `m` and `n` playing the roles of `i` and `j`.

[[fig: A hand-drawn 2D-indexing figure titled "Map 2D + Guard". Center-left: a 3×3 grid of small rounded thread boxes, black-labeled with coordinates (0,0)…(2,2). Overlaid on the top-left 2×2 sub-block, a pale-yellow-hatch region labeled in red "valid: i<2 and j<2". The right column and bottom row of thread boxes are drawn faint/greyed with a red handwritten note "guarded off — no write". To the right: the 2×2 output matrix `out` with pale-yellow hatch (output color), red dimension labels "↔ size = 2" on both axes. Purple code note bottom-left: "i = threadIdx.x ; j = threadIdx.y". A curved blue dashed arrow from thread (1,1) to out[1,1]. Dashed takeaway box: "2D data → 2D threads → guard BOTH axes". || Puzzle 4. A 3×3 launch over a 2×2 array; the surplus row and column are guarded off.]]

## Puzzle 5 — Broadcast: reading is not one-to-one

Broadcast is the first puzzle where the *reads* stop lining up one-to-one with the output. The output is `2 × 2`, but the inputs are a column vector `a` of shape `SIZE × 1` and a row vector `b` of shape `1 × SIZE`. Every output cell is the sum of one element from the column and one from the row:

```python
def call(out, a, b, size) -> None:
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, 0] + b[0, j]
```

Look carefully at the indices: thread `(i, j)` writes `out[i, j]` but reads `a[i, 0]` and `b[0, j]`. The write coordinate and the read coordinates are *different projections* of the same thread ID. This is the moment the one-thread-one-*output*-element model detaches from the inputs: a thread owns exactly one output cell, but it is free to read whatever inputs that cell depends on. `a[i, 0]` gets read by every thread in row `i` — reuse! — and `b[0, j]` by every thread in column `j`. That reuse pattern, where a single loaded value feeds many output threads, is the entire economic argument for shared memory later on.[[sn: In [the naive GEMM](gemm-kernel-1-naive.html) this exact reuse is what we *fail* to exploit: thread `(m, n)` reads all of row `m` of A and all of column `n` of B from global memory, and the same row and column get re-read by every other thread that shares them. Broadcast is the toy version of that redundant-read problem; [shared memory](shared-memory-l1.html) is the grown-up fix.]] Broadcast is small, but it is the first puzzle that is really *about* memory rather than arithmetic.

[[fig: A hand-drawn "broadcast reuse" figure titled "Broadcast: write once, read shared". Center: a 2×2 output matrix `out` drawn with pale-yellow hatch, cells labeled in black (0,0)(0,1)(1,0)(1,1), red dimension labels "↔ size = 2" on both axes. Down the left edge, a column vector `a` of shape 2×1 with blue diagonal hatch (input-A color), cells a[0,0], a[1,0], red shape label "a : size × 1". Across the top edge, a row vector `b` of shape 1×2 with green diagonal hatch (input-B color), cells b[0,0], b[0,1], red shape label "b : 1 × size". From a[0,0] a blue dashed arrow fans RIGHT into BOTH cells of output row 0 (orange callout "one load → whole row reuses it"); from b[0,0] a green dashed arrow fans DOWN into BOTH cells of output column 0. Purple code note bottom-left: "out[i,j] = a[i,0] + b[0,j]". A small orange emphasis star on the mismatch: "write (i,j) ≠ read (i,0),(0,j)". Dashed takeaway box bottom-right: "one output per thread, but inputs are SHARED → this reuse is why shared memory exists". || Puzzle 5. The write index and read indices are different projections of the same thread; each input value feeds a whole row or column.]]

## Puzzle 6 — Blocks and the block-guard: the global index

Everything so far fit in a single block. That can't last, because a block is capped at **1024 threads** and your arrays are millions of elements long.[[sn: The hardware limit is `blockDim.x * blockDim.y * blockDim.z ≤ 1024` per block, and a block must fit on one **Streaming Multiprocessor** (SM). To cover a large array you launch many blocks across the grid; on an H100 there are ~132 SMs to spread them over. Choosing the block size is its own art — see [occupancy](occupancy.html).]] Puzzle 6 sets `SIZE = 9` but launches blocks of only 4 threads, so it needs 3 blocks to cover 9 elements. A thread's `threadIdx.x` now only tells you its position *within* its block — it resets to 0 in every block — so `threadIdx.x` alone is no longer a unique index. You have to reconstruct the **global index** from three pieces:

```python
def call(out, a, size) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < size:
        out[i] = a[i] + 10
```

This one line — `blockIdx.x * blockDim.x + threadIdx.x` — is the single most important formula in CUDA, and it deserves to be read slowly. `blockDim.x` is the width of a block (4 here). `blockIdx.x` is which block you're in (0, 1, or 2). Multiply them to get the offset to the *start* of your block, then add `threadIdx.x` to get your slot inside it. Block 2, thread 1 lands at `2 * 4 + 1 = 9`. And `9` is off the end of our 9-element array (valid indices are 0–8), which is exactly why the guard `if i < size` comes back — with 3 blocks of 4 threads we launched 12 threads for 9 elements, so the last 3 fall off and must be guarded. This is the **block-guard**: the global-index formula *and* the boundary check, together, forming the canonical opening of essentially every 1D CUDA kernel ever written.

[[fig: A hand-drawn "global index" walkthrough titled "Blocks + block-guard". Three block containers drawn as large rounded rectangles side by side, black-labeled "block 0", "block 1", "block 2", each holding 4 small thread cells numbered 0–3 (these are threadIdx.x, red-labeled "resets each block"). Below each block, a green note: "blockDim.x = 4". Under the row of blocks, a single long flat array `out` of 9 pale-yellow cells indexed 0..8, red dimension label "size = 9". Blue dashed arrows map block1/thread2 to the formula box (purple): "i = blockIdx.x·blockDim.x + threadIdx.x = 1·4+2 = 6". Block 2's thread 3 is drawn in red pointing PAST the array to a phantom cell "index 9" with a red warning "✗ out of bounds → guarded". Orange emphasis callout over the formula: "the most important line in CUDA". Dashed takeaway box: "global i = block offset + local offset, THEN guard". || Puzzle 6. Reconstructing a unique global index across three blocks, then guarding the tail.]]

## What these six puzzles actually taught

Line them up and a ladder appears. Map: a thread owns one index. Zip: extra inputs don't change the index. Guards: over-launch on purpose and let surplus threads no-op. Map 2D: index in two dimensions, guard both. Broadcast: the output index and the input indices can differ, and shared reads mean reuse. Blocks: `threadIdx.x` is local, so build the global index, then guard it. That last combination — `int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < N)` — is not a puzzle trick. It is the first two lines of the [naive GEMM kernel](gemm-kernel-1-naive.html), of every activation kernel, of every reduction's load phase. You have now written it half a dozen times.

None of these kernels is *fast*; a pure element-wise add is [memory-bound](the-three-regimes.html) and will sit far below the roofline no matter how you write it, and we haven't touched shared memory, warps, or coalescing yet. But speed was never the point of the first six. The point was to make the indexing automatic, so that when we start caring about memory traffic and profiling with Nsight Compute, the coordinate arithmetic is muscle memory and your whole attention is free for the thing that actually decides performance: how the bytes move. In the next walkthrough the puzzles turn to shared memory and reductions, and the one-thread-one-element model finally has to cooperate with its neighbors.
