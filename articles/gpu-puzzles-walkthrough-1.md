Here is a question that sounds too simple to be interesting, but isn't: **how do you add 10 to every number in an array, using a machine that has thousands of tiny workers all running at the same time?** On a CPU you'd write a loop — `for i in range(N): out[i] = a[i] + 10` — and the CPU would walk the array one element at a time, fast, in order. A GPU refuses to work that way. A GPU wants to throw thousands of workers at the array *simultaneously*, each one touching a different element, and have them all finish at roughly the same instant. To use it, you have to stop thinking "loop over the array" and start thinking "assign one worker per element." That single mental flip is the hardest part of learning GPU programming, and it is what this article is about.

I'm going to teach it the way it finally clicked for me: through the first six of Sasha Rush's [GPU-Puzzles](https://github.com/srush/GPU-Puzzles). Each puzzle is a one-line kernel. They look trivial — genuinely, the "solution" to each is a single line of code — but that's the trap and also the gift. Because the code is so short, there's nowhere to hide. All the difficulty lives in the *indexing*: figuring out which element this particular worker owns. Get the indexing right and you've internalized the one idea that underlies every kernel on this site, from the [naive GEMM](gemm-kernel-1-naive.html) to [FlashAttention](flashattention-1.html). Get it wrong and nothing else you learn will help.

We'll go slowly. By the end of six one-liners you will have written, by hand, the exact two lines of code that open essentially every CUDA kernel ever shipped — the lines that run in vLLM and PyTorch right now, on the H100s serving models as you read this.

[[sn: The puzzles are written in Python against a tiny NUMBA-style CUDA simulator, not real `.cu` files, so you can run every one in a notebook with no GPU at all. The index arithmetic is byte-for-byte identical to CUDA C: `cuda.threadIdx.x` in the puzzle is exactly `threadIdx.x` in a real kernel, `cuda.blockIdx.x` is `blockIdx.x`, and so on. Everything you learn here transfers to the real thing unchanged.]]

## The one mental model: one thread, one element

Before any code, let me plant the picture we'll reuse for the entire article. I'll call it the **stamp model**.

Imagine your output array is a row of empty cells, like an ice-cube tray. And imagine you have a pile of identical rubber stamps, one per cell. Each stamp knows how to do exactly one thing — "read the input here, add 10, write the result here" — but it has no idea which cell it belongs to until you hand it a coordinate. You press all the stamps down at once. Each one lands on its own cell, does its tiny job, and lifts off. There is no order. There is no "first cell, then second cell." They all fire together.

In CUDA the stamps are called **threads**, and the coordinate each one gets handed is called `threadIdx.x`. A thread is the smallest unit of work the GPU runs — one worker, one stamp.[[sn: If you want the full hierarchy — threads group into warps, warps into blocks, blocks into a grid — it's laid out in [threads, warps, blocks, and grids](threads-warps-blocks-grids.html). For the first four puzzles you only need "a thread is one worker with its own coordinate," so I'll introduce the rest only when a puzzle forces us to.]] The crucial, load-bearing fact is this: **`threadIdx.x` is not a loop counter you advance — it is a coordinate you are handed.** In a serial loop, one worker visits index 0, then 1, then 2. Here, four different workers are each *born* knowing their own index, and each one only ever sees its own. Nobody iterates. The grid of threads *is* the loop.

[[fig: A hand-drawn intuition figure titled "The stamp model: one thread, one cell". LEFT PANEL labeled "(A) CPU — one worker, walks the array" in black: a single small figure/box labeled "1 worker" with a curved blue arrow looping over a horizontal array of 5 cells (blue diagonal hatch) numbered 0,1,2,3,4, the arrow labeled in blue "step 0 → 1 → 2 → 3 → 4 (in order, one at a time)". RIGHT PANEL labeled "(B) GPU — many workers, all at once" in black: five separate small rounded stamp boxes stacked, black-labeled "thread 0".."thread 4", each with a straight blue dashed arrow pressing straight DOWN onto its own cell of a pale-yellow-hatch output array numbered 0..4; an orange callout across all five arrows reads "all fire simultaneously — no order". Purple code note under panel B: "out[threadIdx.x] = a[threadIdx.x] + 10". Green spec note top-right: "each thread = 1 worker, handed 1 coordinate". Dashed takeaway box at bottom: "the grid of threads IS the loop — there is no for-loop inside the kernel". || The mental model for the whole article: a CPU walks the array; a GPU stamps every cell at once, one thread per cell.]]

Hold that picture. Every puzzle below is a small deformation of it — a second input, a surplus of stamps, a second axis, shared reads, more than one tray. The stamp never changes; only the bookkeeping around "which cell is mine" does.

One practical note on how the puzzles are run. The harness hands you a function like `call(out, a)`, and — this is the part that trips everyone up the first time — **that function runs once per thread, in parallel, not once total.** You are not writing the loop. You are writing the body of the stamp: what *one* thread does. Inside it you're given `cuda.threadIdx.x` (this thread's coordinate), `cuda.blockIdx.x` (which block it's in — ignore for now), and `cuda.blockDim.x` (how wide a block is — also ignore for now). Your only job is to work out which element of `out` this thread owns, and write it.

## Puzzle 1 — Map: the identity between a thread and an index

The first puzzle: compute `out[i] = a[i] + 10` over an array of `SIZE = 4`, launched with exactly four threads in one block. Here is the whole kernel.

```python
def call(out, a) -> None:
    local_i = cuda.threadIdx.x
    out[local_i] = a[local_i] + 10
```

That's it. That's the entire solution. Let's slow down and watch it happen, because this is where the mental model becomes concrete.

The harness launches four threads. Thread 0 wakes up with `threadIdx.x == 0`, so `local_i = 0`, and it executes `out[0] = a[0] + 10`. At the *same time*, thread 3 wakes up with `threadIdx.x == 3`, sets `local_i = 3`, and executes `out[3] = a[3] + 10`. Neither thread knows the other exists. Neither one loops. Each does one add and one write and is finished. Four stamps, four cells, one press.

Now the question I want you to actually sit with: **why is this a different way of thinking, and not just a fancy loop?** Because in the loop version, the number `i` is a *value that changes over time* — the same variable holding 0, then 1, then 2. In the kernel version, `local_i` never changes. It is 0 forever, in thread 0. It is 3 forever, in thread 3. There is no "over time." The four values of `local_i` exist *simultaneously, in four separate copies of the function.* Once you see that the index is a per-worker constant rather than a shared counter, you've got it — that's the reflex the whole exercise exists to build.

This operation has a name: a **map**. You apply the same function independently to every element. Maps are the friendliest thing a GPU does, because there are zero dependencies between elements — thread 3 never needs anything thread 2 computed. Perfect parallelism. `relu`, `x + 1`, `gelu`, scaling a tensor by a constant — all maps, all this exact shape.

[[fig: A hand-drawn zoom figure titled "Map: from grid to one thread". TOP HALF (the whole picture): four small rounded boxes in a row, black-labeled "thread 0".."thread 3", each with purple note "local_i = threadIdx.x"; an input array `a` of 4 cells with blue diagonal hatch holding values 2,7,1,5, red dimension label "SIZE = 4"; an output array `out` of 4 pale-yellow-hatch cells; four blue dashed arrows carry thread k straight across from a[k] to out[k], each arrow tagged "+10" in orange. Green spec note top-right: "1 block · 4 threads · blockDim.x = 4". BOTTOM HALF (the zoom-in on ONE thread, boxed and connected up to thread 2 with a dashed grey magnifier line): a large box labeled in black "inside thread 2", showing by-hand steps in purple "local_i = 2", then "a[2] = 1", then "out[2] = 1 + 10 = 11", with a red note "local_i never changes — it is 2 for the whole life of this thread". Dashed takeaway box bottom: "a MAP: same function, every element, zero dependencies → perfect parallelism". || Puzzle 1, zoomed into a single thread. The index is a per-thread constant, not a counter.]]

## Puzzle 2 — Zip: a second input changes nothing (and that's the point)

Zip adds two arrays: `out[i] = a[i] + b[i]`, still `SIZE = 4`, still four threads. The solution is what you'd guess.

```python
def call(out, a, b) -> None:
    local_i = cuda.threadIdx.x
    out[local_i] = a[local_i] + b[local_i]
```

Here's the natural objection: *if the answer is this obvious, why is it a separate puzzle at all?* Because the lesson is precisely that it's obvious. A second input buffer changes **nothing** about the indexing. Each thread still owns one index `local_i`, and it reads that *same* index from both inputs. The stamp got a little wider — it now presses down on two input trays instead of one — but it still lands on exactly one output cell. Add a third input, a fourth, ten inputs, and the story is identical: one output index, read straight down from all inputs.

Let's do a little arithmetic, because it foreshadows something important. For each element in Zip, count the memory touches and the math. We read `a[i]`, we read `b[i]`, we write `out[i]` — that's **3 memory operations**. And we do **1** addition. So the ratio of arithmetic to memory movement is 1 flop for every 3 element-touches. Now here's the thing that should bother you: that ratio doesn't improve if you add more inputs. A ten-input sum does 9 adds but 11 memory touches — still roughly one flop per memory touch. The work per element stays proportional to the memory you move.

This is the seed of **arithmetic intensity**, and it's why element-wise (or *pointwise*) kernels — `add`, `mul`, `relu` — are so brutally [memory-bound](the-three-regimes.html). The GPU can do arithmetic vastly faster than it can move bytes from memory. An H100 can do on the order of a thousand floating-point operations in the time it takes to fetch a single number from its main memory.[[sn: Rough but real: an H100 SXM does ~67 TFLOP/s of FP32 and has ~3.35 TB/s of HBM bandwidth. Dividing, that's roughly 20 FLOPs per byte, or ~80 FLOPs per 4-byte float, just to break even. A kernel doing 1 flop per float touched is off that break-even point by a factor of ~80 — it will spend ~99% of its time waiting on memory. This is the whole reason [operator fusion](operator-fusion.html) exists.]] So a kernel that does one add per three memory touches will spend almost all its time *waiting for memory*, and no cleverness in how you write the add can fix that. Zip is where you first feel, in your fingers, the ratio that the entire [roofline model](roofline-model.html) is built around.

## Puzzle 3 — Guards: launching too many threads on purpose

Now the harness does something that looks like a mistake. `SIZE` is still 4, but it launches **8 threads**. You have twice as many stamps as cells. Threads 4, 5, 6, 7 have no element to write. If they run the Puzzle 1 code, thread 4 will try to execute `out[4] = a[4] + 10` — but `a[4]` is off the end of a 4-element array. On the puzzle simulator that's an error; on a real GPU it's an **out-of-bounds** access, which is undefined behavior at best and a hard memory fault that kills your kernel at worst.

The fix is the single most repeated pattern in all of CUDA. It's called a **guard**.

```python
def call(out, a, size) -> None:
    local_i = cuda.threadIdx.x
    if local_i < size:
        out[local_i] = a[local_i] + 10
```

"If my index is inside the array, do the work; otherwise do nothing." Thread 4 checks `4 < 4`, finds it false, and quietly skips the write. It doesn't crash, it doesn't stall its neighbors — it just sits out this instruction and finishes.

But wait — the deeper question is *why would anyone launch 8 threads for 4 elements in the first place?* It looks wasteful and dumb. The answer is that **you almost never get to choose a launch that exactly matches your data.** Threads are launched in whole **blocks**, and block sizes are chosen for hardware reasons — 32, 64, 128, 256 — not to match your array. Your array, meanwhile, is whatever length the model handed you: 4, or 1000, or 50,257 (a real vocabulary size). Those two numbers rarely divide evenly. So the standard move is to compute how many blocks you need by *rounding up* — `gridDim = ceil(N / blockDim)` — which deliberately over-launches, and then let the surplus threads at the tail no-op via the guard.[[sn: Concretely: for `N = 1000` with a block size of 256, `ceil(1000 / 256) = 4` blocks = 1024 threads. The last 24 threads (indices 1000–1023) fall off the end and are guarded off. The guard is essentially free — a single predicated compare that the tail threads resolve to a no-op in one cycle. You pay one cheap instruction to buy the freedom to pick any block size you like.]]

[[fig: A hand-drawn before/after figure titled "Guards: over-launch, then no-op the surplus". Split into two stacked panels sharing the same 4-cell output array. PANEL (A) labeled in red "WITHOUT guard — CRASH": 8 thread boxes in a row, black-labeled 0..7; threads 0–3 have blue dashed arrows into a 4-cell output array (pale-yellow hatch, red label "size = 4"); threads 4–7 have RED arrows pointing PAST the array into phantom greyed cells 4,5,6,7 marked with a red "✗ out of bounds → fault". PANEL (B) labeled in green "WITH guard — safe": same 8 thread boxes; threads 0–3 still write (blue arrows); threads 4–7 are drawn faint/greyed with a red note "if local_i < size → false → do nothing". Purple code note beside panel B: "if local_i < size:". Orange callout between panels: "launch 8 for 4 ON PURPOSE — block sizes are fixed, arrays aren't". Dashed takeaway box: "gridDim = ceil(N / blockDim) over-launches; the guard is the ~free price of rounding up". || Puzzle 3. Without the guard the four surplus threads read off the end; with it they cheaply do nothing.]]

There's a quieter second lesson hiding here, and it matters later. When the eight threads hit the `if`, four take the "write" path and four take the "do nothing" path. That's a **branch**, and when threads that run in lockstep disagree on a branch, the hardware has to handle both paths. Here it's cheap — the two paths are "one write" and "nothing," so the split costs almost nothing. But the general mechanism, a group of threads diverging on a branch, is [SIMT divergence](simt-and-divergence.html), and it is not always this cheap. When the two sides of an `if` are both expensive, divergence can cut your throughput in half. File that away; Puzzle 3 is where the reflex is born, and the sidenote is where the danger lives.

## Puzzle 4 — Map 2D: a second axis, the same idea

The next puzzle moves to a `2 × 2` array and launches a `3 × 3` block of threads, using both `threadIdx.x` and `threadIdx.y`.

```python
def call(out, a, size) -> None:
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, j] + 10
```

Two things get installed at once, and both are small extensions of things you already have.

First, **threads can be laid out in up to three dimensions.** `threadIdx` has `.x`, `.y`, and `.z`. Why offer this? Because so much of what GPUs compute is naturally 2D or 3D — images (height × width), matrices (rows × columns), video (height × width × time). If your data is a 2D grid, it's cleaner to address it with a 2D grid of threads than to flatten everything into one long line and do division-and-modulo to recover the coordinates. Thread `(i, j)` owns cell `(i, j)`. The stamp model is completely unchanged — one thread, one cell — the cell just has two coordinates now instead of one.

Second, **the guard now has to check both axes.** We launched a `3 × 3` block over a `2 × 2` array, so we over-cover on *both* edges: there's a surplus column (i = 2) and a surplus row (j = 2). A thread at `(2, 0)` is off the right edge; a thread at `(0, 2)` is off the bottom. So the guard becomes `if i < size and j < size` — it fails if *either* coordinate is out of range. The rule generalizes cleanly: the guard scales with the dimensionality of your problem. A 1D kernel guards one axis; a 2D kernel guards two; a 3D kernel guards three.

Hold onto this `(i, j)` layout, because it is exactly the shape we use in the [naive GEMM](gemm-kernel-1-naive.html): one thread per output element of a matrix, with `m` (which row of C) and `n` (which column of C) playing the roles of `i` and `j`. When you write your first real matrix-multiply kernel, this puzzle is what your fingers will remember.

[[fig: A hand-drawn 2D figure titled "Map 2D + guard both axes". CENTER-LEFT: a 3×3 grid of small rounded thread boxes, black-labeled with coordinates (0,0),(0,1),(0,2) across the top row down to (2,2). Overlaid on the top-left 2×2 sub-block, a pale-green translucent highlight labeled in green "valid region: i<2 AND j<2 → these 4 write". The rightmost column (i=2 boxes) and bottom row (j=2 boxes) are drawn faint/greyed and enclosed by a red dashed outline labeled "guarded off — surplus row & column, no write". RIGHT: the 2×2 output matrix `out` drawn with pale-yellow hatch, red dimension labels "↕ size = 2" and "↔ size = 2" on the two axes. A curved blue dashed arrow runs from thread box (1,1) to out[1,1]. Purple code note bottom-left: "i = threadIdx.x ; j = threadIdx.y ; if i<size and j<size". Orange callout on the guarded L-shape: "3×3 launch over 2×2 data → over-covers on BOTH edges". Dashed takeaway box: "2D data → 2D threads → guard scales with dimensionality (guard BOTH axes)". || Puzzle 4. A 3×3 launch over a 2×2 array; the extra row and column are guarded off.]]

## Puzzle 5 — Broadcast: the read index and the write index can differ

Everything so far has had a comforting symmetry: thread `(i, j)` reads from position `(i, j)` and writes to position `(i, j)`. Read and write lined up. Broadcast is the first puzzle where they come apart, and it's the most conceptually important of the six.

The output is `2 × 2`. But the inputs are a **column vector** `a` of shape `SIZE × 1` and a **row vector** `b` of shape `1 × SIZE`. Every output cell is the sum of one element from the column and one from the row.

```python
def call(out, a, b, size) -> None:
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, 0] + b[0, j]
```

Read those indices slowly, because this is the whole point. Thread `(i, j)` **writes** `out[i, j]` — same as before. But it **reads** `a[i, 0]` and `b[0, j]`. The write coordinate and the two read coordinates are *different projections of the same thread ID.* The thread throws away `j` when it reads from the column (it only needs the row index `i`), and throws away `i` when it reads from the row (it only needs the column index `j`).

Why does this matter so much? Because it detaches the mental model from a mistake people quietly carry: "one thread owns one element of *everything*." No. **One thread owns one output cell.** What it reads is whatever that output cell depends on — and different threads are free to read the *same* input. Look at what happens: `a[i, 0]` gets read by *every thread in row i*. In a `2 × 2` output, `a[0, 0]` is read by both `out[0,0]` and `out[0,1]`. One value, loaded from memory, feeds two output cells. That's **reuse**.

And reuse is the entire economic argument for the fast on-chip memory we'll spend later articles on. Here it's tiny — one value feeding two cells. But scale it up. In a matrix multiply, one row of A feeds *every* column of the output; one loaded value gets reused across a whole row of results. If every thread re-fetches it from slow main memory, you drown in redundant memory traffic. If instead you load it once into fast [shared memory](shared-memory-l1.html) and let the whole row read it from there, you win.[[sn: This is exactly the redundant-read problem we *deliberately leave unfixed* in the [naive GEMM](gemm-kernel-1-naive.html): thread `(m, n)` reads all of row `m` of A and all of column `n` of B straight from global memory, and every other thread sharing that row or column re-reads the identical values. Broadcast is the two-cell toy version of that waste; [shared memory tiling](gemm-kernel-3-shared-memory.html) is the grown-up fix that turns it into a several-fold speedup.]] Broadcast is small, but it's the first puzzle that is really *about memory* rather than arithmetic — and memory, as Zip already hinted, is where all the performance is won or lost.

[[fig: A hand-drawn reuse figure titled "Broadcast: write once, read shared". CENTER: a 2×2 output matrix `out` with pale-yellow hatch, cells black-labeled (0,0),(0,1),(1,0),(1,1), red dimension labels "↕ size = 2" and "↔ size = 2". DOWN THE LEFT EDGE: a column vector `a` of shape 2×1 with blue diagonal hatch, two stacked cells a[0,0]=3, a[1,0]=5, red shape label "a : size × 1". ACROSS THE TOP EDGE: a row vector `b` of shape 1×2 with green diagonal hatch, two cells b[0,0]=1, b[0,1]=4, red shape label "b : 1 × size". From a[0,0] a single blue dashed arrow FANS OUT to the right into BOTH cells of output row 0, orange callout on it "ONE load → whole row reuses it". From b[0,0] a single green dashed arrow FANS DOWN into BOTH cells of output column 0. A worked cell: out[0,1] shows in purple "= a[0,0] + b[0,1] = 3 + 4 = 7". A small orange star on the index mismatch: "write (i,j)  ≠  read (i,0) & (0,j)". Dashed takeaway box bottom-right: "one output per thread, but inputs are SHARED across a row/column → reuse is WHY shared memory exists". || Puzzle 5. The write index and the read indices are different projections of the same thread; each input value feeds a whole row or column.]]

## Puzzle 6 — Blocks and the global index: the most important line in CUDA

Everything so far fit in a single block. That can't last. A block is capped at **1024 threads**, and it has to fit on a single [Streaming Multiprocessor](streaming-multiprocessor.html) (SM) — the physical processing unit that runs it.[[sn: The exact limit is `blockDim.x * blockDim.y * blockDim.z ≤ 1024` threads per block, and the whole block must fit on one SM because the threads in a block can share memory and synchronize with each other, which only works if they're physically co-located. To cover a big array you launch *many* blocks and let them spread across the GPU's SMs — an H100 has 132 of them. How big to make each block is a real tuning decision; see [occupancy](occupancy.html).]] Your arrays, meanwhile, are millions of elements long. So real kernels always use many blocks, and Puzzle 6 is where that world opens up.

The setup: `SIZE = 9`, but each block has only 4 threads, so you need **3 blocks** to cover 9 elements. And here's the wrinkle that breaks everything you've relied on so far: `threadIdx.x` now only tells you a thread's position *within its own block.* It resets to 0 at the start of every block. So there are three different threads all reporting `threadIdx.x == 1` — one in each block — and if you index with `threadIdx.x` alone, all three of them fight over `out[1]` while `out[5]` never gets written. `threadIdx.x` is no longer a unique index. It's a *local* index.

To recover a unique **global index**, you reconstruct it from three pieces.

```python
def call(out, a, size) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < size:
        out[i] = a[i] + 10
```

This line — `blockIdx.x * blockDim.x + threadIdx.x` — is the single most important formula in CUDA, and it deserves to be read one term at a time, because it's really just a room-and-seat address. `blockDim.x` is how wide a block is — 4 here — like the number of seats in each row. `blockIdx.x` is which block you're in — 0, 1, or 2 — like your row number. Multiply them, `blockIdx.x * blockDim.x`, and you get the offset to the *start* of your block: block 0 starts at 0, block 1 starts at 4, block 2 starts at 8. Then add `threadIdx.x`, your seat within the row, to land on your exact global slot.

Let's do it by hand. Take block 1, thread 2. Global index = `1 * 4 + 2 = 6`. So this thread writes `out[6]`. Take block 2, thread 3: `2 * 4 + 3 = 11`. But our array only has valid indices 0 through 8 — index 11 is off the end! And that's not a bug, it's the guard earning its keep again: with 3 blocks of 4 threads we launched 12 threads for 9 elements, so the last 3 (global indices 9, 10, 11) fall off and get guarded away by `if i < size`. Everything from Puzzle 3 comes back, now that the over-launch is happening across block boundaries.

This combination — the global-index formula *plus* the boundary check — is the **block-guard**, and it is the canonical opening of essentially every 1D CUDA kernel ever written:

```python
i = blockIdx.x * blockDim.x + threadIdx.x
if i < N:
    ...  # do this thread's work
```

[[fig: A hand-drawn "global index" walkthrough titled "Blocks + block-guard: room and seat". Three large rounded-rectangle block containers side by side, black-labeled "block 0","block 1","block 2", each holding 4 small thread cells numbered 0,1,2,3 (red label under the first block: "threadIdx.x — RESETS to 0 in every block"). A green note under each block: "blockDim.x = 4". BELOW the row of blocks, one long flat output array `out` of 9 pale-yellow cells indexed 0..8, red dimension label "size = 9". A blue dashed arrow traces block 1 / thread 2 down to a purple formula box: "i = blockIdx.x·blockDim.x + threadIdx.x = 1·4 + 2 = 6 → writes out[6]". A second RED arrow traces block 2 / thread 3 to "2·4 + 3 = 11" pointing PAST the array to a phantom greyed cell "index 11" with a red warning "✗ off the end → if i<size guards it". Orange banner over the formula: "the most important line in CUDA — block offset + local seat". Dashed takeaway box: "global i = blockIdx·blockDim + threadIdx, THEN guard the tail". || Puzzle 6. Reconstructing a unique global index across three blocks, then guarding the 3 surplus threads at the tail.]]

## The ladder these six puzzles built

Step back and line them up, because a deliberate progression appears — each puzzle is one small idea stacked on the last.

- **Map** — a thread owns one index; the grid of threads is the loop.
- **Zip** — extra inputs don't change the index; and the flop-to-byte ratio is why element-wise ops are memory-bound.
- **Guards** — you over-launch on purpose (`ceil(N / blockDim)`) and let surplus threads cheaply no-op.
- **Map 2D** — index in two dimensions; the guard scales with dimensionality.
- **Broadcast** — the write index and read indices can differ; shared reads mean reuse, which is why shared memory exists.
- **Blocks** — `threadIdx.x` is local, so build the global index, then guard it.

[[fig: A hand-drawn "ladder" summary figure titled "Six puzzles, one reflex". Draw six rungs of a ladder climbing up-right, each rung a small rounded box, black-labeled bottom-to-top: "1 Map", "2 Zip", "3 Guard", "4 Map 2D", "5 Broadcast", "6 Blocks". Beside each rung, a tiny blue handwritten note of its one idea: rung1 "1 thread = 1 index", rung2 "+inputs, same index", rung3 "over-launch + no-op", rung4 "2 axes, guard both", rung5 "read ≠ write → reuse", rung6 "local → global index". At the TOP of the ladder, an orange arrow points to a purple code card holding the two payoff lines: "i = blockIdx.x*blockDim.x + threadIdx.x" and "if (i < N) { ... }". A green note beside the card: "opening lines of the naive GEMM, every activation kernel, every reduction's load phase". Dashed takeaway box: "none of these is FAST — the point was to make indexing automatic, so attention is free for what decides speed: how the bytes move". || The six puzzles form a ladder whose top rung is the two lines that open essentially every CUDA kernel.]]

That last combination — `i = blockIdx.x * blockDim.x + threadIdx.x` followed by `if (i < N)` — is not a puzzle trick. It is the first two lines of the [naive GEMM kernel](gemm-kernel-1-naive.html), of every activation kernel, of every reduction's load phase, of the elementwise epilogue that vLLM fuses onto its matmuls today. You have now written it, by hand, six times. It should feel automatic — that was the entire purpose of the exercise.

And here's the honest scope note. **None of these six kernels is fast.** A pure element-wise add is [memory-bound](the-three-regimes.html) — we did the napkin math back in Zip, one flop per few bytes, roughly a factor of 80 below the point where the H100's compute could keep its memory system busy. It will sit far below the [roofline](roofline-model.html) no matter how cleverly you write it. And we haven't touched the things that actually make kernels fast: shared memory, warps, [memory coalescing](memory-coalescing.html), tiling. But speed was never the point of the first six. The point was to make the *indexing* automatic — to burn the coordinate arithmetic so deep into your hands that, when we start caring about memory traffic and profiling with Nsight Compute, you're not spending any thought on "which element is mine." That whole question is answered reflexively, and your full attention is free for the one thing that decides performance: **how the bytes move.**

In the [next walkthrough](gpu-puzzles-walkthrough-2.html), the puzzles turn to shared memory, pooling, and reductions — and for the first time the one-thread-one-element model has to break, because the threads finally have to *cooperate*: load a tile together, synchronize, and read each other's work. That's where GPU programming stops being bookkeeping and starts being engineering.
