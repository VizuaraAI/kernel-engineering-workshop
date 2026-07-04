In the previous article we lit the tensor cores up with the `wmma` API and jumped past our best CUDA-core kernel — but we ended with a confession: `wmma` was leaving performance on the table, and the profiler knew exactly where. This is the article where we take the training wheels off. We drop from `wmma` down to raw `mma.sync` PTX, get honest about how a fragment is actually laid out across a warp's 32 registers, build a double-buffered `cp.async` software pipeline so the memory system and the tensor cores stop taking turns, and pick the right precision for the job. By the end we are within a hair of `cuBLAS` — and standing at the doorway of Hopper's `wgmma`.

## Why `wmma` was the ceiling

The `wmma` API is a beautiful abstraction and that is exactly its problem. You call `load_matrix_sync`, you get an opaque `fragment`, you call `mma_sync`, and you never once see where the 256 elements of a `16×16` tile physically live. The compiler decides. And because the compiler has to be conservative and portable across every architecture that ever supported `wmma` — the same source compiles for Volta, Turing, Ampere, and Hopper — the code it emits is rarely the code you would write by hand. That portability is bought with fixed fragment layouts and extra shared-memory round-trips that a hand-written `mma.sync` kernel can skip. When I inspected the SASS from the `wmma` kernel, the tensor-core `HMMA` instructions were there, but they were islands in a sea of `LDS` (load-from-shared) and address-arithmetic instructions. The tensor cores were being *fed* badly.

The fix is to stop describing what we want and start saying it. `mma.sync` is the PTX instruction that `wmma` compiles down to anyway — we are just going to write it directly, which means *we* own the fragment layout, *we* own the loads, and *we* can pipeline them.

## `mma.sync`: the warp is the unit

Here is the mental shift that has to click. A tensor-core MMA is not a per-thread instruction and it is not a per-block instruction. It is a **warp-level** instruction: all 32 threads in the warp execute one `mma.sync` cooperatively, and the input and output matrices are *distributed across the registers of those 32 threads*. No single thread holds a whole row or column of anything.

The instruction we start with looks like this:

```
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
```

Read it left to right. `m16n8k16` is the tile shape: the A fragment is `16×16`, the B fragment is `16×8`, and the accumulator C/D fragment is `16×8`. `row.col` says A is row-major and B is column-major in the register layout. The four `f16`s are the types of D, A, B, C respectively.[[sn: The shapes are not free-form. Volta only offered `m8n8k4`; Ampere added the `m16n8k8` and `m16n8k16` family; the exact legal shapes per precision are a table in the PTX ISA, and using an illegal one is a compile error, not a slow path.]] One instruction, executed once by the warp, does a `16×8` block of the output for a `k`-slice of 16 — that is 2,048 multiply-accumulates issued by a single instruction.

The catch is that the operands have to already be sitting in the right registers, in the right threads, in the exact scrambled order the tensor core expects. Getting them there is the entire game.

[[fig: A hand-drawn diagram titled "mma.sync.m16n8k16 — the warp holds the tile". Center: a warp drawn as a horizontal strip of 32 small numbered boxes labeled T0..T31 in black. Above it, matrix A drawn as a 16×16 grid with blue diagonal hatch and red dimension labels "16↕ × 16↔"; matrix B as a 16×8 grid with green diagonal hatch and red dimension labels "16↕ × 8↔"; accumulator D as a 16×8 grid with pale-yellow hatch and red dimension labels "16↕ × 8↔". Thin dashed blue arrows fan out from a handful of A cells to specific threads (T0, T1, T4, T8) with a blue note "each thread holds only a FEW elements — 8 of A, 4 of B". A purple code box reads `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`. Orange emphasis callout: "ONE instruction = 2048 MACs, issued by the whole warp at once". Dashed takeaway box bottom-right: "no thread owns a row — the fragment is smeared across 32 threads' registers". || The `mma.sync` fragment layout. The tile lives in the warp's registers, scrambled across all 32 threads in a fixed pattern.]]

## `ldmatrix`: the loader that scrambles for you

If you tried to shuffle a shared-memory tile into that fixed register pattern by hand — with each thread computing which four `half`s it personally owns — you would drown in address arithmetic. That is precisely the sludge I saw in the `wmma` SASS.

Ampere added an instruction built for exactly this: `ldmatrix`. One `ldmatrix.sync.aligned.m8n8.x4` reads four `8×8` tiles out of shared memory and deposits them into registers already permuted into MMA fragment order, with each of the 32 threads supplying a shared-memory address and receiving its correct handful of elements. It even has a `.trans` variant that transposes on the fly — which is how you get a column-major B fragment out of a row-major shared-memory tile without a separate transpose pass.

The load path per `k`-step becomes: global memory → shared memory (once, shared by the whole block), then `ldmatrix` from shared into registers (per warp), then `mma.sync`. Concretely:

```cpp
// A_smem, B_smem are __shared__ half tiles for this k-slice.
uint32_t a_frag[4], b_frag[2], c_frag[4] = {0};

// each thread hands ldmatrix its own shared-memory address;
// the instruction returns that thread's slice of the fragment.
ldmatrix_x4(a_frag, A_smem_addr_for(threadIdx.x));   // 16x16 of A
ldmatrix_x2(b_frag, B_smem_addr_for(threadIdx.x));   // 16x8  of B, transposed

asm volatile(
  "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
  : "=r"(c_frag[0]), "=r"(c_frag[1]), "=r"(c_frag[2]), "=r"(c_frag[3])
  : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
    "r"(b_frag[0]), "r"(b_frag[1]),
    "r"(c_frag[0]), "r"(c_frag[1]), "r"(c_frag[2]), "r"(c_frag[3]));
```

Two `uint32_t`s per B fragment, four per A fragment — because two `half`s pack into one 32-bit register. This is the level of control `wmma` never gave us. Swapping the hand-managed `ldmatrix` + `mma.sync` inner loop in for the `wmma` calls, and unrolling the `k`-loop, the kernel moves from roughly **25% of `cuBLAS`** to the point where the *next* bottleneck — shared-memory bank conflicts — becomes the thing standing in the way.

## Killing bank conflicts with a swizzle

With the loads explicit, Nsight Compute now points its finger at shared memory. Shared memory on the H100 is **32 banks** wide; a warp gets full bandwidth only when its 32 threads hit 32 distinct banks. Our `8×8` MMA tiles, laid out naively, had all eight rows landing in the same four banks — the profiler reported roughly a **5-way conflict on loads** and a **2.6-way conflict on stores**. Every conflicted access serializes, and the tensor cores stall waiting for operands that shared memory is dribbling out four banks at a time.

The blunt fix is padding — add a few dead columns so consecutive rows offset into different banks. It works, but it wastes precious shared memory (we only get up to **228 KiB** usable per SM, and every wasted byte is a tile we cannot stage). The better fix is a **swizzle**: a permutation of shared-memory indices, computed with a couple of XORs, that spreads each `8×8` tile across all 32 banks with zero wasted bytes.[[sn: The canonical swizzle XORs a few high bits of the row index into the column index — `col ^= (row & mask) << shift`. The exact mask depends on element size and tile width; get it wrong and you either reintroduce conflicts or corrupt the tile. Hopper's TMA can apply an equivalent swizzle in hardware, which is why later kernels stop doing this by hand.]]

[[fig: A hand-drawn two-panel figure titled "Swizzling shared memory". Panel (A) labeled "naive layout — conflicts": a grid of 32 vertical bank columns (numbered 0..31 in green at the top), with an 8×8 tile drawn as red-outlined cells all piling into banks 0-3, and a red warning note "5-way conflict → serialized". Panel (B) labeled "XOR swizzle — conflict free": the same 8×8 tile, but its cells now spread evenly across all 32 banks, each row shifted, blue note "col ^= (row bits) spreads rows across banks". A purple code snippet box: `smem_col ^= ((tid >> 4) & 7) << 3;`. Orange callout between panels: "same bytes, no padding". Dashed takeaway box: "≈2× — from ~25% to ~50% of cuBLAS". || The swizzle. XORing row bits into the shared-memory column index scatters each MMA tile across all 32 banks with no wasted memory.]]

The swizzle is the single biggest jump on the tensor-core half of the ladder: it roughly **doubles throughput, taking us from ~25% to ~50% of `cuBLAS`.** The tensor cores are now being fed cleanly. What is stopping them now is *latency*.

## Double buffering: stop taking turns

Here is the shape of the problem. Our inner loop, per `k`-slice, does two things in sequence: it copies the next tile of A and B from global memory into shared memory, and then it runs the `mma.sync` chain on the current tile. Copy, compute, copy, compute. But the global-memory copy is *slow* — hundreds of cycles of HBM latency — and while it runs, the tensor cores sit idle. Then while the tensor cores run, the memory system sits idle. Two expensive resources, each waiting on the other, neither ever busy at the same time.

The classical answer is a **software pipeline** with **double buffering**. Keep two shared-memory buffers. While the tensor cores chew on buffer 0, kick off the *asynchronous* copy of the next tile into buffer 1. Next iteration, compute on buffer 1 while copying into buffer 0. The two operations now overlap — the memory latency of tile `k+1` is hidden *behind* the compute of tile `k`.

The instruction that makes this possible is `cp.async` (Ampere's asynchronous copy). Unlike a normal load, `cp.async` copies global → shared *without* staging through registers and *without* blocking the thread — you fire it and it runs in the background. You then reach a commit/wait point (`cp.async.commit_group` / `cp.async.wait_group`) only when you actually need the data.[[sn: `cp.async` still runs on the SM's LSU (load/store) pipes, so it is "async" in the sense of not blocking the warp, not in the sense of being free. Hopper's TMA goes further: a single-threaded DMA engine does the whole 2D tile transfer off the warp entirely. That is a section-05 story.]]

```cpp
// prologue: kick off the first tile
cp_async_cg(&A_smem[0], &A_gmem[0], 16);   // 16 bytes = float4 = 8 halfs
cp_async_cg(&B_smem[0], &B_gmem[0], 16);
cp_async_commit();

for (int k = 0; k < K; k += K_TILE) {
    int cur = (k / K_TILE) & 1, nxt = cur ^ 1;
    // prefetch NEXT tile into the other buffer — non-blocking
    if (k + K_TILE < K) {
        cp_async_cg(&A_smem[nxt * TILE], &A_gmem[k + K_TILE], 16);
        cp_async_cg(&B_smem[nxt * TILE], &B_gmem[k + K_TILE], 16);
        cp_async_commit();
    }
    cp_async_wait_group<1>();   // wait until only the newest group is in flight
    __syncthreads();
    // ... ldmatrix + mma.sync on A_smem[cur], B_smem[cur] ...
}
```

Note the copies move `float4`-sized chunks (16 bytes, eight `half`s) so each `cp.async` is one fat, coalesced transaction rather than eight skinny ones. The `wait_group<1>` is the clever part: it blocks only until everything *except* the most recently committed group has landed, which is exactly the double-buffer invariant — the tile we are about to compute on is ready, and the tile we just prefetched is allowed to still be in flight.

[[fig: A hand-drawn pipeline timeline titled "Double-buffered cp.async". Horizontal time axis (red arrow labeled "time →"). Two swim-lanes stacked: top lane "MEMORY (cp.async)" in green, bottom lane "TENSOR CORES (mma.sync)" in blue. In the naive case (small inset labeled "before"), the two lanes alternate with big idle gaps (grey hatched "IDLE" boxes). In the main "after" timeline, the memory lane shows boxes "copy tile 1", "copy tile 2", "copy tile 3" and the tensor lane shows "mma tile 0", "mma tile 1", "mma tile 2" — the copy of tile k+1 sits directly ABOVE the mma of tile k, overlapping. Blue dashed arrows connect "copy tile k+1" down to "mma tile k+1" one step later. Two shared-memory buffers drawn as boxes labeled "buf 0" / "buf 1" with a purple note "ping-pong". Orange callout: "memory latency hidden BEHIND compute". Dashed takeaway box: "≈70% of cuBLAS — the two units are finally busy at the same time". || Double buffering. While the tensor cores compute tile k, `cp.async` prefetches tile k+1 into the other buffer, so HBM latency is hidden behind the math.]]

With the pipeline in place, the two units run concurrently and the profiler's memory-stall time collapses. This is the jump from **~50% to roughly 70% of `cuBLAS`** — the tensor cores are now busy most of the time, and the remaining gap is tuning, not architecture.

## The precision menu

Everything above was written in FP16 for concreteness, but the same `mma.sync` machinery gives you a menu, and picking the right item is free performance and free accuracy.

- **FP16** (`f16.f16`) — inputs and multiply in half precision. Fastest and smallest, but the ~10-bit mantissa and narrow exponent can overflow or lose precision on real training data. Almost always accumulate in FP32 (`.f16.f16.f16.f32`) to keep the sum sane.
- **BF16** (`bf16`) — the same 16 bits but spent as 8 exponent bits instead of FP16's 5. Same tensor-core throughput as FP16, same halved memory traffic, but the wide exponent means it *just works* on deep-learning tensors without loss scaling. This is what modern training actually uses.
- **TF32** (`tf32`) — a 19-bit format the tensor cores use as a drop-in accelerator for FP32 math: it keeps FP32's 8 exponent bits but truncates the mantissa to 10 bits. You feed it FP32 data, it runs on the tensor cores at a large multiple of FP32-CUDA-core speed, and you get "close enough" FP32 results for most workloads.[[sn: TF32 is not an in-memory format — you never store a TF32 array. It is purely what the tensor core does *internally* when it multiplies two FP32 operands in TF32 mode. The H100 does about **989 TFLOP/s** of BF16/FP16 tensor throughput; TF32 runs at roughly half that, still enormously faster than FP32 on the CUDA cores.]]

The rule of thumb: **BF16 inputs, FP32 accumulate** is the default for deep learning; reach for TF32 when a caller genuinely needs FP32 semantics; reserve pure FP16 for inference paths where you have already validated the numerics. Switching precision is mostly a matter of changing the `mma.sync` type suffix and the fragment element type — the pipeline, the swizzle, and the `ldmatrix` structure are identical.

## Where we landed, and the wall ahead

Stack it all together — `mma.sync` with hand-owned fragments, `ldmatrix` loads, a bank-conflict-free swizzle, a double-buffered `cp.async` pipeline, and BF16 inputs with FP32 accumulation — and after tuning the tile shapes the kernel reaches **into the low-to-mid 90s as a percentage of `cuBLAS`**, and on large `8192×8192` problems a well-tuned version of exactly this recipe gets to around **96% of `cuBLAS`.** For a kernel we derived one profiler reading at a time, matching NVIDIA's flagship library to within a few percent is a genuinely satisfying place to stand. That is the same neighborhood the CUDA-core ladder topped out at with warp-tiling at **93.7%**, except now we are doing it through the tensor cores — with real headroom left in the silicon.

And yet the profiler still has a complaint, and it is a familiar one: **register pressure**. The best of these kernels burns something like **165 registers per thread**, which caps occupancy near **18%** and leaves the scheduler reporting "not selected" stalls on a third of its cycles.[[sn: With a **256 KB register file** per SM (65,536 32-bit registers) and a hard ceiling of **255 registers/thread**, 165 registers/thread means only a handful of warps fit per SM. The math is unforgiving: fewer resident warps means fewer independent instruction streams to hide latency behind.]] Every thread in the warp is holding a slice of the A fragment, a slice of B, and a chunk of the accumulator, and there simply are not enough registers to keep many warps resident at once. We are compute-adjacent but occupancy-starved — a classic case of the [three regimes](the-three-regimes.html) tension between doing more per thread and keeping enough threads alive.

This is exactly the wall Hopper was designed to knock down. In the next section we meet `wgmma` — **warp-group matrix multiply-accumulate** — where **128 threads** (four warps) cooperate on one enormous MMA like `wgmma.mma_async.sync.aligned.m64n64k16`, a single instruction issuing **65,536** multiply-accumulates and reading its input operands *straight from shared memory* rather than requiring them pre-staged in registers the way `mma.sync` does. Pair it with the **Tensor Memory Accelerator** (TMA), which does the whole 2D tile copy — swizzle and all — off the warp entirely, and the register-pressure problem that is capping us here largely dissolves. That is where the climb from library-class to genuinely Hopper-native GEMM begins.
