[Kernel 1](gemm-kernel-1-naive.html) left us with a profile and a promise. The profile said we were drowning in global-memory traffic at **1.3% of cuBLAS**; the promise was that the very next kernel would roughly quadruple that with *a one-line change* — no shared memory, no tiling, no new algorithm. This is that kernel. It is my favorite one on the whole ladder, because the payoff-to-effort ratio is absurd and because the change looks, at first, like it does nothing at all.

We keep everything from kernel 1: one thread per output element, FP32, square `N × N` matrices, `32 × 32` thread blocks. The math is identical. All we are going to do is change *which* output element each thread computes — a relabeling of threads onto data. That relabeling is the entire optimization, and understanding *why* it works is understanding the single most important access-pattern rule on the GPU.

## The hypothesis: it's the memory transactions, not the flops

From the [three regimes](the-three-regimes.html) we already know kernel 1 is memory-bound — it does about one flop per float it loads, hundreds of times below the ridge. So the win has to be a *memory* win. But there are two very different memory problems, and it is worth being precise about which one we are attacking here.

The first is **reuse**: every thread re-reads a full row of `A` and a full column of `B` straight from HBM, so we fetch `O(N³)` floats to do `O(N³)` flops. That is real, and it is the biggest problem, and we do *not* fix it in this kernel. Fixing reuse needs on-chip staging — that's [shared memory](gemm-kernel-3-shared-memory.html), kernel 3.

The second problem is **coalescing**: even for the bytes we *do* fetch, are we using each memory transaction fully, or are we throwing most of it away? That is what we fix here. The hypothesis is narrow and testable: *the naive kernel issues far more global-memory transactions than it needs to, each mostly empty, and simply rearranging threads will pack those transactions full.*

To see why, we need one fact about how the hardware reads memory.

## What a warp actually does when it loads

Threads do not execute alone. The **Streaming Multiprocessor** (SM) schedules them in groups of 32 called a **warp**, and a warp issues one instruction at a time for all 32 lanes together. When that instruction is a global load, the memory system looks at the 32 addresses the lanes want and services them in units of a fixed size: the GPU moves memory in **sectors** of 32 bytes, and a full **cache line** is 128 bytes — four sectors.[[sn: On Hopper the L2 line is 128 B, split into four 32 B sectors; the memory controller tracks residency and moves data at sector granularity, so a transaction that touches even one byte of a sector pays for the whole 32 B.]]

Here is the rule that decides everything. If the 32 lanes of a warp request 32 consecutive 4-byte floats — a contiguous, aligned 128-byte span — the hardware fuses them into a **single 128-byte transaction** and every byte it moves is a byte a thread asked for. This is **global memory coalescing**. If instead the 32 lanes request 32 floats scattered `N` apart, the hardware cannot fuse anything: it issues a separate 32-byte sector transaction per lane, and each of those sectors delivers 32 bytes to satisfy a single 4-byte request. You use one float out of eight. Seven-eighths of your precious 3.35 TB/s of HBM3 bandwidth is spent hauling bytes you immediately discard.

[[fig: Two side-by-side hand-drawn panels titled "What a warp asks for". Panel (A) labeled "NAIVE — strided" in orange: a horizontal row of 32 small squares representing warp lanes 0..31 (red label "threadIdx 0→31"), each with a blue dashed arrow diving into a tall matrix A drawn with blue diagonal hatch; the arrows land on cells spaced far apart down a column, red annotation "stride = N floats apart". To the right, 32 separate little 32-byte boxes stacked up, orange note "32 transactions, 7/8 of each WASTED". Panel (B) labeled "COALESCED — contiguous" in orange: the same 32 lanes, blue arrows now landing on 32 adjacent cells in one row of a green-hatch matrix B, red annotation "contiguous 128 B span". To the right, ONE fat box labeled "128 B = 32×4 B" in green, orange note "1 transaction, 100% used". Dashed takeaway box bottom: "same 32 floats — 32 transactions vs 1. The addresses are the whole game." || A warp's 32 lanes either fuse into one 128-byte transaction or splinter into 32 near-empty ones. Nothing else about the load changes.]]

Notice what this panel does *not* depend on: it says nothing about how much math each thread does, or how many times we re-read a value. Two kernels can load the exact same set of floats and differ by 8× in bandwidth purely from the *order* the lanes ask for them. That is the lever.

## The naive mapping puts the warp in the wrong direction

Recall how kernel 1 assigned threads to output elements:

```cpp
const uint m = blockIdx.x * blockDim.x + threadIdx.x;  // row
const uint n = blockIdx.y * blockDim.y + threadIdx.y;  // col
```

The subtle thing is which threads land in the same warp. A warp is 32 threads with *consecutive linear indices*, and for a 2D block the linear index is `threadIdx.y * blockDim.x + threadIdx.x`. With a `32 × 32` block, that means a warp is a set of 32 threads sharing one `threadIdx.y` and running across all 32 values of `threadIdx.x`. In other words, **within a warp, `threadIdx.x` varies and `threadIdx.y` is fixed.** The trap is that the naive kernel wires `threadIdx.x` to the *row* `m` — so the fast-varying lane index runs *down* a column of memory, exactly the wrong direction.

Now trace the two loads inside the inner `k` loop, `A[m*N + k]` and `B[k*N + n]`:

- For `A[m*N + k]`: across the warp, `m = ... + threadIdx.x` varies by 1 while `k` is the same for all lanes at a given loop step. So the 32 lanes request `A[m][k], A[m+1][k], …` — addresses `N` floats apart, straight down a column. The hardware cannot fuse them; it splinters into 32 near-empty sector transactions. This is the bandwidth killer.
- For `B[k*N + n]`: across the warp, `n = ... + threadIdx.y` is *constant* and `k` is the same for all lanes, so all 32 lanes request the *same* address in `B`.[[sn: A same-address broadcast isn't wasteful in the sector sense — the hardware broadcasts one value to all lanes from a single sector — so `B` is not where the naive kernel bleeds bandwidth. The damage is entirely on the strided `A` load and the equally strided store to `C[m*N + n]`, both of which march down a column with stride `N` as `threadIdx.x` sweeps the warp.]]

The upshot, confirmed by the profiler, is that the naive kernel sustains a pitiful **~15 GB/s** of global-memory throughput — a rounding error against 3.35 TB/s. The lanes are pointed the wrong way relative to how DRAM wants to be read.

## The one-line remap

Here is the whole change. Instead of taking `m` and `n` from the 2D `threadIdx.y` / `threadIdx.x`, we flatten the block to a 1D index and split it *ourselves*, so that the fast-varying direction of the warp runs along the output column:

```cpp
const int m = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);  // row
const int n = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);  // col
```

That is it. `BLOCKSIZE` is 32, and we now launch with a 1D block of `32 * 32 = 1024` threads. Read the two lines carefully, because this is the part that puzzles people (it puzzled me): we have *swapped which arithmetic feeds the column*. The column `n` now comes from `threadIdx.x % BLOCKSIZE`, and `%` is exactly the operation that cycles fastest as `threadIdx.x` increments. So within a warp — 32 consecutive `threadIdx.x` from, say, 0 to 31 — the row `m` (`threadIdx.x / 32`) stays *constant* and the column `n` (`threadIdx.x % 32`) sweeps `0, 1, 2, …, 31`.

[[fig: A tiling walkthrough with three numbered panels titled "Remapping the warp onto the tile". Panel (1): a purple code box showing `m = ... + threadIdx.x / 32` and `n = ... + threadIdx.x % 32`, with a red circle (1). Panel (2): a 32×32 grid representing one output tile of C (pale-yellow hatch), with a single warp drawn as 32 numbered lanes laid out HORIZONTALLY along the top row — lane 0 in the top-left cell, lane 31 at the right end of the same row; blue annotation "one warp = one contiguous row, threadIdx.x % 32 sweeps the column index", red labels "m constant" down the side and "n = 0→31" across the top; red circle (2). Panel (3): a zoom of matrix B (green hatch) row k, showing the same 32 lanes' arrows landing on 32 adjacent floats, green note "128 B, one transaction", and matrix C's tile row getting 32 adjacent stores, orange note "the STORE to C coalesces now too". Dashed takeaway box: "warp runs along n → loads of B and stores of C both become contiguous. Same threads, same math, different labels." || The remap lays each warp along a contiguous output row, so the fast-varying lane index drives contiguous columns. The math each thread does is unchanged.]]

The effect chains through all three arrays, and it is exactly the mirror image of the naive kernel. The load `A[m*N + k]` — the strided, splintered access that was killing us — is now a genuine broadcast: every lane in the warp shares row `m`, so all 32 want one address, which the hardware serves from a single sector. The load `B[k*N + n]`, which used to be a broadcast, now has `n` sweeping across the warp, so the 32 lanes read 32 adjacent floats — one full 128-byte transaction. And the store `C[m*N + n]` writes 32 adjacent elements per warp, coalesced where it used to stride. Every global access the warp issues is now either one full 128-byte transaction or one broadcast — nothing splintered.

I want to stress how little happened here to be sure it lands: we did not add a byte of shared memory, we did not change the arithmetic intensity, we still re-read `A` and `B` from HBM `N` times over. We *only* changed the addresses each warp presents to the memory system on the same loads it was already doing.[[sn: You can get the identical coalesced layout by keeping a 2D block and just reading `threadIdx.y` as the column and `threadIdx.x` as the row — the "swap x and y" trick. The explicit `/` and `%` on a 1D block make the warp-to-address mapping impossible to misread, which is why I prefer it in a teaching kernel.]]

## The measurement

Point Nsight Compute at the coalesced kernel and the story is exactly the one we predicted. Global-memory throughput jumps from **~15 GB/s to ~110 GB/s** — the transactions are now full instead of one-eighth full, and the DRAM controller is finally being fed. The memory workload analysis still shows us memory-bound (we haven't touched reuse), but we are moving bytes an order of magnitude more efficiently for each transaction we issue.

Throughput on the benchmark rises from the naive kernel's **~300 GFLOP/s** to about **~1990 GFLOP/s** — from **1.3%** to **8.5% of cuBLAS**.[[sn: Simon Boehm reports 309.0 → 1986.5 GFLOP/s on an A6000; the exact figure moves with the card, but the ~6.4× jump from coalescing alone is remarkably stable, because it comes from a hardware granularity rule, not from tuning.]] A **6.4× speedup from relabeling threads.**

[[fig: A SASS-listing-plus-diagram figure titled "Same loop, different addresses". On the LEFT, a handwritten assembly column showing the inner-loop SASS for both kernels stacked: a purple block labeled "NAIVE" with lines `LDG.E R4, [A]` / `LDG.E R5, [B]` / `FFMA R0, R4, R5, R0`, and below it a purple block labeled "COALESCED" with the IDENTICAL three lines, an orange bracket linking them with the note "inner loop SASS is the same — no new instructions". On the RIGHT, two small bar-gauges: a short red bar labeled "15 GB/s" over a long faint outline of the full scale, and a taller green bar labeled "110 GB/s", with a green annotation "3.35 TB/s HBM3 = full scale" and a red note "still memory-bound — reuse untouched". Numbered circles (1) by the SASS, (2) by the bars. Dashed takeaway box: "the win is entirely in the ADDRESSES, not the instructions — 7.3× GMEM throughput for free." || The inner-loop instructions are byte-for-byte the same across both kernels. The 7× jump in achieved bandwidth comes purely from the addresses each warp presents.]] No new memory, no new instructions in the inner loop — the SASS for the loop body is essentially identical to kernel 1; only the address arithmetic outside the loop changed. That is the entire point of the lesson: the naive kernel was leaving 7/8 of its bandwidth on the floor, and the hardware handed it back the moment the warp faced the right direction.

## What the profiler tells us to do next

We are still memory-bound, and the profiler is blunt about why: **110 GB/s is still 30× short of the 3.35 TB/s the HBM3 can deliver, and even that is the wrong number to chase**, because the deeper problem is that we are hitting HBM at all for data we already read. Coalescing made each trip to global memory efficient; it did nothing about the fact that we take `O(N³)` trips. Element `A[m][k]` is still fetched by all 32 threads that share row `m`'s tile, `N` times over the course of the launch.

So the regime playbook points at the same lever it always does when you're memory-bound and out of easy access-pattern wins: **stop going to HBM.** Stage a tile of `A` and a tile of `B` into on-chip **shared memory** once, then let a whole block of threads reuse them from there — turning `N` global reads of each element into one. The H100 gives us up to `228 KiB` of SMEM per SM to spend on exactly this, and that is where the real climb starts.

That is [kernel 3](gemm-kernel-3-shared-memory.html), and it takes us from **8.5%** to **12.8% of cuBLAS** — the first kernel where we finally attack reuse instead of access pattern, and the first hint of the on-chip data-staging discipline that carries the rest of the ladder all the way to **93.7%**.
