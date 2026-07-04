By the end of the last kernel we were at **93.7% of cuBLAS** with warptiling, and every remaining percent gets harder to find. When you are that close, the profiler stops complaining about the obvious things — coalescing is fixed, the arithmetic intensity is high, the register file is packed with a fat `8×8` accumulator tile per thread. So I did what you always do when the easy wins are gone: I pointed Nsight Compute at the inner loop and asked the question from [the three regimes](the-three-regimes.html) one more time — *what is it waiting on?*

The answer was memory latency. Not bandwidth — we are reusing tiles beautifully by now — but *latency*. At the top of every `K`-tile iteration the whole block stalls, waiting for the next slab of `A` and `B` to arrive from **global memory** (GMEM) into **shared memory** (SMEM) before it can start the math. The tensor-adjacent FMA units sit idle for a few hundred cycles, every iteration, with nothing to chew on. This article is about hiding that stall.

## The hypothesis: compute on tile *k* while tile *k+1* is in flight

Here is the shape of the current inner loop, in words. For each step along the `K` dimension we do two things, strictly in order:

1. **Load** a `BM×BK` slab of `A` and a `BK×BN` slab of `B` from GMEM into a single SMEM buffer, then `__syncthreads()`.
2. **Compute** — every thread reads its slivers out of that SMEM buffer and updates its `8×8` register accumulator.

Step 2 cannot begin until step 1 finishes, because they touch the same buffer. That serialization is the whole problem. The GMEM load has a latency of a few hundred cycles; the compute is fast; so for a good chunk of every iteration the SMs are stalled on `LDG` returns.[[sn: With enough resident warps the scheduler can sometimes hide this by switching to another warp, but at the occupancy a heavily register-tiled GEMM runs at — often only a handful of warps per SM — there simply aren't enough other warps to cover a multi-hundred-cycle GMEM round trip. Latency hiding by occupancy has run out of runway.]]

The fix is a classic **software pipeline**. Use *two* SMEM buffers instead of one. While the FMA units compute on buffer `A` (tile `k`), issue the loads for tile `k+1` into buffer `B`. When compute finishes, the next tile has already landed — swap the buffers and go again. The load latency is now *hidden underneath* the compute of the previous tile instead of stacked in front of it. This is **double buffering**, also called **ping-pong buffering**.

[[fig: A pipeline timeline titled "Double buffering: hide the load under the compute". Top half labeled (A) SINGLE BUFFER: a horizontal time axis with alternating boxes — a blue box "LOAD k" then a pale-yellow box "COMPUTE k" then "LOAD k+1" then "COMPUTE k+1", strictly sequential, with a red squiggly bracket under each LOAD box labeled "STALL — SMs idle ~hundreds of cycles". Bottom half labeled (B) DOUBLE BUFFER: two parallel lanes. Top lane "COMPUTE" has back-to-back yellow boxes "COMPUTE k", "COMPUTE k+1", "COMPUTE k+2" with no gaps. Bottom lane "ASYNC LOAD" has blue boxes "LOAD k+1", "LOAD k+2" each drawn shifted left so it sits UNDER the previous compute box, connected by a thin blue curved dashed arrow labeled "runs in the shadow". A vertical green spec note off to the side reads "GMEM latency ~400 cyc". Orange emphasis note between the lanes: "load k+1 runs WHILE computing k". A dashed takeaway box: "steady state = zero load stalls; only the first tile's load is exposed (the prologue)". || Single-buffered GEMM serializes load and compute. Double buffering runs the next load in the shadow of the current compute.]]

Notice one honest detail in that figure: the very first load — the **prologue** — is still exposed. You have to fill the pipe before you can drain it. But the prologue is one tile out of `N/BK` of them, so its cost amortizes to nothing on a real matrix.

## First attempt: two buffers, still synchronous

The naive way to build this doesn't need any new instructions at all. Allocate two SMEM buffers, and manually reorder the loop so that you prefetch tile `k+1` into the *other* buffer before computing tile `k`:

```cpp
__shared__ float As[2][BM * BK];
__shared__ float Bs[2][BK * BN];

int cur = 0;
load_tile(As[cur], Bs[cur], /*k=*/0);   // prologue
__syncthreads();

for (int k = 0; k < N; k += BK) {
    int nxt = cur ^ 1;                    // ping-pong the buffer index
    if (k + BK < N)
        load_tile(As[nxt], Bs[nxt], k + BK);   // prefetch NEXT tile
    compute_tile(As[cur], Bs[cur]);            // work on CURRENT tile
    __syncthreads();
    cur = nxt;
}
```

The `cur ^ 1` is the ping-pong: one XOR flips between buffer `0` and buffer `1` with no branch.[[sn: salykova's kernel does exactly this XOR trick but on the SMEM *addresses* rather than an index — the copy toggles with `sts_a_addr ^= 8192` (and `^= 4096` for the smaller `B` half), flipping the base pointer between the two halves of the double buffer. The `8192` works precisely because the buffer is a power-of-two-aligned block, so one XOR does the whole swap. Same idea, one instruction, no register spent on a loop-carried index.]] It cost us `2×` the SMEM — with `BM=BN=128, BK=8` that is still well inside the `228 KiB` an H100 SM can give to SMEM, so we have room.

[[fig: A tiling walkthrough titled "Ping-pong: two SMEM buffers, one XOR". Center: a tall red matrix labelled "A (N×K)" with two consecutive horizontal slabs highlighted — slab k in blue hatch labelled "tile k" and slab k+1 in a lighter blue hatch labelled "tile k+1". To the right, an SMEM region drawn as two stacked boxes: buffer 0 (blue hatch) and buffer 1 (green hatch), each with a green spec note "8 KiB". Numbered circles show the reading order: circle (1) a solid blue arrow from "tile k" into buffer 0 with note "computing on this now"; circle (2) a blue DASHED arrow from "tile k+1" into buffer 1 labelled in orange "async prefetch — in flight"; circle (3) a purple curved arrow looping the two buffers labelled `cur ^= 1` with purple note "swap: no branch, 1 instruction". A red dimension arrow marks "↔ BK = 8". Dashed takeaway box: "compute reads buffer cur; loads write buffer cur^1; they never collide". || The two shared-memory buffers alternate every K-step. Compute drains one while the async copy fills the other; a single XOR flips which is which.]]

And it helped, a little. But the profile was disappointing: the `load_tile` still issues ordinary `LDG` (global load) instructions that write into *registers*, and only then `STS` (store-to-shared) instructions that copy register → SMEM. That round trip — GMEM → register → SMEM — burns registers we desperately need for the accumulator, and worse, the `LDG` results still have to *retire into registers the same warp is using*, so the compiler can't move the loads as far ahead as we'd like. The overlap is real but leaky. We are still, in effect, waiting.

## The real tool: `cp.async` — bypass the register file

Ampere introduced the instruction this pattern was crying out for: `cp.async`, an **asynchronous copy** that streams data *directly from global memory into shared memory without passing through registers or the L1 cache line's worth of the register file at all*.[[sn: In SASS this shows up as the `LDGSTS` instruction — "load global, store shared" — a single fused async copy. When you see `LDGSTS` in your disassembly, `cp.async` fired; if you see `LDG` followed by `STS`, the compiler fell back to the synchronous path and you've lost the overlap.]] You fire it and it runs in the background on the memory pipe while your warp keeps issuing math.

The PTX comes in two flavors that differ only in caching policy: `cp.async.ca.shared.global` caches the line in L1/L2 on the way through, and `cp.async.cg.shared.global` (the `cg` = "cache global") bypasses L1 and caches only at L2 — often the better fit for GEMM operands you'll touch exactly once per tile and never revisit. Which one actually wins is empirical, not obvious: salykova's kernel ships the `ca` variant, because on that GPU and tile shape caching the line paid off. Treat `cg` as the reasonable default for streamed-once operands, then measure. In CUDA C++ you rarely write the PTX by hand; you reach for the pipeline primitives:

```cpp
// prefetch tile k+1 into the OTHER buffer — asynchronously
__pipeline_memcpy_async(&As[nxt][row], &A_gmem[...], sizeof(float4));
__pipeline_memcpy_async(&Bs[nxt][row], &B_gmem[...], sizeof(float4));
__pipeline_commit();                 // seal these copies into a "group"

// ... meanwhile, compute on the CURRENT buffer ...
compute_tile(As[cur], Bs[cur]);

__pipeline_wait_prior(0);            // block until the prefetch group lands
__syncthreads();
```

The two verbs are the whole story. `__pipeline_commit()` (PTX `cp.async.commit_group`) draws a line under the async copies you've issued so far and bundles them into a numbered group. `__pipeline_wait_prior(N)` (PTX `cp.async.wait_group`) blocks the warp until all but the most recent `N` groups have completed. With a two-stage pipeline you commit one group per iteration and `wait_prior(0)` to drain the previous one right before you need it — the copy of tile `k+1` was launched a full compute-tile ago, so by the time you wait, it has almost always already finished. The wait is free; the latency was spent under the compute.

[[fig: A SASS-plus-diagram figure titled "cp.async bypasses the register file". Left column, handwritten SASS-style listing in purple: top block labelled "SYNCHRONOUS (old)" shows three lines "LDG.E R4, [A_gmem]" / "LDG.E R6, [B_gmem]" / "STS [As], R4" with a red note "GMEM → REGISTER → SMEM (2 hops, burns regs)". Bottom block labelled "ASYNC (cp.async)" shows one line "LDGSTS.E.128 [As], [A_gmem]" with a green note "GMEM → SMEM (1 hop, 0 regs)". Right side: a small memory pyramid — a green "HBM 3.35 TB/s" slab at the bottom, a red "REGISTER FILE 256 KB/SM" box in the middle with a big red ✗ struck through the path into it, and a blue "SMEM 228 KiB" box at top. A curved blue dashed arrow labelled "cp.async" arcs from HBM straight up to SMEM, skipping the register box entirely. Orange callout: "the register file stays free for the 8×8 accumulator". Dashed takeaway box: "LDGSTS in the SASS = the async copy fired". || The async copy fuses the global-load and shared-store into one instruction that never touches the register file — leaving those registers for the accumulator and letting the copy run in the background.]]

There are two payoffs here, and they compound. First, the **latency hiding**: the copy genuinely runs concurrently with the FMAs, so the steady-state loop has no exposed load stall. Second, the **register relief**: because the data never lands in registers, the compiler stops spilling and every one of those `255`-per-thread registers is available for the accumulator. On a register-starved warptile kernel that second effect is sometimes worth as much as the first.

## The measurement

Swapping the synchronous double buffer for a `cp.async`-driven, `float4`-vectorized two-stage pipeline is the last big structural change on the ladder. The inner loop now looks like the bottom lane of that first figure: compute runs flat-out, loads run in its shadow, and the profiler's "long scoreboard stall" — its name for a warp parked waiting on a memory dependency — collapses toward zero on the steady-state iterations.

The number: this pushes us past warptiling's `93.7%` to roughly **96% of cuBLAS**, and with careful autotuning of `BK` and the vector width the well-tuned open-source GEMMs land within a few percent of the library across the useful shape range.[[sn: salykova reports the `cp.async` kernel actually *beating* cuBLAS by 3–4% at locked clocks on the `128×128×8` tile — but at ~12% higher power, which throttles it back below cuBLAS on large matrices once the clocks are unlocked. "Faster than cuBLAS" and "faster than cuBLAS at the same power budget" are very different claims; the honest one is the second.]] We have gone from a naive kernel at **1.3%** to a hand-written kernel that trades blows with a library NVIDIA has tuned for fifteen years — and every step was a measurement, not a guess.

## Where cuBLAS (and Hopper) get their overlap

So how does the library stay ahead? The same idea, with better hardware. cuBLAS uses *deeper* pipelines — three, four, or more stages instead of two — so that several tiles are in flight at once and even a very long tail of the GMEM latency distribution stays hidden. More stages cost more SMEM, which is exactly why Hopper's `228 KiB` SMEM budget matters: it buys you pipeline depth.

And on Hopper (`sm_90a`) the async-copy machinery gets promoted from an instruction to a dedicated engine. The **Tensor Memory Accelerator** (TMA) issues an entire multi-dimensional tile copy from a single thread, computes all the addresses in hardware, applies the shared-memory **swizzle** for bank-conflict-free layout automatically, and signals completion through an `mbarrier` — so `cp.async`'s per-thread address arithmetic disappears entirely.[[sn: The Hopper trio that supersedes hand-rolled double buffering — TMA for the copies, thread-block clusters + **distributed shared memory** (DSMEM) for cross-SM tile sharing, and `wgmma` for warp-group-wide tensor-core MMAs — is `sm_90a`-only. A `cp.async` pipeline is the portable version of the same idea; TMA is what you graduate to when you commit to Hopper.]] Paired with `wgmma`, the tensor cores consume tiles as fast as the TMA can stage them, and the whole GEMM becomes a producer-consumer pipeline where the copy engine and the math engine never wait on each other.

That is the next rung, and it is a big one: rewriting the pipeline around TMA and `wgmma` is less "one more optimization" and more "a different kernel." But the intuition is the one we just built by hand. Double buffering with `cp.async` is the honest, portable core of every fast GEMM; TMA is that same idea cast into silicon. Once you have felt the load stall vanish under the compute in your own kernel, the Hopper version is just the same trick with a bigger engine and a better view.
