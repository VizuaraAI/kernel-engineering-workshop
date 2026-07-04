Every kernel we have written so far has computed matrix multiplication the honest way: a thread reads two floats, multiplies them, adds the product into an accumulator, and repeats. That is a **SIMT** (Single Instruction, Multiple Threads) GEMM, and by the end of the last ladder we had tuned it hard — up to **93.7% of cuBLAS** with warptiling. But that number is a lie of omission. It is 93.7% of cuBLAS *running on the CUDA cores*, and cuBLAS has not seriously used the CUDA cores for GEMM since 2017. The real library dispatches to the **tensor cores**, matrix-math units that do an entire small matrix multiply-accumulate per instruction and deliver roughly an order of magnitude more throughput than the scalar pipe. So this section starts a new ladder from a new baseline, and the first rung is the gentlest possible on-ramp to that hardware: the **WMMA** (Warp Matrix Multiply-Accumulate) API.[[sn: This ladder rebuilds Alex Armbruster's *"How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores"*. Note that Alex actually skips WMMA and goes straight to raw PTX `mma` — for exactly the reasons we will hit at the end of this article. We start with WMMA anyway, because you should feel the abstraction before you feel its ceiling.]]

We are switching precision too. The tensor core does not multiply FP32 the way the CUDA core does. Its native diet is 16-bit inputs — `half` (FP16) or `bfloat16` — accumulated into an FP32 result. So from here on `A` and `B` are `half`, `C` is `float`, and we compute `C = A · B` with the multiply happening in FP16 and the running sum kept in FP32. That mixed-precision shape is not a compromise we are tolerating; it is the exact shape the silicon was built for.

## What a tensor core actually is

A **tensor core** is a hardware unit inside each **Streaming Multiprocessor** (SM) that computes a fused matrix multiply-accumulate `D = A·B + C` on small fixed-size tiles, in a handful of clock cycles, with a single instruction. The H100 has four tensor cores per SM across its ~132 SMs, and together they are the reason the chip is rated at about **989 TFLOP/s** of BF16 — against a CUDA-core FP32 peak roughly a tenth of that.[[sn: 989 TFLOP/s is the realistic, sparsity-free figure. NVIDIA's headline slides often quote ~1979 TFLOP/s, which assumes 2:4 structured sparsity you almost never have in a dense GEMM. We benchmark against the honest number.]]

The critical structural fact — the one that reshapes how you write the kernel — is that a tensor core operates at **warp scope**, not thread scope. A single `mma` instruction is issued by all `32` threads of a warp together, and the input and output matrices are spread across the registers of the whole warp. No single thread holds a full row or column. The warp is the unit that owns a tile. This is why you can no longer think "one thread, one output element"; you now think "one warp, one output *tile*."

[[fig: A hand-drawn "architecture map" titled "Where the tensor core lives". Left: a large rounded rectangle labeled "H100 die" containing a smaller box "1 of ~132 SMs" in black, with a green margin note "~132 SMs · 989 TFLOP/s BF16". Inside the SM box, four small green squares labeled "Tensor Core ×4" all lit with tiny motion marks, next to a greyed-out row of tiny squares labeled "CUDA cores (idle for GEMM)". A blue dashed arrow points at the tensor cores with a blue note "1 instruction = a whole tile MMA". Right side: a rounded box labeled "1 WARP = 32 threads" drawn as 32 tiny numbered cells, with an orange curved arrow wrapping all 32 into a single big block labeled in orange "the warp owns the tile, not the thread". A red note: "D = A·B + C, per instruction". Dashed takeaway box bottom: "SIMT thinks per-element. Tensor cores think per-tile, per-warp." || The tensor core is a per-SM, warp-scoped matrix unit. The mental shift is from one-thread-one-element to one-warp-one-tile.]]

## The WMMA contract: fragments

CUDA exposes the tensor core to plain C++ through the `nvcuda::wmma` namespace. Its whole design is built to hide the fact that data is scattered across 32 threads' registers. The abstraction it gives you is the **fragment**: an opaque, warp-cooperative container that holds one operand tile. You never index into a fragment. You never know which thread holds which element — that mapping is an undocumented implementation detail, and WMMA guarantees only that if you *load* into a fragment and *feed* it to the matching `mma`, the pieces line up.

A fragment is a template with three roles:

- `fragment<matrix_a, M, N, K, half, row_major>` — a tile of the left operand.
- `fragment<matrix_b, M, N, K, half, col_major>` — a tile of the right operand.
- `fragment<accumulator, M, N, K, float>` — a tile of the output, in FP32.

The `M, N, K` are the **mma shape**: the dimensions of the little multiply each instruction performs. For 16-bit inputs, the shape everyone starts with is `m16n16k16` — a `16×16` chunk of `A` times a `16×16` chunk of `B`, accumulated into a `16×16` chunk of `C`.[[sn: WMMA also allows `m32n8k16` and `m8n32k16` for the same FP16 case — same total work, different tile aspect ratio. Under the hood all three lower to the same `16×8×16` hardware `HMMA` instructions; the WMMA "16×16×16" is a convenience tile the compiler unrolls into two hardware MMAs. You are never as close to the metal as the API's tidy numbers suggest.]] So one warp, executing one `wmma::mma_sync`, computes a `16×16` output tile from a `16×16 × 16×16` product. Note the accumulator fragment carries no layout tag — it lives inside the tensor core's register file and you only get a layout when you store it back out.

There are four operations, and that is nearly the entire API:

- `wmma::fill_fragment(acc, 0.0f)` — zero the accumulator before the K-loop.
- `wmma::load_matrix_sync(a_frag, ptr, ldm)` — the whole warp cooperatively loads a `16×16` tile from memory at `ptr` with leading dimension `ldm` into the fragment.
- `wmma::mma_sync(acc, a_frag, b_frag, acc)` — the tensor core does `acc = a_frag · b_frag + acc`.
- `wmma::store_matrix_sync(ptr, acc, ldm, mem_row_major)` — write the `16×16` result tile back.

Every one of these is a *warp-collective* call. All 32 threads must reach it, with the same arguments. If you wrap one in a divergent `if`, you get undefined behavior, not a compile error.

## The hypothesis

The naive tensor-core kernel is a direct translation of our very first SIMT idea, promoted from elements to tiles: **one warp per `16×16` output tile.** The warp zeroes an accumulator, then walks the `K` dimension in steps of `16`, loading a `16×16` tile of `A` and a `16×16` tile of `B` at each step, issuing one `mma_sync` per step, and finally storing the accumulated tile. It is the "each worker reads a strip of A and a strip of B and marches down K" pattern from [the naive SGEMM](gemm-kernel-1-naive.html) — only now the worker is a warp and the strip is a tile.

Below is the whole kernel. We assume `M`, `N`, `K` are multiples of `16` so there is no ragged edge to guard.

```cpp
#include <mma.h>
using namespace nvcuda;

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

__global__ void wmma_gemm(int M, int N, int K,
                          const half* A, const half* B, float* C) {
    // One WARP per 16x16 output tile. blockDim.x must be a multiple of 32.
    int warpId  = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpRow = warpId / (N / WMMA_N);   // which tile-row of C
    int warpCol = warpId % (N / WMMA_N);   // which tile-col of C

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // March down the K dimension one 16-wide slab at a time.
    for (int k = 0; k < K; k += WMMA_K) {
        const half* a_tile = A + (warpRow * WMMA_M) * K + k;   // row-major A
        const half* b_tile = B + (warpCol * WMMA_N) * K + k;   // col-major B
        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float* c_tile = C + (warpRow * WMMA_M) * N + (warpCol * WMMA_N);
    wmma::store_matrix_sync(c_tile, acc_frag, N, wmma::mem_row_major);
}
```

Two things about the layouts are worth pausing on. `A` is `row_major`, which is natural. But we declare `B` as `col_major` and index it as if it were stored `K × N` transposed — a small trick that lets `load_matrix_sync` read contiguous 16-element runs for both operands. It is the tensor-core equivalent of the coalescing fix from the SIMT ladder: the fragment loader is happiest when each of its cooperating threads reads a contiguous strip.

[[fig: A "tiling walkthrough" in three numbered panels titled "One warp marches down K". Panel (1): three matrices A (M×K, blue diagonal hatch), B (K×N, green diagonal hatch), C (M×N) drawn as rectangles with red dimension labels M, N, K. In C, a single pale-yellow-hatched 16×16 cell is highlighted and labeled in red "one 16×16 tile = one warp". Panel (2): a zoom showing that C-tile fed by a horizontal blue strip of A (labeled "16 rows of A") and a vertical green strip of B (labeled "16 cols of B"), with a red circled "K/16 steps" and a numbered sequence (1)(2)(3) of small 16×16 sub-tiles walking left-to-right across A and top-to-bottom down B, purple note "for k += 16: load, load, mma_sync". Panel (3): a single 16×16 accumulator box labeled in orange "acc_frag stays in registers the whole loop — FP32", with a blue dashed arrow out to C labeled "store_matrix_sync once at the end". Dashed takeaway box: "warp = tile · accumulate in registers · one HBM write per tile". || Kernel 1 on tensor cores. Each warp owns a 16×16 output tile and accumulates across K entirely in the fragment registers, writing to HBM exactly once.]]

We launch it with a flat block of warps. A `256`-thread block is `8` warps, so it covers `8` output tiles; the grid has enough blocks to cover all `(M/16) × (N/16)` tiles.

```cpp
dim3 block(256);
int tiles = (M / WMMA_M) * (N / WMMA_N);
dim3 grid((tiles + 8 - 1) / 8);
wmma_gemm<<<grid, block>>>(M, N, K, dA, dB, dC);
```

## The measurement

It compiles, it is numerically correct against a reference, and — the whole point — it is dramatically faster than anything we built on the CUDA cores. On a large square problem this naive WMMA kernel lands in the low tens of **TFLOP/s** of effective FP16 throughput, comfortably several times the SIMT warptile champion from the previous ladder. We have crossed onto the tensor cores, and even the dumbest possible use of them beats a hard-won scalar kernel. That is the headline: **the floor of the tensor-core ladder is above the ceiling of the SIMT ladder.**

And yet, measured against `cuBLAS` on this same FP16 problem, we are only pulling roughly **8% of the library** — the tensor cores themselves are mostly idle, waiting. The profiler tells us exactly why, and it is the same story as kernel 1 of the SIMT ladder, one level up. Point Nsight Compute at it and **memory** lights up, not the math pipe. Every warp reads its `A` strip and `B` strip straight from **global memory** through `load_matrix_sync`, and just like before, neighboring warps re-read enormously overlapping data. Two warps in the same tile-row of `C` both stream the same `16` rows of `A` from HBM. There is no on-chip staging, so the `3.35 TB/s` of HBM3 is the wall, and a unit rated at nearly a PFLOP/s spends its life stalled on `LDG`.

There is a second, subtler tax. `load_matrix_sync` reading from global memory is not the access pattern that instruction was designed for — WMMA's fragment loaders are tuned to pull from **shared memory**, where the leading dimension is small and the whole tile is already on-chip. Feeding them raw HBM pointers works, but it is the tensor-core analogue of an uncoalesced load: correct, and wasteful.

[[fig: A "SASS listing + diagram" split panel titled "The profiler's verdict". Left column: a handwritten pseudo-SASS listing for the inner loop, black mono lettering — "LDG.E.128 R4, [A_ptr]", "LDG.E.128 R8, [B_ptr]", "HMMA.16816.F32 R16, R4, R8, R16", "IADD K, K, 16", "BRA loop" — with an orange bracket around the two LDG lines labeled in orange "two global loads per ONE mma". Right side: a small memory hierarchy — a big green box "HBM 3.35 TB/s · 80 GB" at the bottom, a fat blue dashed arrow rising straight into a tiny "acc_frag (regs)" box, with the whole `SMEM` level drawn greyed-out and crossed through, red note "SMEM skipped → no reuse". A blue annotation on the HMMA line: "tensor core stalls here, waiting on LDG". Dashed takeaway box bottom-right: "989 TFLOP/s unit fed through a global-memory straw." || The inner-loop SASS: two `LDG`s feed every `HMMA`, straight from HBM with no shared-memory staging. The math unit stalls on loads.]]

## What this tells us to do next

The profile hands us the same to-do list the SIMT ladder did, and we will climb it the same way — one measured step at a time.

- **Stage tiles in shared memory.** Have each thread block cooperatively load a block-sized slab of `A` and `B` from HBM into `SMEM` once, then let all its warps run `load_matrix_sync` out of that fast, reused, on-chip copy. This is the tensor-core version of the shared-memory kernel, and it is where the real climb begins.[[sn: The H100 gives each SM up to `228 KiB` of that `256 KiB` L1/SMEM pool as addressable shared memory across `32` banks — enough to stage genuinely large tiles and keep every one of the four tensor cores fed. Sizing that tile against the register and SMEM budget is most of the game from here.]]
- **Give each warp more than one tile.** A single `16×16` tile per warp is too little work to hide the latency of the loads around it. Just as the SIMT ladder moved from one output element per thread to a `2D` register tile, we will give each warp a grid of accumulator fragments and reuse each loaded `A`/`B` tile across several of them.

And there is a ceiling coming that we should name now. WMMA deliberately hides the fragment layout — which thread holds which element — and that opacity is exactly what stops us at the top. The fastest kernels overlap the *load from shared memory into registers* with the `mma` math, using vectorized loads and hand-scheduled double buffering, and WMMA gives you no handle on that boundary because it owns it. To break through, later rungs drop to the raw `mma.sync.aligned.m16n8k16` PTX instruction and manage the shared-to-register move ourselves — the same reason the source we are following skipped WMMA entirely. But that is a fight for a warp that already has its data on-chip. First we get it there. Next kernel: shared memory.
