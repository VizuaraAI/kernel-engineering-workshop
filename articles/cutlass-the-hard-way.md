By the end of [the GEMM ladder](gemm-recap-the-ladder.html) we had a hand-written kernel that reached **93.7% of cuBLAS** on an H100, and it took eight kernels of increasingly baroque indexing to get there. Every one of those kernels was a fixed choice: this tile size, this thread mapping, this vectorization width, baked into `#define`s and `constexpr`s. That was the point — you learn the machine by nailing every parameter to the wall and reading the profiler.

But nobody ships that kernel. NVIDIA's own `cuBLAS` doesn't. What it ships is a *generator* — a set of C++ templates that, given a tile shape and a target architecture, emit the right kernel. That generator is **CUTLASS** (CUDA Templates for Linear Algebra Subroutines), and the honest reason to learn it is this: our warptiling kernel and a real CUTLASS kernel are the *same algorithm*. The difference is that CUTLASS parameterizes the choices we hard-coded, and — crucially — it knows how to reach the Hopper-only instructions (`wgmma`, `TMA`, thread-block clusters) that our hand-CUDA never touched. This article is the bridge: same vocabulary, industrial machinery.[[sn: I'm following Kapil Sharma's *"Learn CUTLASS the hard way,"* which walks the 2.x `device::Gemm` API up from a naive kernel. The framing here — mapping CUTLASS's levels onto the ladder we already climbed — is the through-line I care about.]]

## The same tiling, one level up

Recall the structure of [kernel 8, warptiling](gemm-kernel-8-warptiling.html). There were four nested tiles, and every performance decision lived in the sizes of those tiles:

1. A **threadblock tile** `BM × BN × BK` — the chunk of `C` one block owns, staged through shared memory.
2. A **warp tile** — the sub-chunk of that block one warp owns, held live enough to feed the math units.
3. A **thread tile** — the `TM × TN` register block each thread accumulates.
4. The **MMA / instruction shape** — the actual `m × n × k` the hardware multiply-accumulate consumes in one issue.

CUTLASS has *exactly these four levels*, and it gives each one a namespace. This is not a coincidence or a rough analogy — the CUTLASS authors and the ladder we climbed are describing the same physical decomposition of a GEMM onto an [SM](streaming-multiprocessor.html), so the layers line up one-to-one.

[[fig: Hand-drawn Excalidraw-style two-column "Rosetta stone" diagram on pure white, fine black ink, hand-lettered Virgil-style labels. Black title top-center "Same tiling, two vocabularies". LEFT column header in orange handwriting "Our ladder (hand-CUDA)" over a vertical stack of four nested hand-drawn rounded rectangles, outermost to innermost: a big pale-yellow-hatch box labeled black "threadblock tile" with its dimensions in RED "BM×BN×BK"; inside it a blue-hatch box "warp tile"; inside that a green-hatch box "thread tile" with RED dims "TM×TN"; innermost a small red-outlined box "MMA" with RED dims "m×n×k". RIGHT column header in orange "CUTLASS" over the identical four nested boxes, each relabeled in PURPLE handwriting (code): "device::Gemm  ·  kernel", "CollectiveMainloop  (threadblock)", "warp::Mma", "arch::Mma  (tensor core)". Four long thin curved dashed arrows connect each left box to its right twin, each arrow tagged with a small BLUE handwritten note: "same chunk of C", "same warp ownership", "same register block", "same hardware issue". Hand-drawn numbered circles (1)(2)(3)(4) down the left stack marking top-to-bottom reading order. A dashed rounded takeaway box at the bottom in black reads: "CUTLASS parameterizes what we hard-coded — same four levels, pulled into template params." Flat, no shadows, no gradients, wide composition, generous white space. || The CUTLASS hierarchy is the warptiling kernel with its knobs pulled out into template parameters.]]

Reading the levels top-down:

- **`cutlass::gemm::device::Gemm`** — the device level. This is the object you actually construct and launch from host code. It owns the grid launch, the workspace, and the choice of everything below it. It is the one-liner that hides the other three levels.
- **Kernel level** — orchestrates a single threadblock's life: allocating shared memory, driving the main loop, running the epilogue. In CUTLASS 3.x this is the `CollectiveMainloop` plus `CollectiveEpilogue` pair.
- **Warp level** (`cutlass::gemm::warp::Mma…`) — how the 32 threads of a warp cooperatively feed the tensor cores. This is where fragment layouts and register mapping live.
- **Arch / MMA level** (`cutlass::arch::Mma`) — a thin wrapper over the literal hardware instruction: `mma.sync` on Ampere, `wgmma` on Hopper.

The whole point is that you pick tile sizes at the top and CUTLASS *specializes the templates all the way down* to instructions that are correct for your architecture. You get the kernel-8 algorithm without writing kernel-8's index arithmetic.

## The device-level one-liner

Here is the entire "kernel" in the CUTLASS 2.x `device::Gemm` API — a type definition and a call.

```cpp
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,      // A: element, layout
    cutlass::half_t, cutlass::layout::ColumnMajor,   // B
    float,           cutlass::layout::RowMajor,      // C
    float,                                           // accumulator
    cutlass::arch::OpClassTensorOp,                  // use tensor cores
    cutlass::arch::Sm90,                             // target Hopper
    cutlass::gemm::GemmShape<128, 128, 32>,          // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,            // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>              // MMA instruction shape
>;

Gemm gemm_op;
gemm_op({ {M, N, K}, {dA, lda}, {dB, ldb}, {dC, ldc}, {dC, ldc}, {alpha, beta} });
```

Look at what the three `GemmShape` lines are. `<128, 128, 32>` is `BM × BN × BK` — the same threadblock tile we spent [kernel 5](gemm-kernel-5-2d-blocktiling.html) hand-tuning. `<64, 64, 32>` is the warp tile from [kernel 8](gemm-kernel-8-warptiling.html). `<16, 8, 16>` is the instruction shape — the tensor-core `m × n × k` we'd otherwise emit by hand via `mma.sync`.[[sn: The `<16, 8, 16>` shape is an Ampere/Ada `mma.sync` tile (Kapil's original uses the `16×16×16` WMMA shape; either is a legal synchronous tensor-core op). On true Hopper (`Sm90`) you'd instead go through the 3.x collective API to reach `wgmma`, whose instruction shape is a *warpgroup* op — `m64nNk16`, with `N` a multiple of 8 up to 256 — where a group of four warps issues one asynchronous matrix op. CUTLASS swaps the arch-level template accordingly; the device-level line barely changes.]] Every knob we nailed to the wall in the ladder is now a template argument. That is the whole trade: we gave up the pedagogy of writing the loop, and we bought the ability to change the tile shape by editing one number.

Choosing those numbers well is not free — it's the same autotuning problem from [kernel 7](gemm-kernel-7-autotuning.html), just moved into template space. CUTLASS ships a profiler that sweeps the configuration lattice for you, but the search space is enormous and the wrong tile can halve your throughput.

## CuTe: layouts as first-class algebra

The device API above is CUTLASS 2.x. The reason 3.x exists — and the reason CUTLASS can target Hopper and Blackwell at all — is a small layout library underneath it called **CuTe** (CUDA Tensors). CuTe is the piece worth slowing down for, because it is genuinely a new idea and not just more templates.

In hand-CUDA, a "layout" is implicit: you write `A[m * N + k]` and the row-major mapping lives in your head and your arithmetic. CuTe makes the mapping an *object*. A **Layout** is a pair of a **Shape** and a **Stride**, both nested integer tuples, and it is literally a function from a logical coordinate to a linear memory offset.

```cpp
// a 4-row, 8-col row-major tile:  offset(i,j) = i*8 + j
auto layout = make_layout(make_shape(4, 8), make_stride(8, 1));

// a Tensor pairs a pointer with a layout:
auto A = make_tensor(ptr, layout);
A(2, 3);   // == ptr[2*8 + 3] — indexing is layout application
```

Row-major versus column-major stops being a special case and becomes "which stride is `1`." That sounds cosmetic until you realize what it buys: because layouts are *composable algebra*, you can express "tile this global tensor into `128 × 32` blocks, then partition each block across warps, then across threads" as a *product of layouts* rather than as a spiral of `blockIdx`/`threadIdx` arithmetic.[[sn: This is CuTe's `local_tile` / `local_partition` machinery. The mental model that finally made it click for me: a Layout is a coordinate-space *reshaping function*, and the tiling hierarchy is just function composition. Once the layout is right, the copy and the MMA are almost boilerplate.]] The hierarchical tiling we hand-derived across four ladder kernels becomes a handful of layout compositions.

[[fig: Hand-drawn Excalidraw-style diagram on pure white, fine black ink, hand-lettered Virgil-style labels, no typeset fonts. A "layout as a function" walkthrough titled in black "CuTe: Shape × Stride → offset". On the left, a hand-drawn 4×8 grid of cells with red row indices 0-3 down the side and red column indices 0-7 across the top; one cell (i=2, j=3) is highlighted pale-yellow. Purple handwritten annotations at top-left list "Shape = (4, 8)" and "Stride = (8, 1)". A blue dashed arrow leaves the highlighted cell to a hand-lettered equation box: "offset = 2·8 + 3 = 19", and a second dashed blue arrow points to a flat 1-D memory strip below (32 little boxes) with box #19 shaded pale-yellow to match. To the RIGHT, a small second panel labeled in orange "compose = tile": the same 4×8 grid overlaid with a bold 2×2 partition, purple note "layouts multiply → hierarchical tiles for free". Dashed takeaway box: "a Layout is a function; tiling is composition." || A CuTe Layout is a Shape paired with a Stride — a pure function from logical coordinate to linear offset. Composing layouts gives you the tiling hierarchy without index arithmetic.]]

## What CUTLASS reaches that we couldn't

Everything so far could be dismissed as "nicer packaging for kernel 8." The part that is genuinely beyond our hand-CUDA is the Hopper feature set, and this is where CUTLASS earns its keep.

Our ladder loaded shared memory with plain vectorized `float4` loads, or at best `cp.async` in [the double-buffering kernel](gemm-double-buffering-cpasync.html). On Hopper, CUTLASS's `CollectiveMainloop` instead uses the **Tensor Memory Accelerator** (TMA) — a dedicated hardware unit that copies whole multi-dimensional tiles from global memory into shared memory asynchronously, addressed by a descriptor rather than by per-thread arithmetic. One thread kicks off a bulk copy of a `128 × 32` tile; the rest of the warpgroup does other work while it lands.[[sn: TMA also handles the reverse (SMEM→global) and the swizzling needed to keep the destination [bank-conflict](bank-conflicts.html)-free. Writing a correct swizzle by hand is miserable; CuTe layouts encode it declaratively and TMA honors it. See [Hopper TMA](hopper-tma.html) for the descriptor mechanics.]] For the math, it issues `wgmma` — a **warpgroup matrix-multiply-accumulate** where four warps (128 threads) cooperate on one *asynchronous* tensor-core op, reading operands straight from shared memory. And the whole threadblock tile can be a **thread-block cluster** spanning several SMs that share data through distributed shared memory.

Here's the thing: you could, in principle, write all of that by hand. `wgmma`, TMA descriptors, and cluster launch are all exposed in PTX. But the async pipeline — issue the next TMA load, wait on the previous one, feed `wgmma`, drain the accumulator — is a warp-specialized producer/consumer state machine that is *extremely* easy to get subtly wrong, and CUTLASS's `CollectiveMainloop` is a battle-tested implementation of exactly that dance.[[sn: This is the [warp-specialization](hopper-wgmma-warp-specialization.html) pattern: some warps in the group are dedicated *producers* driving TMA, others are *consumers* driving `wgmma`, coordinated through shared-memory barriers. It's the single biggest reason a from-scratch Hopper GEMM is so much harder than an Ampere one.]]

[[fig: Hand-drawn Excalidraw-style pipeline-timeline diagram on pure white, fine black ink, hand-lettered Virgil-style labels, flat with no shadows or gradients. Black title "Hopper mainloop: what CUTLASS orchestrates". A horizontal time axis (arrow pointing right, red "time →"). Two swimlanes stacked vertically, each a row of boxes with overlap shading. TOP lane labeled in blue "Producer warps — TMA": a sequence of pale-yellow boxes "load tile k=0", "load k=1", "load k=2" drawn slightly ahead, each with a green note "async, 1 thread issues, ~128×32 tile". BOTTOM lane labeled in blue "Consumer warps — wgmma": boxes "MMA k=0", "MMA k=1", "MMA k=2", each shifted right so it overlaps the NEXT producer load (shaded overlap region marked orange "compute hides load latency"). A purple bracket spanning both lanes labeled "CollectiveMainloop". Small numbered circles (1) at first TMA, (2) at barrier arrive/wait between lanes, (3) at first wgmma. A dashed arrow from a shared-memory box in the middle feeds both lanes, green-labeled "228 KiB SMEM, swizzled". Dashed takeaway box: "producer/consumer + double buffer = latency hidden. Hard to write by hand — CUTLASS ships it." || The Hopper mainloop is a warp-specialized producer/consumer pipeline: TMA loads run ahead while wgmma consumes the previous tile. CUTLASS's CollectiveMainloop is this state machine, done correctly.]]

## The epilogue is where fusion lives

One more level worth naming, because it pays off constantly in practice. After the K-loop finishes, every ladder kernel of ours just wrote the accumulator to `C` with a scale-and-add. CUTLASS calls that final phase the **epilogue**, and it is a pluggable template.

The reason this matters: the accumulator is *already in registers, on-chip*. Any element-wise work you fuse into the epilogue — a bias add, a ReLU, a residual, a cast to `bf16` — happens before the data ever touches HBM. That is exactly the [operator fusion](operator-fusion.html) win from the memory-bound playbook, except CUTLASS hands it to you as a template parameter (`cutlass::epilogue::thread::LinearCombinationRelu`, and friends) instead of a second kernel launch. For the fused GEMMs that dominate real transformer inference, the epilogue is often where the actual speedup over a naive two-kernel implementation comes from.

## So when do you write it by hand?

After all this, the fair question is why we spent eight articles hand-writing kernels at all. The answer is a clean split, and it's worth stating plainly.

**Learn by hand.** You cannot understand what CUTLASS is doing — cannot debug it, cannot choose its tile sizes, cannot read its profiler output — without having felt coalescing, bank conflicts, register pressure, and occupancy in your own kernel first. Every template parameter in that `device::Gemm` line is a decision you already made by hand on the ladder. CUTLASS is illegible to someone who skipped it.

**Ship with CUTLASS** whenever your problem is a GEMM or a GEMM-shaped thing — attention, convolution-as-GEMM, grouped/batched matmul, a fused MLP — *and* you want to actually use Hopper's tensor cores at speed. Re-deriving the TMA/`wgmma`/cluster pipeline by hand to beat CUTLASS is a multi-month project that a team of NVIDIA engineers is already doing full-time. Our [best hand-kernel got to ~94% of cuBLAS](beating-cublas-on-h100.html); a well-tuned CUTLASS config *is* roughly cuBLAS, because cuBLAS's newer kernels are built on the same machinery.

**Write hand-CUDA** when the operation is *not* GEMM-shaped and CUTLASS's abstractions fight you: irregular reductions, custom sparsity, weird data-dependent access, small bespoke fused ops where the template overhead isn't worth it. There, the [three regimes](the-three-regimes.html) and the ladder's toolkit are still exactly what you reach for.

The mental model to leave with: **the ladder taught you the algorithm; CUTLASS is the industrial parameterization of that algorithm plus the Hopper-only instructions you can't easily reach by hand.** They are not competitors. The ladder is how you earn the right to use CUTLASS well — and the next section, on [Blackwell's `tcgen05` and Tensor Memory](blackwell-tcgen05-tmem.html), is where even CUTLASS's abstractions start shifting under the hardware again.
