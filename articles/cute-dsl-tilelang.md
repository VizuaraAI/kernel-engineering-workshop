By the end of the [GEMM ladder](gemm-kernel-1-naive.html) we hand-wrote our way from **1.3% of cuBLAS** to **93.7%**, and every rung was a real thing we typed: a swizzle for shared memory, a `float4` vectorized load, a warptile loop unrolled by hand. That is the right way to *learn* the machine. It is a ruinous way to *ship* a kernel library. The warptile kernel is a few hundred lines of CUDA that works for one tile shape, one dtype, one architecture; move to a different `M×N×K`, or from Hopper's `wgmma` to Blackwell's `tcgen05`, and most of it has to be rewritten. Production kernel authors do not do this. They reach for an abstraction — and the interesting engineering question is *which* abstraction, and what it costs you.

This article is a map of that landscape. There are four altitudes you can write a GPU kernel at, and picking the wrong one is its own kind of performance bug: too high and you leave 40% of the chip on the floor, too low and you spend a month re-deriving what a compiler would have handed you for free. We will walk from the highest useful abstraction (Triton) down through the layout algebra that CUTLASS is built on (CuTe), touch a newer tile-level DSL (TileLang), and end on the honest question every one of them forces: *when do you drop back to raw CUDA?*

[[fig: A vertical "abstraction ladder" diagram titled "Four altitudes for one GEMM". Four stacked horizontal bands, highest at top. Band 1 (orange label) "Triton — ~40 lines, Python": a small box with a green note "compiler owns coalescing · SMEM · pipelining". Band 2 (black label) "CuTe / CUTLASS — C++ templates": a box hatched with layout grids, blue note "you own the layouts, it owns the codegen". Band 3 (black label) "TileLang — tile-level DSL": box with purple note "explicit tiles, inferred schedule". Band 4 (red label) "Raw CUDA / PTX / SASS": box with red note "you own everything". A long thin dashed arrow down the left edge labeled in blue "more control ↓", and a matching dashed arrow up the right edge labeled in green "more productivity ↑". Dashed takeaway box at the bottom: "the skill is choosing the altitude, not maxing it out". || The four altitudes. Every kernel you ship lives on exactly one of these rungs — the engineering is picking which.]]

## Triton: forty lines, and a compiler that has read the [three regimes](the-three-regimes.html)

Start at the top, because it is where most people should start. **Triton** is a Python-embedded DSL from OpenAI: you write a kernel that operates on *blocks* of a tensor, decorate it with `@triton.jit`, and a compiler lowers it through an MLIR-based pipeline to PTX. The unit of thought is not the thread — it is the tile. You never write `threadIdx.x`. You describe what one *program instance* does to a `BLOCK_M × BLOCK_N` slab, and the compiler decides how 32 threads in a warp cooperate to make that happen.

Here is a fused softmax, which in raw CUDA is a fiddly two-pass reduction with shared-memory scratch and a careful max-subtract for numerical stability. In Triton it is legible:

```python
@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptrs = in_ptr + row * n_cols + cols
    x = tl.load(ptrs, mask=cols < n_cols, other=-float('inf'))
    x = x - tl.max(x, axis=0)          # numerically stable
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * n_cols + cols, y, mask=cols < n_cols)
```

That is the whole kernel. One program instance owns one row; `tl.max` and `tl.sum` are *tile* reductions that the compiler realizes as a warp shuffle tree — no shared memory declared, no `__syncthreads()`, no bank-conflict bookkeeping. A tiled matmul is similarly compact: the canonical Triton GEMM is on the order of **40 lines**, versus the few hundred of our hand-tuned warptile kernel. And it is genuinely fast — Triton GEMMs and fused-attention kernels routinely land in the **80–90% of cuBLAS/FlashAttention** range on the shapes they are tuned for, which is to say: *most of the ladder, for a tenth of the code.*

What is the compiler actually doing on your behalf? Three of the exact things we sweated by hand:

- **Coalescing.** When you `tl.load` a contiguous tile, the compiler assigns lanes to addresses so the access is coalesced. The one-line `m`/`n` reassignment that quadrupled kernel 2 — you never write it; it is a lowering decision.
- **Shared memory.** Tiles that need reuse get staged in SMEM automatically, with a swizzle chosen to avoid bank conflicts across the 32 banks. You declared none of it.
- **Pipelining.** With `num_stages`, the compiler builds a software pipeline of `cp.async` loads that overlaps the next tile's global fetch with this tile's tensor-core math — the double-buffering we assembled by hand becomes a knob.[[sn: `num_stages` and `num_warps` are the two big autotuning axes. Triton's `@triton.autotune` sweeps them for you at first launch and caches the winner, which is why the *same* Triton kernel can be near-peak on an A100 and on an H100 without a source edit.]]

[[fig: A pipeline-timeline diagram titled "What @triton.jit lowers into". On the left, a small purple code box labeled "your ~40 lines" showing three handwritten lines "tl.load", "tl.dot", "tl.store". A big black arrow labeled "MLIR lowering" points right into a horizontal timeline with three overlapping stage rows, each a rounded rectangle: row 1 blue-hatch "cp.async load tile k+1", row 2 green-hatch "wgmma on tile k", row 3 yellow-hatch "store tile k-1". The three rows are offset so they overlap in time, with an orange bracket over the overlap labeled "software pipeline — num_stages=3". Blue margin notes with dashed arrows: "coalesced lane→address (you wrote none of this)" pointing at the load row, "SMEM staged + swizzled, 32 banks" pointing between load and compute. Green note bottom-right "≈ 80–90% of cuBLAS". Dashed takeaway box: "you described the tile; the compiler built the schedule". || What the Triton compiler hands you. Forty lines of Python lower into the exact coalesced, SMEM-staged, double-buffered pipeline we assembled by hand across ten kernels.]]

So why isn't every kernel a Triton kernel? Because the compiler owns the schedule, and there are schedules it will not find. The Stanford CRFM group, writing about AI-generated kernels, deliberately worked "in pure CUDA-C without using libraries and DSLs such as CUTLASS and Triton" precisely so they could express tricks the tile abstraction hides — a hand-shaped `cp.async` pipeline, `half2`-vectorized shared-memory writes, precomputed index caches. Triton gives you a *good* pipeline; it does not let you specify an *arbitrary* one. It has (historically) limited access to the newest hardware paths — Hopper's `wgmma` and the TMA descriptor engine, Blackwell's `tcgen05` and Tensor Memory — because those need bespoke lowering the compiler has to grow support for.[[sn: This gap closes with every release — recent Triton has real Hopper TMA and `wgmma` support — but it is *always* one architecture behind the metal, because someone has to teach the compiler each new instruction. The lag is structural, not a bug.]] And the tile abstraction has a floor: if your problem's optimal data movement doesn't decompose into rectangular tiles with a compiler-inferable schedule, Triton can't express the good version at all.

## CuTe: the layout algebra underneath CUTLASS

Drop one altitude. **CUTLASS** is NVIDIA's open-source C++ template library for GEMM and its relatives — it is, in effect, the readable source of the tricks `cuBLAS` keeps closed. And since CUTLASS 3.x, the thing every kernel is *built out of* is **CuTe** (Cooperative Thread Arrays), a small algebra of **Layouts** and **Tensors**. If you understand one idea from this section, make it this one: CuTe turns "how is this data arranged, and who touches which part" from ad-hoc index arithmetic into a composable algebra.

A **Layout** is a pair — a `Shape` and a `Stride` — that is a function from logical coordinates to a linear offset. A row-major `4×8` tile is `make_layout(make_shape(4,8), make_stride(8,1))`: coordinate `(i,j)` maps to offset `8*i + 1*j`. A **Tensor** is just a Layout plus a pointer, `make_tensor(ptr, layout)`, so indexing it does the offset math the layout defines.[[sn: The payoff is that shapes and strides are *hierarchical* — an entry can itself be a `(shape, stride)` pair. That is how CuTe expresses a swizzled shared-memory tile, or the bizarre register fragment layout a tensor-core instruction demands, as one algebraic object instead of a page of hand-derived indices.]] The whole point is that Layouts *compose*: you can partition a global tensor across thread blocks, then across warps, then across the lanes of one `wgmma`, purely by composing layouts — and CuTe's algebra guarantees the offsets line up.

[[fig: A tiling walkthrough titled "A Layout is a function coord → offset", three numbered panels left to right. Panel (1): a 4×8 grid of small cells, red dimension labels "4" (down) and "8" (across), a purple code line underneath "make_layout(Shape<_4,_8>, Stride<_8,_1>)". Panel (2): the same grid with one cell (i=1,j=3) highlighted orange and a blue handwritten annotation "offset = 8·i + 1·j = 11", a thin dashed arrow from the cell to the number. Panel (3) labeled "COMPOSE" in orange: three nested hatched rectangles — a large blue-hatch "global tile" containing a green-hatch "block tile" containing a small yellow-hatch "wgmma fragment" — with blue notes "partition by block", "by warp", "by lane", and a purple note "layouts compose → offsets line up". Dashed takeaway box: "same algebra from HBM down to one tensor-core lane". || CuTe in one picture. A layout is a coordinate-to-offset function, and because layouts compose, the same algebra partitions data from global memory down to a single tensor-core lane.]]

CUTLASS then stacks these into a **collective/kernel/device** hierarchy that mirrors the [thread-block cluster → block → warp](shared-memory-l1.html) structure of the hardware. A `TiledMMA` object wraps the actual tensor-core instruction — a Hopper `wgmma.mma_async` on an `sm_90a` build — and carries the exact register-fragment layouts the instruction expects, so you feed it CuTe tensors and it *cannot* be miswired. In the 3.x API you assemble a kernel from a `CollectiveBuilder` (mainloop) and an epilogue, hand it tile shapes and a `TiledMMA`, and CUTLASS generates the pipelined, TMA-fed, bank-conflict-free mainloop that we spent ten kernels approximating.

The trade against Triton is exact and worth saying plainly. Triton owns the schedule and you accept it; **CuTe hands the schedule back to you** — you choose the tile shapes, the number of pipeline stages, the swizzle, the MMA atom — but you pay for that control in C++ template machinery and a genuinely steep learning curve. This is why a well-written CUTLASS kernel can *match* `cuBLAS` — often within a percent or two, sometimes past it on shapes cuBLAS didn't specialize for — where Triton tends to plateau in the high 80s. You bought the last ten points of the roofline with layout algebra.

## TileLang: tiles you write, a schedule that's inferred

Between "compiler owns the schedule" (Triton) and "you own everything in C++" (CuTe) there is a growing middle. **TileLang** is a Python tile-level DSL, in the lineage of Apache TVM, where you write the kernel as explicit operations on named tiles — allocate a shared-memory tile, `copy` a global slice into it, `gemm` two tiles into an accumulator — but the *schedule* (thread binding, pipelining, layout inference) is filled in by the compiler and exposed as tunable annotations. You get Triton-like brevity for the data-movement skeleton, with CuTe-like explicitness about *which* tiles live *where* in the memory hierarchy.

```python
@T.prim_func
def matmul(A, B, C):
    with T.Kernel(N // BN, M // BM, threads=128) as (bx, by):
        As = T.alloc_shared((BM, BK), "float16")
        Bs = T.alloc_shared((BK, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        T.clear(Cl)
        for k in T.Pipelined(K // BK, num_stages=3):   # scheduler fills this in
            T.copy(A[by*BM, k*BK], As)                  # global → SMEM
            T.copy(B[k*BK, bx*BN], Bs)
            T.gemm(As, Bs, Cl)                          # SMEM → tensor cores
        T.copy(Cl, C[by*BM, bx*BN])
```

Notice the altitude. Shared-memory tiles are *named and allocated by you* (`alloc_shared`) — that is lower than Triton, where SMEM is implicit — but `T.Pipelined(..., num_stages=3)` asks the compiler to build the async pipeline, which is higher than the hand-rolled `cp.async` ring buffer of a raw kernel. It is the same double-buffering idea from the ladder, expressed as intent rather than mechanism. In practice TileLang aims to reach CUTLASS-class numbers on GEMM and attention with a fraction of the code, and its bet is that *tile placement* is the thing worth writing by hand and *scheduling* is the thing worth inferring — the opposite of Triton's split, which infers placement too.[[sn: The DSL space is churning fast — TileLang, Mosaic/Pallas, ThunderKittens, Hidet, and CuTe DSL's own Python frontend are all circling the same target: CUTLASS performance without CUTLASS's C++. None has clearly won; the boundary between "compiler-inferred" and "author-specified" is exactly where they differentiate.]]

## When to drop to raw CUDA

So: four altitudes, more control as you descend, more productivity as you climb. The decision procedure is short, and it is the same predict-then-measure loop from the [three regimes](the-three-regimes.html).

Start high. Write the Triton kernel first — forty lines, autotuned, and for the large majority of shapes it will be within a few percent of the best kernel you could hand-write, which for a memory-bound fusion is *all* of the win. Profile it. If Nsight Compute says you're at 85% of the roofline and the kernel is on the memory-bound side, stop — the remaining 15% is not worth a month of C++.

Drop to **CuTe/CUTLASS** when you are compute-bound, the shape is stable and high-volume, and the profiler shows Triton leaving a real gap — the tensor cores stalling on a pipeline the compiler under-scheduled, or a `wgmma`/`tcgen05` path the DSL doesn't emit yet. Here the layout algebra earns its keep: you specify the exact schedule and reclaim the last points against `cuBLAS`. Reach for **TileLang** or a peer DSL when you want that control over tile placement without the C++, and your kernel's data movement is tile-shaped enough for its scheduler.

Drop to **raw CUDA / PTX** only for the genuinely off-menu: an instruction no DSL exposes yet, a data-movement pattern that isn't rectangular tiles, a bespoke `cp.async` choreography — exactly the territory the Stanford CRFM kernels staked out on purpose. This is where the ladder's hand-skills pay off: you can only tell a compiler is leaving performance on the floor if you know, from having done it yourself, what the floor looks like.

That is the real reason we hand-wrote all ten kernels. Not because you should ship hand-written CUDA — you almost never should — but because every abstraction above it is a *bet about what to hide*, and you cannot evaluate the bet unless you have seen what's underneath. Triton hides the threads; CuTe hides the index arithmetic; TileLang hides the schedule. Knowing which hidden thing is costing you 10% of the H100 in front of you — that is the whole job.
