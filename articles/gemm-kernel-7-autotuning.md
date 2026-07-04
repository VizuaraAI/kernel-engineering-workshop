By kernel 6 we had done something slightly dishonest. The [vectorized kernel](gemm-kernel-6-vectorized.html) reached **78.4% of cuBLAS**, and every one of those tile sizes — `BM`, `BN`, `BK`, `TM`, `TN` — was a number I had *picked*. I picked them the way everyone picks them: `128 × 128` blocktile, `8 × 8` per thread, `BK = 8`, because they looked round and they fit in shared memory. They worked. But "they worked" is not the same as "they are optimal", and the whole premise of this site is that we let measurement, not taste, choose the next move.

So this kernel writes no new CUDA. The kernel body is byte-for-byte the vectorized one. What changes is that we stop guessing the constants and start *searching* them — a grid search over the tile-shape parameter space, run once per GPU, that hands us the best configuration by brute force. This is the least glamorous entry in the ladder and one of the most important, because it is the moment the code stops being a clever thing I wrote and becomes a thing the hardware voted for.

## The hypothesis

The claim is simple: **the tile shape is a free parameter, and the compiler-plus-hardware system has a preference we cannot fully predict.** We have a template kernel with five knobs. Every setting of those knobs produces a *correct* GEMM; they differ only in how the work is partitioned across the memory hierarchy, and therefore in how well the actual silicon digests it. Somewhere in that space is a configuration meaningfully faster than my hand-picked one, and the honest way to find it is to try them all.

Before searching, it pays to name the knobs precisely, because they are not independent — they are three nested tilings, each feeding the next level of the [memory pyramid](shared-memory-l1.html).

- `BM`, `BN`, `BK` — the **blocktile**. One thread block loads a `BM × BK` slab of `A` and a `BK × BN` slab of `B` from global memory into shared memory, then marches the `BK` dimension across the full `K`. These decide how much HBM traffic we amortize.
- `TM`, `TN` — the **threadtile**. Each thread computes a `TM × TN` patch of the output by holding those results in registers and streaming operands out of shared memory. These decide arithmetic intensity *inside* the SM.
- The **thread count** is not free — it falls out of the others: `NUM_THREADS = (BM * BN) / (TM * TN)`. A `128 × 128` blocktile with `8 × 8` threadtiles needs `(128 × 128) / (8 × 8) = 256` threads. Change a tile and you have implicitly changed the block size, the occupancy, and the register pressure all at once.

[[fig: A tiling-walkthrough figure titled "One kernel, five knobs" with three stacked hatched matrices A (blue diagonal hatch), B (green diagonal hatch), C (pale-yellow hatch). On C, a large square tile outlined in orange labeled in red "BM × BN blocktile"; a red dimension arrow ↔ across it reads "128". Inside that tile, a tiny sub-square in darker yellow labeled in red "TM × TN threadtile (8×8)". A blue dashed arrow runs from the A matrix into a small box labeled "SMEM: BM×BK slab" with a green note "BK = the reduction step". Purple handwritten napkin math bottom-left: "NUM_THREADS = (BM·BN)/(TM·TN) = (128·128)/(8·8) = 256". A red note points at the whole thing: "change ONE knob → block size, occupancy AND registers all move". Dashed takeaway box: "the five knobs are not independent — they are one coupled tiling". || The five template parameters are three nested tilings; the thread count is a consequence, not a choice.]]

## Why you cannot just reason your way to the answer

My first instinct was to derive the optimum on paper. It does not work, and it is worth being clear about *why* it does not work, because the failure is the whole lesson.

Bigger blocktiles are better for one reason: they raise arithmetic intensity. A `128 × 128` tile reuses each loaded byte across more math than a `64 × 64` tile, so it moves less HBM traffic per flop. If that were the only force, you would make `BM` and `BN` as large as possible and stop.

But every force pulling toward "bigger" has a twin pulling toward "smaller", and they live in different currencies:

- **Shared memory.** Each block needs `(BM × BK + BK × BN)` floats of SMEM, double that if you double-buffer. An SM offers up to `228 KiB` of SMEM,[[sn: Those `228 KiB` are not a hard architectural constant — the SMEM and L1 share a single `256 KiB` physical block per SM on H100, and how much of it you may declare as SMEM is an opt-in you request with `cudaFuncSetAttribute`. The usable ceiling is a few KiB below `256` because L1 and driver reservations claim the rest.]] and that budget is shared by every block resident on the SM. A bigger tile means fewer blocks fit, or none of your double-buffering does.
- **Registers.** A `TM × TN` threadtile needs at least `TM × TN` accumulator registers per thread, plus operands and indexing. Go past **255 registers per thread** and the compiler *spills* to local memory, which is HBM wearing a costume. The `256 KB` register file per SM (`65536 × 32-bit`) is also split across all resident threads, so more registers per thread means fewer warps resident.
- **Occupancy.** Both of the above throttle **occupancy** — the number of warps the SM can keep in flight to hide latency. A large tile can be so hungry for SMEM and registers that only one or two blocks fit per SM, and then a single stalled warp has nothing to hide behind.

So the parameter space has a genuine interior optimum: a tile large enough to feed the tensor pipeline but small enough that enough warps stay resident to hide the memory latency. Where exactly that optimum sits depends on the register allocator's mood, the exact SMEM carve-out, and the SASS the compiler happens to emit — none of which I can compute in my head.

[[fig: A two-axis tradeoff figure titled "Occupancy vs tile size". A hand-drawn plot with axes: x-axis red-labeled "tile size (BM·BN) →" with a red dimension arrow, left y-axis blue-labeled "arithmetic intensity ↑ (reuse per byte)", right y-axis blue-labeled "resident warps / occupancy ↑". One rising blue line labeled "reuse per byte" climbs to the right; one falling blue line labeled "occupancy (SMEM + register limited)" drops to the right; the two lines cross just right of center where an orange star and orange note read "sweet spot — big enough to feed, small enough to hide". Two green hardware-ceiling labels pinned by dashed green arrows to the falling line: "228 KiB SMEM / SM" and "255 regs / thread → spill to local". A red warning at far right under the blue line: "1 block/SM → nothing to hide a stall". A purple margin note near the axis reads "smem = (BM·BK + BK·BN)·4". Dashed takeaway box: "the optimum is interior — and the compiler, not the algebra, decides where it lands". || The two forces cross somewhere in the middle. Paper can tell you the curves exist; only measurement tells you where they cross on your silicon.]]

## The method: a grid search with a legality filter

The plan is a brute-force sweep, but a *disciplined* one. Most parameter tuples in the naive cross-product are illegal — they overflow SMEM, blow the register file, or produce a thread count that cannot vectorize. Enumerating all of them and letting the launch fail is slow and noisy. Instead we generate candidates and pass each through a legality filter *before* it ever touches the GPU, so we only compile and time the configurations that can actually run.

The candidate grid is small on purpose — powers of two and a few near-them, over sensible ranges:

```python
BM  = [64, 128, 256]
BN  = [64, 128, 256]
BK  = [8, 16, 32, 64]
TM  = [4, 8, 16]
TN  = [4, 8, 16]

def legal(BM, BN, BK, TM, TN, smem_cap=228*1024, max_regs=255):
    nthreads = (BM * BN) // (TM * TN)
    if not (64 <= nthreads <= 1024):            # block-size limits
        return False
    if (BM * BN) % (TM * TN) != 0:              # tile must divide evenly
        return False
    # vectorized float4 SMEM loads: each thread loads whole float4s
    if (BM * BK) % (4 * nthreads) != 0:         # A slab is float4-loadable
        return False
    if (BK * BN) % (4 * nthreads) != 0:         # B slab is float4-loadable
        return False
    smem = (BM * BK + BK * BN) * 4              # bytes, single-buffered
    if smem > smem_cap:
        return False
    est_regs = TM * TN + 8                      # accumulators + overhead
    return est_regs <= max_regs

configs = [c for c in itertools.product(BM, BN, BK, TM, TN) if legal(*c)]
```

That filter is the entire trick. The `float4` divisibility checks are the ones people forget: the vectorized kernel loads each SMEM slab as `128-bit` chunks, so `BM * BK` and `BK * BN` must each be a multiple of `4 * NUM_THREADS`, or a thread's load straddles a `float4` boundary and the whole vectorization collapses.[[sn: This is exactly the constraint Boehm calls out — `BM * BK` divisible by `4 * NUM_THREADS`. It is why some tile shapes that *look* balanced are quietly illegal: a `128 × 128` block with `256` threads and `BK = 8` gives `128·8 / (4·256) = 1`, which is fine, but nudge `BK` to `12` and it stops being an integer.]] After the filter, a grid that starts as a few hundred tuples collapses to a few **dozen legal configurations** — small enough to sweep exhaustively in a few minutes.

The runner itself is embarrassingly simple, and that simplicity is a feature. For each legal config we recompile the template kernel with those constants, run it a few times on the target matrix size, discard the warmup, and keep the median TFLOP/s:

```python
best = None
for cfg in configs:
    binary = compile_kernel(cfg)              # -DBM=... -DBN=... etc.
    for _ in range(warmup): binary.run(A, B, C)
    ts = [time(lambda: binary.run(A, B, C)) for _ in range(iters)]
    tflops = flops(N) / median(ts)
    if best is None or tflops > best.tflops:
        best = Result(cfg, tflops)
```

No cleverness, no gradient, no Bayesian optimizer — the space is small enough that exhaustive beats smart. The one rule I hold to: **autotune on the real problem.** Tune at the matrix size and dtype you will actually run, on the actual card, because the answer does not transfer.

## The measurement

I ran the sweep and let it grind. The winning configuration on this hardware came back as `BM=BN=128`, `BK=16`, `TM=TN=8` — almost my hand-picked guess, except that the grid search doubled `BK` from `8` to `16`, deepening each shared-memory reduction step so more math happens per SMEM round-trip. That one change lifts us from **78.4% to 84.8% of cuBLAS**, a clean **six-point gain for zero new kernel code**.[[sn: Six points from a constant is not a rounding error at this altitude. Going from 78% to 85% closes nearly a third of the *remaining* gap to cuBLAS, and it cost me a bash loop and lunch. The last points are always the expensive ones.]]

Two things about that result are worth sitting with.

First, my intuition was *nearly* right and the mistake it made was invisible to reasoning. I would never have guessed that `BK = 16` beats `BK = 8` here, because the difference lives entirely in how the register allocator and the SMEM-load scheduling interact at that specific occupancy. Reading the SASS afterward, the `BK = 16` build issues the shared-memory loads in a pattern that overlaps better with the FMA pipeline — but I could only *explain* that after the search found it, not predict it beforehand. That gap between "can explain post-hoc" and "could predict a priori" is the entire reason autotuning exists.[[sn: Boehm is refreshingly honest that fully explaining *why* a given tuple wins is "very unsatisfying" without going deeper into compiler and hardware internals. I feel the same. The grid search is, in part, an admission that the system is too complex to model exactly — so we measure it instead.]]

Second — and this is the part that makes autotuning non-optional rather than a nice-to-have — **the answer is per-GPU.** The exact same sweep on an A100 does not return `128/128/16/8/8`; it prefers a smaller `BM=BN=64`, `BK=16`, `TM=TN=4`, because the A100's SMEM carve-out, register file, and warp scheduler strike the occupancy-vs-tile balance at a different point on that crossing-curves picture. Ship the H100 constants to an A100 and you leave measurable performance on the floor. A tile shape is not a property of the algorithm; it is a property of the *machine*, and different machines have different opinions.

[[fig: A pipeline/results figure titled "The sweep" with a left-to-right flow. Box 1 (black) "candidate grid ~hundreds of tuples". A funnel labeled in purple "legality filter: SMEM ≤ 228 KiB, regs ≤ 255, float4-divisible" narrowing to Box 2 "a few dozen legal configs". An arrow into Box 3 "compile · warmup · median TFLOP/s" drawn as a small looping arrow with a green note "run on the REAL size + card". Box 4 an orange-starred winner card reading in red "BM=BN=128 · BK=16 · TM=TN=8". Below it a second grey card labeled "same sweep, A100" reading "BM=BN=64 · BK=16 · TM=TN=4" with a blue note "different machine, different answer". A big orange number floats above the winner: "78.4% → 84.8% of cuBLAS". Dashed takeaway box: "no new kernel — the hardware picked the constants". || The sweep in four stages. The only human input is the candidate grid and the legality rules; the machine chooses the rest.]]

## What this bought us, and what is still on the table

We are at **84.8% of cuBLAS** and we did it by deleting judgment from the hot path. That is the deeper point of this kernel: past a certain level of optimization, *tuning is a step in the build, not a decision a human makes.* Real production kernel libraries — cuBLAS itself, CUTLASS, Triton's autotuner — all ship a search over exactly this kind of space, cached per architecture and per problem shape. We just built the smallest honest version of that machinery.

But the search only rearranged work that a single warp still does the same clumsy way. Every thread in the winning config still reads its operands out of shared memory one `float4` at a time, and threads in a warp step on each other's SMEM banks more than they should. Grid search cannot fix that — no tile shape rewrites the *access pattern within a warp*. To go further we have to stop treating the warp as 32 independent threads and start treating it as a unit: a **warptile** that lays out its 32 lanes to hit all `32` SMEM banks conflict-free and to reuse operands across lanes. That is the next kernel, and it is the one that finally breaks **90% of cuBLAS**.
