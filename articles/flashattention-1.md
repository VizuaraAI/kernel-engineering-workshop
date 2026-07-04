Attention is the layer everyone points at when a transformer is slow, and for the wrong reason. People assume the two matrix multiplies — scores `= QKᵀ` and `out = PV` — are the cost. They are not. On any real sequence length the expensive part of naive attention is the *softmax in the middle*, and specifically the fact that a naive implementation materializes an `N × N` matrix of scores in HBM, reads it back to normalize it, reads it back again to weight `V`, and does all of this at an [arithmetic intensity](the-three-regimes.html) far below the ridge point. It is a textbook memory-bound layer wrapped around two compute-bound GEMMs.

This article is the first of two on **FlashAttention**. The goal here is narrow and conceptual: fuse the *entire* attention operation — the two GEMMs and the softmax between them — into a single kernel so that the `N × N` score matrix **never touches HBM**. We never write it, so we never read it back. The trick that makes this possible is **online softmax**: a way to compute a numerically-stable softmax in a single streaming pass, tile by tile, keeping only a running max, a running sum, and a running output accumulator. We tile over blocks of `Q`, `K`, and `V` in shared memory, and we rescale the accumulator as we go. The second article turns this sketch into a real Hopper kernel with `wgmma` and `TMA`; this one earns the idea.

## Why naive attention is a memory disaster

Write out standard attention for one head, `Q`, `K`, `V` each `N × d` (call `d` the head dimension, typically 64 or 128):

```python
S = Q @ K.T          # (N, N)  scores
P = softmax(S, dim=-1)  # (N, N)  row-wise
O = P @ V            # (N, d)  output
```

Count the traffic. The two GEMMs are fine on their own — their intensity grows with `N`, so in isolation they are compute-bound and happy. The problem is the two intermediates. `S` is `N × N`; for a sequence of `N = 8192` that is 67 million elements, **256 MiB** in FP32, per head, per layer.[[sn: This is why long-context attention hurts so disproportionately — the score matrix is quadratic in `N` while everything else is linear. Doubling the sequence quadruples the intermediate you have to move.]] We write all of `S` to HBM, read it all back to compute the row-wise softmax, write `P` back out, and read `P` a third time to multiply by `V`. The softmax itself does a trivial amount of arithmetic — an exponent and a couple of adds per element — so those three full passes over an `N × N` matrix are pure memory movement with almost no compute to hide behind.

This is exactly the pathology Horace He's *"Making Deep Learning Go Brrrr"* names as the case for fusion: "instead of writing our data to global memory just to read it again, we elide the extra memory accesses by performing several computations at once."[[sn: horace.io/brrr_intro.html — required reading, and the same source behind [the three regimes](the-three-regimes.html). Normalization ops like softmax do on the order of hundreds of times fewer FLOPs than a matmul yet can dominate runtime precisely because they are memory-bound.]] The softmax is the intermediate we should never have written.

[[fig: A hand-drawn "before vs after" memory-traffic diagram titled "Attention: the intermediate that shouldn't exist". LEFT panel labeled (A) NAIVE in black. A vertical stack representing HBM on the far left (green box labeled "HBM 3.35 TB/s"). Three matrices drawn as squares: Q and K small (red dims "N×d"), and a big square S with red hatch labeled "S = N×N scores" and an orange callout "256 MiB @ N=8192!". Blue dashed arrows show the round-trips: "write S → HBM", "read S ← HBM (softmax)", "write P → HBM", "read P ← HBM (·V)", each arrow numbered (1)(2)(3)(4). Blue note "4 passes over an N×N matrix". RIGHT panel labeled (B) FLASHATTENTION in black. The same Q/K/V but now inside a small orange box labeled "on-chip SMEM" with the big S square crossed out in red and annotated "never materialized". A single blue arrow "stream K,V tiles in" and a single arrow out "write O once (N×d)". Green note "traffic ∝ N·d, not N²". Dashed takeaway box bottom: "fuse the whole thing → the N×N scores never reach HBM". || Naive attention makes four trips over an N×N matrix. FlashAttention makes none — the scores live and die on-chip.]]

## The obstacle: softmax wants the whole row

The reason nobody fused this by default is that softmax is *global along a row*. To normalize row `i` of the scores you need its maximum (for numerical stability you subtract the max before exponentiating) and the sum of all its exponentials. Both quantities depend on every column in the row. So the obvious fused loop — stream one tile of `K`/`V` at a time and immediately use it — seems impossible: you cannot divide by a denominator you have not finished computing, and you cannot subtract a max you have not finished finding.

The resolution is **online softmax**, and it is the whole trick. It is the same insight as computing a running mean in one pass: you keep partial statistics and *correct them* each time a new tile changes your knowledge of the row.

Concretely, as we walk across the key blocks for a given query block, we keep three running quantities per query row:

- `m` — the running row max seen so far,
- `ℓ` — the running sum of `exp(score − m)` so far,
- `O` — the running weighted-`V` accumulator so far (a `d`-vector per query row).

When a new key tile arrives, its scores may contain a value larger than our current `m`. If they do, every exponential we have already accumulated was computed against a stale, too-small max and is now too large by a factor of `exp(m_old − m_new)`. So we **rescale**: multiply the old `ℓ` and the old `O` by that correction factor, then fold in the new tile. That single rescale is what lets the softmax denominator and the output accumulator move forward without ever seeing the full row at once.[[sn: Online softmax predates FlashAttention — it comes from Milakov & Gimelshein's 2018 "Online normalizer calculation for softmax". FlashAttention's contribution was realizing you can carry the `O` accumulator through the *same* recurrence, fusing the second GEMM into the streaming pass too.]]

[[fig: A tiling-walkthrough figure in 3 numbered panels titled "Online softmax: stream K/V, rescale as you go". Setup listed top-left in handwriting: "Q block = Br×d (blue hatch), K/V blocks = Bc×d (green hatch), keep m, ℓ, O per row". Panel (1): one Q block (blue hatch) fixed on the left, and a row of K/V blocks (green hatch) drawn left-to-right labeled j=1,2,3,4 with a red arrow "outer loop over key blocks →". Panel (2): a zoom on processing block j — a small Sᵢⱼ tile (pale-yellow hatch) with purple code "Sᵢⱼ = Qᵢ Kⱼᵀ", then "m_new = max(m, rowmax(Sᵢⱼ))" in blue, then "P̃ = exp(Sᵢⱼ − m_new)" in blue. Orange emphasis callout "found a bigger max!". Panel (3): the RESCALE step drawn as two beakers: old ℓ and old O each multiplied by a purple factor "α = exp(m_old − m_new)", then "ℓ = α·ℓ + rowsum(P̃)" and "O = α·O + P̃·Vⱼ" in blue. Red note "correct the past, then add the present". Dashed takeaway box: "one streaming pass · O(N·d) memory · exact softmax, not approximate". || The core recurrence. Each new key block can raise the row max, so we scale the accumulated sum and output by exp(m_old − m_new) before folding the block in.]]

## The kernel sketch

Here is the algorithm as a fused kernel, in the concept-first form. One thread block owns one block of `B_r` query rows and keeps its statistics resident on-chip for the whole computation. It loops over the key/value blocks, and only writes the final `O` (an `N × d` matrix, linear in `N`) back to HBM at the end.

```python
# One CTA owns query block Q_i  (B_r x d), resident in SMEM/registers.
# m: (B_r,) running max, init -inf
# l: (B_r,) running sum, init 0
# O: (B_r, d) accumulator, init 0
m = full(B_r, -inf); l = zeros(B_r); O = zeros(B_r, d)

for j in range(num_k_blocks):          # outer loop over K/V blocks
    K_j = load(K, j)                   # (B_c, d) -> SMEM  (streamed)
    V_j = load(V, j)                   # (B_c, d) -> SMEM

    S = Q_i @ K_j.T                    # (B_r, B_c)  scores, stays in SMEM/regs
    m_new = maximum(m, rowmax(S))      # (B_r,)  updated row max
    P = exp(S - m_new[:, None])        # (B_r, B_c)  unnormalized weights
    alpha = exp(m - m_new)             # (B_r,)  correction factor

    l = alpha * l + rowsum(P)          # rescale old sum, add new
    O = alpha[:, None] * O + P @ V_j   # rescale old output, add new
    m = m_new

O = O / l[:, None]                     # single normalize at the very end
store(O_i, O)                          # (B_r, d) -> HBM   (only write!)
```

Read what did *not* happen. `S` is `B_r × B_c` — a small shared-memory tile sized to fit, never the full `N × N` — and it is consumed immediately, never stored. `P` is likewise ephemeral. The division by the softmax denominator is deferred to a single normalize at the end, because until we have finished the last key block we do not know the true `ℓ`.[[sn: One subtlety the sketch hides: the second GEMM `P @ V_j` accumulates into `O` while `O` is *also* being rescaled by `alpha` each step. On tensor-core hardware this is why FlashAttention keeps the `O` accumulator resident in registers rather than round-tripping it — the rescale is a cheap element-wise multiply fused between two matmuls.]] The `alpha` rescale is the only line that would look strange to someone who has written a normal softmax, and it is the price of never seeing the whole row.

Sizing the tiles is a shared-memory budget problem, exactly like the GEMM ladder's [SMEM tiling step](gemm-kernel-1-naive.html). An H100 gives you up to **228 KiB** of usable SMEM per **Streaming Multiprocessor** (SM). You must fit `Q_i` (`B_r × d`), the current `K_j` and `V_j` (`B_c × d` each), and the score tile `S` (`B_r × B_c`) — plus room for double-buffering the next `K`/`V` block so the loads overlap the math. With `d = 128` and FP16, a natural choice is `B_r = B_c = 64`, which keeps every live tile comfortably on-chip and leaves headroom for the pipeline.[[sn: These are illustrative sizes, not a tuned configuration. The real FlashAttention picks `B_r`/`B_c` from the SMEM cap and the register budget, and the numbers differ across FA-1, FA-2, and the Hopper rewrite. `B_c` is often chosen as ⌈SMEM / (4d)⌉ and `B_r = min(that, d)`.]]

## What we actually bought

The win is not a FLOP win — FlashAttention does the *same* matrix math as naive attention, and even a few *extra* flops for the rescales. The win is entirely in memory traffic. Naive attention moves `O(N²)` bytes for the score matrix alone, several times over. Fused attention moves only `Q`, `K`, `V`, and `O`, each `O(N·d)`, once. For `N = 8192` and `d = 128` that is the difference between shoving a quarter-gigabyte intermediate across HBM four times and never creating it at all.

[[fig: A memory-pyramid / traffic-ledger figure titled "Where the bytes go". LEFT: a stacked memory pyramid, top-to-bottom: "Register file 256 KB/SM" (green), "SMEM 228 KiB/SM" (green), "L2 ~50 MiB" (green), "HBM3 80 GB @ 3.35 TB/s" (green) at the wide base. A blue bracket spans SMEM+registers labeled "S, P live and die here". RIGHT: a two-row ledger drawn as a hand-written table. Row 1 "NAIVE": red entries "write S (N²) · read S (N²) · write P (N²) · read P (N²) · +Q,K,V,O" with orange total "≈ 4N² + 4Nd". Row 2 "FLASH": blue entries "read Q,K,V once · write O once" with orange total "≈ 4Nd". A red dimension arrow between them labeled "ratio ≈ N/d ⇒ ~64× less HBM traffic at N=8192, d=128". Dashed takeaway box: "same FLOPs, far fewer bytes → a memory-bound layer becomes compute-bound". || The ledger. Traffic drops from ~4N² to ~4Nd — for long sequences, dozens of times fewer bytes across HBM.]]

The order-of-magnitude is worth stating in bold. Fusing attention this way turns a hard **memory-bound** layer back into a **compute-bound** one: with the `N × N` intermediate gone, the only traffic left is linear in `N`, the arithmetic intensity climbs above the ridge, and the two GEMMs can finally run the tensor cores near their **989 TFLOP/s** ceiling instead of stalling on HBM. On long sequences this is where the frequently-quoted **several-fold** speedups on the attention layer come from — not from doing less math, but from stopping the math units from waiting on a matrix that never needed to exist. The multiplier depends heavily on `N`: at short sequence lengths the `N²` term is small and fusion barely helps, but the gain grows with context length, which is exactly why FlashAttention landed at the same moment models started chasing long context.

## The bridge

What we have is the right algorithm and the wrong hardware mapping. This sketch would run — and it would already beat naive attention on memory traffic — but it leaves most of the H100 on the table. It says nothing about *how* to feed `K`/`V` blocks into shared memory without stalling (Hopper's `TMA`), nothing about issuing the two GEMMs on the tensor cores (`wgmma`, warp-group asynchronous matrix-multiply), and nothing about overlapping the load of block `j+1` with the compute on block `j` so the pipeline never drains.

That is the entire subject of the next article. We take this exact recurrence and schedule it as a real Hopper kernel: `TMA` for the streaming copies, a producer/consumer **warp specialization** so one warp group loads while another computes, and the `alpha` rescale fused between two `wgmma` accumulations kept resident in registers. Same math as the sketch above — but this time the profiler, not the algebra, drives every decision, exactly the way it did all the way up the [GEMM ladder](gemm-kernel-1-naive.html).
