Softmax is the most-run reduction in all of deep learning. Every attention head computes one over its scores; every classifier ends in one over its logits. And on paper it is trivial — exponentiate a vector, divide by the sum. But that trivial formula, written naively, overflows to `inf` the moment a logit crosses about `88.7` in FP32, and it reads its input from HBM more times than it has any right to. I want to build softmax the way a kernel engineer has to build it: numerically stable first, then down from three passes over memory to two, then to the single fused streaming pass — the **online softmax** — that [FlashAttention](flash-attention.html) is built on top of.

The whole thing is a reduction over the last axis, so this is really two lessons wearing one coat: how to write a correct, stable softmax, and how to think about reduction kernels at all. I'll stay in NumPy and CUDA pseudocode rather than ship a tuned kernel — the point here is the *math* of the passes, because that math is what carries over to attention.

## Why softmax is a memory problem, not a math problem

Start with the arithmetic intensity, because it decides everything. For a row of `N` logits, softmax does on the order of `N` exponentials and a couple of `N`-length reductions — call it a handful of FLOPs per element. It reads `N` numbers and writes `N` numbers. That is an arithmetic intensity of *single digits* of FLOPs per byte, which — from [the three regimes](the-three-regimes.html) — is hundreds of times below the H100's ridge point of ~`295` FLOPs per byte.[[sn: Horace He's "Making Deep Learning Go Brrrr" makes the same point at the network level: normalization and pointwise ops are a rounding error of the FLOPs in a transformer (~0.2%) yet eat a wildly disproportionate share of the wall-clock, precisely because they are memory-bound while the matmuls are compute-bound.]] Softmax is **memory-bandwidth-bound**, hard, on every GPU ever made. The tensor cores are spectators.

That reframes the entire optimization target. We are not trying to do the exponentials faster — `MUFU.EX2`, the hardware exponential, is nearly free relative to the loads. We are trying to *touch HBM as few times as possible*. Every optimization below is a memory optimization in disguise.

[[fig: Hand-drawn Excalidraw-style memory-pyramid diagram on pure white, fine black ink, hand-lettered Virgil-style labels, titled in black "Softmax is bandwidth-bound". On the LEFT, a stacked pyramid of three memory levels drawn as wobbly horizontal boxes, widest at the bottom: bottom box black-labeled "HBM3" with a green handwritten spec "3.35 TB/s"; middle box black-labeled "L2" with green spec "~50 MiB, ~10 TB/s"; narrow top box black-labeled "SMEM / registers" with green spec "on-chip, ~TB/s each SM". On the RIGHT, a row of N logits drawn as a blue diagonal-hatched strip, its length dimension marked in red "↔ N logits". A long blue dashed curved arrow carries the strip UP the pyramid and back DOWN, annotated in blue handwriting "read N, write N — a few FLOPs each". An orange emphasis callout with a curved arrow points at the traffic: "intensity ≈ single-digit FLOP/byte". A red warning tag near the HBM box: "ridge point = 295 FLOP/byte → 100s× below". Bottom-right dashed rounded takeaway box in black: "the win is fewer trips to HBM, not faster exp()". Flat, no shadows, no gradients, generous white space. || Softmax lives at the bottom of the pyramid. The only lever is how many times you walk the input.]]

## The naive formula, and why it explodes

The textbook definition, for a vector `x` of length `N`:

```python
def softmax_naive(x):
    e = np.exp(x)          # <-- overflows the moment max(x) is large
    return e / e.sum()
```

The bug is not subtle once you see it. `exp(x)` in FP32 overflows to `inf` for any `x > ~88.7`, and attention logits after a `QKᵀ / √d` scale routinely blow past that on real models. Once a single element is `inf`, the sum is `inf`, and every output becomes `inf/inf = NaN`. The whole row is poisoned.

The fix is the identity everyone memorizes and few derive: softmax is invariant to shifting the input by a constant. Subtract the row maximum `m = max(x)` before exponentiating. Every exponent is now `≤ 0`, so every `exp` is in `(0, 1]`, and overflow is structurally impossible.[[sn: The identity: softmax(x)_i = exp(x_i) / Σ exp(x_j) = exp(x_i − m) / Σ exp(x_j − m), because the shared factor exp(−m) cancels top and bottom. It is *exact*, not an approximation — you lose no precision, you only move the numbers into a range the hardware can represent.]]

```python
def softmax_stable(x):
    m = x.max()            # pass 1: the max
    e = np.exp(x - m)      # pass 2: shifted exponentials + their sum
    return e / e.sum()     # pass 3: normalize
```

Correct. But count the passes over `x`: one to find the max, one to exponentiate and sum, one to divide. That is **three reads of the input from memory** for one softmax. On a bandwidth-bound kernel, three passes is roughly three times the runtime floor. We can do better without giving up a bit of the stability.

## Two passes: fuse the normalize

The third pass is the easiest to kill. Once we know `m` and the denominator `d = Σ exp(x_j − m)`, the normalization `e_i / d` is pointwise — it needs no reduction, so it can ride along with whatever writes the output. If the consumer of softmax is another kernel (in attention, it is the `P·V` matmul), the division folds into *that* kernel's read and never touches HBM as its own pass. So the honest count is **two passes**: one for the max, one for the exponential-and-sum. That is the standard "safe softmax," and it is what a good library ships when it can't see the whole attention block at once.

But look at those two passes. The first reads all of `x` just to compute a single scalar `m`, throws the data away, then the second reads all of `x` *again*. We loaded every byte twice. The question that opens the door to FlashAttention is: **can we compute the max and the sum in the same pass, before we've seen the max?**

It sounds impossible. The sum `Σ exp(x_j − m)` depends on `m`, and we don't know `m` until we've seen the whole vector. This is the crux.

## Online softmax: one pass, with a running rescale

The trick is to keep a *running* max and a *running* sum, and to correct the sum retroactively every time the max moves. Walk the vector left to right. Maintain two scalars: `m`, the max of everything seen so far, and `d`, the sum of `exp(x_j − m)` over everything seen so far — where that `m` is the *current* running max, not the final one.

When a new element `x_i` arrives, there are two cases folded into one update. Compute the new running max `m_new = max(m, x_i)`. The old `d` was accumulated relative to the old `m`; every term in it is off by a factor of `exp(m − m_new)` now that the reference shifted. So rescale the old sum by exactly that factor, then add the new element's contribution:

```
m_new = max(m, x_i)
d_new = d * exp(m - m_new) + exp(x_i - m_new)
m     = m_new
d     = d_new
```

That `d * exp(m - m_new)` is the entire idea. When the max doesn't change, `m - m_new = 0`, the factor is `1`, and it degenerates to a plain running sum. When a new element sets a new record, the factor `exp(m - m_new)` is `< 1` and it shrinks every previously-accumulated term into the new reference frame — exactly as if we'd known the new max all along.[[sn: This is algebraically exact, not a numerical approximation. After processing the whole row, `d` equals `Σ exp(x_j − m_final)` to the last bit, identical to what the two-pass version computes. The rescale just distributes the max-subtraction across the walk instead of doing it all at the end.]] The result is a **single pass over `x` that produces both `m` and `d`.** One read of the input instead of two.

[[fig: A tiling-walkthrough figure in three numbered panels titled "Online softmax: running max + rescaled sum". Panel (1): a row of logits drawn as hatched blue cells, a scan cursor (orange arrow) at the first cell, with handwritten state below in green "m = -inf, d = 0". Panel (2) circled (2): cursor advanced to a NEW-MAX element highlighted orange, purple code beside it "m_new = max(m, x_i)" and "d = d·exp(m−m_new) + exp(x_i−m_new)"; a curved blue dashed arrow labeled "rescale old sum ↓" pointing back over the already-scanned cells showing them shrink. Panel (3) circled (3): cursor at end, green state "m = m_final, d = Σexp(x−m)", and a red note "= exact two-pass result". Bottom dashed takeaway box: "one pass. the rescale factor exp(m−m_new) corrects history when the max jumps." || The online update. Every time the running max jumps, we retroactively rescale the sum we'd already accumulated.]]

Here it is as a scalar loop — the reference before we parallelize it:

```python
def softmax_online(x):
    m = -np.inf            # running max
    d = 0.0                # running denominator, relative to current m
    for xi in x:                       # single pass
        m_new = max(m, xi)
        d = d * np.exp(m - m_new) + np.exp(xi - m_new)
        m = m_new
    return np.exp(x - m) / d           # pointwise; fuses into the consumer
```

The final `exp(x - m) / d` line is the pointwise normalize we already agreed folds into the downstream kernel. So online softmax is genuinely **one pass to reduce, plus a free pointwise tail** — the minimum possible traffic for a stable softmax.

## The reduction kernel structure

Now the GPU part. A row of `N` logits is reduced by a whole thread block cooperatively, and the online update is *associative*, which is what lets us parallelize it at all. The pattern is the canonical three-tier reduction, and it is worth internalizing because every reduction kernel you write — sum, max, norm, softmax — has this exact skeleton.

**Tier 1 — thread-local.** Each thread strides across the row with a grid-stride loop, folding its slice into a private `(m, d)` pair using the online update above. `float4` vectorized loads here so each thread pulls 16 bytes per transaction and the reads coalesce.[[sn: The load pattern matters more than the arithmetic. If consecutive threads read consecutive elements (`threadIdx.x` maps to consecutive logits), the warp's 32 loads collapse into a few 128-byte sectors. Get this wrong and you replay every transaction — the classic coalescing tax from the GEMM ladder, and on a bandwidth-bound kernel it is the whole ballgame.]]

**Tier 2 — warp-level.** The 32 threads of a warp combine their `(m, d)` pairs without touching memory at all, using `__shfl_down_sync` to pass values between lanes in a tree. But you cannot naively `+` the `d` values — each thread's `d` is relative to *its own* local max. So the warp reduction is the online *merge*: given two `(m_a, d_a)` and `(m_b, d_b)`, the combined max is `max(m_a, m_b)` and the combined sum rescales *both* sides into it.

```cpp
// merge two partial (m, d) states — the associative online combine
__device__ float2 softmax_merge(float2 a, float2 b) {
    float m = fmaxf(a.x, b.x);
    float d = a.y * __expf(a.x - m) + b.y * __expf(b.x - m);
    return make_float2(m, d);
}

// warp reduction over lanes, no shared memory
for (int off = 16; off > 0; off >>= 1) {
    float2 other;
    other.x = __shfl_down_sync(0xffffffff, s.x, off);
    other.y = __shfl_down_sync(0xffffffff, s.y, off);
    s = softmax_merge(s, other);
}
```

**Tier 3 — block-level.** Each warp's lane 0 now holds that warp's partial `(m, d)`. Write those (at most 32 of them, one per warp) to a small **shared memory** (SMEM) staging array, `__syncthreads()`, then have the first warp reduce the partials with the same `softmax_merge` tree. The block's result is one `(m, d)` for the whole row — computed with a single read of the row from HBM and zero intermediate round-trips.[[sn: For very long rows (attention over long context) one block per row can run out of parallelism, and you split the row across blocks with a second small reduction pass over the per-block `(m, d)` pairs. The merge is associative, so this "reduce the reducers" step is the same `softmax_merge` again — it composes cleanly, which is exactly the property FlashAttention exploits across tiles.]]

## Where this lands, and where it goes

The payoff is a bandwidth story, so I'll tell it in bandwidth. The naive stable kernel walks HBM **three times**; the fused two-pass walks it **twice**; the online kernel walks it **once**. On a memory-bound kernel whose floor is set by that traffic, going from three passes to one approaches a **3× speedup** on the softmax itself — the exact factor depends on cache reuse between passes, so treat it as the ceiling, not a promise — and it comes entirely from touching memory less, not computing faster. Profile it and you'll see the achieved HBM bandwidth climb toward the **`3.35 TB/s`** ceiling while the FLOP/s stays microscopic. That is the signature of a reduction kernel doing its job: pinned against bandwidth, ignoring the tensor cores by design.

[[fig: A pipeline-timeline figure titled "Three passes → two → one". Three stacked horizontal timelines against a shared time axis. Top row labeled "naive stable" in black shows three separate blue blocks "read: max", "read: exp+sum", "read: divide", spanning the full width — red tag "3× HBM traffic". Middle row "fused 2-pass": two blue blocks "read: max", "read: exp+sum", divide shown as a faded block merged into the downstream kernel — orange tag "2×". Bottom row "online" in orange: a SINGLE blue block "read: max+sum (running rescale)" plus a faded "normalize fuses downstream" — green tag "1× — bandwidth floor". A vertical dashed line marks where each finishes, bottom row finishing first. Dashed takeaway box: "same math, same stability — one third the memory traffic." || The whole optimization in one picture: identical output, a third of the trips to HBM.]]

The reason this article exists before the attention articles is that online softmax is the load-bearing idea underneath [FlashAttention](flash-attention.html). Attention never has the full row of scores in memory at once — it streams tiles of `K` and `V` through SMEM, computing a slice of the scores at a time. Which means it *cannot* do a two-pass softmax; it never sees the whole row before it has to start accumulating `P·V`. The online update is precisely what rescues it: keep a running `(m, d)` and a running output accumulator, and every time a new tile pushes the max higher, rescale the partial output by the same `exp(m - m_new)` factor we derived here. Softmax and the value-matmul get fused into one streaming pass that never writes the `N × N` score matrix to HBM at all.

So the [reduction skeleton](reductions.html) and the rescale identity on this page are not a warm-up — they *are* FlashAttention's inner loop, minus the matmul. We build that next.
