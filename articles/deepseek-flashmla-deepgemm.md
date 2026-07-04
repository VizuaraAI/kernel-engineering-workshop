Most of this course has been about *making* kernels — climbing the GEMM ladder one profile at a time until we sit at 93.7% of `cuBLAS`. This article is about a different question: why would a frontier LLM lab, whose product is a model and not a compiler, spend engineer-years writing and *open-sourcing* its own kernels? DeepSeek did exactly that. When they shipped their models they also shipped **FlashMLA** — a Hopper decode kernel for their attention variant — and **DeepGEMM**, a small, legible FP8 GEMM library. The DSpark model card doesn't bury these; it puts them in the launch command. That is the tell. For a model this shape, the kernel *is* the product surface.

I want to work through both kernels the way we've worked through everything else: what is the hardware waiting on, what did they change, and what does it buy. But the deeper lesson is architectural. These kernels exist because the model's *shape* — latent-compressed attention, a sparse mixture of experts, FP8/FP4 weights — falls outside what a general-purpose library like `cuBLAS` or a stock attention kernel is tuned for. When the workload is weird enough, the only person who will write your kernel is you.

## Why an LLM lab ships kernels at all

Start from the serving economics. A model like DSpark is **1.6 trillion parameters with 49 billion activated per token** — a **Mixture of Experts** (MoE) where each token only touches a handful of the experts.[[sn: 1.6T total, ~49B active. The whole point of MoE is that total parameters buy capacity while active parameters set the per-token cost — you pay for a 49B forward pass and get a 1.6T model's knowledge.]] At inference you are almost never compute-bound in the [three-regimes](the-three-regimes.html) sense during decode: you generate one token at a time, the GEMMs are tall-skinny (batch of rows against a huge weight matrix), and you spend your life moving weights and KV cache out of HBM. Every byte you don't move is a token you serve faster.

That reframes the whole optimization target. It isn't peak TFLOP/s on a square matmul; it's **bytes moved per token** and **latency per decode step** at low batch. General libraries optimize the former beautifully and the latter incidentally. So DeepSeek wrote kernels aimed squarely at the latter: an attention kernel that respects their latent KV layout, and a GEMM library that runs the MoE experts in FP8 without the accuracy tax. The DSpark card wires both into `vLLM`:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro-DSpark \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention-config '{"use_fp4_indexer_cache": true}'
```

Every flag there is a kernel decision. `--moe-backend deep_gemm_mega_moe` selects the DeepGEMM path for the expert matmuls; `--kv-cache-dtype fp8`, the `--block-size 256` paged layout, and the FP8/FP4 weight split make the memory math work. This isn't configuration — it's kernel engineering surfacing as CLI.

[[fig: An architecture map titled "Why the lab ships its own kernels". Center: a large rounded box labeled "DSpark forward pass" containing two stacked sub-boxes — top sub-box "Attention: MLA (latent KV)" with a blue hatch, bottom sub-box "MoE FFN: many experts, top-k routed" with pale-yellow hatch tiles (draw ~8 tiles), a couple lit orange labeled "active experts". On the left margin, a red box labeled "cuBLAS / stock attention" with a dashed red arrow that STOPS at the boundary of the DSpark box, annotation in red "doesn't fit this shape". On the right margin two green boxes: "FlashMLA → attention" and "DeepGEMM → expert GEMM", each with a blue dashed arrow pointing INTO the matching sub-box. Green spec notes: "1.6T params · 49B active", "FP8 weights, FP4 experts". Numbered circles (1) at MLA, (2) at MoE. Dashed takeaway box bottom: "weird model shape → general libraries leave money on the table → write the kernel yourself". || The two open kernels each target one half of the forward pass that off-the-shelf libraries under-serve.]]

## FlashMLA: attention when the KV cache is compressed

**Multi-head Latent Attention** (MLA) is DeepSeek's answer to the KV-cache tax. In vanilla multi-head attention you cache a full key and value vector *per head* for every past token; at a million-token context that cache is enormous and it is read back in full on every decode step. MLA instead caches a single low-rank **latent** vector per token and reconstructs the per-head keys and values on the fly by projecting that latent up.[[sn: The card notes the optimized cache needs roughly 10% of the KV memory of the prior version at the same context. That is the latent compression paying off directly as bytes not moved from HBM.]] The result is a KV cache a fraction of the size — the card cites about a tenth — which during memory-bound decode translates almost linearly into throughput.

The catch is that MLA is a genuinely awkward kernel to write well. FlashAttention-style kernels assume the keys and values are already sitting in HBM in a clean layout you can stream through tensor cores. MLA hands you a *compressed* latent that has to be up-projected inside the kernel, and it does so over a **paged KV cache** — the KV blocks are scattered across physical memory in fixed-size pages (the card runs `block-size 256`) rather than contiguously.[[sn: Paged KV, from the PagedAttention line of work, is what lets a server pack many sequences of wildly different lengths into one GPU without fragmentation. The cost is that the kernel must gather non-contiguous pages, which a naive attention kernel is not built to do.]] So FlashMLA has to fuse three things a stock kernel keeps separate: gather the scattered latent pages, up-project them to per-head K and V, and run the attention softmax-and-accumulate — all without ever materializing the full-size KV in HBM.

Conceptually the decode-step inner loop looks like this, per query:

```python
# FlashMLA decode, one query token, streaming over KV pages
acc = 0            # output accumulator, per head
denom = 0          # running softmax denominator
m = -inf           # running max, for numerically-stable softmax
for page in kv_pages(seq):          # scattered physical pages
    latent = load(page)             # low-rank cached vector (small!)
    K, V = up_project(latent)       # reconstruct per-head K, V on-chip
    s = q @ K.T * scale             # attention logits
    m_new = max(m, rowmax(s))       # online softmax rescale
    p = exp(s - m_new)
    acc = acc * exp(m - m_new) + p @ V
    denom = denom * exp(m - m_new) + rowsum(p)
    m = m_new
out = acc / denom
```

The `up_project` step is the whole game. Because the cached `latent` is small, the byte traffic from HBM per token is tiny — you are moving the compressed representation, not the reconstructed K and V. The reconstruction happens in registers and shared memory, on-chip, where bandwidth is effectively free. On Hopper, FlashMLA leans on the same machinery the rest of this course has been building toward: **Tensor Memory Accelerator** (TMA) bulk copies to pull pages into [shared memory](shared-memory-l1.html) asynchronously, `wgmma` warpgroup matmuls for the projection and the score computation, and enough software pipelining to keep the copies and the math overlapped so the SM never stalls waiting on a page.

[[fig: A pipeline-timeline figure titled "FlashMLA decode step". Left: a strip of scattered rectangles labeled "paged KV cache" in HBM, each small rectangle labeled in red "latent (compressed)", drawn non-contiguous with gaps, green note "block-size 256". A blue dashed arrow labeled "TMA async copy" pulls three pages into a middle box labeled "SMEM" (up to 228 KiB, green note). Inside SMEM, a purple-labeled step "up-project → K,V (on-chip)" with a small blue hatch matrix for K and green hatch for V. Then a warp-tile box labeled "wgmma: q·Kᵀ then p·V" in pale-yellow hatch feeding an "online softmax" box (orange). At the bottom, a horizontal timeline with three overlapping lanes — "copy page i+1", "up-project page i", "matmul page i-1" — shaded to show they run concurrently. Numbered circles (1) gather, (2) reconstruct, (3) attend. Dashed takeaway box: "move the COMPRESSED latent from HBM; reconstruct full K,V on-chip where bandwidth is free". || FlashMLA fuses page-gather, up-projection, and attention so only the small latent ever crosses HBM.]]

The payoff is measured in the currency that matters for decode: memory moved. By keeping the KV cache at roughly a tenth of a conventional layout and never spilling the reconstructed K/V back to HBM, FlashMLA turns the single biggest bandwidth sink in long-context decode into a small one. On Hopper this is where the kernel earns its keep — decode is memory-bound, and MLA is a **~10× reduction in the dominant byte stream**.

## DeepGEMM: FP8 experts, and nothing you can't read

The other half of the forward pass is the MoE feed-forward. Each token routes to a small set of experts, and each expert is a pair of GEMMs against a large weight matrix. With 49B active parameters, these expert GEMMs are where the FLOPs — and the weight bytes — actually live. DeepGEMM is DeepSeek's library for running them, and its design thesis is almost aggressively simple: **clean, FP8, and small enough to read**. The whole library is a few hundred lines of core CUDA; it does one thing — FP8 GEMM with fine-grained scaling, including the grouped/masked variants MoE needs — and it does it without the sprawling template metaprogramming that makes production GEMM libraries opaque.

Why FP8, and why write a new library for it? The card specifies a mixed regime: **MoE expert parameters in FP4, most other parameters in FP8**.[[sn: The card lists FP4 for the MoE experts and FP8 elsewhere, with an `use_fp4_indexer_cache` attention flag. FP4 here is the block-scaled e2m1 format (NVFP4-style) that Blackwell's tensor cores accelerate natively — this is a big reason the card targets GB300 nodes rather than H100.]] FP8 halves the weight bytes versus BF16, which during memory-bound decode is a near-linear throughput win — but FP8 GEMM is only useful if you can hold accuracy, and that requires *scaling*. A single scale factor for a whole tensor loses too much dynamic range; DeepGEMM uses **fine-grained block scaling** — a separate scale for each small tile of the matrix — computed and applied inside the kernel so the accumulation still happens in higher precision. The Hopper path issues the FP8 `wgmma` instructions, accumulates in FP32, and rescales per block.

The `deep_gemm_mega_moe` backend named in the `vLLM` command is the grouped version. Instead of launching one GEMM per active expert — a swarm of tiny, overhead-bound kernels with different weight matrices — it launches a single **grouped GEMM** that packs all the active experts' matmuls into one kernel, using a masked/segmented layout so each token's rows hit the right expert's weights.

```python
# grouped MoE GEMM — one launch for all active experts
# tokens are sorted by expert; group_offsets marks each expert's slice
deep_gemm.grouped_gemm_fp8(
    lhs        = tokens_fp8,       # [num_tokens, hidden], block-scaled
    rhs        = expert_w_fp8,     # [num_experts, hidden, inter], FP8/FP4
    out        = y,               # [num_tokens, inter], FP32 accumulate
    group_offs = group_offsets,   # per-expert row ranges (masked layout)
    scales     = block_scales,    # fine-grained scale per tile
)
```

This is a direct assault on the third regime — **overhead**. Firing one kernel per expert means dozens of launches, each paying launch latency and each too small to saturate the SMs; the grouped kernel pays launch cost once and keeps every SM busy across the whole expert batch.[[sn: This is the same insight as fusing many small element-wise kernels into one launch, applied to MoE. When the individual GEMMs are small and numerous, the launch and tail overhead dominates — batching them into one grouped kernel is the fix.]] Combined with FP8 weights, the effect is that the MoE FFN — nominally the heaviest part of a 1.6T model — moves half the bytes and launches a fraction of the kernels.

[[fig: A tiling-walkthrough figure titled "deep_gemm_mega_moe: grouped FP8 GEMM", two panels. Panel (A) labeled "naive: one GEMM per expert" — several small separate matmul boxes stacked, each with its own red "launch" tag and a grey idle-SM note, orange warning "overhead-bound: many tiny launches". Panel (B) labeled "grouped: one launch" — a single wide LHS matrix (blue hatch, red dim "num_tokens × hidden") sorted into colored row-bands, each band an arrow to its expert's weight tile in a stacked RHS (green hatch, red dim "num_experts × hidden × inter"), all feeding one pale-yellow output tile. Purple code note beside it "group_offsets → masked layout". Green spec note "FP8 weights, FP32 accumulate, per-tile block scale". Numbered circles (1) sort tokens by expert, (2) one grouped launch, (3) FP8 wgmma per tile. Dashed takeaway box: "half the bytes (FP8) + one launch instead of N → kills the MoE overhead tax". || DeepGEMM packs all active experts into one FP8 kernel, trading a swarm of tiny launches for a single grouped GEMM.]]

## How it plugs into serving

Neither kernel is a research artifact you admire and shelve. They are load-bearing in the serving path, and the integration is the interesting part. `vLLM` owns the outer loop — scheduling, the paged KV allocator, continuous batching, CUDA graph capture — and it *delegates* the two hot inner kernels to DeepSeek's implementations through the backend flags. FlashMLA slots in behind the attention config, consuming the same paged KV blocks `vLLM`'s allocator hands out. DeepGEMM slots in behind `--moe-backend`, consuming the routed, expert-sorted tokens the MoE layer produces.

That division of labor is why the open kernels matter beyond DeepSeek. Because they speak `vLLM`'s interfaces — paged KV, expert-parallel layout, FP8 tensors — anyone serving an MLA-plus-MoE model can adopt them without rewriting their stack. And because DeepGEMM in particular is small and readable, it doubles as a teaching artifact: it is the honest, few-hundred-line answer to "how do you actually do FP8 GEMM with block scaling on Hopper," which is exactly the kind of thing this course exists to demystify.

[[fig: A stacked architecture figure titled "Where the kernels live in the serving stack", drawn as horizontal layers. Top layer (black) "vLLM engine: scheduler · continuous batching · CUDA graphs". Second layer "paged KV allocator (block-size 256)" green note. Third layer split into two halves by a dashed vertical line: left half blue box "Attention layer → FlashMLA" with a blue dashed arrow down to a hardware box; right half yellow box "MoE layer → deep_gemm_mega_moe" with a dashed arrow down. Bottom layer (green, hardware) a wide box "4× GB300 node" with green specs "FP4/FP8 tensor cores · HBM3e" and an orange note "TMA · wgmma". Red margin annotations: "vLLM owns the loop" pointing at top, "lab owns the hot kernels" pointing at the split layer. Numbered circles (1) request in, (2) attention via FlashMLA, (3) MoE via DeepGEMM, (4) token out. Dashed takeaway box: "the framework schedules; the open kernels do the two hot inner loops". || vLLM owns the serving loop and delegates the two hottest inner kernels to DeepSeek's open implementations.]]

## The takeaway

Step back and the pattern is clean. A frontier model has a *shape* — MLA compresses the KV cache, MoE sparsifies the FFN, FP8/FP4 shrinks the weights — and each of those shape choices creates a kernel that general-purpose libraries under-serve. FlashMLA answers the compressed-KV attention; DeepGEMM answers the FP8 grouped-expert GEMM. Both are aimed at the regime that dominates decode: bytes moved and kernels launched, not peak square-matmul FLOP/s.

The lesson for a kernel engineer is that the frontier is no longer "make GEMM fast" in the abstract — `cuBLAS` and the ladder we climbed already do that. It's "make *this specific model's* forward pass fast": reading the architecture, finding the byte stream and the launch storm the libraries miss, and writing the fused kernel that closes them. DeepSeek open-sourced their homework. Everything we built on the [GEMM ladder](gemm-kernel-1-naive.html) — coalescing, shared-memory tiling, `wgmma`, TMA, pipelining — is exactly the vocabulary you need to read FlashMLA and DeepGEMM, and eventually to write the next one.
