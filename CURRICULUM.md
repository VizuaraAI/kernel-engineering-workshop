# Kernel Engineering — Curriculum (v3, hiring-skills-verified)
8 live foundational lectures (2/wk × 3h × 4 wks) + 6 deep-dive workshops (modern topics).
Site content (~72 articles) is INDEPENDENT of lectures; every lecture lists its supporting articles.

## Skill map — what a hiring CEO expects vs where we deliver it
(source: a real kernel-engineering hiring post; every listed resource is integrated)
1. CUDA matmul from scratch (siboehm)            → §3 ladder, L4–L5
2. Tensor-core matmul worklog (alexarmbr)        → §3 tensor-core mini-ladder, L6
3. Outperforming cuBLAS on H100 (cudaforfun+hamza)→ W2, §5 Hopper articles
4. GPU MODE lectures (profiling→CUTLASS→SASS)    → mapped async companion track
5. CUTLASS the hard way (kapilsh)                → W3, §5 CUTLASS/CuTe articles
6. Automated GPU kernel generation (Simon Guo)   → §6 AI × Kernels, W6, capstone
7. Surprisingly fast AI-generated kernels (CRFM) → §6 AI × Kernels, W6, capstone

## The 8 Foundational Live Lectures (3h = 3 × 50-min blocks + breaks)

### L1 — How fast can this go? (mental models + the silicon)
- Block 1: The three regimes (horace/brrr): compute-bound, memory-bound, overhead-bound.
  Napkin math on H100: 989 TF/s BF16 vs 3.35 TB/s HBM → arithmetic intensity ≈ 295.
  Roofline model drawn live. "What % of peak are you at?" as the master question.
- Block 2: Silicon tour, top-down: die → 8 GPCs → SMs (132 SXM) → warp schedulers,
  tensor cores, register file, SMEM/L1 · L2 partitions + crossbar · HBM stacks/interposer.
- Block 3 (live): PyTorch benchmark session — elementwise vs matmul FLOPs/byte; find the
  regime of ReLU, softmax, GEMM at several sizes; predict-then-measure.
- Site articles: §0 all, §1.1–1.6.

### L2 — The CUDA programming model + first kernels
- Grid/block/warp/thread; SIMT & divergence; kernel launch anatomy; nvcc → PTX → ptxas →
  SASS compilation story; compute capability.
- Live: vector add → image RGB→grey → naive reduction. First 6 GPU-Puzzles solved live.
- Site: §2.1–2.6.

### L3 — The memory hierarchy in anger
- Coalescing (32B/64B/128B transactions) measured live; SMEM & bank conflicts with the
  padding fix; occupancy calculus (registers/thread × blocks/SM); register spills to local.
- Live: matrix transpose ladder — naive → coalesced → SMEM → +padding; ncu first contact.
- Site: §1.7–1.10, §2.7–2.9.

### L4 — GEMM worklog I: naive → tiling (ladder kernels 1–4)
- Kernel 1 naive (1.3% cuBLAS) → K2 coalescing (8.5%) → K3 SMEM tiling (12.8%) →
  K4 1D blocktiling (36.5%). Each step: hypothesis → code → ncu → %.
- The arithmetic-intensity ratchet as the through-line.
- Site: §3.1–3.5.

### L5 — GEMM worklog II: registers → warps (kernels 5–10)
- K5 2D blocktiling (68.7%) → K6 vectorized float4/LDS.128 (78.4%) → autotuning (84.8%)
  → warptiling (93.7%). Reading SASS: LDG.E.128, instruction-issue counting.
- Live: SASS diff before/after vectorization (the "8 loads → 2 loads" moment).
- Site: §3.6–3.10.

### L6 — Tensor cores: the second worklog
- Spine = alexarmbr "Fast Matrix Multiplication From Scratch With Tensor Cores":
  mma.sync/wmma, fragment/register layouts, SMEM swizzling to kill bank conflicts,
  the precision menu (TF32/BF16/FP16/FP8), double buffering + cp.async — its own
  hypothesis→profile ladder, just like L4–L5 but on tensor cores.
- Where CUTLASS/CuTe sit; brief Hopper preview (TMA/WGMMA → W2).
- Live: WMMA GEMM beating our best SIMT kernel; ncu tensor-pipe utilization.
- Site: §3.11–3.13 (tensor-core mini-ladder), §5.1 preview.

### L7 — Profiling & debugging like a professional
- Nsight Compute deep-read: SOL section, memory workload analysis, warp stall reasons.
- The vLLM debugging workflow: compute-sanitizer; hanging kernels; user-triggered core
  dumps (`CUDA_ENABLE_USER_TRIGGERED_COREDUMP`), cuda-gdb, `-lineinfo`, nvdisasm -gi.
- Live: three sabotaged kernels (race, misaligned vector load, silent NaN) diagnosed live.
- Site: §5.9–5.10.

### L8 — Attention: the kernel that ate the world (+ capstone kickoff)
- Attention as matmuls + softmax; why naive attention is memory-bound (O(N²) HBM traffic);
  online softmax; FlashAttention v1 tiling built live (simplified, single head);
  what FA2/FA3 change. KV-cache math → why decode is GEMV/memory-bound.
- **Capstone kickoff — "You vs the machine" (CS149 × KernelBench):** pick an op
  (histogram / SwiGLU / FA variant / heat eq), optimize it BY HAND and then run an
  LLM-in-the-loop (propose→compile→profile→iterate) against your own kernel; the
  worklog documents both tracks and what each found that the other missed. Graded
  on process (CS149 model), not raw speedup. This is the workshop's signature.
- Site: §4.1–4.6, §6.1–6.2.

## The 6 Deep-Dive Workshops (modern topics, one per week after/interleaved)

### W1 — FlashAttention from scratch
Full FA forward: tiling, online softmax rescaling, causal masking; benchmark vs PyTorch
SDPA; FA2 work-partitioning; FA3 warp-specialization/pingpong on Hopper (conceptual+code).

### W2 — Hopper deep dive: outperforming cuBLAS on H100
The two real worklogs (hamzaelshafie + cudaforfun "Outperforming cuBLAS on H100")
as case studies: TMA async bulk copies, wgmma m64nNk16, producer/consumer warp
specialization, mbarriers, cluster/DSMEM; Hopper SASS reading; what it actually
takes to beat NVIDIA's own library.

### W3 — The abstraction ladder: Triton → CUTLASS/CuTe → TileLang
Rewrite GEMM + FA in Triton (~40 lines); what the compiler does for you (coalescing,
SMEM, pipelining) and what it can't. Then **CUTLASS the hard way** (kapilsh):
naive GEMM → CuTe layouts/tensors → a real CUTLASS GEMM; reading CUTLASS's
warptiling with our L5 vocabulary. TileLang/CuTe-DSL tour; when to drop to raw CUDA.

### W4 — Inference-serving kernels
Prefill vs decode; GEMV & batched decode; KV-cache layouts, PagedAttention kernel;
fusion patterns (RMSNorm+QKV, SwiGLU); quantized kernels (FP8, W4A16); continuous
batching interaction with kernel shapes.

### W5 — Blackwell & NVFP4
tcgen05, Tensor Memory (TMEM), CTA pairs; microscaling block formats (NVFP4 e2m1 +
FP8 scale factors); the hackathon journey re-run: 2000μs → 22.3μs batched FP4 GEMV
(intrinsics vs bit-twiddling, ILP, PTX fusion); CuTe-DSL vs raw CUDA paths.

### W6 — Frontier finale: DeepSeek, DSpark & AI that writes kernels
Hour 1 — the DeepSeek stack: MLA → FlashMLA decode kernel; DeepGEMM FP8/MoE
(`deep_gemm_mega_moe`); V4-Pro's CSA/HCA sparse attention (27% FLOPs @1M ctx, 10% KV
cache); **DSpark = speculative-decoding module on V4-Pro** — why spec-dec is a kernels
problem (draft passes, parallel verify, acceptance kernels); FP4 experts + FP8 KV on
GB300-class hardware.
Hour 2 — **AI-generated kernels**: KernelBench & fast_p; test-time scaling ("monkeys":
DeepSeek-V3 4%→37% with 100 samples, →72% with feedback); CRFM's branching search with
natural-language optimization ideas (LayerNorm 484% of PyTorch, Conv2D 180% — and the
honest failures: FA at 9%, FP16 matmul 52%); Kevin multi-turn RL; KernelBook; why
human+AI+profiler loops win (the kernel harness ← cross-link to Vizuara's Harness
Engineering workshop).
Hour 3 — capstone demos ("You vs the machine" worklogs) + leaderboard + certificates.

## Site article map (~72) — section → count
§0 Start Here (5: + "The kernel engineer's skill map" career article) ·
§1 GPU From Silicon Up (14) · §2 CUDA Model (12) ·
§3 GEMM Worklog (15: 12 SIMT ladder + 3 tensor-core mini-ladder [alexarmbr spine]) ·
§4 Kernels for Inference (12) ·
§5 Frontier (10: Hopper×3 incl. beating-cuBLAS-on-H100, Blackwell/NVFP4×2,
   DeepSeek/DSpark×2, CUTLASS/CuTe×2, debugging×1... profiling folded into §2) ·
§6 AI × Kernels (4: KernelBench & how to measure · test-time scaling & search ·
   the CRFM experiments · Kevin/RL + KernelBook + what's still human)
(Each article: worklog voice, 3–6 figures, 2–6 sidenotes per STYLE.md.)

## Async layer
GPU-Puzzles guided track (srush) · **GPU MODE lecture companion map** (our § ↔ their
lecture numbers: profiling → kernels → CUTLASS → SASS) · per-article quizzes · worklog
assignment templates · kernel leaderboard (Popcorn-CLI-style later) · recorded deep-dives.
