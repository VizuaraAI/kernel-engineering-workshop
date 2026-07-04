export const meta = {
  name: 'deepen-articles',
  description: 'Rewrite all 72 Book articles far deeper: Socratic, from foundations, simple-but-deep, 6-10 figures each',
  phases: [{ title: 'Deepen' }, { title: 'Verify' }],
}
const DIR = 'kernel-engineering-site'
const PRE = `You are DEEPENING one existing article of the Kernel Engineering Book so it becomes far deeper, more visual, and easier to understand — in the spirit of siboehm.com's data-parallel-training and CUDA-MMM posts. The current article is too short and terse; your job is to rewrite it 2-3x longer and much deeper WITHOUT making it harder.

READ THESE FIRST (non-negotiable):
- ${DIR}/DEEPEN_STYLE.md  (the deepening spec — follow it exactly)
- ${DIR}/STYLE.md         (figure grammar + formatting conventions: [[fig:]] and [[sn:]])
- the CURRENT article at the path below (preserve all its correct facts, numbers, code and cross-links — deepen, never lose or contradict them)

Then REWRITE THE FILE IN PLACE (overwrite it completely) so that it:
- STARTS FROM FOUNDATIONS: establish the prerequisite from scratch in the opening; state the question the article answers; a newcomer must be able to start here and keep up.
- Uses the SOCRATIC method: pose the natural question the reader is asking ("but why does this actually help? let's think about what the hardware is really doing…"), then answer by reasoning from basics; question the obvious; when something is surprising, stop and explain why.
- Builds gradually, one idea at a time, with a central mental model introduced early and reused; derive every formula/number from a tiny concrete by-hand example.
- Stays SIMPLE while going deep: short sentences, conversational, honest caveats, occasional first person; depth from explaining WHY, not jargon.
- Grounds everything in napkin math (bytes/FLOPs/cycles/%) and in what runs in production right now (vLLM, FlashAttention, DeepSeek, H100/B200).
- Is 3,500-6,000 words. Keeps the worklog rhythm for kernel articles (hypothesis -> code -> profile -> BOLD %/x number -> bridge) but expands the "why" at every step.
- Has 6-10 [[fig:]] figures (up from ~3), roughly one every 2-3 sections, and VARIED in type: an intuition/analogy figure early, BEFORE/AFTER side-by-side comparisons for any optimization, a timeline/pipeline figure where things overlap, a zoom-in (whole -> one thread/tile with by-hand numbers), and the precise technical diagram(s). Each [[fig:]] prompt must be detailed, self-contained, and follow the semantic-color Excalidraw grammar (blue=mechanism, green=specs, red=dims/labels, purple=code, orange=emphasis, yellow=packaging/output; hatched matrices; numbered circles; dashed takeaway box).
- Has 5-8 [[sn:]] sidenotes for caveats/exceptions/exact-number corrections.
- Does NOT put an H1 title at the top (the site adds it); start with prose; use ## / ### for internal sections.

If it helps accuracy/depth on a frontier or production topic, FETCH the primary source below.

Return ONLY: the slug, the new word count, and how many [[fig:]] and [[sn:]] the rewrite has. Do not paste the article back.`

function deepenPrompt(a) {
  return `${PRE}

REWRITE THIS FILE IN PLACE: ${DIR}/articles/${a.slug}.md
Primary source to consult for extra depth + accuracy: ${a.src}`
}
function verifyPrompt(a) {
  return `Read ${DIR}/DEEPEN_STYLE.md, then read ${DIR}/articles/${a.slug}.md.
You are the depth editor. Check it against the spec and FIX IN PLACE with Edit if it falls short:
- Is it >= 3,500 words and genuinely deep? If thin, EXPAND it: add Socratic question-and-answer passages, more from-foundations build-up, more napkin math, and more explanation of WHY.
- Does it have >= 6 [[fig:]] of VARIED types (intuition, before/after, timeline, zoom-in, technical)? If fewer or all-same-type, ADD figures with detailed semantic-color Excalidraw prompts.
- Does it start from foundations and stay simple? If it jumps ahead or gets dense, smooth it.
- Are all figure prompts detailed and self-contained? Improve vague ones.
- Technical accuracy: cross-check against your knowledge and ${a.src}; fix any wrong claim; keep all correct facts/code.
Return ONLY: slug, a one-word verdict (CLEAN or DEEPENED), and a terse list of what you changed (max 4 items).`
}

const S = {
  MODAL:'https://modal.com/gpu-glossary', SIBOEHM:'https://siboehm.com/articles/22/CUDA-MMM',
  SALYKOVA:'https://salykova.github.io/gemm-gpu', HORACE:'https://horace.io/brrr_intro.html',
  DAMEK:'https://damek.github.io/random/basic-facts-about-gpus/',
  HAMZA:'https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/',
  CUDAFORFUN:'https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog',
  ALEXARMBR:'https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html',
  KAPILSH:'https://kapilsh.github.io/posts/learn-cutlass-the-hard-way/', GPUPUZZLES:'https://github.com/srush/GPU-Puzzles',
  NVFP4:'https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html',
  VLLMDBG:'https://vllm.ai/blog/2025-12-03-improved-cuda-debugging', SIMONGUO:'https://simonguo.tech/blog/2025-10-automated-gpu-kernels.html',
  CRFM:'https://crfm.stanford.edu/2025/05/28/fast-kernels.html', DSPARK:'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark',
  CS149:'https://github.com/stanford-cs149/asst5-kernels', GPUMODE:'https://github.com/gpu-mode/lectures',
  HORACEDP:'https://siboehm.com/articles/22/data-parallel-training',
}
const ARTICLES = [
  {slug:'the-three-regimes', src:`${S.HORACE}, ${S.DAMEK}`},
  {slug:'gemm-kernel-1-naive', src:`${S.SIBOEHM}`},
  {slug:'why-kernels-run-the-world', src:`${S.HORACE}, ${S.SIMONGUO}`},
  {slug:'speed-of-light-thinking', src:`${S.HORACE}, ${S.DAMEK}`},
  {slug:'the-kernel-engineers-skill-map', src:`${S.SIBOEHM}, ${S.GPUMODE}, ${S.SIMONGUO}`},
  {slug:'how-to-use-this-site', src:`${S.MODAL}`},
  {slug:'streaming-multiprocessor', src:`${S.MODAL}, ${S.HAMZA}`},
  {slug:'cuda-cores', src:`${S.MODAL}, ${S.DAMEK}`},
  {slug:'tensor-cores', src:`${S.MODAL}, ${S.ALEXARMBR}`},
  {slug:'warp-scheduler', src:`${S.MODAL}, ${S.DAMEK}`},
  {slug:'register-file', src:`${S.HAMZA}, ${S.MODAL}`},
  {slug:'shared-memory-l1', src:`${S.HAMZA}, ${S.SIBOEHM}`},
  {slug:'l2-cache', src:`${S.HAMZA}, ${S.MODAL}`},
  {slug:'hbm-global-memory', src:`${S.HAMZA}, ${S.DAMEK}`},
  {slug:'gpc-tpc', src:`${S.HAMZA}, ${S.MODAL}`},
  {slug:'a100-h100-b200-whatchanged', src:`${S.HAMZA}, ${S.NVFP4}`},
  {slug:'roofline-model', src:`${S.HORACE}, ${S.DAMEK}`},
  {slug:'occupancy', src:`${S.SIBOEHM}, ${S.MODAL}`},
  {slug:'arithmetic-intensity', src:`${S.HORACE}, ${S.SIBOEHM}`},
  {slug:'memory-coalescing', src:`${S.SIBOEHM}, ${S.MODAL}`},
  {slug:'threads-warps-blocks-grids', src:`${S.MODAL}, ${S.DAMEK}`},
  {slug:'simt-and-divergence', src:`${S.MODAL}`},
  {slug:'kernel-launch-anatomy', src:`${S.HORACE}, ${S.MODAL}`},
  {slug:'memory-spaces', src:`${S.HAMZA}, ${S.MODAL}`},
  {slug:'ptx-vs-sass', src:`${S.HAMZA}, ${S.MODAL}`},
  {slug:'compute-capability', src:`${S.MODAL}, ${S.HAMZA}`},
  {slug:'bank-conflicts', src:`${S.SIBOEHM}, ${S.SALYKOVA}`},
  {slug:'atomics-and-reductions', src:`${S.MODAL}, ${S.GPUPUZZLES}`},
  {slug:'streams-and-async', src:`${S.SALYKOVA}, ${S.MODAL}`},
  {slug:'your-first-kernel', src:`${S.GPUPUZZLES}, ${S.DAMEK}`},
  {slug:'gpu-puzzles-walkthrough-1', src:`${S.GPUPUZZLES}`},
  {slug:'gpu-puzzles-walkthrough-2', src:`${S.GPUPUZZLES}`},
  {slug:'gemm-kernel-2-coalescing', src:`${S.SIBOEHM}`},
  {slug:'gemm-kernel-3-shared-memory', src:`${S.SIBOEHM}, ${S.SALYKOVA}`},
  {slug:'gemm-kernel-4-1d-blocktiling', src:`${S.SIBOEHM}`},
  {slug:'gemm-kernel-5-2d-blocktiling', src:`${S.SIBOEHM}`},
  {slug:'gemm-kernel-6-vectorized', src:`${S.SIBOEHM}, ${S.HAMZA}`},
  {slug:'gemm-kernel-7-autotuning', src:`${S.SIBOEHM}`},
  {slug:'gemm-kernel-8-warptiling', src:`${S.SIBOEHM}, ${S.HAMZA}`},
  {slug:'gemm-double-buffering-cpasync', src:`${S.SALYKOVA}, ${S.HAMZA}`},
  {slug:'gemm-cublas-baseline', src:`${S.SIBOEHM}, ${S.KAPILSH}`},
  {slug:'gemm-benchmark-methodology', src:`${S.SALYKOVA}, ${S.SIBOEHM}`},
  {slug:'gemm-recap-the-ladder', src:`${S.SIBOEHM}`},
  {slug:'tc-kernel-1-wmma-intro', src:`${S.ALEXARMBR}, ${S.MODAL}`},
  {slug:'tc-kernel-2-fragments-swizzling', src:`${S.ALEXARMBR}, ${S.SALYKOVA}`},
  {slug:'tc-kernel-3-mma-sync-fast', src:`${S.ALEXARMBR}, ${S.HAMZA}`},
  {slug:'prefill-vs-decode', src:`${S.HORACE}, ${S.DAMEK}`},
  {slug:'operator-fusion', src:`${S.HORACE}`},
  {slug:'softmax-from-scratch', src:`${S.HORACE}`},
  {slug:'rmsnorm-from-scratch', src:`${S.HORACE}, ${S.CRFM}`},
  {slug:'attention-naive', src:`${S.HORACE}`},
  {slug:'flashattention-1', src:`${S.HORACE}`},
  {slug:'flashattention-2', src:`${S.HORACE}`},
  {slug:'flashattention-3', src:`${S.HAMZA}, ${S.CUDAFORFUN}`},
  {slug:'kv-cache-and-paged-attention', src:`${S.VLLMDBG}, ${S.DSPARK}`},
  {slug:'quantization-kernels-fp8-int4', src:`${S.NVFP4}, ${S.DSPARK}`},
  {slug:'swiglu-kernel', src:`${S.CS149}, ${S.HORACE}`},
  {slug:'batched-decode-matvec', src:`${S.HORACE}, ${S.DSPARK}`},
  {slug:'hopper-tma', src:`${S.HAMZA}, ${S.CUDAFORFUN}`},
  {slug:'hopper-wgmma-warp-specialization', src:`${S.HAMZA}, ${S.CUDAFORFUN}`},
  {slug:'beating-cublas-on-h100', src:`${S.HAMZA}, ${S.CUDAFORFUN}`},
  {slug:'blackwell-tcgen05-tmem', src:`${S.NVFP4}`},
  {slug:'nvfp4-microscaling', src:`${S.NVFP4}`},
  {slug:'deepseek-flashmla-deepgemm', src:`${S.DSPARK}`},
  {slug:'deepseek-dspark-speculative', src:`${S.DSPARK}`},
  {slug:'cutlass-the-hard-way', src:`${S.KAPILSH}`},
  {slug:'cute-dsl-tilelang', src:`${S.KAPILSH}, ${S.CRFM}`},
  {slug:'debugging-kernels-vllm-workflow', src:`${S.VLLMDBG}`},
  {slug:'kernelbench-and-fast-p', src:`${S.SIMONGUO}`},
  {slug:'test-time-scaling-and-search', src:`${S.SIMONGUO}, ${S.CRFM}`},
  {slug:'crfm-fast-kernels-experiments', src:`${S.CRFM}`},
  {slug:'kevin-rl-and-kernelbook', src:`${S.SIMONGUO}, ${S.GPUMODE}`},
]

phase('Deepen')
log(`Deepening ${ARTICLES.length} articles (rewrite -> verify), Socratic + from-foundations + 6-10 figures each…`)
const results = await pipeline(
  ARTICLES,
  a => agent(deepenPrompt(a), { label: `deepen:${a.slug}`, phase: 'Deepen' }),
  (prev, a) => agent(verifyPrompt(a), { label: `verify:${a.slug}`, phase: 'Verify' }).then(v => ({ slug: a.slug, v })),
)
const ok = results.filter(Boolean)
log(`Deepened ${ok.length}/${ARTICLES.length} articles.`)
return { total: ARTICLES.length, completed: ok.length }
