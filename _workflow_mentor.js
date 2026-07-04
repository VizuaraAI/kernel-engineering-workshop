export const meta = {
  name: 'mentor-chapters',
  description: 'Write the 35 remaining Mentor Handbook chapters (teach-the-teacher, metaphor-rich, from scratch)',
  phases: [{ title: 'Write' }],
}
const DIR = 'kernel-engineering-site'
const PRE = `You are writing ONE chapter of "The Mentor's Handbook" for the Kernel Engineering workshop — a teach-the-teacher companion for Dr. Raj Dandekar and Shubham Panchal. The goal: after reading this chapter, a mentor who started knowing nothing about the topic can confidently LEARN it and then STAND UP and TEACH it.

READ THESE FIRST (non-negotiable):
- ${DIR}/MENTOR_STYLE.md  (the handbook style spec — follow it exactly)
- ${DIR}/mentor/mg-matmul-from-scratch.md  (exemplar chapter — match this voice and depth)
- ${DIR}/mentor/mg-cpu-vs-gpu.md            (exemplar chapter — match this voice and depth)
- the grounding article(s) listed below, for the correct facts, numbers and code (read them so the chapter is technically accurate).

Then WRITE the file at the path below, following MENTOR_STYLE.md:
- Warm, simple, second-person-to-the-mentor voice. Explain from ZERO. Short sentences. Over-explain rather than under-explain. It is fine (encouraged) to be expansive.
- Every concept gets the seven ingredients: plain words -> a METAPHOR -> a tiny BY-HAND number -> the real math built up gently -> "in production TODAY" link (vLLM/FlashAttention/DeepSeek/H100-B200) -> teaching notes (board plan, sequence, the demo, the jaw-drop number) -> the common student confusion + the fix.
- Use the callout blocks liberally, EXACT syntax on their own line: [[note: TYPE || content]] where TYPE is one of metaphor, example, production, teach, say, demo, confusion, aha. Aim for 6-10 callouts spread through the chapter.
- 6-9 [[fig:]] figures. MIX two flavors: (1) warm METAPHOR illustrations that draw the analogy itself (kitchens, marching bands, warehouses, post offices, highways — charming, hand-drawn, friendly), and (2) technical Excalidraw diagrams with the semantic-color grammar (blue=mechanism, green=specs, red=dims/labels, purple=code, orange=emphasis, yellow=packaging/output; hatched matrices; numbered circles; dashed takeaway box). Often pair a metaphor figure with its technical translation. Every [[fig:]] prompt must be detailed, self-contained, white background, hand-lettered labels.
- A few [[sn:]] sidenotes for nuances are welcome.
- Open with the ONE-sentence goal ("By the end of this chapter you can teach …"). End with a "## You can now teach" section: a 4-6 bullet checklist of what the mentor can now deliver.
- Length 1,600-3,000 words. Do NOT put an H1 title at the top (the site adds it); start with prose; use ## / ### internally.

For the delivery-craft chapters (lecture/workshop plans), also read ${DIR}/CURRICULUM.md and give concrete minute-by-minute timings, the board sequence, the one live demo per block, and checkpoint questions.

Return ONLY: the slug, the word count, and how many [[fig:]] and [[note:]] callouts you included. Do not paste the chapter back.`

function chPrompt(c) {
  return `${PRE}

WRITE THIS FILE: ${DIR}/mentor/${c.slug}.md
CHAPTER TITLE (do not repeat as H1): "${c.title}"
WHAT THIS CHAPTER TEACHES THE MENTOR: ${c.blurb}
GROUNDING ARTICLES to read for correct facts/numbers/code: ${c.ground || '(use the book generally + your knowledge)'}`
}

const A = s => `${DIR}/articles/${s}.md`
const CH = [
  {slug:'mg-how-to-use-this-guide', title:'How to use this handbook', blurb:'Meta: the seven ingredients, how to read it (learn first then teach), and how to prep a session from a chapter.', ground:`${DIR}/MENTOR_STYLE.md, ${DIR}/CURRICULUM.md`},
  {slug:'mg-the-mentors-mindset', title:'The mentor’s mindset: teaching hard things simply', blurb:'You only understand something when you can teach a beginner. Building from zero, using metaphors, turning abstraction into board arithmetic.', ground:''},
  {slug:'mg-the-whole-arc', title:'The whole arc: one story, fourteen sessions', blurb:'The single narrative (make the matmul fast, then make the model fast) that ties every lecture and workshop together, drawn as one map.', ground:`${DIR}/CURRICULUM.md`},
  {slug:'mg-why-matmul-is-everything', title:'Why matrix multiply is the whole game', blurb:'Every linear layer, attention score and MLP is a matmul; trace a transformer and count them; why kernels matter. Production link.', ground:`${A('why-kernels-run-the-world')}, ${A('attention-naive')}`},
  {slug:'mg-what-fast-means', title:'What ‘fast’ actually means: time, FLOPs, bytes', blurb:'Operations, FLOP/s, bytes moved, why a stopwatch is not enough. The units a mentor must own, taught from scratch.', ground:`${A('arithmetic-intensity')}, ${A('the-three-regimes')}`},
  {slug:'mg-compute-vs-memory', title:'Compute vs memory: the kitchen that governs everything', blurb:'The three regimes (compute/memory/overhead) as a kitchen story — cooking vs fetching ingredients. The master mental model.', ground:`${A('the-three-regimes')}, ${A('roofline-model')}`},
  {slug:'mg-numbers-to-know-cold', title:'The numbers to know cold', blurb:'989 TFLOP/s, 3.35 TB/s, 132 SMs, 228 KiB, 32 threads — the handful of H100 numbers that turn hand-waving into intuition.', ground:`${A('hbm-global-memory')}, ${A('tensor-cores')}, ${A('streaming-multiprocessor')}`},
  {slug:'mg-the-roofline-simply', title:'The roofline, drawn simply', blurb:'The one plot that says the fastest a kernel could ever go — a compute ceiling and a memory ramp — with no intimidation.', ground:`${A('roofline-model')}, ${A('speed-of-light-thinking')}`},
  {slug:'mg-threads-warps-blocks', title:'Threads, warps & blocks: the marching band', blurb:'One thread = one musician; a warp of 32 marches in lockstep; blocks share a room. The execution hierarchy as a parade.', ground:`${A('threads-warps-blocks-grids')}, ${A('simt-and-divergence')}`},
  {slug:'mg-the-memory-hierarchy', title:'The memory hierarchy: desk, drawers, shelves, warehouse', blurb:'Registers, shared memory, L2, HBM as distances you walk for a tool; why the whole craft is keeping data close.', ground:`${A('memory-spaces')}, ${A('shared-memory-l1')}, ${A('hbm-global-memory')}`},
  {slug:'mg-tensor-cores-simply', title:'Tensor cores: the matmul machine inside the machine', blurb:'A special unit that eats little matrices whole; where 95% of the FLOPs live; why kernels are built around it.', ground:`${A('tensor-cores')}`},
  {slug:'mg-latency-hiding', title:'Latency hiding: the short-order cook juggling tickets', blurb:'Memory is slow, so the GPU keeps dozens of jobs in flight and works on a ready one; why occupancy matters.', ground:`${A('warp-scheduler')}, ${A('occupancy')}`},
  {slug:'mg-coalescing-simply', title:'Coalescing: everybody boards the same bus', blurb:'32 threads reading neighbouring addresses fetch in one trip; scattered reads waste the bus; the cheapest big win.', ground:`${A('memory-coalescing')}`},
  {slug:'mg-teach-naive-gemm', title:'Teaching Kernel 1: the honest, slow start', blurb:'One thread per output; run it, show 1.3% of cuBLAS, make students hungry to fix it. Board plan + demo.', ground:`${A('gemm-kernel-1-naive')}`},
  {slug:'mg-teach-coalescing', title:'Teaching Kernel 2: the one-line miracle', blurb:'Remap two indices, 6x faster; build the suspense and reveal; draw coalesced vs scattered bus.', ground:`${A('gemm-kernel-2-coalescing')}, ${A('memory-coalescing')}`},
  {slug:'mg-teach-shared-memory-tiling', title:'Teaching Kernel 3: the shared whiteboard', blurb:'Stage tiles in shared memory so the block reuses them; the reuse idea drawn; why we block over K.', ground:`${A('gemm-kernel-3-shared-memory')}, ${A('shared-memory-l1')}`},
  {slug:'mg-teach-register-tiling', title:'Teaching Kernels 4-5: pockets full of work', blurb:'Each thread computes many outputs from register-held values; 1D then 2D tiling; the intensity climb to 68.7%.', ground:`${A('gemm-kernel-4-1d-blocktiling')}, ${A('gemm-kernel-5-2d-blocktiling')}, ${A('arithmetic-intensity')}`},
  {slug:'mg-teach-vectorization', title:'Teaching Kernel 6: carry four boxes at once', blurb:'float4 loads and the SASS eight-loads-become-two moment — the most satisfying reveal in the course.', ground:`${A('gemm-kernel-6-vectorized')}, ${A('ptx-vs-sass')}`},
  {slug:'mg-teach-warptiling', title:'Teaching Kernels 7-8: the org chart', blurb:'Autotuning and warptiling — making block/warp/thread explicit to reach 94% of cuBLAS; how to land the finale.', ground:`${A('gemm-kernel-8-warptiling')}, ${A('gemm-kernel-7-autotuning')}, ${A('gemm-recap-the-ladder')}`},
  {slug:'mg-teach-tensor-core-gemm', title:'Teaching the tensor-core rebuild', blurb:'Do the ladder again on tensor cores: fragments, swizzling, mma.sync; teach WMMA without drowning students.', ground:`${A('tc-kernel-1-wmma-intro')}, ${A('tc-kernel-2-fragments-swizzling')}`},
  {slug:'mg-teach-fusion', title:'Teaching fusion: stop driving to the warehouse', blurb:'Why round-tripping tensors through HBM between tiny ops is the waste fusion removes; the biggest inference win.', ground:`${A('operator-fusion')}`},
  {slug:'mg-teach-softmax-online', title:'Teaching softmax & the online trick', blurb:'Stable softmax by hand, then the running-max/running-sum trick that makes FlashAttention possible; teach as a running average.', ground:`${A('softmax-from-scratch')}`},
  {slug:'mg-teach-attention', title:'Teaching attention as three matmuls', blurb:'QK^T -> softmax -> V, and why the giant N x N scores are the problem; set up FlashAttention as the fix.', ground:`${A('attention-naive')}`},
  {slug:'mg-teach-flashattention', title:'Teaching FlashAttention: never write the scores down', blurb:'Streaming attention in tiles so the N x N matrix never touches memory; the metaphor, the tiling, the production stakes.', ground:`${A('flashattention-1')}, ${A('flashattention-2')}`},
  {slug:'mg-teach-kv-cache-decode', title:'Teaching the KV cache: why chat is memory-bound', blurb:'Prefill vs decode, the KV cache, why each token re-reads a growing memory; PagedAttention as the fix.', ground:`${A('kv-cache-and-paged-attention')}, ${A('prefill-vs-decode')}`},
  {slug:'mg-teach-hopper', title:'Teaching Hopper: the async delivery truck (TMA & WGMMA)', blurb:'The H100 hardware copy engine and warpgroup matmul in words a student can hold; why FlashAttention-3 needs it.', ground:`${A('hopper-tma')}, ${A('hopper-wgmma-warp-specialization')}`},
  {slug:'mg-teach-blackwell-nvfp4', title:'Teaching Blackwell & NVFP4: four-bit numbers that work', blurb:'Tensor memory, and micro-scaled 4-bit formats; the 2000us->22us hackathon story as your hook.', ground:`${A('nvfp4-microscaling')}, ${A('blackwell-tcgen05-tmem')}`},
  {slug:'mg-teach-deepseek-dspark', title:'Teaching DeepSeek & DSpark: kernels behind a frontier model', blurb:'FlashMLA, DeepGEMM, and DSpark = speculative decoding; how an open lab ships its own kernels — the ultimate production link.', ground:`${A('deepseek-dspark-speculative')}, ${A('deepseek-flashmla-deepgemm')}`},
  {slug:'mg-teach-ai-writes-kernels', title:'Teaching AI that writes kernels', blurb:'KernelBench, the monkeys that sample 100 kernels, the honest wins and failures; the human+AI+profiler loop and the tie to harnesses.', ground:`${A('kernelbench-and-fast-p')}, ${A('crfm-fast-kernels-experiments')}, ${A('test-time-scaling-and-search')}`},
  {slug:'mg-teach-debugging', title:'Teaching debugging: war stories that stick', blurb:'The race, the misaligned load, the silent NaN; compute-sanitizer, core dumps and cuda-gdb as detective stories.', ground:`${A('debugging-kernels-vllm-workflow')}`},
  {slug:'mg-lecture-plans-l1-l4', title:'Lecture plans: L1-L4, minute by minute', blurb:'Timings, board sequence, the one demo, and checkpoint questions for the first four lectures.', ground:`${DIR}/CURRICULUM.md`},
  {slug:'mg-lecture-plans-l5-l8', title:'Lecture plans: L5-L8, minute by minute', blurb:'The GEMM finale, tensor cores, profiling/debugging, and attention — paced for a three-hour room.', ground:`${DIR}/CURRICULUM.md`},
  {slug:'mg-workshop-plans-w1-w6', title:'Workshop plans: the six deep-dives', blurb:'FlashAttention, Hopper, the abstraction ladder, serving kernels, Blackwell/NVFP4, DeepSeek+AI finale — how to run each hands-on.', ground:`${DIR}/CURRICULUM.md`},
  {slug:'mg-student-questions', title:'The questions students always ask (with crisp answers)', blurb:'A ready bank of recurring questions and tight, correct answers so you are never caught flat.', ground:''},
  {slug:'mg-running-the-capstone', title:'Running the ‘You vs the machine’ capstone', blurb:'How to set up, mentor, and grade the by-hand-plus-LLM kernel capstone on the worklog, not the raw speedup.', ground:`${A('kernelbench-and-fast-p')}`},
]

phase('Write')
log(`Writing ${CH.length} mentor handbook chapters in parallel…`)
const results = await parallel(CH.map(c => () => agent(chPrompt(c), { label: `mentor:${c.slug}`, phase: 'Write' })))
const ok = results.filter(Boolean)
log(`Wrote ${ok.length}/${CH.length} mentor chapters.`)
return { total: CH.length, completed: ok.length }
