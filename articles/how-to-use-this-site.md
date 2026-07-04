This site is built the way I actually learned GPU kernels: by writing the dumbest version of a thing, profiling it, being embarrassed by the number, and then earning each improvement one measurement at a time. Nothing here is a lecture that hands you a finished answer. Every optimization is a small argument — a hypothesis, a kernel, a profile, and a bold number that either vindicates the hypothesis or tells me I was wrong. The whole thing is a worklog you can read over my shoulder, and, more importantly, one you can *run*.

Before you dive in, it helps to know how the site is put together, because there are two surfaces here that look nothing alike on purpose, and there are a couple of parallel tracks you can move through at different speeds. This page is the map.

## Two surfaces: the terminal and the paper

The first thing you'll notice is that the site has a split personality, and that split is deliberate.

The **shell** — the home page, the section indexes, the left sidebar with the collapsible article tree — is a dark terminal. Phosphor green on near-black, monospace everything, an ASCII-art GPU die on the landing page, term chips like `GPC` and `SASS` you can click.[[sn: The shell is modeled on Modal's excellent GPU Glossary, which organizes everything into device-hardware / device-software / host-software / performance. If you want a pure reference — "what exactly is a warp scheduler" — that glossary is the companion to this site, and I link into it constantly.]] The shell is where you *navigate*. It is the index, the terminal you keep open in the corner, the thing that tells you where the kernels live.

The **article pages** — like the one you are reading — are the opposite: warm off-white paper, a serif body, a wide right margin full of sidenotes, generous leading. This is the notebook. It is where I *think*. The contrast between the two is the whole point: terminal on the outside, notebook on the inside. When you're hunting for a topic you're in the green terminal; when you're actually working through an idea you're on paper.

[[fig: A hand-drawn Excalidraw-style diagram on pure white, titled in black "Two surfaces", two side-by-side panels labeled (A) and (B). Panel (A): a dark near-black rounded rectangle labeled "THE SHELL", its interior filled with monospace-style hand-lettered green lines mimicking a phosphor terminal — a small ASCII GPU die sketch, a sidebar tree with items "00 Start Here", "GEMM ladder", "▸ Kernel 1", and three small outlined green term chips (GPC)(SASS)(PTX). A green spec annotation with a dashed arrow to the chips reads "monospace · #78f09a on #0b1a10". A blue handwritten margin note with a long thin dashed curved arrow points at the sidebar tree: "navigate here — index + terminal". Panel (B): a warm off-white rounded rectangle labeled "THE PAPER" with wavy serif text lines and a wide right gutter; a red superscript "1." sits in that gutter connected by a thin dashed arrow to a small pale-yellow hatched note box labeled "sidenote". A blue note with a dashed arrow points at the serif lines: "think here — the worklog". A green spec annotation reads "serif · warm #FDFCF9 paper". Between the two panels, a horizontal double-headed dashed arrow labeled in orange "terminal outside · notebook inside". A dashed rounded takeaway box at the bottom in black: "same site, two moods — one for finding, one for reading". Flat, no shadows, hand-lettered labels throughout. || The two surfaces. The green shell is for navigation; the paper articles are for working through an idea.]]

## The worklog method

Every kernel article on this site follows the same loop, and once you see it you'll see it everywhere. It is not a stylistic tic — it is the actual method of performance engineering, compressed into a page.

1. **Hypothesis.** I state, in one sentence, why the next change should help and what regime I think we're in. "This is memory-bound, so staging tiles in shared memory should cut the HBM traffic." Predicting out loud is non-negotiable; a prediction is a thing that can be *wrong*, and being wrong is where the learning is.
2. **Concept, then code.** I explain the idea in prose first — the tiling, the swizzle, the async copy — and only *then* show the kernel. Code before concept teaches you to copy; concept before code teaches you to derive.
3. **Profile as evidence.** Then I point Nsight Compute at it, or read the **SASS** (the actual machine assembly the GPU runs), and let the profiler — not my intuition — say what the bottleneck is. SASS listings here are evidence, not decoration.
4. **A bold number.** Every step ends with a number in bold: a fraction of `cuBLAS`, a speedup, a percentage of peak. Numbers are how we keep score, and they live in the prose, not in a table.
5. **Bridge.** The profile hands us the next hypothesis, and the loop repeats.

The clearest example is the GEMM ladder, which starts at a genuinely humiliating **1.3% of cuBLAS** with the naive one-thread-per-output kernel and climbs, one measured step at a time, to **93.7%** — a library NVIDIA has been tuning for fifteen years, reached from first principles. You can watch the number move: coalescing takes us to **8.5%**, shared memory to **12.8%**, a 1D thread-tile to **36.5%**, a 2D tile to **68.7%**, vectorized loads to **78.4%**, autotuning to **84.8%**, and warp-tiling to **93.7%**. No step is magic; each one is a memory or occupancy fact the profiler forced on us.

[[fig: A hand-drawn "pipeline timeline" titled "The worklog loop" showing five boxes left-to-right connected by black arrows: (1) "HYPOTHESIS" with a blue note "predict the regime out loud", (2) "CONCEPT → CODE" with a purple note "prose first, kernel after", (3) "PROFILE / SASS" with a green note "ncu says the bottleneck", (4) a big orange box "BOLD NUMBER" showing "→ 93.7% of cuBLAS", (5) "BRIDGE" with a curved dashed arrow looping all the way back to box (1) labeled in red "repeat". Above the boxes, a small rising staircase sketch of the GEMM ladder with red rung labels "1.3 → 8.5 → 12.8 → 36.5 → 68.7 → 78.4 → 84.8 → 93.7". Dashed takeaway box: "the profiler picks the next move, not your gut". || The worklog loop. Each kernel is one turn of this cycle, and the number only moves when the profiler agrees.]]

If you read only one thing before the ladder, read [the three regimes](the-three-regimes.html) — compute, memory, and overhead — because the entire method rests on being able to name which of the three you're bottlenecked on, usually in under a minute.

## The GPU-Puzzles async track

Reading a worklog is passive. To actually build the muscle you have to write kernels, and that's what the **GPU-Puzzles** track is for. It runs *alongside* the articles as a self-paced, do-it-when-you-want strand: small, self-contained CUDA puzzles that each isolate exactly one idea — a coalesced load, a shared-memory tile, a reduction, a `float4` vectorized access — with a test harness that either goes green or tells you your indexing is off.

The puzzles are async by design. There's no cohort you have to keep pace with and no unlock gating; each article that introduces a mechanism links to the puzzle that drills it, and you can do them in any order.

[[fig: A hand-drawn "architecture map" of the site's tracks, titled "How the tracks connect". Down the left, a vertical stack of black rounded boxes labeled the reading spine: "00 Start Here" → "Three Regimes" → "Hardware primer" → "GEMM Kernel 1" → "Kernel 2 …" with a green note beside the stack "the knowledge base — standalone". To the right of each kernel box, a small pale-yellow hatched box labeled "GPU-Puzzle" connected by a short blue dashed arrow labeled "drill it now"; a blue margin note reads "async track — do any, any time". Off to the far right, a separate orange rounded box labeled "LIVE LECTURES" with a purple note "synchronous · optional" and a long thin dashed arrow curving back toward the spine labeled in orange "makes it faster, not required". Numbered circles (1)(2)(3) mark the beginner path down the spine. Dashed takeaway box bottom-right: "spine stands alone · puzzles reinforce · lectures accelerate". || How the tracks connect. The article spine is self-contained; puzzles drill each rung; live lectures accelerate but are never required.]]

My honest advice: do the matching puzzle *immediately* after reading a kernel, while the hypothesis is still warm. The gap between "I understood the tiling diagram" and "I can write the tiling and get the right answer" is much larger than it feels, and the puzzles are where you close it. They are also the fastest way to internalize the ugly parts — off-by-one boundary conditions, `threadIdx.x` vs `threadIdx.y` mixups, the moment you forget a `__syncthreads()` and get a race — that no amount of reading will fix.

## Live lectures vs. the knowledge base

There are two ways to consume the material, and they are genuinely independent.

The **live lectures** are the cohort experience: scheduled sessions where we build kernels together in real time, I profile things live, mistakes happen on screen, and you can ask "why is that number so bad" in the moment. They're synchronous, energetic, and the best way to absorb the *judgment* — when to stop optimizing, how to read a red panel in the profiler, which of ten ideas to try first.

The **knowledge base** — this collection of articles — is the permanent, standalone artifact. It does not depend on the lectures and never assumes you attended one. Every concept the lectures cover is written up here in full, with its own figures and its own profiles, so a reader who has never seen a session can still go from the naive GEMM to the warp-tiled one entirely on their own.[[sn: This is a hard rule for the site: no article may say "as we saw in the lecture" as its only explanation of a concept. The knowledge base has to stand completely on its own, or it isn't a knowledge base — it's lecture notes.]] The lectures make the knowledge base faster to absorb; the knowledge base makes the lectures re-readable forever. Use whichever fits how you learn, or both.

## A suggested reading order

The site is a tree, not a line — you can jump anywhere from the sidebar — but if you want a path, here are two, depending on where you're starting.

**If you're new to CUDA**, don't rush to the fast kernels. Start here, then build the mental model before the mechanics:

1. This page, then [the three regimes](the-three-regimes.html) — the single most important idea on the site. Learn to name compute-, memory-, and overhead-bound before anything else.
2. The hardware primer next — what a **Streaming Multiprocessor** (SM) is, what a **warp** (32 threads) is, how the memory hierarchy stacks from registers to shared memory to L2 to HBM. An H100 has ~132 SMs; that number will start to mean something.
3. Then the GEMM ladder *in order*, starting at [the naive kernel](gemm-kernel-1-naive.html). Do not skip kernel 2's coalescing fix — it's the highest payoff-to-effort change in the whole sequence.
4. Do the matching GPU-Puzzle after each rung. Green test, then next article.

**If you already know CUDA** — you've written kernels, you know what `threadIdx` and `__syncthreads()` do, you've launched a grid — you can move faster and more surgically:

1. Skim [the three regimes](the-three-regimes.html) anyway to calibrate on the vocabulary I use for bottlenecks; it's a two-minute read and everything downstream references it.
2. Jump straight to the point on the GEMM ladder where the numbers get interesting — the shared-memory kernel and the 2D thread-tile, where we go from **12.8%** to **68.7%**. That's where the real ideas are.
3. Then the Hopper-specific material: thread-block clusters, distributed shared memory, **TMA** (the Tensor Memory Accelerator for async bulk copies), and `wgmma`, all of which are new in `sm_90a` and change how the fastest kernels are structured.[[sn: If you've done kernel work on Ampere but not Hopper, the async-copy and cluster material is where your instincts will be most out of date — the fastest H100 GEMM does not look like the fastest A100 GEMM. Blackwell (`tcgen05`, Tensor Memory, NVFP4) moves the target again, and gets its own later section.]]
4. Cherry-pick puzzles for the mechanisms you haven't used — most CUDA veterans have never hand-written a `wgmma` tile or a TMA descriptor, and those are the puzzles worth your time.

Either way, the meta-instruction is the same one that runs through every article: **predict the regime, then measure it.** When your prediction is right, you understood the kernel. When it's wrong, you've just found the most valuable thing on the page — a hidden copy, an occupancy cliff, a launch you didn't expect. That habit is the whole course; the kernels are just where we practice it.

Now open the sidebar, pick a rung, and let's go make a number move.
