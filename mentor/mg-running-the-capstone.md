By the end of this chapter you'll be able to set up, mentor, and grade the "You vs the machine" capstone — the final project where each student writes a kernel by hand, points a language model at the same problem, and keeps a **worklog** of the fight. And you'll be able to do the one thing that makes this capstone work: grade the *thinking*, not the raw speedup.

This is where the four weeks come together. The capstone is a supervised sprint, and your job shifts from "explainer at the whiteboard" to "coach on the sideline." Let's build the coaching muscle.

## The one-sentence version of the capstone

Every student picks one kernel problem and does two things with it. **First**, they write the kernel themselves, by hand, measuring as they go — the predict-then-measure loop we've drilled all month. **Second**, they hand the *exact same* PyTorch reference to a language model and ask it to write the kernel too. Then they compare, in a worklog: where the human won, where the model won, why, and what each got stuck on.

The deliverable is **not** the fastest kernel. The deliverable is the worklog.

[[note: metaphor || The capstone is a **chess match with an engine, written up as a diary**. A beginner who plays the computer and loses learns nothing if all they record is "I lost." A student who writes down *"on move 14 I hung my knight because I didn't see the pin — the engine punished it instantly"* has learned chess. We are not grading who won the game. We are grading the quality of the diary. The student who honestly writes "the model beat me and here is exactly why" gets a higher grade than the student who squeaked out a 1.05× win and can't explain it.]]

[[fig: A warm hand-drawn illustration titled "You vs the machine". Center: a friendly student figure at a desk with a pencil, facing a friendly robot figure across a small table, a chess-clock between them. Both are working on the SAME sheet pinned above labeled in green "one PyTorch reference module". The human's side of the table has a handwritten scratchpad labeled blue "by-hand kernel + measurements"; the robot's side has a little screen labeled blue "LLM-generated .cu". Below the table, front and center, a thick open notebook labeled in orange "THE WORKLOG — this is what gets graded", with a red arrow pointing to it and a note "not the winner — the write-up". Excalidraw style, white background, charming, handwritten labels. || The capstone framed as a match plus a diary: human and model attack the same reference, but the graded artifact is the worklog, not the winner.]]

## Why a worklog and not a leaderboard

Here is the trap you must steer students away from. If you grade on raw speedup, you teach students to chase a *lucky number* instead of understanding. Kernel speed is noisy, hardware-specific, and often decided by one lucky guess about a tile size. A leaderboard rewards the coin-flip. A worklog rewards the *reasoning*.

A real benchmark already made this exact design decision, and you should tell your students about it — it makes the capstone feel like the real world, because it is. It's called **KernelBench**.

[[note: production || KernelBench (from Stanford's Hazy Research group — Anne Ouyang, Simon Guo and collaborators) is the serious benchmark for exactly this task: hand a model a PyTorch `nn.Module`, ask it to emit a functionally identical kernel in inline CUDA, faster. It's not academic trivia — it's the yardstick people actually use to ask "can an LLM write GPU kernels yet?" Your capstone is a hand-run, human-scale version of the same experiment. When a student runs their own "You vs the machine" match, they are reproducing, on one problem, what a whole research field runs on hundreds.]]

The most important thing KernelBench got right is its **scoring metric**, and it's the backbone of how you'll grade. It refuses to reward the wrong thing. Let me build it up the way you'll build it for students.

## The two gates: correct AND faster

A kernel can fail in two completely different ways, and students always forget one of them. Say this line out loud at the board and let it sit.

[[note: say || "A slow-but-correct kernel is useless. A blazing-fast-but-wrong kernel is *worse* than useless — it's a bug that ships. So there are two gates, not one. Your kernel has to be **correct**, and it has to be **faster** than what you started with. Miss either gate and you scored zero. This is the whole job."]]

Gate one is **correctness**. You run both the reference and the candidate on random inputs and check the outputs match — not bit-for-bit, but within a tolerance (an `allclose` with a sensible `atol`/`rtol`). Why a tolerance? Because a legitimate fast kernel might add up numbers in a different order, or use slightly lower precision, and floating-point arithmetic isn't perfectly associative. The tolerance *forgives honest floating-point reordering* while still *catching a real bug*.

Gate two is **speed**. You time both forward passes and take the ratio: `speedup = t_reference / t_candidate`. Above 1× means the candidate is genuinely faster than PyTorch.

[[note: example || Do the arithmetic on the board so "speedup" stops being a vibe and becomes a number. Say PyTorch's forward pass takes **4.0 milliseconds** and the student's kernel takes **1.6 milliseconds**. Speedup = 4.0 / 1.6 = **2.5×**. Now flip it: if the student's kernel is *slower*, say 5.0 ms, then 4.0 / 5.0 = **0.8×** — below 1×, it lost to PyTorch, gate two fails, score zero even if the numbers were perfectly correct. Make them feel that a "correct" kernel can still score zero.]]

[[fig: A hand-drawn 2D scatter titled "Two gates: correct AND faster". X-axis (red label) "speedup = t_ref / t_cand" with a red dashed vertical line at x=1 labeled "1x (beat PyTorch)". Y-axis (red label) "correct? passes allclose" with a horizontal blue dashed line splitting CORRECT (top) from WRONG (bottom). Four quadrants with hand-drawn dots: bottom-left "wrong AND slow — worthless"; bottom-right "fast but WRONG — worse, ships a bug" with a small skull doodle; top-left (correct, left of 1x) "correct but SLOWER — still zero, fails gate 2"; top-right green-shaded region "the only cell that counts — correct AND faster" with an orange star. Dashed takeaway box: "two independent gates — you must clear BOTH". Excalidraw style, white background, handwritten. || The scoring picture students must internalize: correctness and speed are orthogonal, and only the top-right quadrant scores.]]

## fast_p: one dial that sets the bar

KernelBench folds both gates into one clean idea called **fast_p**. Here `p` is a number you pick — the speedup you're demanding. fast_p is "the fraction of problems where the kernel is correct *and* at least `p×` faster than PyTorch." Slide the dial and the meaning changes:

- **fast_0** is basically just "is it correct?" — any speed counts.
- **fast_1** is the headline: correct *and* strictly faster than PyTorch (speedup `> 1×`). This is "did you actually beat the framework?"
- **fast_2** is a real engineering win: correct *and* more than twice as fast (`> 2×`).

[[note: aha || Here's the number that motivates the whole capstone, and it lands hard. Point today's frontier language models at KernelBench, ask for one kernel per problem, and score fast_1: they produce a kernel that's correct *and* faster than plain PyTorch **less than 20% of the time.** Four out of five attempts either don't compile, give wrong numbers, or give right numbers *slower* than the framework. Say the punchline: "The machine you're racing in this capstone fails four times out of five. You have spent four weeks learning the exact skill it's worst at. Go find out where you beat it."]]

[[fig: A hand-drawn "dial" figure titled "fast_p: one knob, three bars". A large hand-drawn rotary dial in the center with three labeled detents. At the left detent, red "fast_0 = just correct (any speed)". At the middle detent, orange "fast_1 = correct AND > 1x — beat PyTorch". At the right detent, green "fast_2 = correct AND > 2x — a real win". A blue arrow shows the dial pointing at fast_1 with a callout "the headline bar". Below the dial, a big hand-lettered stat in an orange box: "frontier LLMs clear fast_1 < 20% of the time — 4 in 5 attempts fail". Dashed takeaway box: "sliding p right raises the speed bar; correctness is always required". Excalidraw style, white background, handwritten. || fast_p as a single dial: p sets how much speedup you demand, and even the best models clear the fast_1 detent less than a fifth of the time.]]

## Setting up the capstone (the logistics)

Now the practical part — how you run this in the room. Keep the moving parts small.

**Pick the problem tier deliberately.** KernelBench sorts problems into three levels, and the level decides how the match goes — so choose it to make a *teachable* fight, not an impossible one.

- **Level 1 — a single operator** (one matmul, one softmax, one layernorm). Brutal, because PyTorch's reference already dispatches to a tuned library like cuBLAS. Even a strong hand-written matmul only reaches about **93.7% of cuBLAS**, so "just beat PyTorch" is a high bar here. Assign Level 1 only to your strongest students, and set the expectation that *losing to the library is a normal, publishable result.*
- **Level 2 — an operator sequence** (`conv → bias → scale → ReLU`). This is the sweet spot for most students. The win is **fusion**: collapse the chain into one kernel so the intermediate results never leave the chip. It's a memory-movement win, it's learnable, and it's where students most often *beat* both PyTorch and the model.
- **Level 3 — a full architecture block** (a ViT block, a Mamba block). Dozens of ops, multiple bottlenecks. The skill is *finding where the time actually goes* before optimizing. Great for a team, hard for a solo student.

[[note: teach || Default everyone to **Level 2** unless they ask for a harder fight. It's the level where the human has a genuine edge over the model — fusion is a memory-logistics win, exactly the "feed the cooks" lesson from week one — so students get the satisfying experience of beating the machine for a reason they can articulate. Level 1 is where they'll usually *lose* to the library, and that's a fine lesson too, but don't make it the default or half the room goes home demoralized.]]

[[fig: A hand-drawn three-tier pyramid titled "Which fight to assign", three stacked bands. BOTTOM (widest), red "LEVEL 1 — single op (matmul, softmax)", blue note "you're racing a tuned library — expect to lose, that's data". MIDDLE, red "LEVEL 2 — op sequence (conv→bias→relu)", green note "FUSION — the human's best shot to win — DEFAULT". TOP (narrowest), red "LEVEL 3 — whole block (ViT / Mamba)", purple note "find the real bottleneck first — team project". A curved orange arrow up the side labeled "harder ≠ better for learning — pick the teachable fight". Excalidraw style, white background, handwritten. || Choosing the capstone tier: Level 2 fusion is the default because it's the fight the human is most likely to win for an explainable reason.]]

**The measurement harness.** Give students a ready-made harness so nobody loses a week to timing bugs. It does three things and nothing more: (1) runs both modules on several random inputs and checks `allclose`, (2) times both forward passes and reports the speedup, (3) warms up the GPU before timing so the first-run overhead doesn't pollute the number.

[[fig: A hand-drawn "kitchen inspector" metaphor illustration titled "The harness checks two things". A friendly inspector figure with a clipboard stands between two identical plates of food on a counter, one labeled green "reference (PyTorch)" and one labeled blue "candidate (student/model kernel)". Above the plates, gate one: a red magnifying glass over both plates with a note "taste-test: do they match? (allclose)". To the side, gate two: a hand-drawn stopwatch timing both with a note "which is served faster? (t_ref / t_cand)". The inspector's clipboard shows two checkboxes: "correct?" and "faster?" with a note "both ticked or it doesn't pass". Excalidraw style, white background, charming, handwritten labels. || The harness as a kitchen inspector: it tastes both dishes for a match, then times which is served faster — both boxes must be ticked.]]

[[note: demo || The one live demo to run on setup day: take a trivial element-wise kernel, run it through the harness twice. First time, *don't* warm up the GPU — show the wildly inflated time. Second time, warm up first — show the honest number. The gap makes them believe the harness matters. Then run a deliberately buggy kernel (off-by-one in the indexing) and watch `allclose` catch it and print FAIL. Now they trust both gates before they've written a line themselves.]]

## What the worklog must contain

This is the graded artifact, so be explicit about its skeleton. A good worklog has five sections — hand them this list on a slip of paper.

1. **The problem and the plan.** Which PyTorch reference, which tier, and the student's *predicted* bottleneck before writing anything — memory-bound or compute-bound? (The predict-then-measure loop starts here.)
2. **The by-hand attempt.** Each kernel version, its measured time, and *why* they changed it. Not "v2 was faster" — "v2 tiled into shared memory because v1 was re-reading the same row from HBM 32 times."
3. **The model's attempt.** The exact prompt, what the model emitted, and whether it passed both gates on the first try. Did it even compile? Did the numbers match?
4. **The head-to-head.** The final speedups side by side, and — this is the heart of it — *an honest explanation of the gap in either direction.*
5. **What each got stuck on.** The single most valuable page. Where did the human waste an afternoon? Where did the model confidently produce garbage?

[[note: confusion || The most common student mistake in the write-up: they treat a *slower* result as a failure to hide, so they fudge the prompt until the model loses, or they cherry-pick their fastest lucky run. Head this off directly. Tell them: "A clean, honest 'the model beat me by 1.4× and here is exactly the tensor-core trick it used that I couldn't' is an **A**. A suspicious 3× win you can't explain is a **C**. I am grading the diary, not the scoreboard — you cannot lose this capstone by losing the race." Say it on day one and repeat it at the midpoint.]]

## Where the machine reliably breaks (coach with this)

You'll mentor better if you know where the language model tends to fail — that's where you steer students to look for their win. The failure profile from KernelBench is remarkably consistent.

The model does *relatively well* on **Level 2 fusion**, where the win is simply "delete a round-trip to HBM" — a pattern all over its training data. It does *badly* wherever the win requires **hardware-specific intrinsics and tensor-core utilization** — orchestrating things like `wgmma`, shared-memory swizzling, and async copies with no margin for error. That deep orchestration is hard to learn from text, and it's precisely the skill your students spent four weeks building by hand.

[[note: production || This is the genuinely encouraging note to end the workshop on. In real kernel-engineering work right now — the code inside vLLM, inside FlashAttention, inside the stacks serving DeepSeek and Llama on H100 and B200 clusters — the parts a model can auto-generate are the easy fusions. The parts that still need a human are exactly the deep tensor-core, memory-orchestration parts your students just learned. The capstone isn't a party trick. It's a live demonstration of which half of this job is still theirs.]]

[[fig: A hand-drawn "where the machine breaks" figure, two columns. LEFT column headed green "model does OK here": a box "Level 2 element-wise fusion" with a note "win = delete an HBM round-trip — common in training data". RIGHT column headed red "model reliably fails here": a box "tensor-core intrinsics (wgmma, swizzling, async copy)" with a note "no margin for error — hard to learn from text", and a small broken-gear doodle. A big orange arrow from the right column pointing down to a banner "<- YOUR students' edge — steer them here to find their win". Dashed takeaway box: "the human still owns the deep hardware orchestration". Excalidraw style, white background, handwritten. || The model's failure map, used as a coaching tool: point students at tensor-core-heavy problems where their four weeks of hand-tuning is the decisive advantage.]]

## The grading rubric (make it concrete)

Give yourself and the students a rubric that matches everything above. A workable split:

- **Correctness & measurement discipline (30%)** — did both attempts actually pass gate one, and is the timing done honestly (warmup, multiple runs, reported ratio)?
- **Depth of the by-hand worklog (30%)** — does each kernel version have a *reason*, tied to a measurement, not a guess?
- **Quality of the head-to-head analysis (30%)** — is the gap, in either direction, explained with real mechanism (memory movement, tensor cores, fusion), not vibes?
- **Honesty (10%)** — losing to the model, cleanly explained, scores full marks here; a hidden or fudged result loses all of it.

[[sn: Notice the raw speedup number appears in *zero* of these buckets on its own. It only matters as evidence inside "measurement discipline" and "analysis." A student can score 100% while losing every single race, as long as the worklog is honest and mechanistic. That is the whole philosophy of the capstone in one rubric.]]

## You can now teach

- The **capstone framing**: a "You vs the machine" match on one shared PyTorch reference, where the graded deliverable is the **worklog**, not the winner.
- The **two-gate scoring idea** — correct *and* faster — and **fast_p** as one dial (fast_0 / fast_1 / fast_2), with the honest allclose-plus-timing harness behind it.
- The **jaw-drop number**: frontier LLMs clear fast_1 less than **20%** of the time — the emotional hook that tells students the machine is beatable.
- **Choosing the tier** (Level 1 / 2 / 3) to set up a *teachable* fight, and why Level 2 fusion is the default where the human most often wins.
- **The worklog skeleton and rubric** — five sections, four grading buckets, raw speedup deliberately not on the scoreboard.
- **Where the model reliably breaks** (tensor-core intrinsics, deep memory orchestration) and how to steer students toward the win their four weeks earned them.
