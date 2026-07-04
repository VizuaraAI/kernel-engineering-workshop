By the end of this chapter you can stand in a three-hour room and run lectures L5 through L8 — the GEMM finale, tensor cores, professional profiling, and attention — minute by minute, knowing exactly what goes on the board, which single demo to run, and the one number that makes each block land.

This is a delivery-craft chapter. You already learned the *ideas* in the earlier chapters — matmul as a grid of dot products, the GPU as a cafeteria that must be fed. Here you learn how to *pace* them. A three-hour lecture is not a document you read aloud. It is a performance with a rhythm: reveal, demo, breathe, checkpoint. Get the rhythm right and even hard material feels easy.

## The shape of every three-hour night

Every one of our live lectures is built the same way: **three 50-minute blocks with two 10-minute breaks.** Never fight this structure. A block is one idea, one board sequence, one demo, one checkpoint question. If you find yourself with four ideas in a block, you have two blocks pretending to be one.

[[note: metaphor || Think of a three-hour lecture like a three-course meal, not a firehose. Each block is a course: an appetizer that makes them hungry (the hook), a main (the board build), and a taste of dessert (the demo — the thing that makes them go "ohh"). The 10-minute breaks are when the last course digests. A cook who serves all three courses at once has not saved time; they have ruined dinner. Pace is not padding — it is how the food gets absorbed.]]

[[fig: A warm hand-drawn illustration titled "The shape of a 3-hour lecture". A horizontal timeline drawn as a friendly train with three carriages, each carriage labeled "Block 1 (50 min)", "Block 2 (50 min)", "Block 3 (50 min)", separated by two small station signs reading "break 10". Inside each carriage, three little icons in a row: a spark labeled "hook", a chalkboard labeled "build", a play-button labeled "demo", and a question-mark labeled "checkpoint". Above the whole train, a green handwritten banner "one block = one idea, one demo, one checkpoint". A dashed takeaway box below: "never put four ideas in one carriage." Excalidraw style, white background, charming, handwritten labels. || The universal rhythm: three blocks, each a self-contained idea-demo-checkpoint unit, with breaks to digest.]]

[[note: teach || Write the block plan on a corner of the board before you start and leave it there all night: "Block 1: X — Block 2: Y — Block 3: Z". Students relax when they can see the map. Tick each one off as you finish it. This costs you thirty seconds and buys you three hours of a calm room.]]

Now let's walk the four lectures.

---

## L5 — GEMM worklog II: the finale (registers → warps)

**The one-sentence goal for the room:** *we take our matmul from one-third of NVIDIA's own library to matching it, and we do it by climbing one rung at a time and reading the machine code to prove each rung worked.*

L5 is the emotional peak of the whole GEMM story. In L4 the students got to 36.5% of cuBLAS. Tonight they get to **93.7%.** That climb is the payoff of four weeks. Your job is to make each rung feel earned, not magical.

### Block 1 (0:00–0:50) — Registers: the fastest shelf (K5, K6)

Start by re-drawing the memory hierarchy from memory — HBM far away, shared memory close, and **registers**, the tiny shelf *inside* each worker's own hands. Then state tonight's whole thesis in one line.

[[note: say || "In L4 we learned to carry a tile of data from the far pantry into shared memory so the whole block could reach it. Tonight we go one step further: we grab little pieces out of shared memory and hold them *in our hands* — in registers — so each thread reuses them many times without asking anyone. That is the last, fastest shelf. There is nothing closer than your own hands."]]

Build kernel 5, **2D blocktiling**, on the board as a picture, not as code first. Each thread now computes not one output element but a small square of them — say an 8×8 patch. Why does that help? Because once a value is in your hand, you can multiply it into *eight* results instead of one. You paid to fetch it once; you cash it in eight times.

[[note: aha || The number that lands Block 1: kernel 5 hits **68.7% of cuBLAS** — we nearly *doubled* K4's 36.5% just by having each thread compute a patch instead of a point. Say it plainly: "We didn't do less math. We did the same math, but we stopped re-fetching numbers we already held." That is the whole game of arithmetic intensity in one sentence.]]

[[fig: A hand-drawn metaphor illustration titled "Registers are your own two hands". A friendly cook figure at a counter. On the far wall, a shelf labeled in green "HBM (far pantry)"; a nearer counter shelf labeled blue "shared memory (block's table)"; and the cook's own two cupped hands, glowing orange, labeled "registers (my hands)". An ingredient is shown being carried once from the table into the hands, then a purple note "used 8 times without asking again" with eight little arrows fanning out to eight finished trays. A red annotation: "fetch once, reuse many". Dashed takeaway box: "the closer the shelf, the more reuse — hands are closest." Excalidraw style, white background, warm, handwritten. || The register tier taught as a cook's own hands: fetch a value once from the table, reuse it eight times without asking again.]]

Then kernel 6, **vectorized loads** (`float4` / `LDS.128`). The idea in plain words: instead of asking the pantry for one number four times, you ask for four numbers in one trip. Same data, one quarter of the requests.

[[note: demo || The signature live demo of L5 is the **SASS diff.** Compile the scalar-load kernel and the `float4` kernel, dump both to assembly, and put them side by side on the projector. In the old one, the inner loop shows eight separate `LDG.E` load instructions. In the new one — two `LDG.E.128`. Say: "Eight loads became two. The compiler is now asking for four numbers per trip." Watching real machine code change in front of them is the moment students stop believing this is abstract.]]

Kernel 6 reaches **78.4%.** Close Block 1 with a checkpoint.

[[note: teach || Checkpoint question for Block 1, asked to the whole room, hands up: "Kernel 5 didn't reduce the number of multiplies at all. So why did it get twice as fast?" The answer you're fishing for: "because it reused each fetched value more times — fewer trips to memory per multiply." If three hands give that answer, you're clear to move on. If not, re-draw the hands picture.]]

### Block 2 (1:00–1:50) — Autotuning and warptiling (the summit)

Come back from break with **autotuning**. Plain words: we have knobs — tile sizes, patch sizes — and nobody is smart enough to guess the best combination. So we let the machine try hundreds of combinations and keep the fastest. Autotuning takes us to **84.8%.**

[[note: metaphor || Autotuning is like tuning a guitar by ear versus with a tuner. You *could* guess the peg positions from theory, but the fastest way to a perfectly tuned string is to try, listen, adjust, try again — automatically, hundreds of times. The machine has no ego about the "right" tile size; it just measures and keeps the winner.]]

Then the summit: **warptiling.** This one needs the warp concept fresh, so re-draw it — a **warp is a gang of 32 threads that always move in lockstep**, like a rowing crew pulling on the same stroke. Warptiling adds a middle layer of organization: the block is split into warps, each warp owns a region, and within a warp the 32 threads cooperate tightly. It hits **93.7% of cuBLAS.**

[[note: aha || Here is the jaw-drop line for the whole GEMM arc. Put the ladder on the board as a staircase: "Naive 1.3% → coalescing 8.5% → shared memory 12.8% → blocktiling 36.5% → register tiling 68.7% → vectorized 78.4% → autotuned 84.8% → warptiling 93.7%." Then: "Same math the whole way. Same GPU. We went from using one-hundredth of the machine to nearly all of it — purely by feeding it better. *That* is what a kernel engineer sells." Let the staircase sit on the board while it sinks in.]]

[[fig: A technical Excalidraw diagram titled "The GEMM staircase to cuBLAS". A rising staircase of eight steps drawn left to right, each step a labeled tile with its percent-of-cuBLAS in large red: "naive 1.3%", "coalesce 8.5%", "SMEM tile 12.8%", "1D blocktile 36.5%", "2D blocktile 68.7%", "vectorize 78.4%", "autotune 84.8%", "warptile 93.7%". The lower steps hatched blue (memory-focused), the upper steps hatched orange (reuse/compute-focused), with a green dashed horizontal line across the top labeled "cuBLAS = 100%". Small purple annotations on the two biggest jumps: "+SMEM reuse" and "+register reuse". A dashed takeaway box: "same math, same chip — every step just feeds the cooks better." Excalidraw style, white background, hand-lettered. || The whole GEMM arc as one staircase: eight rungs from 1.3% to 93.7% of NVIDIA's own library, each rung a better way to feed the GPU.]]

### Block 3 (2:00–2:50) — Live: read the SASS, count the instructions

The third block is hands-in. Have students run the profiler (`ncu`) on kernels 5 and 6 themselves and read the instruction-issue counts. The goal is not new theory — it's confidence that the numbers on the board are real and reproducible.

[[note: production || Land the production link before break: "The kernel you just wrote at 93% is not a toy — this exact climb, in this exact order, is what engineers do at NVIDIA, at DeepSeek, at every serving company. cuBLAS itself was written by people doing what you did tonight. When you match cuBLAS by hand, you have done the actual job that pays $300k+." That sentence is why they signed up.]]

---

## L6 — Tensor cores: the second worklog

**The one-sentence goal:** *there is a second, hidden engine on the GPU built only for matmul — the tensor core — and tonight we learn to feed it, beating our own best kernel from L5.*

The framing that makes L6 click: L4–L5 was a worklog on the *normal* math units (the CUDA cores). L6 is the **same kind of ladder, but on a different engine.** Same rhythm — hypothesis, code, profile, percent — just aimed at a specialized machine.

[[note: metaphor || Until tonight, our thousand cafeteria cooks have each been scooping one spoonful at a time. A tensor core is a **specialized machine that plates a whole 16×16 tray in a single motion** — it does a small matrix-multiply as one hardware instruction. It is breathtakingly fast, but fussy: it will only accept ingredients arranged in an exact, peculiar layout. Most of L6 is about arranging the tray so the machine will accept it.]]

[[fig: A warm hand-drawn illustration titled "The tensor core is a tray-stamping machine". A cook feeds a neatly arranged 4x4 grid of ingredients (labeled "fragment — exact layout required") into a big friendly stamping machine labeled blue "tensor core: does a whole small matmul in one press". Out the other side pops a finished tile labeled yellow "16x16 result, one instruction". A red warning sign beside the input: "will jam if the layout is wrong!". A green note: "way faster than scooping one at a time". Dashed takeaway box: "one press = a whole tile — but only if you arrange the tray exactly right." Excalidraw style, white background, charming, handwritten. || Tensor cores taught as a tray-stamping machine: it plates a whole tile in one press, but only accepts a precise ingredient layout (the fragment).]]

### Block-by-block

**Block 1 (0:00–0:50) — The instruction and the fragment.** Introduce `wmma` / `mma.sync`: the single instruction that multiplies two small matrices. Introduce the **fragment** — the specific way each thread must hold its slice of the data in registers. Do a tiny by-hand example: a 16×16 by 16×16 tile becomes *one* conceptual instruction instead of thousands of FMAs. Checkpoint: "How many multiply-adds is one `wmma` doing under the hood?" (Thousands — that's the point.)

**Block 2 (1:00–1:50) — The precision menu and swizzling.** Walk the precision menu on the board: **TF32, BF16, FP16, FP8** — fewer bits means more throughput but less numerical room. Then the hard part: **SMEM swizzling** to kill bank conflicts. Plain words: when 32 threads all grab from shared memory, if they collide on the same "bank" they queue up; swizzling shuffles the layout so nobody collides.

[[note: confusion || The number-one confusion in L6: students think the tensor core *replaces* everything from L5. It doesn't. Say: "All of L5 still runs — tiling, shared memory, vectorized loads — because the tensor core is starving unless you feed it fast, and feeding it is exactly the L5 skillset. Tensor cores don't retire your kernel craft; they raise the ceiling you're feeding toward." This unlocks the whole lecture.]]

**Block 3 (2:00–2:50) — Live: WMMA beats our best SIMT kernel.** Run a WMMA GEMM and watch it beat the 93.7% kernel from L5. Open `ncu` and show the **tensor-pipe utilization** metric climbing. End with where this goes next.

[[note: production || "Every modern serving stack lives on tensor cores. FlashAttention runs on them. When DeepSeek serves a model in FP8, it's feeding tensor cores in the exact precision menu we drew tonight. CUTLASS and CuTe — the libraries you'll meet in the workshops — exist mostly to arrange that fussy fragment layout for you. On Hopper it's called WGMMA and it's fed by a hardware copy engine called TMA; on Blackwell it's tcgen05. Same idea, bigger stamp." Plant these as a preview, not a deep dive.]]

---

## L7 — Profiling and debugging like a professional

**The one-sentence goal:** *tonight students stop guessing and start diagnosing — reading the profiler like a doctor reads a chart, and hunting three real bugs to ground.*

L7 is different in flavor: less new theory, more craft. The metaphor to open with is medical.

[[note: metaphor || A profiler is a **medical scan**, and you are teaching students to be doctors, not fortune-tellers. A bad engineer guesses: "I think it's slow because of memory." A good one runs the scan, reads the chart, and *knows*: "warp stall reason is 'long scoreboard' — we're waiting on memory, here's the proof." Nsight Compute's SOL section is the vital-signs panel; the stall-reasons breakdown is the diagnosis. Never guess what a scan can tell you.]]

[[fig: A warm hand-drawn illustration titled "The profiler is a doctor's scan". A friendly GPU character lying on a medical bed, with a big monitor beside it showing three readouts labeled in green: "SOL — speed of light (how full is the pipe?)", "memory workload", "warp stall reasons". A doctor figure (the kernel engineer) points a stethoscope at the monitor with a purple speech bubble "stalled on memory — proven, not guessed". A crossed-out red thought-bubble on the side shows a blindfolded figure guessing "maybe it's... memory?". Dashed takeaway box: "read the chart, don't read tea leaves." Excalidraw style, white background, charming, handwritten. || Profiling taught as diagnosis: the good engineer reads the scan (SOL, memory workload, stall reasons) instead of guessing at the cause.]]

### Block-by-block

**Block 1 (0:00–0:50) — Reading Nsight Compute.** Do a slow, guided read of one real `ncu` report on the projector. **SOL section** first ("what percent of peak are we at — the master question from L1, now measured"). Then **memory workload analysis.** Then **warp stall reasons** — the single most useful panel, because it tells you *why* threads are waiting. Checkpoint: "A kernel is at 20% SOL and the top stall reason is 'long scoreboard.' Compute-bound or memory-bound?" (Memory-bound — they're waiting on data.)

**Block 2 (1:00–1:50) — The debugging toolkit.** Walk the real vLLM-style workflow: **compute-sanitizer** for races and bad memory access, handling **hanging kernels**, user-triggered core dumps (`CUDA_ENABLE_USER_TRIGGERED_COREDUMP`), **cuda-gdb**, and `-lineinfo` so the assembly points back to your source lines. Keep it practical — these are tools, introduced by the problem each one solves.

**Block 3 (2:00–2:50) — Live: three sabotaged kernels.** This is the block students remember. Hand them three broken kernels and diagnose each live: a **race condition**, a **misaligned vector load**, and a **silent NaN**.

[[note: demo || Run the three sabotaged kernels one at a time and let the *tool* find the bug, not your intuition. For the race: `compute-sanitizer` prints the exact conflicting threads. For the misaligned `float4` load: the sanitizer flags the misaligned address. For the silent NaN: turn on the checks and watch where it first appears. The lesson lands itself — "I didn't stare at the code and get lucky. I asked the tool and it told me." That is the entire ethos of L7.]]

[[note: confusion || Students think debugging is about being clever enough to spot the bug by reading. Reframe hard: "The professional move is to make the *machine* find the bug. Your cleverness goes into choosing the right tool and reading its answer, not into out-staring 200 lines of CUDA." This flips them from hero-debugging to systematic debugging.]]

[[note: production || "This exact toolkit is what keeps vLLM and every production inference server alive. When a serving cluster hangs at 3am, nobody reads the code hoping for inspiration — they trigger a core dump, open cuda-gdb, and read the scan. The skill you practiced tonight is literally the on-call skill." ]]

---

## L8 — Attention: the kernel that ate the world (+ capstone kickoff)

**The one-sentence goal:** *tonight we build the single most important kernel in modern AI — attention — see why the naive version chokes on memory, fix it with FlashAttention's tiling, and launch the capstone.*

This is the finale of the eight lectures, so it carries weight. Structure it as: build attention → show why it's broken → fix it → hand them the capstone.

### Block 1 (0:00–0:50) — Attention is just matmuls and a softmax

Build attention from parts they already own. **Q, K, V are three matrices.** Scores = Q times K-transpose (a matmul they can already do). Softmax turns scores into weights (a normalization — big scores win, everything sums to 1). Output = weights times V (another matmul). That's it.

[[note: say || "You already know every ingredient in attention. It's a matmul, then a softmax, then a matmul. There is no new arithmetic here tonight — the only new thing is a memory problem, and fixing that memory problem is what made FlashAttention the most-cited kernel of the decade." ]]

Then the catch. For a sequence of length N, the score matrix is **N×N.** At N = 8192 that is 67 million numbers — for *one* head, *one* layer. Writing that giant matrix out to HBM and reading it back is the whole cost.

[[note: aha || The jaw-drop of L8: "Naive attention is memory-bound because it writes and re-reads an N-by-N matrix that never needed to exist. At N=8192 that's 67 million numbers per head, dragged to far memory and back. The math is cheap; the *moving* is what kills you." This is the same lesson as the whole course — feed the cooks — now on the most important kernel in AI.]]

[[fig: A technical Excalidraw diagram titled "Why naive attention is memory-bound". Left: three green-hatched input tiles labeled "Q", "K", "V" with a red dim label "seq len N". Center: a huge red-outlined hatched square labeled "scores N x N" with an orange annotation "67 million numbers at N=8192!" and a blue double-headed arrow to a distant box labeled "HBM (far)" annotated "written out, then read back — twice the traffic". Right: a small yellow output tile "O". A purple note under the big square: "this giant matrix never needed to exist all at once". Dashed takeaway box: "the math is cheap; moving the N x N matrix is what costs." Excalidraw style, white background, hand-lettered. || The core problem of attention: the N-by-N score matrix is enormous, and shuttling it to far memory and back is the real cost.]]

### Block 2 (1:00–1:50) — Online softmax and FlashAttention tiling

The fix has two parts. First, **online softmax:** you can compute a softmax in streaming chunks, keeping a running max and running sum and rescaling as you go — you never need the whole row at once. Do a tiny by-hand example with three numbers arriving one at a time so they *see* the running max update.

[[note: example || By hand on the board, softmax of [1, 3, 2] arriving one at a time. See number 1: running max = 1. See number 3: max jumps to 3, rescale the earlier partial sum by e^(1-3). See number 2: max stays 3, just add e^(2-3). The final answer matches the all-at-once softmax exactly. "We never held all three at once — yet we got the identical result. That's the trick that lets us never write the N×N matrix." ]]

Then **FlashAttention tiling:** walk over K and V in blocks, compute partial results in shared memory, rescale with the online-softmax bookkeeping, and accumulate. The N×N matrix is never materialized in HBM. Build FA v1 live, simplified, single head.

[[note: metaphor || FlashAttention is a **buffet you eat one station at a time while keeping a running total on your napkin.** You don't lay all the food out on one impossibly long table (the N×N matrix). You visit each station (a block of K and V), take what you need, and update the running total on your napkin (the online-softmax max and sum). At the end your plate is exactly right — and you never needed the giant table.]]

[[fig: A warm hand-drawn illustration titled "FlashAttention: a buffet with a running total". A friendly figure walking past several buffet stations in a row, each labeled blue "block of K,V". At each station they take a scoop and update a small napkin they carry, labeled orange "running max + running sum". A crossed-out red giant table in the background labeled "the N x N table we DIDN'T build". At the end, a finished plate labeled yellow "exact output O". Green note: "one station at a time, never the whole table". Dashed takeaway box: "stream the blocks, keep a running total, never materialize N x N." Excalidraw style, white background, charming, handwritten. || FlashAttention as a buffet eaten one station at a time with a running total on a napkin — the online-softmax trick that avoids ever building the giant matrix.]]

[[note: confusion || The confusion here is deep: "if we process in blocks, how is the softmax still correct? Softmax needs the whole row!" The fix is the by-hand three-number demo above. Make them watch the running max rescale the earlier terms. Once they see that rescaling makes streaming *exactly* equal to all-at-once, the doubt dissolves. Do not move on until every hand agrees the two answers matched.]]

Briefly name what **FA2 and FA3** change (better work partitioning across warps; on Hopper, warp specialization and asynchrony) — as a preview to W1, not a deep dive. Then the **KV-cache** note: during generation each new word attends to all previous ones, so decode becomes a skinny matrix-times-vector (GEMV) that is **memory-bound**, not compute-bound. This is the bridge to the inference workshops.

[[note: production || "FlashAttention is not one paper — it is *the* kernel running underneath every chatbot you've used. The instant it appeared, the whole industry adopted it within months because it fed the cooks better on the exact operation that dominates transformer cost. Every time you talk to Claude or DeepSeek or Llama, a descendant of the kernel we just built on this board is running. You just built the most important kernel of the decade, simplified — and you understood every line." ]]

### Block 3 (2:00–2:50) — Capstone kickoff: "You vs the machine"

Close the eight lectures by handing over the signature capstone. Explain it slowly because it defines the rest of the workshop.

Each student picks one operation — **histogram, SwiGLU, a FlashAttention variant, or a heat-equation kernel.** They optimize it **by hand**, keeping a worklog. Then they run an **LLM in the loop** — propose, compile, profile, iterate — against their own kernel. The deliverable documents *both* tracks and, crucially, **what each one found that the other missed.**

[[note: teach || Say the grading rule out loud and let it land: "You are graded on *process*, not raw speedup. A careful worklog that explains why each step helped beats a fast kernel you can't explain. This is the CS149 philosophy — we care that you can reason about the machine." This removes the fear of the fast-kid-wins race and refocuses the room on understanding.]]

[[note: production || "The human-plus-AI-plus-profiler loop you're about to run is exactly the frontier right now. AI-generated kernels have hit 484% of PyTorch on some ops and failed embarrassingly on others — 9% on FlashAttention. The winning pattern everywhere is human judgment plus AI proposals plus a profiler telling the truth. You're not doing a class exercise; you're running the actual 2026 workflow. And it cross-links straight into Vizuara's Harness Engineering workshop — the loop *is* the product." ]]

[[fig: A hand-drawn illustration titled "The capstone: you vs the machine". A friendly split-screen race. Left lane: a human figure at a chalkboard labeled blue "you: optimize by hand + worklog". Right lane: a little robot labeled purple "LLM in the loop: propose -> compile -> profile -> iterate". Both lanes point at the same finish line labeled orange "same kernel, two paths". Below, a green magnifying-glass note "the real grade: what did each path find that the other missed?". A red banner across the top: "graded on PROCESS, not raw speed". Dashed takeaway box: "human + AI + profiler — the actual 2026 workflow." Excalidraw style, white background, charming, handwritten. || The capstone framing: optimize a kernel by hand and with an LLM-in-the-loop, then compare what each path discovered — graded on process, not raw speedup.]]

[[sn: If the room is running behind on L8, protect Block 2. You can compress the FA2/FA3 preview and the KV-cache aside to two minutes each, but never rush the online-softmax by-hand demo — it is the one thing that makes FlashAttention click, and a confused room here undermines the capstone.]]

[[sn: For L5, if `ncu` access is flaky in the room, pre-record the SASS diff and the profiler screens the night before. The demo's power is in *seeing eight loads become two*; a screenshot delivers that just as well as a live run, and removes the risk of a failed live command killing your momentum at the summit.]]

---

## You can now teach

- **L5 minute by minute:** the register-and-warp climb from 36.5% to 93.7% of cuBLAS, with the SASS-diff demo (eight loads → two) as the centerpiece and the eight-rung staircase as the jaw-drop.
- **L6 minute by minute:** tensor cores as a tray-stamping machine, the fragment layout and precision menu, swizzling to kill bank conflicts, and the live WMMA kernel beating your best SIMT kernel.
- **L7 minute by minute:** the profiler as a doctor's scan (SOL, memory workload, stall reasons), the real debugging toolkit, and the three-sabotaged-kernels live diagnosis.
- **L8 minute by minute:** attention as matmul-softmax-matmul, why naive attention is memory-bound (the 67-million-number N×N matrix), online softmax by hand, FlashAttention tiling built live, and the "you vs the machine" capstone kickoff.
- **The universal pacing craft:** three 50-minute blocks, one idea and one demo and one checkpoint per block, the block map left on the board, and the discipline to protect the one by-hand demo that makes each lecture click.
