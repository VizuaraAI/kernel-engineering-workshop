By the end of this chapter you can walk into the room for any of the first four lectures with a wristwatch, a marker, and a plan — knowing exactly what to draw, when to draw it, which single demo to run, and how to check the room is still with you before you move on. This is the delivery playbook. The concepts live in the other chapters; here we choreograph them into three-hour lectures that never sag.

Each lecture is **three fifty-minute blocks with two short breaks**. Your job is to protect the rhythm: one big idea per block, one live thing per block, one checkpoint question before the break. If a block runs long, cut *content*, never the demo.

[[note: metaphor || A three-hour lecture is a three-course meal, not a firehose. Block 1 is the appetizer — a sharp idea that opens the appetite. Block 2 is the main — the substance, the silicon, the code. Block 3 is the dessert everyone stays for — the live demo where numbers move on screen. Pace the courses and clear the plates between them (the break + checkpoint). Nobody remembers a meal served all at once, cold.]]

[[fig: A warm hand-drawn illustration titled "One lecture = a three-course meal". A long dinner table drawn left to right with three plates. Plate 1 labeled "Block 1 · the idea (50 min)" holds a small tidy appetizer. Plate 2 labeled "Block 2 · the substance (50 min)" holds a big hearty main course. Plate 3 labeled "Block 3 · the live demo (50 min)" holds a glossy dessert with a little "▶ run it live" flag stuck in it. Between plates, two little brooms labeled "break + checkpoint Q" sweeping crumbs away. Above the table a handwritten banner "one big idea per course, clear the plate between". Excalidraw style, white background, charming, handwritten labels. || The universal shape of every lecture: three courses, cleared between with a break and a checkpoint question.]]

Below, each lecture gets a minute-by-minute spine, the board sequence, the one demo, and the checkpoint questions. Keep this page open on your laptop while you teach.

## L1 — "How fast can this go?" (mental models + the silicon)

The whole lecture answers one question, and you should write it on the board and leave it there for three hours: **"What percentage of peak are you at?"** Everything L1 teaches is machinery for answering that.

### Block 1 (0:00–0:50) — The three regimes + the roofline

- **0:00–0:08.** Cold open. Ask: "If I gave you the best GPU on Earth, how fast could this run?" Let them flounder — most say "as fast as the GPU's FLOPs." Plant the seed that the answer is usually *no*.
- **0:08–0:25.** The three regimes. Draw three buckets: **compute-bound** (waiting on math), **memory-bound** (waiting on data), **overhead-bound** (waiting on the launch/Python). One everyday example each. Every kernel lives in one bucket.
- **0:25–0:45.** Napkin math on the H100, live and slow. Two headline numbers: **989 TF/s** BF16 compute vs **3.35 TB/s** HBM bandwidth. Divide them. That ratio — about **295** — is the *arithmetic-intensity break-even*: you need ~295 math ops per byte fetched, or the memory pipe starves the math units.
- **0:45–0:50.** Draw the **roofline** live: a flat ceiling (compute limit) and a slanted ramp (memory limit) meeting at ~295. Where your kernel sits under that roof *is* its regime.

[[note: example || Do the division on the board: 989 × 10¹² FLOP/s ÷ 3.35 × 10¹² byte/s ≈ 295 FLOP/byte. In plain words: "For every byte this chip drags out of memory, it can do about 300 sums in the same time. So if your kernel does fewer than 300 sums per byte, the math units sit idle — you're memory-bound, and a faster-math GPU won't help you one bit." That last clause is the aha.]]

[[fig: A hand-drawn roofline diagram, technical style. X-axis in red labeled "arithmetic intensity (FLOP per byte)", Y-axis in red labeled "achievable TFLOP/s". A blue slanted line rising from the origin labeled "memory-bound ramp (3.35 TB/s HBM)" that bends into a flat orange horizontal ceiling labeled "compute roof (989 TF/s BF16)". The bend point marked with a numbered circle ① and a red label "ridge ≈ 295 FLOP/byte". Two example dots plotted: a green dot low-left labeled "ReLU (memory-bound)" and a green dot up-right under the roof labeled "big GEMM (compute-bound)". A purple note points at each dot: "left of ridge = feed it faster; right of ridge = do math faster". Dashed takeaway box: "your kernel's job is to climb toward the roof". Excalidraw style, white background, handwritten labels. || The roofline drawn live: the ridge at ~295 splits every kernel into memory-bound (left) or compute-bound (right).]]

[[note: say || "This one drawing is the map for the whole course. Every optimization for four weeks slides your kernel up-and-right on this chart — toward the roof. 'Is this kernel good?' really means: how close to the roof is it, and which wall is it stuck under?"]]

- **Checkpoint (before break):** "I give you a kernel that reads a huge array and adds 1 to each element. Compute-bound or memory-bound?" (Answer: memory-bound — one add per byte, far left of the ridge.)

### Block 2 (1:00–1:50) — Silicon tour, top-down

- **1:00–1:10.** Set the rule: go *top-down*, big to small. Students drown if you start with transistors.
- **1:10–1:40.** Peel the onion, one layer per few minutes: the **die** → **8 GPCs** → **SMs** (the H100 has **132**) → inside one SM: **warp schedulers**, **tensor cores**, **register file**, shared **SMEM/L1**. Then zoom back out to the plumbing: **L2 + crossbar**, and the **HBM stacks** on the interposer.
- **1:40–1:50.** The punchline: almost all that silicon is *arithmetic*, and memory sits *far away*. This is why the roofline ridge exists — physical distance.

[[note: metaphor || The silicon tour is a Google-Earth zoom, not a stamp collection. Start at the continent (the die), fly to cities (GPCs), then neighborhoods (SMs), then a single house (one SM's cores), then pull up to see the highways (L2, crossbar, HBM). Students remember a zoom; they forget an acronym list. Narrate a flight, not an inventory.]]

[[fig: A hand-drawn "top-down zoom" figure of an H100, technical style, drawn as four nested framed panels connected by big orange zoom arrows. Panel 1 (widest): a rectangle labeled in green "the die · 132 SMs · HBM around the edge". Zoom arrow into Panel 2: "8 GPCs" shown as 8 blue blocks. Zoom arrow into Panel 3: "1 SM" showing four small sub-boxes labeled in blue "warp schedulers", "tensor cores", "register file", "SMEM / L1". Zoom arrow back OUT to Panel 4: the shared plumbing — a blue block "L2 + crossbar" wired to green stacks labeled "HBM stacks on interposer". Red dimension labels throughout. A dashed takeaway box: "mostly math units up close; memory lives far away → that distance IS the roofline ridge". Excalidraw style, white background, handwritten labels. || The silicon as a top-down zoom: die to GPC to SM to plumbing, ending on why memory is far.]]

- **Checkpoint (before break):** "How many SMs does an H100 have, and what lives inside one?" (132; warp schedulers, tensor cores, register file, SMEM/L1.)

### Block 3 (2:00–2:50) — LIVE: predict-then-measure in PyTorch

This is the block they came for. **The one demo:** benchmark three operations and place each on the roofline *before* measuring.

- **2:00–2:10.** The game: "We predict the regime, *then* measure. Predicting first is the whole skill."
- **2:10–2:40.** Benchmark **ReLU**, **softmax**, and **GEMM** at several sizes. For each: compute FLOPs and bytes on the board, place a dot on the roofline, then run the timing and read the achieved TF/s. Watch ReLU stick on the memory ramp while big GEMM climbs to the compute roof.
- **2:40–2:50.** Debrief with the master question, per op: "What % of peak were you at?"

[[note: demo || A tiny PyTorch script that times `torch.relu`, `torch.softmax`, and `torch.matmul` (via `torch.cuda.Event`) and prints achieved TFLOP/s next to the H100's 989 peak. Predict the regime first, THEN run. The jaw-drop: ReLU on a big tensor hits maybe 2–3% of peak FLOPs and is totally fine there — it's memory-bound, already near the memory roof. "Low FLOP % isn't failure; being far from the *right* roof is."]]

[[note: confusion || Students conflate "low % of peak FLOPs" with "bad kernel." Fix: "There are two roofs. A memory-bound kernel near the memory roof is excellent even at 3% of the compute roof. You only measure against the roof you're actually under." Say it twice; it's the subtlest idea in L1.]]

## L2 — The CUDA programming model + first kernels

L2 is where they write GPU code for the first time. The emotional goal: **demystify the launch.** By the end they should feel that a kernel is just "the same function, run by thousands of threads at once, each told who it is."

### Block 1 (0:00–0:50) — Grid / block / warp / thread

- **0:00–0:20.** Build the hierarchy bottom-up here (deliberately opposite of L1's top-down silicon tour; software builds up, hardware zooms down). **Thread** = one worker. **Warp** = 32 threads in lockstep. **Block** = warps sharing SMEM. **Grid** = all the blocks in one launch.
- **0:20–0:35.** The key move: every thread runs the *same code* but computes its *own index* from `blockIdx`, `blockDim`, `threadIdx`. That index is "who am I" — how one function fans out over a million elements.
- **0:35–0:50.** **SIMT and divergence.** The 32 threads in a warp share one instruction pointer. Hit an `if` that splits them and the warp runs *both* sides with half asleep each time. That's divergence — a tax.

[[note: metaphor || A grid of threads is a stadium card-stunt. Everyone got the *same instruction sheet* ("hold up the color for your seat number"); nobody coordinates — each just reads their seat number (`blockIdx * blockDim + threadIdx`) and acts. A warp is one row of 32 who must flip on the same beat. "Odd seats do X, even seats do Y" means the row can't flip together — X while evens wait, then Y while odds wait. That waiting is divergence.]]

[[fig: A warm hand-drawn stadium card-stunt illustration for the CUDA thread hierarchy. A curved stadium stand full of little seated figures each holding a colored card. One row of 32 figures is bracketed in blue and labeled "a warp = 32, flip on the same beat". A block of several rows is bracketed in green labeled "a block = warps sharing a scratchpad (SMEM)". The whole stand is boxed in red labeled "the grid = all blocks in one launch". One figure has a speech bubble "my seat # = blockIdx·blockDim + threadIdx → that's who I am". A small inset shows one row split into odd/even doing different colors with a red note "an if-branch = the row can't flip together → divergence tax". Dashed takeaway box: "same instructions, everyone computes their own seat number". Excalidraw style, white background, charming, handwritten. || The thread hierarchy as a stadium card-stunt: one instruction sheet, everyone reads their own seat, warps flip together.]]

- **Checkpoint (before break):** "A warp hits `if (threadIdx.x < 16)`. What does the hardware do?" (Runs both branches serially, masking off the inactive half — divergence.)

### Block 2 (1:00–1:50) — Launch anatomy + the compile story

- **1:00–1:25.** Anatomy of a kernel launch: the `<<<grid, block>>>` syntax, what a `__global__` function is, how the index math turns into work. Write a full `vector_add` kernel on the board, line by line.
- **1:25–1:50.** The compilation story, drawn as a pipeline: **nvcc → PTX → ptxas → SASS**. PTX is the portable "assembly-ish" intermediate; SASS is the real machine code for *this* GPU. Introduce **compute capability** (the "sm_90" tag) as "which GPU dialect."

[[note: teach || Draw the compile pipeline as an airport baggage belt: your `.cu` is the suitcase, nvcc is check-in, PTX is the tagged bag on the belt (portable, any airport), ptxas is the final sort at *this* airport, SASS is the bag at the actual gate for *this* GPU. Students conflate PTX and SASS constantly; "portable tag vs final gate" fixes it. You'll thank yourself in L5 reading SASS live.]]

[[fig: A hand-drawn compile-pipeline figure, technical style, as a left-to-right conveyor. A purple box "kernel.cu (your code)" → blue stage "nvcc (check-in)" → yellow tile "PTX (portable IL — any GPU)" → blue stage "ptxas (final sort)" → yellow tile "SASS (real machine code, THIS GPU)" → green box "runs on sm_90". Each stage numbered ①②③④ in circles. A red annotation over PTX vs SASS: "PTX = portable bag tag · SASS = arrives at the actual gate". Dashed takeaway box: "one source → portable PTX → GPU-specific SASS". Excalidraw style, white background, handwritten labels. || The compile pipeline as an airport belt: source to portable PTX to GPU-specific SASS.]]

- **Checkpoint (before break):** "What's the difference between PTX and SASS?" (PTX = portable intermediate; SASS = final machine code for a specific GPU.)

### Block 3 (2:00–2:50) — LIVE: three kernels + first GPU-Puzzles

**The one demo:** write and run three real kernels in ascending difficulty, live.

- **2:00–2:15.** **Vector add** — the hello-world. Launch, verify, celebrate. Their first GPU code runs.
- **2:15–2:30.** **RGB → grayscale** — same pattern on a real image, so the output is *visible*. Show the picture turn grey; visible output is a morale win.
- **2:30–2:45.** **Naive reduction** (sum an array) — the first kernel where threads must *cooperate*, setting up the memory lecture.
- **2:45–2:50.** Solve the **first GPU-Puzzles** together as a cool-down.

[[note: demo || The signature L2 moment: run the RGB→grey kernel on an actual photo and show before/after on the projector. Compute doesn't feel real until pixels change. Keep it tiny — one thread per pixel, `grey = 0.21R + 0.72G + 0.07B`. When the image goes grey in front of them, that's the "I made the GPU do something" moment they'll tell people about.]]

[[note: production || Tie it forward: "The vector-add you just wrote is the *exact* shape of an elementwise kernel inside PyTorch — a bias add, a ReLU, a residual connection. When you use an LLM, thousands of kernels this simple fire per token. You're not writing toys; you're writing the smallest real thing in the stack."]]

- **Checkpoint (before break):** "In `vector_add`, how does thread number 5000 know which array element it owns?" (From its global index: `blockIdx.x * blockDim.x + threadIdx.x`.)

## L3 — The memory hierarchy in anger

L3 is the turning point. L1 said "feeding the cooks is the whole game"; L3 is where they *feel* it live, with a profiler. Emotional goal: **the same math can be 10× faster purely by moving data smarter.**

### Block 1 (0:00–0:50) — Coalescing

- **0:00–0:20.** The idea: the GPU fetches memory in fixed **chunks** — 32/64/128-byte transactions. If a warp's 32 threads read 32 *neighboring* addresses, that's one tidy transaction. If they read 32 *scattered* addresses, that's up to 32 separate transactions — same data, many times the traffic.
- **0:20–0:40.** Draw coalesced vs strided access with a warp of 32 arrows.
- **0:40–0:50.** The payoff: coalescing is often the single biggest free speedup in a naive kernel.

[[note: metaphor || Coalescing is the mail carrier. 32 letters to houses 1–32 on the *same street* = one trip down one block. The same 32 letters scattered one-per-neighborhood = 32 separate drives. The GPU's memory system delivers a whole block-worth per trip — so line your addresses up on one street.]]

[[fig: A warm hand-drawn mail-carrier illustration of memory coalescing. Top row "COALESCED": a single mail truck driving down one straight street where houses numbered 1–32 sit in a neat row, all 32 letters delivered in "1 trip" (green, happy). Bottom row "STRIDED": the same 32 letters but the houses are scattered across a sprawling city map, the truck drawing a tangled path labeled "up to 32 trips for the SAME letters" (red, exhausted). A warp of 32 little arrows shown feeding each case. Blue note: "the GPU fetches a whole 32/64/128-byte block per trip". Dashed takeaway box: "line your 32 threads up on one street → 1 trip instead of 32". Excalidraw style, white background, charming, handwritten. || Coalescing as a mail carrier: neighboring addresses = one trip; scattered addresses = many trips for the same data.]]

- **Checkpoint (before break):** "Thread `t` reads `A[t]` vs `A[t * 1000]`. Which coalesces?" (`A[t]` — neighbors; the strided one scatters.)

### Block 2 (1:00–1:50) — Shared memory, bank conflicts, occupancy

- **1:00–1:20.** **SMEM** as the on-chip scratchpad: a small, fast, *shared* space a block uses to avoid re-fetching from far-away HBM. "Keep the ingredients on the counter, not in the far pantry."
- **1:20–1:35.** **Bank conflicts.** SMEM is 32 banks; two threads in a warp hitting the same bank serialize. Show the classic case and the **padding fix** (one dummy column so the stride dodges the banks).
- **1:35–1:50.** **Occupancy calculus.** Blocks-per-SM depends on registers-per-thread and SMEM-per-block. More resident warps = more latency hiding — but push registers too high and you **spill** to slow local memory.

[[note: metaphor || SMEM is the kitchen counter; HBM is the walk-in pantry down the hall. You don't run to the pantry for every pinch of salt — bring a bowl to the counter once and work from there. Bank conflicts are two cooks grabbing the *same* jar at the same instant: they take turns. The padding fix spaces the jars so no two cooks grab the same one on the same beat.]]

[[fig: A hand-drawn technical figure of shared-memory bank conflicts and the padding fix. Left panel "CONFLICT": a grid of SMEM drawn as 32 vertical banks (blue), with several red arrows from a warp all landing in the SAME bank, labeled "threads collide → serialize (slow)". Right panel "PADDED FIX": the same grid but with one extra hatched dummy column (orange) labeled "+1 pad column", now the red arrows fan out to distinct banks labeled "each thread → its own bank (parallel)". A green note: "SMEM = 32 banks; a warp wants 32 different banks at once". A purple code snippet: "__shared__ float tile[32][33]; // 33 not 32". Dashed takeaway box: "pad the stride so the warp hits 32 distinct banks". Excalidraw style, white background, handwritten labels. || Bank conflicts and the one-column padding fix: spread the warp across all 32 banks.]]

- **Checkpoint (before break):** "Why does adding one unused column to a shared tile speed things up?" (It shifts the access stride so warp lanes hit distinct banks — kills the conflict.)

### Block 3 (2:00–2:50) — LIVE: the transpose ladder + first Nsight Compute

**The one demo:** the matrix-transpose ladder, climbing rung by rung, profiled live. This is a dress rehearsal for the GEMM worklog in L4.

- **2:00–2:12.** **Naive transpose.** Run it, time it. Slow — the writes are strided (uncoalesced).
- **2:12–2:24.** **Coalesced transpose** via SMEM: read coalesced, transpose in the scratchpad, write coalesced. Time it — big jump.
- **2:24–2:36.** **+ padding** to kill the bank conflict the SMEM version introduced. Time again — another jump.
- **2:36–2:50.** **First contact with `ncu` (Nsight Compute).** Open the memory section; show achieved bandwidth and the "uncoalesced access" warning appearing and vanishing as you climb.

[[note: demo || The L3 jaw-drop: same transpose, same math, three versions — wall-clock drops roughly 10× from naive to coalesced+padded, *purely* by moving data smarter, not changing a single multiply. Then open ncu and let them SEE the memory-throughput bar fill. "We didn't make the math faster. We fed the cooks better. That's the whole job." This is where L1's promise pays off.]]

[[note: confusion || Students expect faster code to mean "fewer or cleverer calculations." Here the arithmetic is *identical* across all three versions. Fix: "You optimized nothing about the math — you optimized the *logistics of the data*. Kernel engineering is 90% logistics." Return to this every time they reach for a math trick when the fix is a memory trick.]]

- **Checkpoint (before break):** "The naive and coalesced transpose do the exact same multiplies. Why is one 10× faster?" (Memory access pattern — coalesced writes vs strided writes.)

## L4 — GEMM worklog I: naive → tiling (the ladder begins)

L4 is the heart of the course made visible: a **worklog ladder** where each rung is *hypothesis → code → profile → new % of cuBLAS*. The emotional goal: **optimization is a disciplined loop, not magic.** They watch a number climb and learn the method that made it climb.

### Block 1 (0:00–0:50) — The ratchet + Kernel 1 (naive)

- **0:00–0:15.** Frame the worklog. Write the ladder as empty rungs with a "% of cuBLAS" column. The loop: guess the bottleneck (**hypothesis**), change one thing (**code**), measure with ncu (**profile**), record the %. Repeat. cuBLAS (NVIDIA's own library) is 100% — the boss we're chasing.
- **0:15–0:35.** **Kernel 1: naive.** One thread per output cell, each doing a full dot product straight from HBM — the three-nested-loop matmul they already know, one thread per (i, j). Write it.
- **0:35–0:50.** Profile: **~1.3% of cuBLAS.** Diagnose — every thread re-reads whole rows/columns from far-away HBM. Memory-bound, terribly fed.

[[note: metaphor || The worklog ladder is a video-game score bar. Kernel 1 is your first pathetic score (1.3% of the boss). You don't rage — you study the replay (the profiler), form ONE theory about what killed you, change ONE thing, play again. Each rung is a run. Students love that the number is honest and public; it turns optimization into a game with a scoreboard — exactly what it is in industry.]]

[[fig: A hand-drawn "worklog ladder scoreboard" figure, technical style. A vertical ladder with rungs, each rung a labeled bar showing "% of cuBLAS": rung 1 "K1 naive — 1.3%" (tiny red bar), rung 2 "K2 coalescing — 8.5%", rung 3 "K3 SMEM tiling — 12.8%", rung 4 "K4 1D blocktiling — 36.5%" (growing green bars). A dashed 100% line at the top labeled in orange "cuBLAS (the boss)". To the side, a blue loop diagram with numbered circles: ① hypothesis → ② code → ③ profile (ncu) → ④ record % → back to ①. A purple note: "change ONE thing per rung". Dashed takeaway box: "optimization = a disciplined loop, watched on a scoreboard". Excalidraw style, white background, handwritten labels. || The GEMM ladder as a scoreboard: each rung is one hypothesis-code-profile-measure loop climbing toward cuBLAS.]]

- **Checkpoint (before break):** "Kernel 1 does the right math and gets 1.3% of cuBLAS. What's the bottleneck?" (Memory — every thread re-fetches its whole row/column from HBM.)

### Block 2 (1:00–1:50) — K2 coalescing + K3 SMEM tiling

- **1:00–1:20.** **Kernel 2: coalescing.** Reassign which thread handles which cell so neighboring threads read neighboring memory (straight from L3). One indexing change. Profile: **8.5%** — a ~6× jump from *nothing but access pattern*.
- **1:20–1:50.** **Kernel 3: SMEM tiling.** The big conceptual rung. Instead of each thread hitting HBM for the full dot product, the block cooperatively loads a **tile** of A and a tile of B into SMEM once, and every thread reuses them. The arithmetic-intensity ratchet in action — more math per byte. Profile: **12.8%.**

[[note: metaphor || Tiling is a potluck. Without it, all 256 cooks each drive to the far pantry for the *same* ingredients — absurd waste. With tiling, the block agrees: everyone brings one dish to the shared counter (SMEM) *once*, then all 256 eat from the shared spread many times. Each ingredient fetched from the far pantry once, reused dozens of times. That reuse ratio is *arithmetic intensity* — raising it is the through-line of the whole worklog.]]

[[fig: A warm hand-drawn "potluck" illustration of SMEM tiling. Left "NO TILING": a crowd of little cook figures each individually driving a long road to a distant pantry labeled "HBM (far)" for the same crate of ingredients — a tangle of redundant trips, red "everyone fetches the SAME data separately". Right "TILING": the cooks load one shared table in the middle labeled "SMEM counter (fetched once)" with a green arrow "one trip to the pantry", then many small arrows show every cook reusing the shared table over and over, labeled "reuse many times = high arithmetic intensity". A purple note: "fetch once from far memory, reuse from the counter". Dashed takeaway box: "tiling raises math-per-byte → climbs the roofline". Excalidraw style, white background, charming, handwritten. || SMEM tiling as a potluck: load the shared counter once, reuse it many times — the arithmetic-intensity ratchet.]]

- **Checkpoint (before break):** "K2 to K3 barely changed the FLOPs. Why did it get faster?" (SMEM tiling reuses each fetched byte many times — higher arithmetic intensity, fewer HBM trips.)

### Block 3 (2:00–2:50) — LIVE: K4 1D blocktiling + climb the ladder

**The one demo:** build **Kernel 4 (1D blocktiling)** live and profile every rung, watching the % of cuBLAS climb in real time.

- **2:00–2:20.** **Blocktiling**: give each *thread* more than one output element, amortizing its loads across several results. More work per byte still — the ratchet turns again.
- **2:20–2:40.** Build it, run it, profile with ncu. Record: **36.5% of cuBLAS.** From 1.3% to 36.5% in one lecture.
- **2:40–2:50.** Debrief the ladder. Point at the scoreboard: every jump came from *feeding the cooks better*, never from changing the multiply. Preview L5: registers and warptiling reach 93%+.

[[note: demo || The L4 finale: put all four %s on one live bar chart — 1.3 → 8.5 → 12.8 → 36.5 — and say it out loud: "We made the same matrix multiply **28 times faster** in three hours, and never touched the arithmetic. Next lecture we get past 90% of NVIDIA's own hand-tuned library." Let that 28× sit. It's the proof the method works.]]

[[note: production || Anchor it: "cuBLAS is the library that runs when PyTorch calls `matmul` on a server. Getting to 90%+ of it by hand isn't academic — the H100 kernels behind vLLM, FlashAttention, and DeepSeek's serving stack are exactly this game played to the last percent. The loop you learned today — hypothesize, change one thing, profile, record — is *the job*."]]

- **Checkpoint (before break):** "We went 1.3% → 36.5% without changing the math. In one word, what did we optimize?" (Memory — data movement / feeding.)

## Teaching notes that apply to all four lectures

[[note: teach || Three rules that save every lecture. (1) **One demo per block, and it runs.** Test every demo the night before on the actual machine; a demo that fails live costs you the room. (2) **Predict before you measure — always.** The predict-then-measure ritual from L1 is the spine of L3 and L4 too; it turns watching into thinking. (3) **Protect Block 3.** If you're running long, cut a slide from Block 2, never a minute from the live demo. They forget slides; they remember the number that moved.]]

[[note: confusion || The meta-confusion across all four lectures: students think "faster kernel = cleverer math." Every single one of these lectures teaches the opposite — L1 (roofline), L3 (transpose), L4 (GEMM ladder) all make the same point: the math is fixed; you win by feeding it better. If they leave L4 believing "kernel engineering is 90% logistics," you've done your job.]]

## You can now teach

- **L1** minute by minute: the three regimes, the H100 napkin math (989 TF/s vs 3.35 TB/s → ~295), the roofline drawn live, and the predict-then-measure PyTorch demo.
- **L2** minute by minute: the grid/block/warp/thread hierarchy as a card-stunt, launch and compile anatomy (nvcc→PTX→ptxas→SASS), and the vector-add / RGB→grey / reduction demo.
- **L3** minute by minute: coalescing as a mail carrier, SMEM + bank conflicts + the padding fix, occupancy, and the live transpose ladder profiled in ncu.
- **L4** minute by minute: the worklog loop (hypothesis→code→profile→%), the naive→coalesced→SMEM→blocktiling ladder from 1.3% to 36.5% of cuBLAS, built and profiled live.
- The **universal lecture shape**: three courses, one big idea and one live demo per block, a checkpoint question before each break — and the discipline to protect the demo when time runs short.
- The **through-line** to repeat in every lecture: the math is fixed; you win by feeding the cooks — kernel engineering is 90% logistics.
