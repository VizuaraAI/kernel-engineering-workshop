By the end of this chapter you can stand up on day one, draw a single map on the whiteboard, and tell the students the *whole* story of the workshop in five minutes — so that every lecture and every workshop afterward feels like a step on one road they can already see, not a pile of disconnected tricks.

This is the most important chapter for *you*, the mentor, to own — because a workshop without a spine feels like fourteen unrelated topics, and students quietly drown. With the spine, every session has an obvious "you are here." So before you learn any of the pieces, learn the shape of the whole thing. It is simpler than it looks.

## The whole workshop is one sentence

Here is the entire four weeks in one line, and you should write it on the board on day one and leave it there:

**First we make the multiply fast. Then we make the model fast.**

That's it. Everything — every kernel, every profiler screenshot, every fancy GPU feature — lands in one of those two halves. The first half (roughly the first two weeks) takes the single operation matrix-multiply and drags it from painfully slow to nearly as fast as the hardware allows. The second half (the back two weeks) takes that fast multiply and uses it to make a *whole AI model* run fast — attention, serving, the newest chips, and even AI that writes kernels for us.

[[note: say || "This whole workshop is two words long: *faster, then fatter.* First we take one tiny operation — multiplying two grids of numbers — and make it scream. Then we zoom out and make an entire language model scream, using everything we learned on the small problem. If you ever feel lost, ask yourself one question: am I still making the multiply fast, or am I now making the model fast? That's the whole map."]]

[[fig: A warm hand-drawn "one road, two halves" illustration. A single winding country road runs left to right across a white page, drawn like a friendly treasure map. The left half of the road is labeled in blue "PART 1: make the MULTIPLY fast" and has little milestone signposts along it reading "matmul by hand", "the GPU", "CUDA", "memory", "GEMM ladder", "tensor cores", "profiling". The right half of the road is labeled in green "PART 2: make the MODEL fast" with signposts reading "attention", "FlashAttention", "Hopper", "serving", "Blackwell", "AI writes kernels". At the far right the road ends at a big orange star labeled "CAPSTONE: you vs the machine". A dashed takeaway box at the bottom reads "faster, then fatter — one road, fourteen stops". Excalidraw style, hand-lettered, charming. || The whole workshop as one road with two halves: make the multiply fast, then make the model fast.]]

## Why start with the multiply at all

Students often wonder why a whole workshop obsesses over one operation. Here is the answer you give them, and it is the emotional hook of the entire course.

Running a neural network is, almost entirely, matrix multiplication. When you chat with a model, your words become a grid of numbers, and that grid gets multiplied by the model's learned weight-grids, hundreds of times, for every single word it says back. Generating one word is *trillions* of little multiply-and-add steps. So if you can make the multiply even a little faster, you make *all of AI* faster — and cheaper, and cooler.

[[note: aha || The number that makes the room go quiet: multiplying two modest 1000×1000 grids is already **one billion** multiply-adds. A real model does grids far bigger than that, millions of times, for every word. This is why NVIDIA became one of the most valuable companies on Earth — they build the best chips for this one loop. Say it plainly: "We are spending four weeks on this operation because the entire AI economy is spending its electricity bill on it right now."]]

[[note: production || This is not academic. In data centres full of H100 and B200 GPUs, the overwhelming majority of the money and power spent on AI goes into exactly this multiply. The gap between a badly-written kernel and a great one is the gap between a GPU running at 10% of what you paid for and 90%. Closing that gap — nine-tenths of a multi-million-dollar cluster — is literally what a kernel engineer is hired to do.]]

## Part 1, step by step: the climb to a fast multiply

Now walk the first half of the road with the students, sign by sign, so each lecture has a home. Keep it a story, not a syllabus.

We start with **the multiply itself** — by hand, on a receipt, a 2×2 grid of dot products (L-matmul). Then we ask *what hardware do we run it on?* and meet the **GPU**: not a smarter chip than your laptop's, just a chip with thousands of tiny simple workers instead of a few clever ones (L-cpu-vs-gpu, L1). Then we learn to *talk* to those workers — the **CUDA programming model**: grids, blocks, warps, threads; our first real kernels like vector-add and a colour-to-grey image (L2).

Then comes the twist that runs through everything: the workers are so fast that the real problem is *feeding them data* fast enough. So we study the **memory hierarchy** — coalescing, shared memory, bank conflicts (L3). And now we're ready for the heart of Part 1: the **GEMM worklog**, a ladder of matmul kernels where each rung is a measured speed-up.

[[note: metaphor || The GEMM ladder is a *home-renovation montage.* You start with a shack — a naive kernel running at about 1% of the pro library's speed. Then, one weekend project at a time — better plumbing (coalescing), a pantry near the kitchen (shared-memory tiling), keeping tools in your hands instead of the drawer (register tiling), carrying four things at once (vectorized loads) — the shack becomes a mansion at ~94% of the pros. Same house, twelve upgrades. The students watch the number climb every rung, and *that climbing number* is the addictive spine of the whole course.]]

[[fig: A hand-drawn ladder or staircase climbing from bottom-left to top-right, titled "The GEMM ladder: same multiply, one upgrade per step". Each step is a rung labeled in purple with the kernel name and in orange with its percentage of the pro library cuBLAS: "1. naive — 1.3%", "2. coalesced — 8.5%", "3. shared-mem tiling — 12.8%", "4. 1D block tiling — 36.5%", "5. 2D block tiling — 68.7%", "6. vectorized float4 — 78.4%", "7. autotuned — 84.8%", "8. warp tiling — 93.7%". A little stick figure climbs the steps. A green note beside the ladder reads "every rung: guess -> code it -> profile -> read the new %". A dashed takeaway box: "the whole GEMM worklog is one number climbing from 1% to 94%". Excalidraw style, white background, hand-lettered. || The GEMM ladder as a staircase — each rung is one measured optimization, from 1% to ~94% of the pro library.]]

The last rung of Part 1 is **tensor cores** — special hardware on the GPU that does a tiny matmul in a single instruction, blowing past even our best hand-tuned kernel (L6). And to make sure students can improve any kernel and not just ours, we teach them to **profile like professionals** — reading the GPU's own diagnostic tools to find exactly why a kernel is slow (L7). That closes Part 1: they can take a multiply and make it fly, and they can *prove* it with numbers.

[[note: teach || The single habit that ties all of Part 1 together is the loop: **guess → code → profile → read the percentage → repeat.** Draw this little cycle on the board on day one and point back to it at the start of every GEMM lecture. Students should leave able to say the loop in their sleep. It is the scientific method for kernels, and it is the actual daily job. Every rung of every ladder in this course is one turn of this loop.]]

## Part 2, step by step: from a fast multiply to a fast model

Now the road bends. We have a screaming-fast multiply. What do we *build* with it?

The pivot is **attention** — the operation at the heart of every modern language model, the thing that lets a model look back over everything you've said (L8). Attention is, wonderfully, *made of matmuls plus one softmax*. So everything from Part 1 pays off immediately. But naive attention has a nasty habit: it writes a giant grid to slow faraway memory and reads it back, which chokes the GPU. The fix, **FlashAttention**, is one of the most famous kernels in the world — it does the whole thing in fast on-chip memory without ever writing the giant grid (W1). This is the "aha" of Part 2: the same feed-the-workers logistics from Part 1, now applied to a whole layer.

[[note: metaphor || If Part 1 was "learn to cook one dish fast," Part 2 is "run the whole restaurant." Attention is the signature dish. FlashAttention is realising you don't need to plate every course onto the counter and pick it back up — you keep it in your hands and finish it in one motion. Serving (W4) is the *dining room*: some customers place a big order once (prefill), others order one bite at a time all night (decode), and you learn to keep everyone fed at once. Same kitchen skills, now feeding a crowd.]]

[[fig: A warm hand-drawn "restaurant" metaphor illustration for Part 2, titled "From one dish to the whole restaurant". Left: a single chef holding one glowing plate labeled in blue "the fast multiply (Part 1)". An arrow labeled "now scale it up" points right into a busy restaurant scene: a kitchen labeled "attention = the signature dish", a clever waiter labeled "FlashAttention: keep it in your hands, don't set it down", a dining room with two kinds of tables — one big table labeled green "prefill: one huge order" and many small tables labeled green "decode: one bite at a time" — and a fancy new oven in the corner labeled orange "Hopper / Blackwell: newer, faster kitchens". A dashed takeaway box: "same cooking skills, now running the whole restaurant = a whole model". Excalidraw style, white background, charming, hand-lettered. || Part 2 as running the whole restaurant: attention is the dish, FlashAttention keeps it in-hand, serving feeds the whole dining room.]]

From there Part 2 climbs into the frontier. We do a **Hopper deep dive** — the H100's newest features, and what it actually takes to beat NVIDIA's own library (W2). We tour the **abstraction ladder** — higher-level tools like Triton and CUTLASS that write some of the kernel for you, and when you still must drop down to raw CUDA (W3). We study **inference-serving kernels** — the real machinery behind serving a model to millions: KV-caches, paged attention, quantization (W4). We peek at **Blackwell and NVFP4** — the very newest chips and number formats, where one hackathon dragged a kernel from 2000 microseconds down to 22 (W5).

And we finish at the frontier finale: the **DeepSeek stack** and, the mind-bender, **AI that writes kernels** — models that propose an optimization, compile it, profile it, and improve — the exact guess-code-profile loop from day one, now run by a machine (W6).

[[note: production || The finale is real and current. DeepSeek serves models with custom kernels like FlashMLA and DeepGEMM. And AI-generated kernels already beat hand-written PyTorch on some ops — CRFM's experiments hit 484% of PyTorch on LayerNorm — while honestly failing on others (FlashAttention at 9%). The punchline for students: the human + AI + profiler loop is winning, and it is the same loop you learned on a 2×2 by hand in week one. It even cross-links to Vizuara's Harness Engineering workshop — the machine needs a harness, and a harness is engineering.]]

## The thread that never breaks: "what % of peak are you at?"

There is one question that runs the entire length of the road, both halves, and you should make it the workshop's catchphrase: **"What percentage of the hardware's peak speed are you actually getting?"**

Every rung of every ladder is an answer to that question. The naive matmul: 1%. The tuned one: 94%. Naive attention: memory-choked. FlashAttention: much closer to peak. It is the same yardstick from the first day to the last. If a student can always tell you their percentage of peak and *why* it isn't higher, they have become a kernel engineer.

[[note: confusion || The number-one way students get lost in a fourteen-session workshop is losing the thread — they think L6 tensor cores and W4 serving are unrelated worlds. Fix it with the map. At the start of every single session, walk to the road drawing and physically point: "We are HERE. Behind us we made the multiply fast. Right now we're using it to make the model fast. The yardstick hasn't changed — we still just want a higher % of peak." One pointing finger per session keeps the whole cohort oriented.]]

[[fig: A technical Excalidraw diagram titled "One yardstick, fourteen sessions". A tall vertical thermometer/gauge on the left labeled in red "% of hardware peak" running 0% at bottom to 100% at top. Blue dots plotted up the gauge, each labeled with a session: "naive matmul 1%" near the bottom, "tiled 68%", "warptiled 94%" high up, then a green cluster of dots for Part 2 labeled "attention", "FlashAttention", "serving", "frontier" clustered near the top. An orange arrow runs alongside the whole gauge labeled "the ONLY question, every session: what % of peak are you at?". A purple note at the side: "guess -> code -> profile -> read % -> repeat". A dashed takeaway box: "the workshop is one yardstick applied fourteen times". Excalidraw style, white background, semantic colors, hand-lettered. || The single yardstick — percentage of hardware peak — that measures every session from the first to the last.]]

## How to actually open the workshop (the 5-minute board plan)

Here is the exact opening you deliver on day one, before any content, to plant the map.

1. **(1 min) Write the sentence.** "First we make the multiply fast. Then we make the model fast." Leave it up all four weeks.
2. **(1 min) The hook number.** 1000×1000 = a billion multiply-adds; a model does far more, per word; the whole AI economy pays for this loop. This is *why we care.*
3. **(2 min) Draw the road.** Sketch the two-halves map. Name the milestones out loud but do not explain them — just let students see how many stops there are and that they connect.
4. **(1 min) The yardstick.** "There is one question all four weeks: what percentage of peak are you at? You'll answer it with the loop — guess, code, profile, read the number, repeat." Draw the little loop.

[[note: demo || If you have a laptop and a GPU handy, end the opening with a ten-second live jaw-dropper: run a naive matmul and the built-in fast one back to back, and show the wall-clock gap on screen — often 50–100×. Then say: "By week two, *you* will close most of that gap by hand. That's Part 1. Let's begin." Nothing motivates a cohort like seeing the mountain they're about to climb, measured in real milliseconds.]]

[[note: teach || Resist the urge to explain any milestone during the opening map. The whole job of the first five minutes is *orientation, not content* — students should leave knowing the shape of the journey and the one question, nothing more. Trust the later sessions to fill each signpost. A mentor who tries to teach coalescing on day one loses the map; a mentor who just points at the road keeps the whole cohort with them for four weeks.]]

[[sn: The curriculum splits into 8 foundational live lectures (L1–L8) and 6 deep-dive workshops (W1–W6), often interleaved week by week. You don't need students to memorize that structure — the two-halves road is the mental model that matters. The L/W numbering is your bookkeeping, not theirs.]]

## You can now teach

- The **one-sentence spine** of the entire workshop — "first make the multiply fast, then make the model fast" — and how to write it on the board so it anchors all four weeks.
- **Why we obsess over one operation**: the billion-multiply-add hook and the fact that the whole AI economy runs on this loop.
- **Part 1 as a climb** — matmul → GPU → CUDA → memory → the GEMM ladder → tensor cores → profiling — told as one home-renovation montage where a number climbs from 1% to 94%.
- **Part 2 as running the whole restaurant** — attention → FlashAttention → Hopper → serving → Blackwell → AI-that-writes-kernels — reusing every Part 1 skill at model scale.
- The **single yardstick** — "what % of peak are you at?" — and the guess→code→profile→repeat loop that answers it in every session.
- The **5-minute day-one opening** and the "you are here" pointing habit that keeps a fourteen-session cohort from ever losing the thread.
