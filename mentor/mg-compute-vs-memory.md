By the end of this chapter you'll be able to stand at a whiteboard and teach the single most important question in all of GPU performance: *what is this code waiting on?* You'll be able to name the three things a kernel can be stuck on, tell a room which one they're in, and — the real prize — explain why knowing the answer collapses a hundred possible optimizations down to two or three. This is the master mental model of the whole workshop. Everything else hangs off it. So let's build it slowly, with a kitchen.

## The one question that runs the whole course

A GPU is a kitchen full of cooks. We built that picture in the last chapter: thousands of simple line cooks, all doing the same tiny multiply-add at once. But a kitchen full of cooks is not automatically a *fast* kitchen. A kitchen can be slow for three completely different reasons, and the fix for each is different.

Reason one: **the cooks are genuinely busy cooking.** Every hand is moving, every pan is full. The only way to serve food faster is to cook faster or cook less. This is a good problem — it means you're actually using the expensive kitchen you paid for.

Reason two: **the cooks are standing around waiting for ingredients.** The pantry is far away, the hallway is narrow, and the rice trickles in. The cooks could cook ten times faster, but there's nothing in their hands. The kitchen isn't slow because cooking is slow — it's slow because *fetching* is slow.

Reason three: **almost nobody is in the kitchen yet.** You ordered one sandwich. By the time you unlock the doors, turn on the lights, and walk to the station, the actual sandwich took two seconds. All your time went to *getting started*, not to cooking.

[[note: metaphor || The kitchen is the whole chapter. A slow kitchen is either (1) **cooking** — hands full, this is the goal; (2) **fetching** — cooks idle, waiting on ingredients from a far pantry; or (3) **opening up** — the order was so tiny that just walking to the station cost more than the cooking. Every slow GPU kernel is one of these three. Teach the mentor to ask, every single time: *are we cooking, fetching, or opening up?*]]

[[fig: A warm hand-drawn three-panel kitchen illustration titled "Why is the kitchen slow?". Panel 1 labeled "COOKING (compute-bound)" in orange: a busy kitchen line of small cook figures all actively stirring pans and plating, with motion marks and a green handwritten note "every hand moving — this is the GOAL". Panel 2 labeled "FETCHING (memory-bound)" in orange: the same cooks standing idle with little 'zzz' and foot-tapping marks, staring down a long thin winding hallway to a distant pantry labeled "far pantry", only a couple of ingredient boxes trickling down it, blue note "cooks wait on rice". Panel 3 labeled "OPENING UP (overhead-bound)" in orange: a nearly empty kitchen with one tiny sandwich on the counter dwarfed by a huge purple bracket over the door labeled "unlock · lights · walk to station", red note "getting started cost more than the food". Dashed takeaway box across the bottom: "every slow kernel is cooking, fetching, or opening up — one of three." Excalidraw style, white background, charming, handwritten labels. || The master mental model as one kitchen with three failure modes: cooking, fetching, and opening up.]]

Those three have proper names, and you should write them on the board only *after* the kitchen lands: **compute-bound** (cooking), **memory-bandwidth-bound** (fetching), and **overhead-bound** (opening up). The word "bound" just means "held back by." Compute-bound means "held back by how fast we can do math." Say it exactly that plainly.

[[note: teach || Draw the three kitchen panels first, act them out — mime a cook furiously stirring, then a cook tapping his foot staring down a hallway, then a lone cook flicking lights on for one sandwich. *Then* write the three real names underneath. Never lead with "memory-bandwidth-bound." The name is a label you paste onto a picture the room already understands.]]

## Why "fetching" is the surprise — and the whole reason kernels are hard

Here's the thing students never expect, and it's the emotional center of the chapter: **most of the time, a GPU is fetching, not cooking.** Newcomers assume that making a GPU fast is about clever math. It almost never is. The math units are so absurdly fast that the real struggle is shoveling data to them quickly enough. The cooks are faster than the hallway — and it gets more true every year.

Make "fetching" concrete with the simplest kernel there is. Take a big list of numbers and add 1 to each. `x + 1`. For every number, the GPU must **read** it from far-away memory, **add** one (a single trivial flop), and **write** it back. Two trips down the hallway for one tiny stir of the spoon. No number of extra cooks helps — they'd just wait together.

[[note: example || On the board: element-wise `x + 1` on a million numbers. Work = 1 million adds (trivial). Data movement = 1 million reads + 1 million writes = 2 million memory trips. That's *2 memory operations for every 1 flop.* Write the ratio big: "1 flop per 2 numbers moved." Then say: "the math finished instantly. We spent the whole time in the hallway." This one example teaches memory-bound better than any definition.]]

Now contrast it with the operation the whole workshop is about: a big matrix multiply. We learned that a matmul is a grid of dot products — a mountain of multiply-adds. And crucially, once you've fetched a row of A and a column of B, you *reuse them*. A big matrix multiply does an enormous amount of cooking for each trip to the pantry. That's the good kind of kernel: the cooks are actually cooking.

[[fig: A hand-drawn side-by-side comparison titled "Two kernels, two fates", Excalidraw semantic-color style, white background. Left box labeled in orange "x + 1 (element-wise)": a small green flop-square "one add" sitting between two fat blue arrows labeled "READ from HBM" and "WRITE to HBM", red annotation "2 trips per 1 flop → memory-bound, cooks idle". Right box labeled in orange "big matmul": one blue arrow "READ a tile once" feeding a dense grid of many green flop-squares hatched and lit up labeled "reused for thousands of multiply-adds", red annotation "tons of cooking per trip → compute-bound, cooks busy". A dashed takeaway box: "the ratio of cooking-to-fetching decides your fate." Hand-lettered labels. || The two archetypes: element-wise ops fetch constantly (memory-bound); big matmuls reuse each fetch heavily (compute-bound).]]

## The one number that decides it: cooking per fetch

Everything above collapses into a single ratio, and this is the number to build the lesson around. It has an intimidating name — **arithmetic intensity** — but it means something you can say in six words: **how much math per byte moved.** How much cooking do you do for each trip down the hallway?

- High arithmetic intensity = lots of cooking per fetch = the cooks stay busy = **compute-bound**. Good.
- Low arithmetic intensity = barely any cooking per fetch = the cooks wait = **memory-bound**.

[[note: aha || The whole diagnostic is one ratio. **Arithmetic intensity = flops done ÷ bytes moved.** It's "cooking per fetch." If it's high, you're cooking; if it's low, you're fetching. `x + 1` has an intensity near **0.5** — half a flop per number, basically nothing. A large matmul at size 4096 has an intensity in the *thousands*. Same chip, same day: one is starving, one is feasting, and the ratio told you which before you ran anything.]]

[[fig: A warm hand-drawn metaphor illustration titled "Cooking per fetch". Center: a balance scale. On the left pan, a single small cook figure carrying one ingredient box up a long hallway labeled in blue "one fetch (a trip to the pantry)". On the right pan, a pile of many little green multiply-add symbols labeled "the cooking you get to do with it". Above, a big handwritten formula "arithmetic intensity = flops ÷ bytes = cooking per fetch". Two example tags hang below: a red one "x+1 → ≈0.5 (almost no cooking per trip — STARVING)" with a sad idle cook, and a green one "big matmul → thousands (feast per trip — BUSY)" with a happy stirring cook. Dashed takeaway box: "high ratio = cooks stay busy · low ratio = cooks wait." Excalidraw style, white background, charming, hand-lettered labels. || Arithmetic intensity in plain words: how much cooking you get out of each trip to the pantry.]]

Now — how much cooking per fetch do you need before the cooks stop waiting? That depends on the kitchen. A kitchen with blazing-fast cooks and a narrow hallway needs a *lot* of cooking per fetch to keep everyone busy. The crossover point — the exact "cooking per fetch" where the cooks are perfectly balanced against the hallway — is a property of the hardware, and it has a name we'll meet in a moment: the **ridge point**.

Let's put real numbers on the kitchen so it stops being a cartoon.

[[note: example || The H100 kitchen, two hardware facts to write on the board. Cooking speed (peak compute): about **989 TFLOP/s** of BF16 through the tensor cores — 989 trillion multiply-adds per second. Hallway width (peak memory bandwidth): about **3.35 TB/s** from HBM. Divide cooking by hallway: `989e12 / 3.35e12 ≈ 295`. So on an H100 you need to do about **295 flops for every byte** you fetch, just to break even. Below 295, the hallway is your wall. That's a *brutal* bar — and it's the number that runs modern kernel work.]]

## The roofline: the kitchen drawn as one chart

Here's the picture that turns all of this into something you can *read off a wall*. It's called the **roofline**, and it's just the kitchen drawn as a graph.

Put "cooking per fetch" (arithmetic intensity) on the bottom, left to right. Put "how fast you're actually going" (achievable flops per second) up the side. The kitchen imposes two ceilings. The first is a flat line across the top: **the cooks can't cook faster than top speed** — 989 TFLOP/s, however much you feed them. That's the *compute roof*. The second is a slanted line rising from the corner: **if you're waiting on the hallway, then the more cooking you do per fetch, the faster you go** — proportionally, climbing the slope until it bumps into the flat roof. That slope *is* the hallway width — the *memory roof*.

The two roofs meet at one corner — the **ridge point**, at 295 flops per byte on an H100. Left of the corner, on the slope: **memory-bound, fetching.** Right, under the flat roof: **compute-bound, cooking.**

[[fig: A hand-drawn roofline chart in Excalidraw style on white paper, titled "The kitchen as one chart (H100 roofline)". X-axis hand-lettered "cooking per fetch — arithmetic intensity (flop/byte), log scale", Y-axis "how fast you actually go (flop/s), log scale". A blue diagonal line rising from bottom-left labeled in blue "hallway roof: slope = bandwidth 3.35 TB/s (you're FETCHING here)". A flat black horizontal line across the top labeled in green "cook roof = 989 TFLOP/s (top cooking speed)". The two meet at a hand-drawn orange circle labeled "RIDGE POINT" with red note "≈ 295 flop/byte". Region left of ridge shaded with light blue hatch labeled "MEMORY-BOUND — on the slope, cooks wait". Region right of ridge shaded light green labeled "COMPUTE-BOUND — under the roof, cooks busy". A little cook icon standing idle on the slope, a busy cook icon under the flat roof. Dashed vertical line drops from ridge to x-axis. Dashed takeaway box: "left of the corner = fetch fewer bytes · right of the corner = do faster math." Hand-lettered labels. || The roofline is the kitchen as a graph: a slanted hallway roof, a flat cook roof, and a ridge point where they meet.]]

[[note: say || At the board, trace it with your finger: "Start at the bottom left — barely any cooking per fetch. You're stuck on this slope; you're fetching. As I move right, I do more cooking per fetch, so I climb — faster, faster — until I hit *this corner*. That's the ridge. Past it, the flat roof takes over: now I'm cooking as fast as the cooks physically can, and fetching more doesn't help. Your kernel is a single dot on this chart. Your whole job is to find where the dot sits."]]

## The same kernel, three different fates

Here's the demo that makes it click, and it's beautiful because it's the *same* matrix multiply every time — only the size `N` changes.

- **Tiny (N=128).** There's almost no work. By the time the GPU launches the kernel and gets going, it's done. The kitchen barely opened before the order was filled. This is **overhead-bound** — opening up. It sits *below* both roofs, because the roofline can't even see this problem; it's too small to matter.
- **Medium (N=1024).** Now there's real work, and a well-written kernel lands *right at the ridge*. Neither cleanly cooking nor cleanly fetching. This is the ambiguous zone where small changes tip you either way.
- **Huge (N=8192).** Cooking per fetch is now in the thousands — far to the right. The kernel is pressed flat against the cook roof, **compute-bound**, feasting. Every remaining ounce of speed comes from feeding the cooks better, never from touching memory.

[[fig: A hand-drawn roofline with three plotted matmul dots, Excalidraw style, white background, titled "One matmul, three fates". Same axes: blue sloped hallway roof, flat black cook roof at 989 TFLOP/s, orange ridge circle at 295. Three red-labeled dots: dot ① a small grey dot floating BELOW both roofs near the origin labeled "N=128: overhead-bound — the kitchen barely opened", dot ② sitting right at the ridge corner labeled "N=1024: right at the corner, ambiguous", dot ③ far right pressed under the flat roof labeled "N=8192: compute-bound, cooks feasting". A curved blue dashed arrow sweeps left-to-right through the dots labeled "bigger N → more cooking per fetch → dot moves RIGHT". Dashed takeaway box: "same algorithm, different size = different regime." Hand-lettered. || The identical matmul lands in all three regimes depending only on its size — regime is a property of the workload, not just the code.]]

[[note: confusion || The number-one confusion: students think a kernel is *permanently* one type — "matmul is compute-bound, period." It isn't. The **same** matmul is overhead-bound when tiny and compute-bound when huge. The fix: say "a kernel doesn't *have* a regime, it *lands in* a regime — and the size of the data moves it." Show the three dots on one chart. Once they see one algorithm plotted in three places, the confusion evaporates.]]

## Why this is the highest-leverage skill you can teach

Here's the payoff, and it's why this chapter is the spine of the course. **Once you know your regime, the menu of useful fixes collapses to almost nothing.** That's the gift.

If you're **fetching** (memory-bound), you widen the hallway or fetch less: fuse operations so you don't run back to the pantry between every step, cache ingredients close by, use smaller numbers (lower precision) so each trip carries less. You do **not** reach for faster cooking — the cooks are already idle.

If you're **cooking** (compute-bound), you cook faster or cook smarter: use the tensor cores, pick the right precision, keep every station busy. You do **not** obsess over a few extra pantry trips — the hallway isn't your problem.

If you're **opening up** (overhead-bound), you batch orders together and stop reopening the kitchen for every sandwich: bigger batches, fuse many tiny kernels into one launch.

[[note: aha || The whole reason this is the master model: *diagnosis before treatment.* Optimize the wrong regime and you will work incredibly hard to make nothing happen — polishing the cooks' knife skills while they starve in the hallway. A student who can name the regime in under a minute has the single highest-leverage skill in performance engineering. Everything else in the workshop is just *techniques for a specific regime.* This chapter tells them which drawer to open.]]

[[note: production || This is where the money is, right now. When DeepSeek or Meta serve a model to millions, most of the work is *memory-bound* — and it gets more so every year, because each GPU generation adds cooking speed faster than hallway width. The A100's ridge was ~210 flops/byte; the H100's climbed to ~295. The compute grew ~3×, the bandwidth only ~2×. So the "just cook" club keeps shrinking, and clever *fetching* — this is exactly why FlashAttention became famous — keeps getting more valuable. The kernel engineer's job is, more and more, a logistics job wearing a math hat.]]

[[fig: A hand-drawn technical "diagnosis → treatment" table titled "Name the regime, then the fix picks itself", Excalidraw semantic-color style, white background. Three stacked rows, each a rounded box. Row 1 header in orange "FETCHING (memory-bound)": blue mechanism note "widen the hallway / fetch less" with purple code-style chips "fuse ops", "cache close", "lower precision", "coalesce"; red warning "do NOT reach for faster math". Row 2 header in orange "COOKING (compute-bound)": green note "cook faster / smarter" with purple chips "tensor cores", "right precision", "keep every station busy"; red warning "do NOT chase extra pantry trips". Row 3 header in orange "OPENING UP (overhead-bound)": yellow note "stop reopening the kitchen" with purple chips "bigger batches", "fuse tiny launches", "CUDA graphs". A numbered circle ① on the left of each row and a big orange arrow labeled "regime → drawer". Dashed takeaway box: "diagnosis before treatment — wrong regime = hard work, no result." Hand-lettered labels. || The payoff table: once you name the regime, the menu of useful optimizations collapses to a handful.]]

[[sn: A subtlety worth a sidenote for the sharper students: the "bytes" in cooking-per-fetch means bytes moved *to and from the far pantry (HBM)* — not bytes touched. If an ingredient is already sitting on the nearby counter (cache), grabbing it is free. That's *why* caching helps: it doesn't reduce cooking, it reduces trips to the far pantry, which slides your dot rightward on the roofline.]]

## The one habit to drill into them

Before touching any kernel, **predict the regime out loud, then measure.** "This is a little element-wise op, so it should be memory-bound — the cooks will be idle, I bet we hit maybe 5% of peak cooking speed." Then run the profiler and check.

When the prediction is right, they *understand* the kernel. When it's wrong, they've found something worth knowing — a hidden trip to the pantry, a kitchen that never filled up. That predict-then-measure loop is the heartbeat of every optimization in this course.

[[note: demo || The live demo to run once: take element-wise `x + 1` and a big matmul, and show both their achieved-flops as a fraction of the 989 TFLOP/s peak. The `x + 1` will limp in at a few percent — "see, the cooks are idle, we're fetching." The big matmul will hit 70–90% — "now they're cooking." Same GPU, same minute. The gap between those two numbers *is* the whole chapter, and it lands harder than any slide.]]

That's the model. A kitchen that's either cooking, fetching, or opening up; one ratio — cooking per fetch — that decides which; and one chart, the roofline, that lets you read a kernel's fate off two axes. If a student walks out able to ask "are we cooking, fetching, or opening up?" and knows the fix is different for each, you've handed them the mental spine for the entire workshop.

## You can now teach

- The **three regimes** as a kitchen: **cooking** (compute-bound), **fetching** (memory-bound), and **opening up** (overhead-bound) — and that "bound" just means "held back by."
- The surprise at the heart of the course: **GPUs mostly fetch, not cook** — demonstrated with `x + 1` (two memory trips per trivial flop) versus a big reuse-heavy matmul.
- **Arithmetic intensity** as "cooking per fetch," and the **ridge point** (~295 flops/byte on an H100) as the break-even bar, built from 989 TFLOP/s ÷ 3.35 TB/s.
- The **roofline** as the kitchen drawn as one chart — a slanted hallway roof, a flat cook roof, and the ridge where they meet — and how to read a kernel's fate as a dot on it.
- Why the **same matmul** lands in all three regimes depending only on its size (N=128 vs 1024 vs 8192) — regime is a property of the workload, not a fixed label.
- The **payoff**: naming the regime collapses the fix-menu to a handful, so you treat the right disease — plus the production stakes (memory-bound work is growing, which is why FlashAttention-style fetching tricks keep winning).
