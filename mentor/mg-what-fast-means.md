By the end of this chapter you can stand at a whiteboard and teach *what "fast" actually means* on a GPU — not with a stopwatch, but with the three honest units a kernel engineer lives by: **operations**, **FLOP/s**, and **bytes moved**. And you'll be able to explain why a stopwatch, on its own, tells you almost nothing worth knowing.

This is the chapter that turns "it feels slow" into "it's memory-bound at 3% of peak." Once a student owns these units, every later chapter — tiling, coalescing, the whole GEMM ladder — becomes a story about one of three numbers going up. So let's build the units from zero, slowly, until they feel like common sense.

## Why a stopwatch is not enough

Imagine a student runs their kernel, sees "12 milliseconds," and beams. Is that good? *You have no idea.* Twelve milliseconds might be world-class for a huge matrix, or it might be a catastrophe for a tiny one. A raw time answers "how long did it take?" but never "how close was it to the best this machine could do?" — and the second question is the whole job.

[[note: metaphor || A stopwatch is like being told a truck took two hours for its delivery. Good or bad? You can't say until you know two more things: *how much cargo* it carried (the work), and *how fast the truck can possibly go* (the machine's ceiling). Two hours to haul a full load across a state is heroic. Two hours to drop one envelope across the street is a disgrace. Time alone hides both facts. A kernel engineer never quotes time without quoting the work and the ceiling next to it.]]

[[fig: A warm hand-drawn illustration titled "Why a stopwatch lies". Center: a big cartoon stopwatch reading "12 ms" with a question mark hovering over it. Two delivery trucks branch off from it with dashed arrows. Top truck is overflowing with cargo boxes, a green handwritten label "full load, long haul -> 12 ms is HEROIC", a little trophy doodle. Bottom truck carries a single tiny envelope, a red handwritten label "one envelope across the street -> 12 ms is TERRIBLE", a little snail doodle. A dashed takeaway box at the bottom reads "time means nothing without: how much WORK + the machine's CEILING". Excalidraw style, white background, charming, handwritten labels. || The same time can be triumph or disaster. To judge it you need the work done and the ceiling it could have hit.]]

So we need to measure two things the stopwatch hides: **how much work** the kernel did, and **how much traffic** it moved. Those are our first two units.

## Unit one: operations (the work)

An **operation** here means one piece of floating-point arithmetic — one multiply, or one add. In AI, the atom is the **multiply-add**: `a × b + c`. That's two operations bundled together (one multiply, one add), and it's the beating heart of every neural network.

[[note: example || Do the smallest possible count on the board. A dot product of two length-3 lists — `[1,2,3] · [4,5,6]` — is (1·4)+(2·5)+(3·6). That's **3 multiplies and 2 adds** = 5 floating-point operations. We usually round it to "2 operations per term," so a length-3 dot product ≈ 6 operations. Tiny, countable, done by hand. This is the unit everything else is built from.]]

Now scale it. A matrix multiply of two `N × N` matrices is a grid of `N²` dot products, each of length `N`. Each dot-product term is one multiply and one add — 2 operations. So the total work is:

```
   operations  ≈  2 · N³
```

That `2N³` is the most important formula in the whole course. It says the *work* in a matmul grows with the cube of the size. Double `N`, and you do eight times the arithmetic.

[[fig: A hand-drawn technical figure titled "Counting the work", Excalidraw style, white background. Left: a small dot product drawn as two hatched row-boxes of 3 cells each (blue and green), with purple annotations "3 multiplies + 2 adds ~= 2 ops per term". A blue arrow labels this "one term = 1 multiply-add = 2 ops". Center: a 2x2 grid labeled in red "C = N x N cells", each cell tagged "one dot product of length N". Right: a bold orange formula box "operations ~= 2 x N^3", with a purple worked line "N=1000 -> 2 billion ops". A green sticky note: "work is fixed by the PROBLEM, not your code". Dashed takeaway box: "count one multiply-add, scale to the grid, get 2N cubed." Numbered circles (1) count a term (2) count the cells (3) multiply out. || Building the work count from the atom up: one multiply-add, scaled across the grid of dot products, gives 2N³ operations — fixed by the problem, not the code.]]

[[note: aha || Put a real number on it and watch the room. For `N = 1000`, that's `2 × 1000³` = **2 billion** operations — for one matrix multiply. A real model does matrices bigger than this, millions of times, for every word it writes. Say it plainly: "Before we ever talk about *speed*, understand the *amount*. This is a mountain of arithmetic, and our whole job is to move that mountain efficiently."]]

Here is the key discipline to teach: the operation count is a property of the **problem**, not your code. A matmul of size `N` is `2N³` operations whether you write it beautifully or terribly — nobody can make it do *fewer* multiply-adds. That fixedness makes it a fair yardstick: it's the numerator we'll divide everything by.

## Unit two: FLOP/s (the rate)

Now we combine work and time. **FLOPs** (with a capital-S) means "floating-point operations" — a *count*, the thing we just measured: `2N³` of them. **FLOP/s** (with a slash) means "floating-point operations *per second*" — a *rate*, work divided by time.

```
   FLOP/s  =  operations performed  /  time taken
```

[[note: confusion || This is the single most confused pair of terms in the field, and you must nail it on day one. **FLOPs = a count** (how much work — like "kilometres"). **FLOP/s = a speed** (work per second — like "kilometres per hour"). Write them stacked on the board with the units spelled out: "FLOPs → operations. FLOP/s → operations ÷ seconds." Say: "One is distance, one is speed. If a student ever says 'this kernel does 900 teraflops' — stop and ask, flops or flop-per-second? They almost always mean the rate."]]

[[note: example || Make it concrete. Our `N=1000` matmul is `2 × 10⁹` operations. Suppose it runs in 2 milliseconds (0.002 s). Then the rate is `2e9 / 0.002` = `1e12` FLOP/s = **1 TFLOP/s** (one trillion operations per second). Now the "12 ms" from earlier finally *means* something — you divide the fixed work by the time and get a speed you can compare against the machine.]]

[[fig: A warm hand-drawn illustration titled "FLOPs vs FLOP/s — distance vs speed", white background, charming. Left half: a road with distance markers, a green label "FLOPs = a COUNT of work, like KILOMETRES", a signpost reading "2 billion ops". Right half: a car with a speedometer, a blue label "FLOP/s = a RATE, like KM PER HOUR", the speedo reading "1 TFLOP/s". A red divider between them with a big handwritten "not the same!". Under each, a small formula in purple: left "how much", right "how much ÷ seconds". Dashed takeaway box: "one is distance, one is speed. always ask: flops, or flop-per-second?" Excalidraw style, handwritten labels. || The confusion killer, drawn: FLOPs is a distance (a count of work), FLOP/s is a speed (work per second). Never let a student blur them.]]

And the machine has a top speed. An **NVIDIA H100** GPU can sustain about **989 TFLOP/s** in BF16 through its tensor cores — roughly a *thousand trillion* multiply-adds per second. That's the ceiling. So the real scoreboard isn't time at all; it's:

```
   how good is my kernel  =  my FLOP/s  /  the machine's peak FLOP/s
```

[[note: say || "Forget the stopwatch. Here's the only score that matters: take the work your kernel did, divide by the time, and you get *your* speed. Divide *that* by the fastest this chip can go — 989 teraflops on an H100 — and you get a percentage. Ninety percent means you're a hero. Three percent means the chip is asleep, and your job is to wake it up." That percentage — "percent of peak" — is the number this whole workshop is about moving.]]

[[fig: A hand-drawn "speedometer of peak" figure, technical Excalidraw style. A large arc gauge like a car speedometer, hand-lettered. The far right of the arc is a bold green wall labeled "989 TFLOP/s — H100 peak (BF16)". A red needle points at about 3% of the arc, labeled in red "naive kernel: 30 TFLOP/s = 3% of peak". A second faint orange needle points near the far right, labeled "well-tuned: ~90% of peak". Blue annotation box on the left shows the formula: "my FLOP/s = (2N cubed ops) / (time)". A purple note underneath: "score = my FLOP/s ÷ peak FLOP/s". Dashed takeaway box: "we don't chase low time — we chase HIGH % of peak". White background, handwritten labels. || The real scoreboard: your achieved FLOP/s as a fraction of the machine's peak. Percent-of-peak, not raw time, is the score.]]

## Unit three: bytes moved (the traffic)

Here's where beginners get ambushed. You'd think high percent-of-peak just means "do the math fast." It doesn't — because before the math units can chew on a number, that number has to *arrive*. Numbers live in memory, off to the side of the chip, and must be carried in. That carrying has a cost and a speed limit all its own.

A single 32-bit float is **4 bytes**. To read a matrix, every one of its numbers must travel from memory to the chip — that's traffic, measured in **bytes moved**. And the pipe that carries it, **HBM** (High-Bandwidth Memory), has a top speed too: an H100 pulls about **3.35 TB/s** — 3.35 trillion bytes per second. Fast, but finite.

[[note: metaphor || The kitchen and the pantry. Your math units are a row of blazing-fast cooks. But the ingredients (the numbers) sit in a pantry down the hall (HBM). Every ingredient has to be carried up the hallway before a cook can touch it. The hallway has a width — a *bandwidth* — and it does not care how fast your cooks are. If the cooks can chop faster than the hallway can deliver rice, the cooks stand idle. The chip's real struggle is almost never "can the cooks chop?" It's "can we get ingredients up the hallway fast enough?"]]

[[note: example || Count the traffic by hand for a matmul. To multiply two `N × N` matrices you must, at minimum, read `A`, read `B`, and write the result `C` — three matrices of `N²` numbers, 4 bytes each: `3 × N² × 4 = 12N²` bytes. For `N = 1000` that's `12 million` bytes ≈ 12 MB of *unavoidable* traffic, minimum, before any wastefulness. Write it next to the `2N³` operations. Now you have both ingredients: work on top, traffic on the bottom.]]

[[fig: A warm hand-drawn illustration titled "The cooks and the hallway". Right side: a kitchen counter packed with many small fast cook figures holding knives, little speed-motion marks, a green label "math units — 989 TFLOP/s, blazing fast". Left side: a distant pantry drawn as shelves stacked with ingredient boxes, labeled in green "HBM memory — 80 GB pantry". Connecting them: a long hallway of fixed width with a few ingredient boxes trickling down it, labeled in blue "bandwidth — 3.35 TB/s, the hallway width". Some cooks on the right are tapping their feet with little 'zzz' idle marks. Red annotation over the idle cooks: "cooks faster than hallway -> they WAIT". Dashed takeaway box: "fast math is cheap. fast FEEDING is the hard part." Excalidraw style, white background, charming, handwritten. || Two speed limits, not one: the cooks (math) and the hallway (bandwidth). A GPU is usually starved by the hallway, not the cooks.]]

## Putting them together: the one ratio that predicts everything

Now the payoff. You have two counts: **operations** (work) and **bytes moved** (traffic). Divide them and you get the single most useful number in performance engineering — **arithmetic intensity**: how much math you do for every byte you carry.

```
                 operations (FLOPs)
   intensity  =  ────────────────────
                   bytes moved
```

[[note: example || Two extremes, both by hand. (1) An element-wise "add 1 to every number": for each number you read 4 bytes, do 1 add, write 4 bytes. That's 1 operation per 8 bytes ≈ **0.1 FLOPs/byte** — almost no math per byte carried. (2) A big matmul: `2N³` operations over `12N²` bytes = `N/6` FLOPs/byte. For `N=4096` that's *hundreds to thousands* of FLOPs per byte. Same chip, wildly different intensity.]]

Why does this one ratio matter so much? Because the machine has a matching ratio — its own balance point. Take the H100's two ceilings and divide them:

```
   ridge point  =  989 TFLOP/s  /  3.35 TB/s  ≈  295 FLOPs / byte
```

This **ridge point** (≈295 on an H100) is the break-even intensity. It's the whole diagnostic:

- Your kernel's intensity is **below 295** → the hallway runs dry before the cooks run out of work. You're **memory-bound**. The cooks idle. No amount of faster math helps; you must move *fewer bytes* (fuse, cache, lower precision).
- Your kernel's intensity is **above 295** → the cooks are the wall. You're **compute-bound**. This is the good place — the expensive silicon is actually busy.

[[note: aha || Here is the sentence that reframes the entire course: **"You can predict whether a kernel will be fast or slow before you write a single line of it — just by counting operations, counting bytes, dividing, and comparing to 295."** The element-wise op at 0.1 is hundreds of times below the ridge — hopeless, no matter how you code it. The big matmul at thousands is far above — a gift. The regime is decided by the *algorithm*, not your cleverness. Students think optimization is guesswork. This ratio makes it arithmetic.]]

[[fig: A hand-drawn roofline chart, technical Excalidraw style, white background. X-axis hand-lettered "arithmetic intensity — FLOPs per byte (log scale)". Y-axis "achievable FLOP/s (log)". A steep diagonal blue line rising from the origin labeled "memory roof: slope = 3.35 TB/s bandwidth". A flat horizontal green line across the top labeled "compute roof: 989 TFLOP/s". They cross at a bold black dot labeled in orange "RIDGE POINT ~= 295 FLOP/byte". Two red workload dots on the X-axis: one far LEFT near 0.1 labeled "element-wise (add 1)", sitting low on the blue diagonal, red note "memory-bound: cooks idle". One far RIGHT past the ridge labeled "big GEMM N=4096 -> thousands", sitting up on the green flat roof, red note "compute-bound: the goal". A purple formula box: "ridge = peak FLOP/s ÷ peak bandwidth". Dashed takeaway box bottom-right: "left of ridge = starved for bytes. right = busy with math. know which BEFORE you code." || The roofline. Compare your intensity to the ridge point (≈295 on an H100) and you know your regime before compiling.]]

## Why the stopwatch failed, in one clean sentence

Now you can close the loop you opened. A stopwatch gives you *time*. But "good" needs three things stacked: the **operations** (was there a lot of work?), the **FLOP/s** it implies (how fast, as a rate?), and the **bytes moved** (was the machine even *allowed* to run fast, or was the hallway the wall?). Time is one number that quietly folds all three together and hides them. The kernel engineer's craft is unfolding it back into the three units — and *that's* what tells you what to fix.

[[note: production || This isn't academic — it's how every serving stack is tuned right now. When vLLM serves Llama or DeepSeek to millions of users, engineers profile with tools like Nsight Compute that report exactly these units: achieved FLOP/s, percent of peak, bytes moved, and the resulting intensity. The famous **FlashAttention** kernel won by *raising arithmetic intensity* — it fused operations so fewer bytes crossed the slow hallway, moving attention from memory-bound toward compute-bound. And because each GPU generation (H100 → B200) grows compute faster than bandwidth, the ridge point keeps climbing — so more kernels fall into the memory-bound basin every year, and these three units only get more valuable.]]

## Teaching notes: how to deliver this at the board

Here's the sequence that lands cleanly, built to move from "obvious" to "wow" without a single leap.

**Board plan, in order.** (1) Write "12 ms — good or bad?" and let them squirm; nobody can answer. (2) Draw the truck metaphor: work + ceiling are missing. (3) Build unit one — count operations on a length-3 dot product by hand, then reveal `2N³`. (4) Build unit two — divide work by time to get FLOP/s, then divide by 989 TFLOP/s to get "percent of peak." (5) Build unit three — count bytes (`12N²`), introduce the hallway/bandwidth (3.35 TB/s). (6) Divide work by bytes → intensity. (7) Divide the two ceilings → 295. (8) Land the roofline.

[[note: teach || Reveal the units one at a time, never all three at once. The whole lesson is a *slow reveal* of what the stopwatch was hiding. Keep both running counts — `2N³` operations and `12N²` bytes — visible on the board the entire time, side by side, because the finale (intensity) is literally dividing the top of the board by the bottom. When you write the ratio, physically point up at the operations and down at the bytes. The gesture does the teaching.]]

[[note: demo || The one live demo: take any small kernel (an element-wise add) and a matmul, and for each, count operations and bytes *out loud on the board*, divide to get intensity, and predict the regime before running anything. Then run both under a profiler and show the element-wise op sitting at ~3% of peak FLOP/s (memory-bound, exactly as predicted) and the big matmul up near peak. The jaw-drop: "we called it right without touching the code — because the units, not the stopwatch, told us the answer."]]

[[note: confusion || The confusion to expect: "isn't a lower time always better?" Fix it with the truck. A kernel that finishes in less time but does less work per byte can still be *wasting* the machine — sitting at 3% of peak while a slower-looking but better-fed kernel sits at 90%. "Fast" means high percent-of-peak, not low wall-clock on a small toy input. Always quote time *with* the work and the ceiling beside it, never alone.]]

## You can now teach

- Why a **stopwatch is not enough** — time hides the work done and the machine's ceiling, and the truck metaphor makes that obvious.
- **Operations (FLOPs)** as the unit of work, counted by hand on a dot product and scaled to the `2N³` cost of a matmul.
- **FLOP/s** as work-over-time, and the real scoreboard: your FLOP/s divided by the machine's peak (989 TFLOP/s on an H100) — "percent of peak."
- **Bytes moved** as traffic across a finite hallway (HBM bandwidth, 3.35 TB/s), and why the cooks idle when the hallway runs dry.
- **Arithmetic intensity** — operations ÷ bytes — and the **ridge point** (≈295) that predicts memory-bound vs. compute-bound *before* a line of code is written.
- The **production link**: these exact units drive how vLLM, FlashAttention, and every serving stack are profiled and tuned today — and why they matter more each hardware generation.
