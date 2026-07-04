By the end of this chapter you'll be able to stand at a whiteboard, draw one plot, and teach a student who has never seen it *the single most important picture in kernel engineering* — the one that says, before you write any code, the fastest a kernel could ever go and which wall is stopping it. Students think this plot is scary. It has log axes and a Greek-sounding name. It is not scary. It is two lines and a corner. That's the whole thing. Let's build it so it feels obvious.

## The one thing the roofline answers

Here is the question every kernel engineer is secretly asking: *is my code slow because the machine is doing too much math, or because it can't move the data fast enough?* Those are the only two walls. The roofline is the picture that tells you which wall you're hitting — and, better, tells you the *ceiling*, the fastest you could ever possibly go.

[[note: say || "There are only two reasons a kernel is slow. Either the calculator is maxed out — it's doing math as fast as it physically can. Or the calculator is sitting there bored, waiting for numbers to arrive. The roofline is one drawing that tells you which of those two is true. That's it. Two walls, one picture."]]

Don't say the word "roofline" yet. Draw the picture first. Name it last.

## Metaphor: the kitchen with a narrow doorway

You already have the kitchen from the CPU-vs-GPU chapter. Reuse it. The GPU is a cafeteria — thousands of cooks who can chop and scoop blindingly fast. But the ingredients live in a pantry down the hall, and there's only one narrow doorway between the pantry and the kitchen.

Two things can slow lunch down. Maybe the cooks are genuinely maxed out — every cook chopping as fast as human hands allow. That's a **compute** limit; you literally cannot cook faster. Or maybe the cooks are standing around, and the real jam is the doorway — ingredients trickle in too slowly. That's a **bandwidth** limit; the cooks are fine, the *hallway* is the problem.

[[note: metaphor || The kitchen has a fixed top cooking speed (the cooks) and a fixed top delivery speed (the doorway). Lunch is limited by whichever one runs out first. If your recipe needs a LOT of chopping per ingredient — say, one carrot chopped a thousand times — the doorway keeps up easily and the cooks are the limit. If your recipe barely touches each ingredient — one carrot, one scoop, out the door — the cooks fly through it and the doorway is the limit. The recipe decides which wall you hit.]]

[[fig: A warm hand-drawn kitchen illustration titled "Two ways lunch gets slow". On the right, a cafeteria kitchen full of small green cook figures at counters. On the left, a pantry labeled in green "pantry (memory)" connected to the kitchen by a single narrow doorway labeled in blue "the doorway = bandwidth". TOP scenario, boxed: cooks all frantically chopping one carrot many times, a green note "recipe needs lots of chopping per carrot -> COOKS are the limit (compute-bound)", doorway calm. BOTTOM scenario, boxed: cooks standing idle with little zzz marks while single ingredient boxes trickle through the doorway, a blue note "recipe barely touches each carrot -> the DOORWAY is the limit (memory-bound)". A dashed takeaway box spanning the bottom: "lunch is capped by whichever runs out first: the cooks or the doorway." Excalidraw style, white background, charming, handwritten labels. || The whole roofline in a kitchen: a fixed cooking speed and a fixed doorway speed, and the recipe decides which one caps you.]]

That's the entire idea. Two speeds — how fast you can cook, how fast you can deliver — and the recipe decides which one bites. Now let's put numbers on the two speeds.

## The two numbers, from the machine

Every GPU hands you exactly two hard ceilings. Write them on the board and circle them; the whole chapter hangs off these two.

**Peak compute** — how much math the chip can do per second. On an H100, the tensor cores do about **989 TFLOP/s** in BF16. That's 989 trillion multiply-adds every second. That's the cooks' top speed.

**Peak bandwidth** — how many bytes per second you can pull from the GPU's memory (called HBM). On an H100, that's about **3.35 TB/s**. That's the doorway's top speed.

[[note: production || These are real H100 numbers, and they're the numbers a working kernel engineer at NVIDIA, DeepSeek, or a vLLM shop actually quotes. One honesty note to pass to students: the marketing slide says ~1979 TFLOP/s, but that assumes a trick called "sparsity" you almost never have. Use 989. Teaching the honest number is a small act of respect for the student — they'll thank you when their measurements match.]]

Now do the one division that makes the whole model click. How many math operations can the cooks do in the time it takes *one byte* to squeeze through the doorway?

```
989e12 FLOP/s  ÷  3.35e12 byte/s  ≈  295 FLOPs per byte
```

[[note: aha || Say this slowly and let it land: "For every single byte that crawls through the doorway, the cooks can do about 295 multiply-adds in the time it takes to arrive." The machine can compute roughly 295 times faster than it can feed itself. That imbalance is not a bug — it's the defining fact of every modern GPU, and it's why kernel engineering is mostly about the *doorway*, not the math.]]

[[fig: A hand-drawn "two speeds" figure titled "The two numbers, and the one ratio". LEFT box labeled COMPUTE in orange: a small block packed with tiny green squares labeled "H100 tensor cores", green spec underneath "989 TFLOP/s (BF16, honest / dense)". RIGHT box labeled MEMORY in orange: a box labeled "HBM3 memory" with a fat blue arrow leaving it carrying hatched 2-byte blocks, green spec "3.35 TB/s". A big red division sign between them, and a red circled result "= 295 FLOPs / byte". A blue dashed arrow curves from the result to a handwritten note "this magic number is called the RIDGE POINT". Bottom dashed takeaway box: "the machine can do ~295 sums in the time ONE byte arrives." Excalidraw style, white background, handwritten. || Two hardware numbers, one division, and the ridge point that falls out of it.]]

## The recipe's number: arithmetic intensity

The 295 belongs to the *machine*. Now the *recipe* — your kernel — has its own matching number: how many math operations does it do for each byte it drags in from memory? That's called **arithmetic intensity**. Long name, dead-simple idea: FLOPs done, divided by bytes moved.

```
arithmetic intensity  =  total math done  /  total bytes moved from memory
```

[[note: teach || Teach arithmetic intensity as "reuse." A high number means you loaded a byte once and squeezed lots of work out of it before letting it go. A low number means you loaded a byte, used it once, and threw it away — wasteful of the doorway. The whole craft of kernel engineering is *raising this number*: load each ingredient once, chop it many times. Write "arithmetic intensity = reuse" on the board and box it.]]

Let's do it by hand with a tiny example, because a number you compute yourself is a number you believe.

[[note: example || Tiny by-hand case. Add two lists of 4 numbers: `c = a + b`. You read 4 numbers of `a`, read 4 of `b`, write 4 of `c`. In BF16 each number is 2 bytes, so that's 12 numbers × 2 = 24 bytes moved. And the math? Just 4 additions. So arithmetic intensity = 4 FLOPs ÷ 24 bytes ≈ **0.17 FLOPs per byte**. Compare to the ridge of 295. You're not close — you're nearly two thousand times below it. This kernel will *never* trouble the cooks. It's all doorway.]]

Now compare that to a big matrix multiply — the operation the whole workshop exists to speed up. A square `N × N` matmul does about `2N³` FLOPs but only moves about `3N²` numbers (the three matrices). So its arithmetic intensity is roughly `N/3` — and crucially, it *grows with N*. For a small matrix it's tiny. For `N = 8192` it's in the thousands, far past 295.

[[note: aha || Here's the jaw-dropper for students: "The exact same operation — matrix multiply — changes which wall it hits depending only on how big it is. Small matrix? Doorway-limited. Huge matrix? Cook-limited. Nothing about the code changed. Only the size." That's why one algorithm can live in two different worlds, and why the roofline is a *plot* and not a single verdict.]]

## Now draw the plot — two lines and a corner

Here's the reveal. Take the two speeds, and instead of two separate facts, draw them as two lines on one chart.

- **Sideways axis:** arithmetic intensity (the recipe's reuse). Left = low reuse, right = high reuse.
- **Up axis:** how fast the kernel actually runs (FLOP/s).

Draw two lines:

1. **The bandwidth ramp** — a diagonal line rising from the bottom-left. It's the doorway limit. If you're doorway-limited, then the more reuse you have, the more real work you get per byte, so speed rises steadily. Slope = your bandwidth.
2. **The compute ceiling** — a flat horizontal line across the top at 989 TFLOP/s. It's the cooks' limit. No matter how much reuse you have, the cooks cannot chop faster than this. Flat roof.

The ceiling is *whichever line is lower* at your recipe's spot. That's it. That's the roofline — it literally looks like the roof of a house: a slope going up, then a flat top.

[[note: say || As you draw the two lines: "Your kernel can never go above the sloped line — that's the doorway. And it can never go above the flat line — that's the cooks. So your true ceiling is whichever of these two is lower where your kernel sits. Draw a house roof: it goes up, then it flattens. Your kernel lives underneath that roof, always."]]

[[fig: A hand-drawn roofline chart in Excalidraw style on white paper, titled "The roofline: two lines and a corner". X-axis hand-lettered "arithmetic intensity (FLOPs per byte) — how much you reuse", Y-axis "how fast the kernel runs (FLOP/s)". A blue diagonal line rising from bottom-left labeled in blue "the doorway ramp: slope = bandwidth 3.35 TB/s". A flat black horizontal line across the top labeled in green "the cook ceiling = 989 TFLOP/s". The two lines meet at a hand-drawn circle marked in orange "RIDGE POINT ~295 FLOPs/byte". The region LEFT of the ridge shaded light blue hatch, bracketed and labeled "MEMORY-BOUND — you're on the ramp, doorway is the wall". The region RIGHT of the ridge shaded light green, bracketed and labeled "COMPUTE-BOUND — you're under the flat roof, cooks are the wall". A dashed vertical line drops from the ridge circle to the x-axis. Bottom dashed takeaway box: "left of the corner = move fewer bytes. right of the corner = do faster math." Excalidraw style, white background, handwritten labels. || The finished roofline. A sloped bandwidth ramp, a flat compute ceiling, and the ridge point where they meet.]]

## The corner is the whole model

The two lines cross at exactly one spot. That spot is the ridge point — and it's the 295 we already computed, `peak compute ÷ peak bandwidth`. It splits the world in two.

- **To the left of the corner** (arithmetic intensity below 295): you hit the sloped ramp first. You're **memory-bound**. The doorway is your wall. Faster cooks won't help — they're already idle.
- **To the right of the corner** (arithmetic intensity above 295): you hit the flat roof first. You're **compute-bound**. The cooks are your wall. A faster doorway won't help — the ingredients are already waiting.

[[note: confusion || The number-one student confusion: "so I should always make my kernel faster at math, right?" No — and this is the sentence that fixes it: "First find which side of the corner you're on. If you're memory-bound, speeding up the math does *nothing* — the math units are already sitting idle. You have to fix the wall you're actually hitting, not the one that feels important." Make them point at the plot and say which side they're on *before* proposing any fix. That single habit prevents weeks of wasted effort.]]

[[note: production || This isn't academic. When DeepSeek or Meta serves a model to millions of people, an engineer plots the kernel on exactly this chart to decide where to spend the week. The famous FlashAttention kernel is, geometrically, someone dragging attention's dot to the *right* on this plot — raising its arithmetic intensity by keeping data on-chip instead of round-tripping to HBM. Every serving stack in production, vLLM included, is full of kernels that were tuned by reading their position on a roofline.]]

## Reading a real kernel off the chart

Now make it concrete by plotting an actual kernel. The naive matrix-multiply kernel — the very first rung of this workshop's GEMM ladder — reuses almost nothing. It re-reads a whole row and column from memory for every single output number. Its arithmetic intensity is about `0.25` FLOPs per byte. That's *1200 times* below the ridge. On the plot, it's a dot pinned to the far bottom-left of the ramp.

[[note: aha || The roofline *predicts the score before you run anything*. That naive kernel's ceiling on the ramp is `0.25 × 3.35 TB/s ≈ 0.8 TFLOP/s` — a rounding error next to 989. And when you actually run it, it measures at a humiliating **1.3% of cuBLAS** (the vendor's fast library). The chart told you it would be terrible *before you profiled it*. That's the magic students remember: "the plot knew."]]

And every optimization in the four-week workshop — tiling, shared memory, register blocking, vectorized loads — is the *same single motion*: drag that dot to the right (more reuse, higher arithmetic intensity) until it climbs the ramp and hits the flat roof, then push it up the roof toward the vendor's library. The ladder goes 1.3% → 8.5% → 12.8% → 36.5% → 68.7% → **93.7% of cuBLAS**, and on the roofline it's one dot walking up and to the right.

[[fig: A hand-drawn roofline with plotted kernel dots, Excalidraw style, titled "Optimization = dragging the dot up and right". Same axes: blue sloped doorway ramp, flat black cook ceiling at 989 TFLOP/s, orange ridge circle at 295. Dot ① far bottom-left ON the ramp, red label "naive matmul, AI~0.25 -> just 1.3% of cuBLAS". A curved blue dashed arrow sweeps rightward and upward through several small dots labeled in blue "tiling, shared memory, register blocking...", ending at dot ③ pressed against the flat roof, red label "well-tiled, big N -> 93.7% of cuBLAS". A small grey note near dot ③: "the last 6% is polish, not a new regime". Dashed takeaway box: "same algorithm. every optimization just moves the dot toward the corner and up the roof." Excalidraw style, white background, handwritten. || The GEMM ladder as one motion on the roofline: drag the dot right by reusing bytes, then climb the roof.]]

[[sn: The bytes in arithmetic intensity mean bytes moved *to and from the GPU's main memory (HBM)* — not bytes touched. Data that stays on-chip in a cache or shared memory is free. That's exactly why tiling raises arithmetic intensity: it turns memory round-trips into on-chip reuse. You don't do less math; you move fewer bytes.]]

## The roofline as a "when do I stop?" rule

Here's the part that makes senior engineers love this plot. The roofline doesn't just tell you what to fix — it tells you when to *quit*. When your dot is pressed against the roof, the vendor's own library is sitting on that same roof. The last few percent between you and it aren't a new wall to break; they're the asymptote. You're done.

[[note: teach || This is the emotional payoff to deliver last, and it reframes the whole discipline. "The roofline saves you from grinding for weeks on a wall you already hit, while the *other* wall — the one with room above it — goes untouched. Before you write a kernel, you compute its arithmetic intensity, find which side of the corner it's on, and predict the regime. Then you measure, plot the real dot, and look at the gap to the roof above it. That gap is your remaining headroom — the single most honest number in performance work." Put it as: optimizing without a roofline is fishing; with one, it's a checklist.]]

[[sn: The honest roofline has *lowered* roofs. The 3.35 TB/s and 989 TFLOP/s are theoretical peaks you never quite reach — real coalescing and operand-delivery losses pull both ceilings down. A working engineer draws their *achieved* bandwidth and compute as dashed lines just under the theoretical ones and measures headroom against those. A kernel at 70–90% of the honest floor is genuinely excellent; one at 5% has a real bug.]]

## Teaching notes: how to run this at the board

Reveal it in this order and it lands every time:

1. **The question (2 min).** "Two reasons a kernel is slow: cooks maxed out, or doorway too narrow." Don't say "roofline" yet.
2. **The kitchen (4 min).** Draw the pantry, the narrow doorway, the cooks. Act out both failure modes — frantic chopping vs. idle waiting.
3. **The two numbers (3 min).** Write 989 TFLOP/s and 3.35 TB/s. Circle them. Do the division live: `≈ 295 FLOPs per byte`. Let the imbalance shock them.
4. **The recipe's number (4 min).** Define arithmetic intensity as "reuse." Compute the tiny 4-element add by hand (≈0.17). Then say matmul's is `N/3` and grows with size.
5. **Draw the plot (5 min).** Sideways = reuse, up = speed. Draw the ramp, then the flat roof. Point out it looks like a house. Mark the corner at 295.
6. **The one live demo (3 min).** Plot the naive matmul dot at 0.25, read its ceiling off the ramp (~0.8 TFLOP/s), then reveal it measures 1.3% of cuBLAS. "The plot knew before we ran it."
7. **The stop rule (2 min).** Dot on the roof = done. That's the payoff.

[[note: demo || The one demo that makes jaws drop: put the naive-matmul dot on the far-left ramp, compute its ceiling in front of them, and *then* show the 1.3%-of-cuBLAS measurement matching it. The plot predicted a real, embarrassing number before any code ran. If you have a projector, plot the whole GEMM ladder as a row of dots walking up and to the right toward 93.7%. Nothing sells the roofline like watching the dot climb.]]

Two confusions to head off. First, students mix up which axis is which — drill "sideways is reuse, up is speed." Second, they think the ridge point is a goal to *land on*. It isn't; it's just the border between the two worlds. You want to be under whichever roof is lower, as close to it as you can get — not necessarily at the corner.

## You can now teach

- The **one question** the roofline answers: is my kernel cook-limited (compute) or doorway-limited (memory)?
- The **kitchen metaphor** — a fixed cooking speed and a fixed doorway speed, and the recipe decides which caps you.
- The **two machine numbers** (989 TFLOP/s, 3.35 TB/s), the division that gives the **ridge point at ~295 FLOPs/byte**, and why the machine computes ~295× faster than it feeds itself.
- **Arithmetic intensity** as "reuse," computed by hand on a tiny example, and why big matmuls have high intensity while element-wise ops have almost none.
- How to **draw the roofline** — a sloped bandwidth ramp, a flat compute ceiling, a corner — and read which wall a kernel hits from its position.
- The **production hook and the stop rule**: this is the exact plot FlashAttention and vLLM engineers reason with, and a dot pressed to the roof means you're done — stop grinding the wall you already hit.
