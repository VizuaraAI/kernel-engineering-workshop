By the end of this chapter you'll be able to stand at a whiteboard and teach the single biggest speedup in the whole GEMM ladder — the one move that takes a kernel from 12.8% of the expert library all the way to 68.7% — without any student getting lost. And the beautiful part is that the idea is small. So small you can say it in one sentence: *make each worker compute many answers from a handful of numbers it's already holding.* That's it. That's the whole chapter. Let's build it up so gently that it feels obvious.

## Where we are on the journey

Remind your students where they've climbed to, because this chapter only makes sense as the next step. So far the story has been about *moving data closer*. First we made each trip to the far-away pantry count (coalescing). Then we carried a whole box of ingredients onto the kitchen counter so everyone nearby could share it (shared memory). Each move got the data a little closer to the workers.

But we hit a new wall. Even with the box on the counter, each worker was still reaching over to grab an ingredient for *every single* multiply. Grab, multiply. Grab, multiply. The reaching itself became the bottleneck. This chapter is about a different kind of move — not "get the data closer," but "once you've grabbed it, *use it more.*"

[[note: say || "We've spent two kernels getting the ingredients close. Now we stop asking: how do we move data faster? And we start asking: once I'm holding a number in my hand, how many answers can I squeeze out of it before I put it down? That question — reuse — is the biggest win on the whole ladder."]]

## The core idea: a pocket full of work

Give every student this picture. A worker has **pockets** — a tiny number of them, right on their person, instant to reach into. These pockets are called **registers**: the fastest storage on the whole chip, private to each worker, right next to the hands that do the math. Reaching into a pocket is free. Reaching over to the shared counter is not.

Up until now, each worker computed *one* answer. They'd grab a number from the counter, multiply, grab the next, multiply. One answer per worker, and a reach for every multiply. Wasteful.

The new plan: each worker grabs a *small handful* of numbers, stuffs them in their pockets, and then computes *many* answers out of that same handful — without reaching for the counter again in between.

[[note: metaphor || Think of a **sandwich shop at lunch rush**. The slow way: a worker walks to the fridge, gets one slice of cheese, walks back, puts it on one sandwich. Walks to the fridge again for the next sandwich. All walking, no building. The fast way: grab a whole stack of cheese and a whole stack of bread in your two hands *once*, then stand at the counter and build a dozen sandwiches from what you're holding. Same fridge, same food — but now the walking almost disappears and the building never stops. Register tiling is grabbing the stack and building many sandwiches from it.]]

[[fig: A warm hand-drawn sandwich-shop illustration titled "Grab once, build many". LEFT half labeled "the slow way (kernel 3)": a small cook figure walking a long dashed path back and forth between a fridge on the far left (labeled in green "shared counter") and a single sandwich, with tired "walking..." marks and a red note "one trip per sandwich → all walking". RIGHT half labeled "the fast way (register tiling)": the same cook standing still at a counter, holding a tall stack of cheese in one hand and a stack of bread in the other (each labeled in purple "held in pockets = registers"), rapidly assembling a whole ROW of sandwiches, with a green note "grab once, build many → no walking between". A dashed takeaway box across the bottom: "the food didn't change — the WALKING did. reuse what's in your hands." Excalidraw style, white background, charming, handwritten labels. || The whole chapter in one picture: stop walking to the fridge per sandwich; grab a stack and build many from your hands.]]

Say the punchline plainly: **we are not doing less math.** A matrix multiply always needs exactly the same number of multiply-adds — that never changes. What changes is how many times we have to *reach for memory* to feed those multiplies. Fewer reaches, same math, faster kernel.

## Kernel 4: one worker, a column of eight answers (1D tiling)

Start here, because it's the simpler version and it makes the 2D version trivial afterward.

In kernel 4, we stop giving each worker one answer. We give each worker a little **column of 8 answers** stacked vertically in the output grid. Why a column, and why does that help? Here's the magic, and it's worth doing by hand.

All 8 of those answers sit in the *same column* of the result. That means — trace it on the board — all 8 of them need the *same* value from matrix `B`, but 8 *different* values from matrix `A`.

[[note: example || Draw it tiny. One column of 8 output cells. To compute all 8 on one step of the dot product, each cell wants: (a row of A) times (a column of B). The column of B is the SAME for all 8 cells — because they share a column. Only the A rows differ. So the worker loads that ONE B value into a pocket, loads the 8 A values into pockets, and does 8 multiply-adds. Count it: 1 + 8 = 9 numbers grabbed, 8 answers advanced. Nine grabs, eight useful multiplies.]]

Compare that to before. In kernel 3, one answer per step cost *two* grabs (one from A, one from B) to feed *one* multiply. That's a 2-to-1 ratio of reaching-to-math. Now it's roughly 9-to-8 — almost one grab per multiply. And if you're clever about the loop, that single B value gets reused across the *whole* column, so it barely counts at all. The useful work squeezed out of each grab went up about **eightfold**.

[[note: aha || Here's the number that lands it. Kernel 3 was stuck at 12.8% of the expert library, stalling on memory reaches. This one change — one worker owns a column of 8 instead of a single cell — jumps it to **36.5%**. That's nearly a 3× speedup, and we didn't touch the math at all. We only changed the *shape* of what each worker owns. Let the room sit with that: same math, triple the speed, just by holding your ingredients longer.]]

[[fig: A three-panel hand-drawn tiling walkthrough titled "Kernel 4: one worker, a column of 8". Panel (1): the full C matrix as a square with red dimension labels N×N, one block tile outlined in orange labeled "one thread block = 64×64", green note "a team of workers". Panel (2): zoom into that block — a grid of small cells with a single VERTICAL strip of 8 stacked cells highlighted in pale-yellow hatch, red label "TM = 8 answers, ONE worker", purple note "1 worker → 8 outputs down a column". Panel (3): the per-step zoom — on the left a blue-hatched column of 8 A values bracketed, blue note "8 A values → 8 pockets", on the right a single green-hatched B cell, green note "1 B value → 1 pocket, REUSED", a fat orange arrow labeled "×8 multiply-adds" pointing from the pockets to the yellow output strip. Numbered circles (1) grab B once (2) grab 8 A's (3) do 8 FMAs. Purple napkin math at the bottom: "9 grabs → 8 answers (was 2 grabs → 1 answer)". Dashed takeaway box: "reuse the one B value down the whole column → ~8× more math per grab". Excalidraw style, white background, handwritten. || Kernel 4: each worker owns a column of 8 outputs, holds 8 A values plus 1 reused B value in registers, and fires 8 multiply-adds per step.]]

### The one line of code that makes it legal

Show the loop, but explain it as an ordering trick, not as code. The secret is: put the dot-product step on the *outside* and the 8 answers on the *inside*. That way you grab the shared B value *once* at the top, then sweep it across all 8 answers before grabbing anything new.

```cpp
float threadResults[TM] = {0.0f};        // TM=8 answers, all in pockets

for (uint k = 0; k < BK; ++k) {          // the dot-product step, OUTSIDE
    float tmpB = Bs[k * BN + threadCol]; // grab ONE B value into a pocket
    for (uint i = 0; i < TM; ++i) {      // the 8 answers, INSIDE
        threadResults[i] += As[(threadRow*TM + i)*BK + k] * tmpB;
    }
}
```

[[note: teach || Draw the two loops as boxes, one inside the other, and physically point. "The `tmpB` line is OUTSIDE the inner loop — so we grab that B value once, then reuse it eight times without grabbing again." Circle `tmpB` in orange. Then say: "if you moved that line *inside* the inner loop, you'd grab it eight times, and the whole trick evaporates. The magic isn't the pockets — it's putting the grab in the right place so the pockets get reused."]]

[[fig: A hand-drawn pipeline-timeline figure titled "The loop order IS the trick". TWO stacked horizontal timelines. TOP labeled "kernel 3: grab, multiply, grab, multiply" — a repeating unit of three boxes across the row: a blue "grab A", a green "grab B", an orange "multiply", repeated, with grey shading over the grab boxes and a red note "2 grabs per 1 multiply → the reaching is the bottleneck". BOTTOM labeled "kernel 4: grab once, multiply eight times" — one green box "grab B → pocket" and one blue box "grab 8 A's" at the LEFT, then a long tight run of eight orange "multiply" boxes packed together, blue note "B reused ×8, no grabbing in between", green note "≈1 grab per multiply". A purple arrow between them: "move the B grab OUTSIDE the inner loop". Dashed takeaway box: "hoisting the grab out of the inner loop is what makes reuse happen". Excalidraw style, white background, handwritten. || Making the dot-product step the outer loop lets one B grab feed a tight, uninterrupted run of eight multiply-adds.]]

## Kernel 5: one worker, a whole rectangle (2D tiling)

Now the students are ready for the big one, and it's just kernel 4's idea applied *twice at once*.

Ask the question out loud: "If reusing one B value down a column of 8 was good, why not reuse in *both* directions — reuse B values across a row AND A values down a column, at the same time?" That's the whole leap. Instead of a column of 8 answers, each worker now owns a small **8×8 rectangle** of the output — 64 answers, all held in pockets.

Here's the by-hand magic, and it's gorgeous. To advance all 64 answers on one step, the worker grabs **8 values of A** (one per row of the rectangle) and **8 values of B** (one per column of the rectangle) — 16 grabs total. Then it does the **outer product**: every one of the 8 A values times every one of the 8 B values. That's 8 × 8 = **64 multiply-adds** from just 16 grabs.

[[note: example || Do this on the board with a 2×2 rectangle first so the arithmetic is tiny. Two A values: a1, a2. Two B values: b1, b2. The four answers in the little square are a1·b1, a1·b2, a2·b1, a2·b2. Count: you grabbed 4 numbers (2 A's + 2 B's), you produced 4 answers. Now scale to 8×8: grab 16, produce 64. Every grabbed number gets used 8 times — reused across a whole row or a whole column of the rectangle. That's the outer product, and it's the engine of the whole kernel.]]

Now count the reaches-to-math ratio and watch it improve. Kernel 4 was 9 grabs for 8 multiplies — about 1 flop per grab. Kernel 5 is **16 grabs for 64 multiplies — 4 flops per grab.** Four times the useful work squeezed out of every reach into shared memory. Same math (a matmul is always a matmul); four times fewer reaches per answer.

[[note: aha || The number to hang on the wall: kernel 4 was 36.5% of the expert library. Kernel 5 — this single change from a column to a rectangle — hits **68.7%.** That's a 1.9× speedup, nearly doubling, and it's the biggest single jump in the entire eight-step ladder. For the first time we cross the *halfway line* to a library NVIDIA has been hand-tuning for fifteen years. Say it slowly: past halfway, with nothing but arithmetic the students derived themselves from one idea about pockets.]]

[[fig: A hand-drawn "per-worker rectangle" zoom titled "Kernel 5: one worker owns an 8×8 rectangle". CENTER: a pale-yellow hatched 8×8 grid, red label "64 answers, ALL in pockets (registers)". To its LEFT a thin blue-hatched vertical strip of 8 cells labeled "8 values of A → pockets", purple note "grabbed this step". ABOVE the grid a thin green-hatched horizontal strip of 8 cells labeled "8 values of B → pockets". A blue dashed arrow from the left strip and a green dashed arrow from the top strip both sweep INTO the grid, meeting at one highlighted cell, orange annotation "outer product: A[i] × B[j] → answer[i][j]". Purple napkin math bottom-left: "64 multiplies from 16 grabs = 4 per grab (was ~1)". Dashed takeaway box bottom-right: "same math, 4× fewer grabs per answer → memory stops being the wall". Excalidraw style, white background, handwritten. || The per-worker register rectangle: sixteen grabs feed sixty-four multiply-adds — the whole kernel in one picture.]]

[[fig: A warm hand-drawn metaphor illustration titled "The multiplication table you fill from the edges". A big square grid like a school times-table, 8 across and 8 down. Down the LEFT edge, a hand-drawn column of 8 number cards labeled in blue "A values (held in one hand)". Along the TOP edge, a row of 8 number cards labeled in green "B values (held in the other hand)". Inside every cell, a small handwritten "×" and faint arrows showing each interior cell is just its row-card times its column-card, orange note in the middle "fill all 64 cells from just 16 edge numbers". A cook/worker figure at the corner holding the two stacks of cards. Dashed takeaway box: "16 numbers on the edges → 64 answers inside. that's the outer product." Excalidraw style, white background, charming, handwritten labels. || The outer product taught as filling a times-table: 8 edge numbers on each side generate all 64 interior answers.]]

### The code, same shape as before

Show it, and point out it's kernel 4 with a second inner loop:

```cpp
float threadResults[TM * TN] = {0.0f};   // 8×8 = 64 answers, in pockets
float regM[TM] = {0.0f};                 // 8 A values
float regN[TN] = {0.0f};                 // 8 B values

for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint i = 0; i < TM; ++i)        // grab the 8 A's ONCE
        regM[i] = As[(threadRow*TM + i)*BK + dotIdx];
    for (uint j = 0; j < TN; ++j)        // grab the 8 B's ONCE
        regN[j] = Bs[dotIdx*BN + threadCol*TN + j];

    for (uint i = 0; i < TM; ++i)        // the outer product: 64 FMAs
        for (uint j = 0; j < TN; ++j)
            threadResults[i*TN + j] += regM[i] * regN[j];
}
```

[[note: teach || Point at the structure: "The grabs are hoisted ABOVE the double loop — same trick as kernel 4, just done for both A and B. If those grabs happened inside the `i,j` loop, we'd do 64 grabs. By pulling them out into the two little arrays first, we collapse 64 grabs down to 16." That hoisting is the entire performance win, and it's the same lesson as before: the pockets only pay off if the grab sits in the right place.]]

## Why does this let you fit everything? A word on the pockets

A student will worry: "How can 64 answers plus 16 grabbed values all live in pockets at once?" Good question, and here's the honest answer. Each worker on an H100 gets up to 255 pockets (registers). Our 8×8 tile needs 64 + 8 + 8 = 80 of them, plus a few for bookkeeping. Comfortable.

[[note: confusion || The trap students fall into: "if bigger tiles are better, let's make it 16×16!" Stop them gently. A 16×16 tile needs 256 pockets *just for the answers* — over the 255 ceiling. When you run out of pockets, the extra values get shoved back out to slow memory ("spilling to local memory," which is really HBM wearing a disguise), and performance falls off a cliff. The lesson: reuse is powerful, but the register file is small. Bigger is better only up to the size of your pockets. This is why the tile sizes get *tuned* in a later kernel instead of guessed.]]

[[note: sn || A worker owning more answers means the block needs fewer workers to cover the same output. Kernel 5 uses a 128×128 output block with 8×8 tiles per worker, so it needs only (128×128)/(8×8) = 256 workers per block — and every one of them is busy, no idle threads. The arithmetic closes perfectly.]]

## The one number that ties it all together: arithmetic intensity

This is the concept to leave ringing in their ears, because it explains *why every rung of this whole workshop works*. There's a single number called **arithmetic intensity** — the amount of math you do per byte of memory you move. Math on top, bytes on the bottom. A fraction.

The math on top *never changes* for a matmul — it's always the same pile of multiply-adds. So the only way to raise the fraction is to shrink the bottom: move fewer bytes. And that is *exactly* what register tiling does. Every rung of the ladder is the same fraction with a smaller and smaller denominator.

[[note: production || This isn't a classroom exercise — it's the exact climb that runs inside every AI serving stack on Earth right now. When you chat with a model on H100 or B200 GPUs, the matrix multiplies underneath are register-tiled just like this, and libraries like cuBLAS and CUTLASS live or die on getting these tile sizes right. FlashAttention, the kernel behind fast long-context inference in vLLM and every major stack, is register tiling applied to attention — keeping the working data in registers and SRAM so it barely touches slow memory. DeepSeek's efficiency, the reason they serve a frontier model cheaply, comes largely from squeezing this same intensity number as high as it will go. The mentor can say with a straight face: "the idea on this whiteboard is what decides whether a model costs a dollar or a penny to run."]]

[[fig: A hand-drawn "intensity climb" figure titled "The ladder is one number rising". A vertical staircase of steps climbing left-to-right, each step a labeled rung with its percentage of the expert library in red: "1.3% naive", "8.5% coalesce", "12.8% shared memory", "36.5% column tiling (kernel 4)", "68.7% rectangle tiling (kernel 5)". The kernel 4 and kernel 5 steps are highlighted in orange with a bold callout "← this chapter: the biggest jump". A big blue arrow rising alongside the staircase labeled "arithmetic intensity = math ÷ bytes moved". A green sticky note pinned to the top math ÷ bytes with "math on top NEVER changes → we only shrink the bytes". Dashed takeaway box: "same multiply-adds every rung. we just feed them from closer, faster memory each time." Excalidraw style, white background, handwritten. || Read the ladder as one fraction climbing: the math on top is fixed forever, so each rung shrinks the bytes on the bottom.]]

## The board plan, in order

Give the mentor a concrete sequence so this delivers cleanly:

1. **Recap the wall.** "We got the box onto the counter — but we still reach for every multiply. The reaching is now the bottleneck." 2 minutes.
2. **The sandwich metaphor.** Grab a stack, build many. Draw the two cooks. Land "same food, less walking." 3 minutes.
3. **Kernel 4 by hand.** One column of 8. Show all 8 share the same B value. Count 9 grabs → 8 multiplies. Reveal the jaw-drop: 12.8% → 36.5%. 6 minutes.
4. **The loop-order trick.** Draw the two nested boxes, circle `tmpB`, show why it must be outside. 3 minutes.
5. **Kernel 5, the leap.** "Why not reuse in both directions?" Draw the 8×8 rectangle. Fill the times-table from the edges. Count 16 grabs → 64 multiplies. Reveal 68.7% — past halfway. 6 minutes.
6. **The pocket limit.** Answer "why not 16×16?" with the 255-register ceiling. 2 minutes.
7. **Tie it off with intensity.** One fraction, math fixed, bytes shrinking — and that's what runs in vLLM and FlashAttention today. 3 minutes.

[[note: demo || The one live demo: run kernel 3, kernel 4, and kernel 5 back-to-back and let the GFLOP/s number jump on screen — roughly 8,474 GFLOP/s for kernel 4, then about 15,972 GFLOP/s for kernel 5. If you can show the profiler too, point at the shared-memory stall bar shrinking between the runs: the "reaching" bottleneck visibly draining away. Numbers moving live beat any slide.]]

**Checkpoint questions** to fire at the room: "In kernel 4, why does one worker owning a *column* let us reuse a B value?" (Because a column shares the same B column.) "In kernel 5, how many multiplies do we get from 16 grabbed values?" (64 — the outer product.) "Did we change the amount of math between kernel 4 and 5?" (No — only the bytes moved.)

## You can now teach

- **Register tiling as the sandwich-shop trick**: grab a stack of ingredients into your pockets once, then build many answers from your hands instead of walking to the fridge per answer.
- **Kernel 4 (1D tiling)**: one worker owns a column of 8 outputs, reuses one B value across all of them, and jumps the kernel from 12.8% to **36.5%** — demonstrated by hand as 9 grabs → 8 multiplies.
- **The loop-order secret**: why the shared-memory grab must be hoisted *outside* the inner loop for reuse to happen at all.
- **Kernel 5 (2D tiling)**: one worker owns an 8×8 rectangle and fills it with an outer product — 16 grabs → 64 multiplies — carrying the kernel to **68.7%**, past the halfway mark and the biggest jump on the ladder.
- **The register-file limit**: why bigger tiles aren't free (the 255-pocket ceiling and spilling), and why tile sizes eventually get tuned.
- **Arithmetic intensity as the unifying number**: math-per-byte, with the math fixed and the bytes shrinking every rung — the exact climb that makes vLLM, FlashAttention, and DeepSeek efficient in production today.
