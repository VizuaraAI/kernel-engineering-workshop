By the end of this chapter you'll be able to stand at a whiteboard and teach why a four-bit number — a number so coarse it can barely count to six — became the fastest way to run AI on NVIDIA's newest chip, and why a real kernel went from 2000 microseconds to 22 microseconds using them. You need no chip-design background. You need one story, one metaphor about a shared price tag, and the honesty to admit the trick sounds impossible until you draw it.

This is a *frontier* chapter. It sits at the very top of the workshop's ladder. So don't rush it, and don't pretend it's simple. Instead, lean into the wonder: the whole point is that something which *shouldn't* work, works beautifully — once you see the hidden helper.

## Open with the hook: 2000 → 22

Start class with a single number on the board and nothing else.

[[note: say || "Someone entered a hackathon on NVIDIA's Blackwell chip. Their first working version of one AI operation took two thousand microseconds. Their final version took twenty-two. Same math. Same chip. Same answer coming out. Ninety times faster. By the end of today you'll understand *exactly* where those ninety times came from — and it is not where you'd guess."]]

[[note: teach || Write "2000 µs → 22.3 µs (90×)" huge, at the top, and leave it there the entire lecture. Every time you finish a section, walk back and cross off another chunk of that 90×. The lecture becomes a countdown. Students love watching the number fall.]]

[[fig: A warm hand-drawn illustration titled "The 90x climb". A staircase descending left to right, each step a hand-drawn box with a number on it and a tiny stick figure standing on it: "2000µs" (top, figure looking exhausted), then "443", then "39" (this step circled in orange with a star), then "27", then "22.9", then "22.3µs" at the bottom (figure celebrating with arms up). A red dashed vertical ruler on the left shrinking as the steps descend. A big orange handwritten banner across the top: "same math, same chip, 90x faster". A dashed takeaway box at the bottom: "where did the 90x come from? (spoiler: not the math)". Excalidraw style, white background, charming, handwritten labels. || The whole lecture is a countdown down this staircase.]]

## Plain words: what is a four-bit number, and why is it strange?

Every number a computer stores costs *bits*. More bits, more room, more precision — but also more weight to carry around. For years, AI chips have been putting numbers on a diet. They went from 32 bits, to 16, to 8. Each time you halve the bits, you can move numbers around twice as fast and pack twice as many onto the chip. Blackwell — NVIDIA's newest data-center GPU — takes the diet to its extreme: **four bits per number**. The format is called **NVFP4**.

Here's the strange part. Four bits can only make sixteen different patterns. So a single NVFP4 number can only be one of about **sixteen values**, and the biggest it can represent is roughly **6**. That's it. You cannot store a real AI weight — which might be 0.003 or 512 — in a number whose whole world stops at six.

[[note: metaphor || Imagine a ruler with only sixteen marks on it, and the last mark is "6". Someone hands you this ruler and says "measure everything in the building with it — the height of a doorknob, the length of the hallway, the thickness of a hair." Impossible. The ruler is too coarse and far too short. That is a lone four-bit number. On its own, it's useless. So how does Blackwell run the world's best AI models on it? That's the mystery we're about to solve.]]

## The secret: a shared price tag (microscaling)

Here is the entire trick, and it's beautiful once you see it. You don't store one number in four bits and expect it to be accurate. You store a *little group* of sixteen four-bit numbers **together**, and you give the whole group one **shared scale factor** — a single, more-precise multiplier that stretches all sixteen of them to the right size.

This is called **microscaling**. "Micro" because the scale is shared over a *small* block — just 16 values — not the whole giant tensor.

[[note: metaphor || The supermarket price tag. Imagine a shelf of sixteen apples. Instead of printing a full price on every apple ("$1.20", "$0.95", "$1.05"…) you print tiny relative numbers on each apple — "12", "9", "10" — and put ONE big sign over the whole shelf: "× $0.10". Now apple "12" means $1.20. The little number on each apple carries the *shape* (which apple is bigger); the shared shelf sign carries the *magnitude* (what ballpark we're in). Sixteen cheap tags plus one good sign gives you accurate prices — without printing a full price on every apple. That shared sign is the microscale.]]

In NVFP4 the "little number on each apple" is the four-bit value (its technical name is **e2m1** — 1 sign bit, 2 exponent bits, 1 mantissa bit). The "shelf sign" is an eight-bit number in a format called **FP8 e4m3**, shared across the block of 16. The real value you reconstruct is just:

```
real value = (4-bit value) × (shared 8-bit block scale)
```

[[fig: A hand-drawn "shared price tag" metaphor illustration titled "Microscaling = one sign over sixteen apples". A wooden shelf drawn with sixteen apples in a row, each apple with a small handwritten number on it (12, 9, 10, 7, 14, ...). Above the shelf, one big rectangular sign hanging by two strings, labeled in green "× 0.10 (the block scale)". A blue dashed arrow fans out from the sign down to all sixteen apples, labeled "shared by all 16". One apple pulled out to the side with an arrow: "apple '12' × 0.10 = $1.20 (real price)". A red bracket under the whole shelf labeled "1 block = 16 values". Dashed orange takeaway box: "cheap tag per item + one good shared sign = accurate, for almost no storage". Excalidraw style, white background, warm and friendly, handwritten. || The four-bit values are the cheap per-apple tags; the FP8 scale is the one shared shelf sign.]]

[[fig: A hand-drawn technical diagram titled "NVFP4 = e2m1 values × per-block e4m3 scale", the direct translation of the price-tag picture. A long horizontal strip of 16 small cells, each drawn as a 4-bit box hatched blue and labeled in red "e2m1 (s·ee·m)", the whole strip bracketed underneath with a red dimension arrow "16 elements = 1 block". To the right of the strip, a single fatter box hatched green labeled "FP8 e4m3 scale (8 bits)". A blue dashed arrow runs from the green scale box back across the whole strip with a note "one shared exponent for all 16". Below, a purple code line "real = fp4_val × fp8_scale". A yellow packaging note top-left: "16 × 4 bits + 1 × 8-bit scale = 9 bytes / 16 elems ≈ 4.5 bits each". Dashed orange takeaway box: "coarse values + fine shared scale ⇒ ~4.5 effective bits, dynamic range restored". Excalidraw style, white background, handwritten. || The same idea in the chip's own language: sixteen e2m1 values riding on one shared FP8 scale.]]

[[note: aha || Here's the number that makes it click. Sixteen values at 4 bits is 8 bytes, plus one 8-bit shared scale is 1 more byte — so **9 bytes carries 16 numbers**, about **4.5 bits per number**. Against the old BF16 format at 16 bits each, you move roughly **3.5× fewer bytes** — and moving bytes is the slow, expensive part of AI. The apples got 3.5× lighter to carry, and the shared sign kept them accurate. That's the deal.]]

## The tiny by-hand example

Do this on the board. It takes two minutes and it kills all the mystery.

[[note: example || A block of just four four-bit values (pretend the block is 4, not 16, so it fits on the board): the little tags are `[3, 1, 4, 2]`. The one shared block scale is `0.5`. To get the real numbers, multiply every tag by the shared scale: 3×0.5 = **1.5**, 1×0.5 = **0.5**, 4×0.5 = **2.0**, 2×0.5 = **1.0**. So four almost-free tags plus ONE good number reconstructed four real values. Now change the shared scale to `0.01` and redo it: 0.03, 0.01, 0.04, 0.02 — the *same tags* now describe tiny numbers. The tags say "which is bigger"; the shared scale says "how big we're talking." One knob rescales the whole block.]]

That is the whole numeric idea. Coarse shapes, one shared magnitude, multiply to reconstruct. Everything else in this chapter is about making that multiply *free*.

## Where the accumulator lives: Tensor Memory (TMEM)

Now the second Blackwell idea, and it's a plumbing idea, not a numbers idea. To teach it you first remind students what a GPU does with matrix multiplication: it multiplies pairs of numbers and *adds up the running totals*. Those running totals are called the **accumulator** — the scratchpad where partial sums pile up.

For four generations of GPUs, that scratchpad lived in the chip's **registers** — the tiny, ultra-fast slots next to each worker. But registers are a cramped shelf, and modern tensor cores are so fast and their tiles so big that the answers no longer fit. The register file *became the wall*: you could make the math faster, but you had nowhere to put the answers.

Blackwell's fix is blunt: **give the accumulator its own room.** A brand-new, dedicated memory space called **Tensor Memory**, or **TMEM** — 256 KB per SM, whose only job is to hold tensor-core operands and answers.

[[note: metaphor || The dedicated prep counter. Picture a tiny kitchen where the chef had to keep every half-finished dish balanced on the same small cutting board as the ingredients, the knives, and the recipe card. Everything fought for that one board. TMEM is management finally bolting a *second, separate counter* to the wall whose only job is to hold the dishes-in-progress. The knives and ingredients (the registers) get their whole board back; the half-finished sums get a counter of their own. Nobody's elbowing anybody anymore.]]

[[fig: A hand-drawn illustration titled "TMEM = a counter just for the half-finished dishes". A cramped kitchen scene. Left: a tiny overcrowded cutting board with knives, vegetables, a recipe card, AND wobbly stacked plates all piled on it, labeled in red "OLD (Hopper): accumulator crammed in with the registers — the wall". A frazzled chef. Right arrow to: the same kitchen but now with a NEW separate wall-mounted counter drawn in orange, holding only neat stacks of half-finished plates, labeled "NEW (Blackwell): TMEM — a counter just for the running sums". The original board now tidy, labeled green "registers, free again". Dashed takeaway box: "the answers moved out of the registers into their own room — 256 KB TMEM". Excalidraw style, white background, warm, handwritten. || Blackwell gives the accumulator its own dedicated counter so it stops fighting the registers for space.]]

[[note: confusion || Students will immediately ask "so can my code just read TMEM like a normal variable?" No — and this is the exact thing to nail down. TMEM is written *only* by the tensor core, and to look at the answer at all you must explicitly *copy it out* into registers with a special instruction. There's no `x = tmem[5]`. Draw it as a one-way counter: the tensor core puts dishes ON the counter; the waiters must physically carry them OFF to serve them. Say: "You don't read TMEM. You drain it." That single verb — *drain* — fixes the confusion.]]

[[note: sn || There's one more grown-up rule worth a sidenote: TMEM is a real, tiny pool you must *allocate* and *free*, like reserving counter space and giving it back. Forget to free it and the next kernel that needs the counter stalls. In practice you reserve it once at the start and release it once at the end — never inside the inner loop.]]

You can mention two more Blackwell moves lightly, without dwelling: the new tensor core (`tcgen05`) is fired by a *single thread*, and the biggest tiles are so wide that **two neighboring SMs pair up** to feed one multiplication. Both are the same theme — the math unit got so fast the whole chip is redesigned just to keep it fed. Name the theme and move on.

## Now the payoff: walking the 2000 → 22 staircase

Here's where you cash in the hook. The hackathon operation was a **GEMV** — a big matrix streamed past a small vector, one of the most *memory-bound* jobs there is. NVFP4 makes the bytes 3.5× lighter, so this *should* be fast. Walk the staircase and show why it wasn't — until it was.

[[note: demo || Walk the room down the staircase, crossing numbers off your banner as you go:
**2000 µs** — the naive version. It reconstructs every four-bit value *by hand*: mask the sign, shift the exponent, grab the mantissa, reassemble, multiply by the scale. Correct, but every apple is being price-checked with a dozen fiddly instructions.
**443 µs** — fix the access pattern first (coalesce the loads, let 32 threads share one row). Dull but mandatory plumbing. Already 4.5× faster.
**39 µs** — THE BIG ONE. Stop twiddling bits by hand. Blackwell has a *hardware instruction* that decodes two four-bit values straight to real numbers in one shot. Replace a dozen instructions with one. **11× faster** from code that got *shorter*.
**27 → 22.9 → 22.3 µs** — the polish: hand-schedule the multiply-add, and overlap the next chunk's loading with this chunk's math so the memory pipe never idles.]]

[[note: aha || The jaw-drop is this: the single biggest win — 11×, from 443 down to 39 — came from writing *less* code, not more. The whole kernel was memory-bound, and yet the giant speedup was about **reducing instructions**, because decoding four-bit numbers by hand was secretly costing about 10× too many instructions. The bytes were already light; the decode was the anchor. Say it out loud: "The format hands you the byte savings for free — but they're worthless until the *decode* is free. And 'free' means a hardware unit, not clever code."]]

[[fig: A hand-drawn two-column figure titled "The 11× step: bit-twiddling vs hardware decode". LEFT column header "BY HAND (naive)": a handwritten pseudo-assembly listing of ~10 lines — "mask sign", "shift exponent", "grab mantissa", "reassemble", "multiply scale", "...×16 per block" — the whole list bracketed in red with a note "≈10× the instructions". A tired stick figure checking price tags one by one. RIGHT column header "HARDWARE (Blackwell)": a single purple code box "__nv_cvt_fp4x2_to_halfraw2()" with a blue arrow into a green box labeled "hardware FP4→real unit" and out to two yellow cells labeled "two real numbers". A relaxed stick figure. A blue dashed arrow between the columns: "same answer, one instruction". Dashed orange takeaway box: "443µs → 39µs (11×): the decode was never free — the silicon makes it free". Excalidraw style, white background, handwritten. || The naive kernel reconstructed every four-bit value by hand; Blackwell has a conversion unit for exactly this, and using it is an 11× win.]]

[[fig: A hand-drawn pipeline-timeline figure titled "The last mile: hide the loading behind the math". Two horizontal lanes over a time arrow (red, → t). TOP lane "one chunk at a time": alternating boxes LOAD (green) then a red hatched STALL gap then MATH (yellow), repeating, red note "the workers wait for data". BOTTOM lane "two chunks overlapped": LOAD-chunk0 and LOAD-chunk1 drawn overlapping, then MATH-chunk0 and MATH-chunk1 back-to-back with no gap, blue note "next load already in flight". A purple aside: "tried 3–4 chunks → too crowded, slower". A small handwritten bar chart on the right: bars for 2000, 443, 39, 27, 22.9, 22.3 shrinking, last two nearly equal, orange label "90× total". Dashed takeaway box: "once decode is free, the whole game is keeping the pipe full". Excalidraw style, white background, handwritten. || The final polish is pure logistics: overlap the next chunk's loading with this chunk's math so the memory pipe never idles.]]

## The production link: this is running today

Make sure students know this isn't a lab toy.

[[note: production || Blackwell — the B200 and GB200 chips — is what the biggest labs are buying right now to serve models to millions of people. NVFP4 is how they run those models cheaply: four-bit weights mean you move a quarter of the bytes and fit far bigger models on each chip, and the microscale is what keeps the accuracy intact. DeepSeek's low-precision training, NVIDIA's own inference stacks, and every serious Blackwell serving pipeline lean on exactly these microscaled formats and this decode-in-hardware trick. When your students make that four-bit decode disappear into the silicon, they're touching the exact mechanism that decides whether a frontier model costs a dollar or a dime per million tokens.]]

## Teaching notes: the board plan

[[note: teach || Sequence for a 45-minute block. (1) 0–3 min: write "2000 → 22.3 µs (90×)" and read the hook line. (2) 3–10 min: the useless ruler — one four-bit number can't do the job. Let them feel the impossibility. (3) 10–20 min: the shared price tag — draw the apples and the shelf sign, then do the by-hand `[3,1,4,2] × 0.5` example, then rescale it to `× 0.01`. This is the heart; go slow. (4) 20–28 min: the 4.5-bits-per-element storage number (the "aha"). (5) 28–36 min: TMEM as the dedicated prep counter, and the word *drain*. (6) 36–45 min: walk the staircase, crossing off the banner, landing on "the 11× came from *less* code." End on the production line. Checkpoint questions along the way below.]]

[[note: confusion || The deepest confusion: students think NVFP4 is "just a smaller, worse number." Fix it by insisting on the pairing. Say: "NVFP4 is never one number. It's always sixteen cheap numbers *plus* one good shared scale. Judge the format on the block of sixteen, not on a lone value. The block, together, is accurate." Draw a circle around the whole block of sixteen apples and the sign, and tap the circle: "This whole thing is the number."]]

Checkpoint questions to toss out: "Why can't a single four-bit value store an AI weight?" (too coarse, tops out at 6). "What does the shared scale carry that the four bits don't?" (the magnitude / ballpark). "Where did the biggest speedup come from — moving fewer bytes, or fewer instructions?" (fewer instructions — the hardware decode). "What's the one word for reading a result out of TMEM?" (drain — you copy it out; you can't read it directly).

## You can now teach

- **Why a lone four-bit number is useless** — the sixteen-mark ruler that stops at six — and why that's not the whole story.
- **Microscaling** as a shared supermarket price tag: sixteen cheap per-apple tags plus one good shared shelf sign, with the by-hand `× scale` example that makes it concrete.
- The **4.5-bits-per-element** storage math and the ~3.5× byte savings that make NVFP4 fast on memory-bound work.
- **Tensor Memory (TMEM)** as a dedicated prep counter for the accumulator, and the crucial rule that you *drain* it rather than read it.
- The **2000 → 22.3 µs staircase**, and the punchline that the 11× win came from *less* code — a hardware decode, not clever bit-twiddling.
- The **production stakes**: this exact format and decode trick is how frontier models run cheaply on Blackwell today.
