By the end of this chapter you can teach *any* hard idea in this workshop the way the best teachers do — from zero, with a metaphor, and with a tiny number on the board — because you'll have a repeatable recipe for turning something scary into something obvious. This is the one chapter that isn't about GPUs. It's about *you*, standing at the front of the room, making a beginner feel smart.

Everything else in this handbook is a topic. This chapter is the *method*. Read it first, and every other chapter will read like an example of it.

## The whole idea in one sentence

You only truly understand something when you can teach it to a beginner.

Not when you can *use* it, nod along to a paper, or pass the quiz. You understand it when a person who knew nothing walks away *also* understanding it. That's the bar. It's high, and it's the kindest thing about teaching: preparing to teach is what forces the gaps in your own understanding out into the light.

[[note: metaphor || Think of understanding like knowing a city. You can *use* a city with GPS — turn left when the phone says turn left, and you arrive. But if a lost tourist asks you for directions and you can only say "follow your GPS," you don't actually know the city. You know it when you can draw the map on a napkin from memory: "the river's here, the market's two blocks up, cut through the alley." Teaching is drawing the napkin map. If you can't draw it, you were just following GPS.]]

[[fig: A warm hand-drawn two-panel illustration titled "Using vs Understanding". Left panel labeled "using it": a small figure walking while staring down at a phone showing a blue GPS arrow, with a thought bubble that is blank except for a single arrow "turn left". Right panel labeled "understanding it": the same figure standing confidently, drawing a little city map on a paper napkin for a second figure labeled "the lost tourist (a beginner)", the napkin showing a river, a market, and a dashed shortcut alley, all hand-labeled. A dashed takeaway box spanning both: "you understand it when you can draw the map from memory for someone lost." Excalidraw style, white background, charming, handwritten labels. || The bar for real understanding: not following the GPS, but drawing the map for a beginner.]]

Say this before every block you prepare: *if I can't get a beginner there, I don't have it yet — and that's information, not failure.* When you sit down to teach the roofline model or online softmax and feel a soft fog where the explanation should be, that fog is a gift. It points at the exact spot you skated over. Chase it down — that chase is where your own mastery gets built.

## The recipe: seven ingredients, every single time

This handbook runs on a fixed recipe — the one *you* will run in your head at the whiteboard. Every hard concept gets all seven, in roughly this order:

1. **Plain words** — say it like you're talking to a smart friend who knows zero GPU.
2. **A metaphor** — a real-world picture they can redraw: a kitchen, a marching band, a post office.
3. **A tiny number** — a 2×2 matrix, "4 threads vs 32," something you finish by hand on the board.
4. **The real math** — built up gently *from that tiny number*, never dropped from the sky.
5. **In production today** — where this exact thing runs and earns money right now.
6. **Teaching notes** — the board plan, the reveal order, the one demo, the jaw-drop number.
7. **The common confusion** — the exact place students get lost and the sentence that frees them.

[[note: teach || Don't memorize the seven as a list to recite. Memorize them as a *ladder you climb*. You always start on rung 1 (plain words) and you never skip to rung 4 (math) without stepping on rungs 2 and 3 first. Ninety percent of bad teaching is someone standing on rung 4 shouting math down at people still standing on the ground. The metaphor and the tiny number are the rungs that carry them up.]]

[[fig: A hand-drawn ladder figure titled "The seven-ingredient ladder". A tall friendly ladder leaning against a wall, with seven rungs, each labeled in handwriting from bottom to top: rung 1 "plain words" (blue), rung 2 "a metaphor" (green), rung 3 "a tiny number by hand" (green), rung 4 "the real math" (blue), rung 5 "in production today" (orange), rung 6 "how to teach it" (purple), rung 7 "the common confusion" (red). A small figure labeled "the beginner" is climbing, currently standing on rung 3. To the side, a second figure standing on the top rung 4 shouting downward is crossed out in red with a note "don't start here!". A dashed takeaway box: "always climb from the bottom; the metaphor and the tiny number are the rungs that carry them." Excalidraw style, white background, handwritten labels. || The recipe as a ladder: start at plain words, and never leap to the math without stepping on the metaphor and the number.]]

The exemplar chapters in this handbook are *literally* this recipe. The matmul chapter starts with a shopping receipt (metaphor), does a 2×2 by hand (number), *then* shows the three nested loops (math). The CPU-vs-GPU chapter starts with a fine-dining kitchen versus a cafeteria before the word "parallelism" is ever said. That ordering isn't decoration — it's the load-bearing structure. Copy it.

[[fig: A hand-drawn "before and after" figure titled "Same idea, two orders". Left panel labeled in red "BAD: math first" — a whiteboard crammed with dense formulas and a row of small student figures with confused squiggles over their heads and one leaving. Right panel labeled in green "GOOD: recipe order" — a sequence of four little frames left to right: a kitchen sketch, then a small 2x2 grid, then a clean formula, then a dollar sign, with the same students now leaning in and nodding. A blue arrow labels the good sequence "metaphor -> number -> math -> money". A dashed takeaway box: "the topic is identical; only the order changed — and the order is the whole lesson." Excalidraw style, white background, charming, handwritten labels. || The same hard idea taught two ways: math-first loses the room; metaphor-then-number-then-math-then-money keeps it. The order is the lesson.]]

## Building from zero: the discipline of forgetting

The hardest part of teaching an expert topic is that *you are not a beginner anymore*, and you've forgotten what it was like to not know. Experts skip steps unconsciously — the steps became so automatic they stopped being visible. This is called the **curse of knowledge**, and it is the single biggest reason smart people teach badly.

[[note: metaphor || You've climbed a staircase so many times you now take it two steps at a time without looking. A first-time visitor needs every single step, and needs the light on. Building from zero means going back and turning the light on over every step you now skip — even the ones that feel insultingly obvious to you. The step that's obvious to you is exactly the one the beginner falls on.]]

The fix is mechanical, and you can do it every time. Before you teach a thing, write down the *chain of tiny prerequisites* it secretly depends on, and don't stop until every link is something a bright fifteen-year-old already knows. Then teach the chain from that end.

[[note: example || Take "why decode is memory-bound" — a real L8 idea that sounds terrifying. Walk the chain backwards: to get it, they need "arithmetic intensity" → which needs "FLOPs vs bytes moved" → which needs "the GPU can do math faster than it can fetch data" → which needs "math and memory are two different speeds" → which is just "the cooks are faster than the hallway that feeds them" from the CPU-vs-GPU chapter. That last link is something everyone already feels. *Start there.* Now you have a staircase from a kitchen to a frontier serving optimization, and no step is more than one small step up.]]

[[fig: A hand-drawn "prerequisite staircase" figure titled "Build the chain back to something everyone knows". A staircase of five steps climbing left to right. Bottom step (green, widest, labeled "everyone already knows this"): "the cooks are faster than the hallway (kitchen metaphor)". Step 2: "math speed vs memory speed are different". Step 3: "FLOPs vs bytes moved". Step 4: "arithmetic intensity". Top step (orange, labeled "the scary target"): "why decode is memory-bound". A small figure walks up from the bottom. A red dashed arrow shows an expert trying to jump straight from the ground to the top step and stumbling, annotated "the curse of knowledge: skipping steps you can't see". Dashed takeaway box: "list the hidden prerequisites; teach from the one everyone already owns." Excalidraw style, white background, handwritten labels. || Beat the curse of knowledge by walking the chain of prerequisites back to something the room already knows, then climbing up one small step at a time.]]

## Metaphors are the product, not the packaging

New mentors think the metaphor is a cute wrapper around the "real" content. It's the reverse. For a beginner, **the metaphor *is* the understanding**, and the math is the wrapper that makes it precise later. A student who leaves with "the GPU is a cafeteria line and the hard part is feeding the cooks" understands more than a student who can recite the HBM bandwidth number but has no picture to hang it on.

A good metaphor has three properties, and it's worth checking each one out loud when you invent one:

- It's **concrete and everyday** — kitchens, mail, highways, warehouses, marching bands. Not another abstraction.
- It **maps piece-for-piece** onto the real thing, so you can extend it. Cooks → arithmetic units. Hallway → memory bandwidth. Rice → data. Idle cooks → wasted GPU.
- It **breaks honestly** at some point, and you say where. Every metaphor lies a little; naming the lie is what keeps it trustworthy.

[[note: confusion || The danger with metaphors is *over-driving* them — stretching the kitchen until it says something false about the GPU. The fix is to say the break out loud: "the cafeteria metaphor is perfect for *why* the GPU is fast, but a real GPU's cooks work in tight groups of 32 that must all do the same step — that part the cafeteria doesn't capture, so here's a new picture for it." Naming where a metaphor ends is not a weakness. It's the thing that makes students trust the metaphors that *do* hold.]]

[[note: aha || Here's the reframe that changes how you prepare: your job at the whiteboard is not to *transfer information*. Textbooks transfer information and students bounce off them. Your job is to *hand over a picture* — a thing they can see with their eyes closed on the drive home. If, a week later, a student can redraw your cafeteria or your shopping receipt from memory, you have taught them more than a perfectly-worded paragraph ever could. Metaphors are what survive the drive home.]]

## Turning abstraction into board arithmetic

The metaphor gets them believing. The **tiny by-hand number** makes them *own* it. There's a magic in doing an operation slowly, with your hand, on the board, that no slide can replace: students watch abstraction become arithmetic, live, and realize *they could have done that themselves*.

The rule: pick the smallest example that still shows the pattern. A 2×2 matmul, not a 512×512. "4 threads vs 32," not "tens of thousands." Small enough to finish in ninety seconds, big enough that the shape of the idea is visible.

[[note: say || As you do the by-hand number, narrate every single step in plain words and *slow down at the boring parts*: "one times five is five, write it here. Two times seven is fourteen, write it under it. Add them: nineteen. That number goes in this box. That's it — that's one cell. Now watch me do it three more times and we're done." The slowness is the teaching. When you go slow on the trivial arithmetic, students realize the scary operation *is only* trivial arithmetic, repeated. That realization is the whole point.]]

[[fig: A warm hand-drawn illustration titled "Abstraction becomes arithmetic on the board". Left side: a big intimidating cloud labeled in red "MATRIX MULTIPLICATION" with scary swirls and a nervous small student figure looking up at it. A bold blue arrow labeled "do the tiny number by hand" points right. Right side: a clean whiteboard showing a small 2x2 times 2x2 worked out cell by cell in friendly handwriting, "(1x5)+(2x7)=19" circled in orange, and the same student now smiling with a thought bubble "oh — that's all it is?". A dashed takeaway box: "the smallest by-hand example turns a scary abstraction into 'oh, I could do that.'" Excalidraw style, white background, charming, handwritten labels. || The by-hand number is the moment a scary abstraction collapses into arithmetic the student realizes they could have done themselves.]]

Then — and only then — you generalize. Point at the 2×2 and say "now imagine this is 1000×1000." The math you write next isn't dropped from the sky; it's the *same thing they just watched*, with letters where the numbers were. That's "build the math up gently": the formula feels like a re-description of the example, never a new object.

## Always tie it to money and machines running today

A beginner's quiet question is always "why should I care?" Answer it before they ask, by tying every idea to something alive and expensive *right now*. This is the rung that turns a lesson into a reason to lean forward.

[[note: production || You have this ammunition in every chapter. The three nested loops of matmul are what runs when someone chats with DeepSeek or Llama on a rack of H100s — and NVIDIA became one of the most valuable companies on Earth by building the best chip for that loop. FlashAttention, which your students build by hand in W1, was adopted by the entire industry within months because it fed the GPU's cooks better than anything before it. The gap between a GPU kept 40% fed and one kept 85% fed is, directly, half the electricity bill. When a mentor can say "and this exact thing is where the money is won or lost today," the room stops taking notes to pass a test and starts learning to build a career.]]

[[note: aha || The jaw-drop number is a teaching tool, not a flourish — plan one per block. "A 1000×1000 matmul is a *billion* multiply-adds." "DeepSeek-V3 went from 4% to 37% of expert kernel performance with 100 samples, and to 72% with feedback." "That hackathon kernel went from 2000μs to 22.3μs — a 90× speedup on one operation." Write the number big, pause, and let the room feel it. The number is what they'll quote to a friend, and quoting it is how the lesson leaves the room.]]

## Putting it together: how to prep any block

Here's the concrete routine to run before you teach anything in this workshop — a lecture block or a whole workshop hour. It's the seven ingredients turned into a checklist you actually do the night before.

[[note: teach || **The night-before routine (about 30 minutes per block):** (1) Write the *one sentence* a student should be able to say afterward — if you can't, you're not ready to teach it. (2) Walk the prerequisite chain back to something everyone already knows; that's your starting point. (3) Pick or invent the metaphor, and name where it breaks. (4) Choose the smallest by-hand number that shows the pattern; actually work it once yourself on paper. (5) Line up the *one* live demo and the *one* jaw-drop number for the block. (6) Write down the single most likely place students get confused and the one sentence that fixes it. Six index cards. That's a taught block.]]

[[fig: A hand-drawn "six index cards" figure titled "How to prep any block". Six overlapping index cards fanned out, each hand-labeled: card 1 (blue) "the one sentence they leave with", card 2 (green) "the prerequisite chain -> where I start", card 3 (green) "the metaphor (+ where it breaks)", card 4 (purple) "the smallest by-hand number", card 5 (orange) "one live demo + one jaw-drop number", card 6 (red) "the likely confusion + the fix". Below the fan, a small confident mentor figure at a whiteboard. A dashed takeaway box: "six cards = one block. If you can't fill card 1, you're not ready to teach it yet." Excalidraw style, white background, charming, handwritten labels. || The night-before routine as six index cards — the seven ingredients turned into a checklist you actually run.]]

Run this on the sabotaged kernels of L7, the online-softmax build in L8, or the roofline model in L1, and the topic that felt un-teachable becomes six calm cards. The topics here are genuinely hard. The *method* for teaching them is not — it's a ladder, a chain, a picture, a number, a reason, and a fix, in that order, every time.

[[sn: The routine also protects you from the most common mentor failure mode: over-preparing the math and under-preparing the entry. Beginners rarely get lost in the math itself; they get lost at the *first step*, before you've handed them a picture. Spend your prep budget on rungs 1–3, not rung 4.]]

[[sn: If two mentors are co-teaching (as Raj and Shubham will), split by rung, not by topic: one owns the metaphor and the by-hand number, the other owns the math and the production link. Handing the chalk back and forth at the seam keeps energy up and models for students that even experts hand off.]]

## You can now teach

- **The core belief** — you only understand something when you can teach a beginner — and how to treat your own fog as a map to the gaps in your understanding.
- **The seven-ingredient ladder** and why you must climb it from plain words, never leaping straight to the math.
- **Building from zero** by beating the curse of knowledge: walk the prerequisite chain back to something the room already knows, then climb one small step at a time.
- **Metaphors as the product** — concrete, piece-for-piece, and honest about where they break — because the picture is what survives the drive home.
- **Turning abstraction into board arithmetic** with the smallest by-hand number, narrated slowly, then generalized as a re-description rather than a new object.
- **The night-before routine**: six index cards that turn any hard block in this workshop into a calm, teachable plan — with one live demo and one jaw-drop number every time.
