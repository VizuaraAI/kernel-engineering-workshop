By the end of this chapter you can pick up any chapter in this handbook, learn its topic from zero in an afternoon, and walk into a room the next day and teach it well. This chapter is the one that teaches you how to use all the others. Read it once, slowly. Then keep it near you, because everything else in this book is built on the pattern you're about to learn.

Here's the promise of this whole handbook, said plainly. You are Dr. Raj and Shubham — the mentors. You are smart, but you may be starting fresh on GPUs and kernels. That's fine. That's the design. Each chapter takes one idea, hands it to you gently, and then hands you the tools to hand it onward to your students. Learn first. Teach second. Never skip the first.

## The one big idea: learn it, THEN teach it

There are two jobs hiding inside every chapter, and you must do them in order.

**Job one is to learn.** Read the chapter as a student would. Do the tiny number by hand. Draw the metaphor on paper. Don't move on until the idea feels obvious to *you*. If a sentence needs a re-read, that's a signal — sit with it until it clicks.

**Job two is to teach.** Only once the idea is yours do you flip into teaching mode: plan the board, pick the demo, rehearse the exact words. The chapter gives you both jobs, clearly labeled, so you never confuse "I sort of get it" with "I can stand up and deliver it."

[[note: metaphor || Think of a chef learning a new dish before service. First they cook it alone, at home, tasting as they go until they *own* it. Only then do they stand at the pass and call it out to the line. You never teach a dish you haven't cooked yourself. This handbook is written the same way: the first half of every chapter is you cooking at home; the second half is you calling it out to the line.]]

[[fig: A warm hand-drawn two-panel illustration titled "Learn it, then teach it". Left panel labeled "1. cook it at home" shows a single chef figure alone in a small kitchen tasting from a spoon, with a blue handwritten note "read, do the number by hand, draw the metaphor until it's yours". Right panel labeled "2. call it to the line" shows the same chef now standing at a pass calling out to a row of line cooks, with a green handwritten note "board plan, the demo, the exact words". A dashed arrow labeled "only after it clicks" leads from panel 1 to panel 2. A dashed takeaway box at the bottom reads "never teach a dish you haven't cooked yourself." Excalidraw style, white background, charming, handwritten labels. || The core workflow of this handbook: learn the idea alone until it's yours, then teach it to the room.]]

## The seven ingredients — the recipe every chapter follows

Every chapter is built from the same seven parts, always in roughly the same order. Once you see the pattern, you can read any chapter fast, because you know what's coming next. And when you eventually write or adapt a chapter yourself, this is your checklist.

Here are the seven, in plain words:

1. **Plain words first.** The idea explained as if to a smart friend who knows zero about GPUs. Short sentences. Any new word gets defined the moment it appears.
2. **A metaphor.** A picture from real life — a kitchen, a marching band, a post office, a highway. This is the part students remember a week later. The metaphor is the product.
3. **A tiny concrete number.** A 2×2 matrix, a "4 threads vs 32 threads" count — something small enough to do by hand on the board so the abstraction becomes arithmetic.
4. **The real math**, built up gently from that tiny number. Never dropped from the sky. Always grown from the small example the student just watched you do.
5. **In production, right now.** The line to where this lives today — vLLM, FlashAttention, DeepSeek, an H100 or B200 cluster. So you can say "this isn't academic; here's where it earns money."
6. **Teaching notes.** How to actually deliver it: what to draw, in what order, which one demo to run, and the number that makes jaws drop.
7. **The common confusion + the fix.** The exact spot students get lost, and the one sentence that unlocks them.

[[note: aha || Here's the trick that makes this handbook powerful: the seven ingredients are the same seven whether the topic is a dot product or a Blackwell tensor core. The *content* changes wildly across chapters; the *shape* never does. So once you've truly read three chapters, you can read the other thirty at double speed — your eyes already know where the metaphor lives, where the by-hand number lives, where the confusion-fix lives. The structure is a map you only have to learn once.]]

[[fig: A hand-drawn "recipe card" figure titled "The seven ingredients of every chapter", drawn as a tall recipe card with seven numbered rows stacked vertically. Each row has a small hand-drawn icon and a short label in a distinct color: (1) "plain words" in blue, (2) "a metaphor" in purple with a little lightbulb, (3) "a tiny number by hand" in green with a 2x2 grid icon, (4) "the real math" in blue with a little sigma, (5) "in production today" in orange with a tiny GPU chip, (6) "teaching notes" in green with a chalkboard, (7) "the common confusion + fix" in red with a warning triangle. A dashed takeaway box at the bottom reads "same seven ingredients, every chapter — the content changes, the recipe doesn't." Excalidraw style, white background, handwritten labels. || The fixed recipe behind every chapter: seven ingredients, always the same order, so you can read any chapter fast.]]

## The callout blocks — your teaching cheat sheet in the margins

As you read, you'll see colored callout cards scattered through the text. These are not decoration. Each one is a specific kind of help, tagged so you can find it at a glance. Learn the eight tags and you can skim a chapter and pull exactly what you need.

[[note: metaphor || Think of the callouts as colored sticky-tabs in a well-used cookbook. The blue tabs mark the pictures, the green tabs mark "say this out loud," the red tabs mark "here's where it goes wrong." When you're prepping tomorrow's class in a hurry, you don't re-read the whole chapter — you flip to the tabs. The callouts are pre-placed tabs, put there for you.]]

Here is what each tag means:

- **metaphor** 🧠 — the analogy, the picture you'll redraw on the board.
- **example** 🔢 — the tiny by-hand number, worked out step by step.
- **production** 🏭 — where this exact thing runs in the real world today.
- **teach** 🎓 — how to present it: what to draw, in what order, how to pace.
- **say** 🎤 — the *exact words* to speak at the board. Steal these verbatim.
- **demo** ▶️ — the one live thing to run in front of the room.
- **confusion** ⚠️ — the misunderstanding students hit, and the fix.
- **aha** ✨ — the moment or number that makes the whole thing click.

[[note: teach || When you prep a session, do a "tag pass" first. Read only the **say**, **demo**, **confusion**, and **aha** callouts in the chapter. In ten minutes those four give you your script, your live moment, your rescue line, and your emotional peak — the entire spine of a lecture block. Then go back and read the full chapter to fill in the muscle. Tags first for the skeleton, prose second for the flesh.]]

[[fig: A hand-drawn illustration titled "The eight callout tags", drawn as a legend of eight colored cards in two columns. Each card is a small rounded rectangle in its semantic color with its icon and name hand-lettered: a purple "metaphor" card with a brain, a green "example" card with numbers, an orange "production" card with a factory, a blue "teach" card with a mortarboard, a red "say" card with a microphone, a blue "demo" card with a play triangle, a red "confusion" card with a warning sign, a yellow "aha" card with a sparkle. To the right, a hand-drawn cookbook with colored sticky-tabs poking out matching those colors, labeled "skim the tabs when you prep". A dashed takeaway box reads "eight tags = your margin cheat sheet; flip to them under time pressure." Excalidraw style, white background, handwritten. || The eight callout tags, drawn as colored cookbook tabs you flip to when prepping.]]

## Figures come in two flavors — and often in pairs

You'll notice the figures aren't all the same. That's on purpose. There are two kinds, and you should learn to tell them apart, because you'll redraw them differently on the board.

The first kind is the **metaphor illustration** — a kitchen line, a cafeteria, a post office, a highway. Warm, friendly, a little charming. This is the picture that carries the *feeling* of the idea.

The second kind is the **technical diagram** — matrices drawn as hatched grids, arrows in specific colors, numbered steps, a dashed takeaway box. This carries the *mechanism*.

Often the book pairs them: a metaphor figure right next to its technical translation. That pairing is a teaching move you should copy. Draw the friendly picture first to hook the room, then draw the precise version right beside it and say "and here's the same thing in real terms."

[[note: say || When you move from the metaphor drawing to the technical one, use a bridge sentence out loud: "That was the story. Now watch me draw the exact same thing the way the machine sees it — same idea, real symbols." That one sentence tells students the two pictures are the same truth in two languages. Without it, some students think you changed topics.]]

[[sn: The technical figures use a fixed color grammar you'll see everywhere in this book: blue for mechanism, green for specs, red for dimensions and labels, purple for code, orange for emphasis, yellow for outputs and packaging. You don't have to memorize it to teach — but if you use the same colors consistently on your own whiteboard, students start reading your drawings faster.]]

[[fig: A hand-drawn "paired figures" illustration titled "Metaphor, then mechanism", drawn as two side-by-side panels connected by a bold arrow labeled "same idea, two languages". The left panel is a warm friendly cafeteria line of small cook figures with a purple label "the story (metaphor figure)". The right panel is a precise technical diagram of a chip floorplan with hatched green math blocks and a blue control sliver, labeled "the mechanism (technical figure)". A dashed connecting bracket underneath both reads "draw the friendly one first to hook them, then the exact one beside it". Excalidraw style, white background, handwritten labels, the left panel charming and the right panel crisp and diagrammatic. || The two figure flavors side by side: warm metaphor on the left, precise mechanism on the right — draw them in that order.]]

## How to prep a single session from a chapter

Here is the concrete workflow. Suppose tomorrow you're teaching one 50-minute block, and a chapter in this handbook covers it. This is what you do, start to finish.

**The evening before — learn it (about 45 minutes).** Read the chapter straight through as a student. Do every by-hand number yourself, on paper, with a pencil. If you can't reproduce the tiny example without peeking, you're not ready — re-read until you can. This is the non-negotiable step. You cannot teach what you can't do.

**Still the evening before — build the plan (about 20 minutes).** Do the tag pass. Pull the **teach** and **say** callouts into a one-page board plan: what you draw first, second, third. Circle the one **demo** you'll run live. Circle the one **aha** number — the jaw-dropper — and decide exactly when in the block you'll reveal it. Read the **confusion** callout twice and memorize the fix, because that moment *will* come.

**In the room — deliver it.** Follow the CURRICULUM's block timing: a 3-hour lecture is three 50-minute blocks with breaks. Inside a block, the rhythm is always the same — metaphor on the board, tiny number by hand, grow the real math, drop the production hook, run the one demo, land the aha, end with a checkpoint question. The chapter already ordered these for you. Trust the order.

[[note: demo || Rehearse your one live demo *the night before*, on the actual machine, end to end. The single most common way a class loses momentum is a demo that won't run — a missing package, a GPU that's busy, a typo in the benchmark. Run it once cold the night before and you remove the only thing that can genuinely derail your block. One demo per block, tested in advance. That's the rule.]]

[[note: aha || Every chapter hands you one number designed to make the room gasp — matmul on 1000×1000 matrices is a *billion* multiply-adds; a well-fed GPU versus a starved one is *nine-tenths* of a million-dollar cluster. Do not bury it in the middle of your explanation. Save it. Build to it. Say it slowly, then pause and let the silence do the work. The gasp is what students remember on the drive home, and it's what makes them believe the topic matters.]]

[[fig: A hand-drawn "session prep timeline" figure titled "Prepping one block from a chapter", drawn as a horizontal timeline with four labeled stations. Station 1 "evening: learn it (~45 min)" in blue with a little pencil-and-paper icon and note "do every number by hand". Station 2 "evening: build the plan (~20 min)" in green with a one-page board-plan icon and note "tag pass -> board order + demo + aha". Station 3 "night before: rehearse the demo" in orange with a small terminal icon and note "run it once cold". Station 4 "in the room: deliver" in purple with a chalkboard icon and note "metaphor -> number -> math -> production -> demo -> aha -> checkpoint". A dashed takeaway box reads "learn first, plan second, rehearse the demo, then teach the fixed rhythm." Excalidraw style, white background, handwritten labels. || The end-to-end prep workflow for one lecture block, from learning the night before to the in-room delivery rhythm.]]

## The rhythm inside a block

Zoom in on the in-room part, because it's the same shape every time and worth memorizing. A single teaching block moves through the seven ingredients in order: hook with the **metaphor**, ground it with the **tiny number by hand**, grow it into the **real math**, connect it to **production** so it feels real, run the **one demo**, land the **aha number**, and close with a **checkpoint question** to confirm the room is with you.

[[fig: A hand-drawn illustration titled "The rhythm inside every block", drawn as a curving path with seven numbered stops, like a small board-game track, each stop a labeled circle in its semantic color. Stop 1 "metaphor" (purple, lightbulb), stop 2 "tiny number by hand" (green, 2x2 grid), stop 3 "the real math" (blue, sigma), stop 4 "production today" (orange, GPU chip), stop 5 "one demo" (blue, play triangle), stop 6 "the aha number" (yellow, sparkle, drawn slightly bigger as the peak), stop 7 "checkpoint question" (red, question mark). The path climbs gently to the aha at stop 6 then ends at the checkpoint. A dashed takeaway box reads "same seven stops, every block — never end on your own voice, end on a question." Excalidraw style, white background, handwritten labels. || The fixed in-room rhythm of a teaching block: seven stops that climb to the aha number and close on a checkpoint question.]]

That checkpoint question matters. Never end a block on your own voice. End it on a question you throw back — "so, in one sentence, why must the inner dimensions match?" — and wait. If the answers come back clean, move on. If they're muddy, that's your signal to re-draw the metaphor before the break, not after.

[[note: confusion || The most common *mentor* mistake — not student mistake — is skipping straight to the real math because you, having now learned it, find the metaphor and the by-hand number too slow. Resist this completely. You already climbed the ladder last night; the students haven't. The tiny number feels obvious to you *precisely because* you did the slow version once. Give them the same climb you had. Never kick away the ladder you just used.]]

[[note: production || Keep one production fact per topic in your back pocket, current and specific — "FlashAttention was adopted across the whole industry within months because it fed the GPU better," or "the gap between a 40%-fed and an 85%-fed GPU is literally half the electricity bill." Each chapter hands you these. Say at least one out loud per block. It's the difference between students thinking they're doing homework and students thinking they're touching the thing that runs ChatGPT.]]

## A few honest reminders

Read the two exemplar chapters — *Matmul from scratch* and *CPU vs GPU* — before you write or teach anything else. They are the tuning fork. Everything in this handbook is pitched to their voice: warm, unhurried, metaphor-first, always tied to something running in production today.

And keep the hard rule in view: **keep it simple.** If your own explanation needs a re-read, split it. The whole design bet of this handbook is that if the mentor gets it, the students will too. So the moment *you* feel fuzzy, don't push past it — that fuzziness is exactly what your students would feel, magnified. Clear it up for yourself first, every time. That's the job.

## You can now teach

- The **learn-then-teach workflow**: do the by-hand number yourself before you ever plan the board — cook the dish at home before you call it to the line.
- The **seven ingredients** every chapter follows, in order, so you can read any chapter at double speed and know what's coming.
- The **eight callout tags** and how to do a "tag pass" to pull a lecture's skeleton — say, demo, confusion, aha — in ten minutes.
- The **two figure flavors** (warm metaphor vs. technical diagram), why they're often paired, and the bridge sentence that connects them.
- The concrete **session-prep timeline** — learn the night before, build the plan, rehearse the one demo cold, then deliver.
- The **in-room rhythm** — metaphor, number, math, production, demo, aha, checkpoint — and the mentor mistake of kicking away the ladder you just climbed.
