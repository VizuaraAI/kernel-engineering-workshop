# Mentor Handbook — Style Spec
"Teaching Kernel Engineering: A Mentor's Handbook" — the teach-the-teacher companion.
Audience: the MENTOR (Raj Dandekar + Shubham Panchal) who must first LEARN each idea from zero,
then STAND UP and TEACH it. Every chapter must leave the mentor able to deliver that piece of the
workshop confidently. Different from the reference Book (worklog voice): this is warm, simple,
metaphor-first, and full of teaching scaffolding.

## The seven ingredients — EVERY concept gets all seven
1. **Plain words first.** Explain it like you're talking to a smart friend who knows zero GPU. Short
   sentences. No jargon until it's earned; when you introduce a term, define it in the same breath.
2. **A metaphor.** A vivid real-world picture the mentor can redraw on a whiteboard. Metaphors are the
   product here — kitchens, factories, marching bands, post offices, libraries, highways, orchestras.
3. **A tiny concrete number.** A 2×2 or 3×3 matrix, a "4 vs 32 threads" count — something you can do by
   hand on the board so the abstraction becomes arithmetic.
4. **The real math**, built up gently from the tiny example — never dropped from the sky.
5. **In production, right now.** Always tie it to something live: "this is exactly what runs when you
   chat with Llama / DeepSeek / ChatGPT", vLLM, FlashAttention in every serving stack, H100/B200 clusters,
   DeepSeek's DSpark, NVFP4 on Blackwell. The mentor must be able to say "and this is not academic —
   here's where it's earning money today."
6. **Teaching notes.** How to actually deliver it: the board drawing, the order to reveal things, the
   one demo to run, the number that makes jaws drop.
7. **Common confusion + the fix.** The exact place students get lost, and the sentence that unlocks them.

## Callout blocks — use liberally (custom markdown, the build renders them as colored cards)
Syntax on its own line: `[[note: TYPE || the content]]`  where TYPE is one of:
- `metaphor`   🧠  the analogy/picture
- `example`    🔢  the tiny by-hand number
- `production` 🏭  where this runs in the real world today
- `teach`      🎓  how to present it (board work, sequence, pacing)
- `say`        🎤  the exact line to say out loud at the board
- `demo`       ▶️  the live demo/code to run
- `confusion`  ⚠️  the common student misunderstanding + the fix
- `aha`        ✨  the moment/number that makes it click

## Voice
- Second person to the mentor ("You'll want to draw this first…", "Don't say 'coalescing' yet —").
- Kind, unhurried, confidence-building. Assume the mentor is smart but starting fresh on this topic.
- It is OK — encouraged — to over-explain. Expansive beats terse. The mentor re-reads until it's theirs.
- Every chapter ends with a short **"You can now teach:"** checklist (3–5 bullets) of what the mentor
  can confidently deliver after this chapter.

## Structure of a chapter
Open with the ONE sentence goal ("By the end of this you can teach why a GPU is fast without hand-waving").
Then: plain intro → metaphor → tiny example → real math → production link → teaching notes/board plan →
common confusions → "You can now teach:" checklist. Weave 6–9 figures throughout. 1500–3000 words.

## Figures — 6–9 per chapter, TWO kinds (this is a heavily-illustrated book)
Same `[[fig: <scene> || <caption>]]` syntax. Two flavors, mix them:
1. **Metaphor illustrations** — draw the analogy itself: a kitchen line, a marching band, a warehouse
   with shelves, a post office sorting mail, a highway with cars. Warm hand-drawn Excalidraw style,
   friendly, labeled in handwriting, a little charming. These make the book feel human.
2. **Technical diagrams** — the same semantic-color Excalidraw grammar as the main Book (blue=mechanism,
   green=specs, red=dims/labels, purple=code, orange=emphasis, yellow=packaging/output tiles, hatched
   matrices, numbered circles, dashed takeaway box).
   Often pair them: a metaphor figure AND its technical translation side by side.
Every figure prompt must be detailed and self-contained (drawable without the text), on a white
background, hand-lettered labels, no photorealism.

## Hard rule
Keep it SIMPLE. If a sentence needs a re-read, split it. The mentor learning from zero is the whole point —
if the mentor gets it, the students will too. Expansive + simple + visual + always-tied-to-production.
