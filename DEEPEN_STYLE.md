# Deepening Spec — make every Book article DEEP, SIMPLE, and VISUAL
Reference blogs (match this feel): siboehm.com/articles/22/data-parallel-training and
siboehm.com/articles/22/CUDA-MMM — deep, thorough, first-principles, heavily illustrated, yet
so clearly written a beginner keeps up. The reader must finish thinking: **"wow, I finally
understand this, the visuals made it easy."** Our current articles are too short and too terse.
This spec REWRITES each one to be 2–3× longer and far deeper WITHOUT becoming harder.

## The five commitments (every article, no exceptions)
1. **Start from the foundations.** Do not assume the reader already knows the prerequisite. In the
   first few paragraphs, establish the minimal background from scratch (or recap it in one tight
   paragraph and link the sibling article). State, explicitly, the QUESTION this article answers.
   A student who is new to the topic must be able to start here and keep up.
2. **Socratic method.** Repeatedly pose the natural question a curious reader is already asking —
   *"But why does this actually help? Let's think about what the hardware is really doing when 32
   threads ask for memory at once…"* — then answer it by reasoning from basics, not by asserting.
   Question the obvious. When a result is surprising, STOP and explain why it's surprising and then
   why it's true. Anticipate the exact place the reader gets confused and address it out loud.
3. **Build gradually, one idea at a time.** Simple first. Each section builds on the previous. Never
   drop a formula or a number from the sky — derive it from a tiny concrete example the reader can
   follow by hand. Introduce a **central mental model early** (the article's "pebble graph") and
   reuse it throughout so everything hangs on one picture.
4. **Stay simple while going deep.** Short sentences. Conversational. Honest caveats ("this won't be
   exactly equal, and here's why that's fine"). Occasional first person ("when I first profiled
   this, I expected X — I was wrong"). Depth comes from explaining *why*, not from jargon density.
   If a sentence needs a re-read, split it.
5. **Ground everything in numbers and production.** Do the napkin math out loud (bytes, FLOPs,
   cycles, %). And keep tying it to what runs in the real world right now (vLLM, FlashAttention,
   DeepSeek, H100/B200 fleets) so it never feels academic.

## Length & shape
- Target **3,500–6,000 words** (the current versions are ~2,500 and feel thin — go much deeper).
- Keep the worklog rhythm for kernel articles (hypothesis → code → profile → **bold %/× number** →
  bridge) but EXPAND the "why" at every step — a paragraph of reasoning where there was a sentence.
- Keep the article's correct facts, numbers, code, and cross-links — DEEPEN, never lose or contradict
  them. Add more cross-links. Do NOT add an H1 title (the site adds it); start with prose; use ## / ###.

## Figures — go from ~3 to **6–10 per article**, one roughly every 2–3 sections
Use the same `[[fig: <detailed scene> || <caption>]]` syntax and the semantic-color Excalidraw grammar
(blue=mechanism, green=specs, red=dims/labels, purple=code, orange=emphasis, yellow=packaging/output;
hatched matrices; numbered circles; dashed takeaway box). Crucially, VARY the figure types — do not
draw only the one final technical diagram. Include:
- an **intuition figure** early (the mental model / the analogy, drawn simply),
- **before/after side-by-side** comparisons (the naive way vs the good way — this is the single most
  powerful "aha" device; use it wherever there's an optimization),
- a **timeline/pipeline** figure where anything overlaps or streams,
- a **zoom-in** figure (whole picture → one thread/one tile, with by-hand numbers),
- the precise **technical diagram(s)** as before.
Every figure prompt must be detailed and self-contained (drawable without reading the text), white
background, hand-lettered labels, no photorealism.

## Sidenotes — use more (aim 5–8)
`[[sn: …]]` for the "one exception", the exact-number correction, the compiler/hardware nuance, the
honest caveat. They keep the main line clean while rewarding the careful reader.

## Hard rule
Deeper AND simpler at the same time. The measure of success: a motivated beginner reads it top to
bottom, never gets lost, and understands the concept better than from any other single source —
because you started from zero, questioned everything, drew the picture, and did the math.
