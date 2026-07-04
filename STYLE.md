# Kernel Engineering — Canonical Style Spec
Extracted from hamzaelshafie.bearblog.dev "Worklog: Optimising GEMM on NVIDIA H100" (Raj-approved reference)
+ the Modal GPU Glossary (site shell) + Raj's attached figure screenshots.
EVERY article and EVERY figure must follow this spec. Agents: do not improvise outside it.

---

## PART A — Article page layout (the "bearblog Tufte" look)

- **Two-surface site**: the site SHELL (home, section indexes, sidebar tree) is Modal-glossary
  dark terminal green. ARTICLE pages are the opposite: warm white paper, serif, calm.
  The contrast is intentional: terminal outside, notebook inside.
- **Article page**:
  - Background: warm off-white `#FDFCF9`. Text `#1a1a1a`.
  - Main column ~680px, centered-left, leaving a ~280px right gutter for sidenotes.
  - Body font: serif — EB Garamond / Crimson Pro (Google Fonts), 19–20px, line-height 1.75.
    Paragraphs justified-ish (text-align: left is fine; the feel comes from measure + leading).
  - Headings: same serif, bold, sentence case. H2 for "Kernel N: Name" sections.
  - **Inline code**: monospace (JetBrains Mono / Menlo), colored **crimson red `#b3382c`**,
    slightly smaller than body, no background pill. This red inline-code is a signature detail.
  - Code blocks: light grey card `#f6f4ef`, thin border, mono 14px, language-tagged.
  - **Sidenotes (the citations on the right)**: numbered superscripts in **red** in the body;
    the note text sits in the right gutter aligned with its reference, font ~13.5px,
    color `#555`, numbered `5.` in red. On screens <1100px they collapse to tap-to-expand
    footnotes at paragraph end. Sidenotes carry: hardware caveats, compilation internals,
    non-obvious exceptions ("registers are private to each thread, with one exception…"),
    exact-number corrections ("those 228 KiB are not exact…").
  - Figures: full-width of main column (may bleed slightly wider, up to ~900px), white
    background, thin light border or none, generous vertical margin. Caption optional,
    small grey serif italic.
  - Benchmark numbers live IN the prose in bold ("**4.2 TFLOP/s**, about **8.2%** of cuBLAS"),
    not in standalone tables (tables allowed only for final kernel-by-kernel summary).

## PART B — Writing voice (the worklog voice)

1. **First person, honest, incremental.** "I start with the most basic kernel…",
   "This initially puzzled me because…". Admitting confusion is part of the style.
2. **Every optimization section follows the loop:**
   hypothesis (why this change) → code (after the concept, never before) →
   profile (ncu / SASS evidence) → number (% of cuBLAS, × speedup, bold) →
   bridge ("To go further, we need to…").
3. **Terms**: bold on first mention with expansion — "**Streaming Multiprocessors** (SMs)" —
   then plain. Registers/variables always inline-mono: `threadIdx.x`, `float4`, `sharedA`.
4. **Math explained narratively**, with napkin arithmetic shown: "(64 × 64) / (32 × 8 × 4 × 4) = 1".
5. **SASS/PTX as evidence**, not decoration: "inspecting the SASS… every loop iteration
   generates a separate instruction issue" → then a figure showing the listing.
6. **Figures are directed**: prose says "Below is a visualisation of…", figure follows
   immediately, prose then re-references it. Never orphan a figure.
7. Article length: 1,500–3,500 words, 3–6 figures, 2–6 sidenotes.
8. Opening pattern: stakes in one line ("Matrix multiplication sits at the core of modern
   deep learning") → what we'll build → honest scope note.

## PART C — Figure grammar (the hand-drawn Excalidraw style)

Rendered via the parallel Gemini pipeline (gemini-3-pro-image-preview). Base style prompt:

> Hand-drawn technical diagram in Excalidraw style, drawn with a fine black ink pen on a
> pure white background. Slightly wobbly hand-drawn rounded rectangles, thin strokes,
> hand-lettered handwriting-style text for ALL labels and annotations (like the Virgil font).
> Colored annotations in a strict palette: BLUE handwritten notes for mechanisms and data
> movement, GREEN for hardware specs/bandwidth numbers, RED for dimensions and matrix labels
> (A, B, C) and warnings, PURPLE for code snippets and advanced configuration tips, ORANGE
> for emphasis callouts. Long thin dashed arrows with a slight curve connect each margin
> annotation to the exact component it describes. Matrices drawn as rectangles with diagonal
> hatch fills (blue hatch for A/sharedA, green for B/sharedB, pale yellow hatch for output C
> and warp tiles). Dimension arrows with numbers (↔ 128). Hand-drawn numbered circles
> (1)(2)(3) marking the reading order. Constants/parameters listed top-left in handwriting.
> Optional panels labeled (A), (B). Key takeaways inside a dashed rounded box. Flat, no
> shadows, no gradients, no photorealism, no typeset fonts. Wide composition, generous
> white space.

Semantic color rules (keep consistent across ALL articles):
- **black** = structure, titles, main labels
- **blue** = how data moves / mechanism explanations / matrix-A things
- **green** = specs, sizes, bandwidths / matrix-B things
- **red** = dimensions, matrix letters, iteration markers (i = 0), warnings
- **purple** = code lines (`constexpr uint …`), config tips (`cudaDeviceSetLimit`)
- **orange** = "look here" emphasis ("8 GPCs", "we split the tile…")
- **yellow fill** = physical/packaging elements (solder balls) and output/warp tiles

Recurring figure archetypes (reuse these compositions):
1. **Architecture map** — nested boxes die→GPC→SM with margin annotations on both sides.
2. **Tiling walkthrough** — 2–3 numbered panels: matrices with a highlighted tile, arrows
   showing the split, per-thread zoom with napkin math in purple.
3. **Memory pyramid / stacked packaging** — layered boxes with per-layer specs in green.
4. **SASS listing + diagram** — handwritten assembly column on the left, memory diagram on
   the right, dashed takeaway box bottom-right.
5. **Timeline/pipeline** — stages as boxes with overlap shading (for async/double buffering).

Pipeline notes: bulk parallel direct-Gemini calls (ThreadPoolExecutor, max_workers≈14,
idempotent skip on existing files, retry+backoff), AQ Gemini key from [[paperbanana-config]].
NO text-free constraint here — unlike other Vizuara pipelines, these figures REQUIRE
handwritten labels; accept ~10% regen rate for garbled text, regenerate failures.

## PART D — Site shell (Modal glossary look)

- Near-black green `#0b1a10`; text phosphor `#78f09a` / dim `#3f8f58`; accent lime `#a4f77f`.
- Monospace everything (Berkeley Mono vibe → use "IBM Plex Mono"/"JetBrains Mono").
- Left sidebar: full article tree, collapsible sections with `−`/`+`, active item = filled
  light-green row with dark text; term chips (`GPC`, `PTX`, `SASS`) as small outlined tags.
- Content links as `→ Article Title` arrow-lists. Top bar: logo · "Kernel Engineering" ·
  theme toggle (Terminal / Light) · search (⌘K) · Enroll button.
- ASCII-art hero on the landing page (GPU die drawn in ASCII), README → arrow like Modal.
- Footer: © Vizuara AI Labs · vizuara.ai · team@vizuara.com.
