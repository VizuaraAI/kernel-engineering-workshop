By the end of this chapter you'll be able to stand at a whiteboard and teach the finale of the whole workshop: how a real frontier AI lab — DeepSeek — writes and *gives away* its own GPU kernels, and how a trick called speculative decoding makes a giant model answer faster. This is the payoff chapter. Everything students learned about feeding the cooks and fusing launches comes together here, in code that is serving millions of people right now.

Don't be scared of the big names — FlashMLA, DeepGEMM, DSpark. By the end they'll feel like old friends, built from the same three tricks the students already know.

## The one-sentence frame to open with

Say this first, and let it hang in the air:

[[note: say || "For most of this course we've been building kernels for our own tiny examples. Today I'm going to show you a real AI lab that builds kernels for its own real model — and then puts them on the internet for free. When you understand *why* they do that, you'll understand what a kernel engineer is actually for."]]

A frontier lab like DeepSeek sells a *model*, not a compiler. So why spend engineer-years writing GPU kernels? Because their model has a weird *shape*, and off-the-shelf tools (like NVIDIA's `cuBLAS`) aren't tuned for weird shapes. When your car is a weird shape, no factory sells parts that fit — you machine your own. That's the whole chapter in one metaphor.

## What DSpark actually is (keep it simple)

DSpark is DeepSeek's inference-optimized model. Two numbers are all students need:

- It has **1.6 trillion** total parameters — an enormous brain.
- But it only uses **49 billion** of them to answer any single word — a small, fast slice.

[[note: metaphor || Think of a giant hospital with **1.6 trillion** doctors on staff, but any one patient only sees **49 billion** of them — the handful of specialists their case actually needs. You get the wisdom of the whole hospital while only paying for the few doctors in the room. That's a "Mixture of Experts" — MoE — and it's why a model this huge can still be affordable to run.]]

[[fig: A warm hand-drawn illustration titled "Mixture of Experts = a huge hospital, a few doctors". A large friendly hospital building drawn on the left, its windows densely filled with tiny doctor figures, a green handwritten banner across it "1.6 TRILLION doctors on staff". On the right, a small cozy exam room with just a few doctor figures gathered around one patient on a bed, an orange handwritten note "only ~49 BILLION seen per patient". A blue dashed arrow from the hospital to the room labeled "a router picks the right specialists". A dashed takeaway box at the bottom reads "huge brain, small bill: total params buy knowledge, active params set the cost". Excalidraw style, white background, charming, hand-lettered labels. || Mixture of Experts taught as a hospital: enormous staff, but each patient only sees the few specialists they need.]]

## Why answering is the slow, painful part

Remind students of the cafeteria idea. Generating text happens **one word at a time**. To produce a single word, the GPU has to drag a mountain of numbers out of far-away memory, do a trickle of math, and emit one word. Then do it all again for the next word.

[[note: aha || Here's the number that reframes it: producing one word barely uses the GPU's math muscles at all. The chip can do nearly a thousand trillion math operations a second, but during word-by-word generation it sits mostly idle, *waiting for numbers to arrive from memory.* The cooks aren't tired — they're starving. Say it plainly: **"the machine isn't limited by how fast it can think, but by how fast you can feed it."**]]

So the entire game of fast AI serving is: move fewer numbers, and stop making the GPU wait between tiny jobs. Hold that thought — every trick below is one of those two things.

## The big idea: speculative decoding (guess ahead, then check)

Here's the star of the show. Instead of grinding out one word per expensive step, what if we could produce *several*?

[[note: metaphor || Picture a slow, careful **editor** and a fast, cheap **intern**. Normally the editor writes every word himself — slow. Instead, the intern quickly scribbles a guess at the next **seven words**. Then the editor reads all seven *at once* — one glance — and says "yes, yes, yes, yes... no." He keeps the words that match what he would have written, throws away the rest from the first mistake, and writes the next one himself. Most guesses are right, so the editor does far less writing. That's speculative decoding: **guess cheap, check in one glance, keep the good prefix.**]]

Why is "check all seven at once" so much cheaper than writing seven? Because checking seven words you already have in hand has no waiting-in-line — you can look at all of them in parallel. In GPU terms, writing one word is a skinny, memory-starved job; checking seven is a fat, math-friendly job that lights up the tensor cores. **You turned a starving job into a well-fed one.**

[[note: example || Do it by hand on the board. The intern drafts 7 tokens. The editor verifies and the first mismatch is at position 5. So you *keep* tokens 1–4, plus the editor writes the correct token at position 5 as a "bonus." That's **5 real words** produced from **one** expensive editor-pass instead of five. Write it big: "accept n → get n+1 words per pass." That's the whole win.]]

[[fig: A warm hand-drawn comic-strip illustration titled "Speculative decoding: the intern and the editor". Three panels left to right. Panel 1 labeled "DRAFT": a small hurried intern figure at a desk scribbling a row of 7 numbered word-cards very fast, a blue note "cheap & quick — guesses 7 words". Panel 2 labeled "CHECK": a calm editor figure with glasses reading all 7 cards in a single glance (draw one big eye-beam sweeping all 7 at once), a green note "reads all 7 AT ONCE — one pass". Panel 3 labeled "KEEP": the same 7 cards, the first 4 stamped with a green check, card 5 replaced by the editor's own card, cards 6-7 crumpled and tossed in a bin, an orange note "keep the matching prefix + 1 bonus word". A dashed takeaway box: "guess cheap, check in one glance, keep the good ones". Excalidraw style, white background, friendly, hand-lettered. || The core metaphor: a cheap intern guesses several words, a careful editor checks them all in one glance and keeps the run that matches.]]

Now show the technical translation of the same picture, so they can connect the cartoon to the real pipeline.

[[fig: A hand-drawn Excalidraw pipeline on white, fine black ink, hand-lettered labels. Title "Speculative decode: one big pass, many words". Three stacked lanes. Top lane "DRAFT (intern)": a small blue-hatch box "draft head" with a curved dashed arrow to a chain of 7 tiny numbered circles left to right, blue note "sequential, greedy, cheap". Middle lane "VERIFY (editor)": one wide green-hatch box "TARGET model — checks all 7 at once" spanning under the tokens, green note "GEMM shape → tensor cores happy, ONE launch". Bottom lane "ACCEPT": the 7 tokens redrawn, first 4 filled green (kept), last 3 struck through in red, orange note "keep matching prefix + 1 bonus". A red double-arrow under the kept run labeled "≈ n+1 words per pass". Dashed takeaway box: "skinny memory-bound job → fat math-friendly job". Excalidraw style, white background. || The technical translation: draft is sequential and cheap, verify is one batched pass, accept keeps the matching prefix plus one bonus token.]]

## The three stages, in the mentor's own words

You'll teach speculative decoding in three beats. Here they are, each with the confusion to watch for.

**Beat 1 — Drafting.** The intern (a small "draft head" bolted onto the model) writes seven guesses, one after another. This *must* be sequential — word 2 depends on word 1. And each guess is a tiny job, so the danger is the GPU spending all its time *starting* jobs rather than doing them.

[[note: confusion || A student will ask "if checking is parallel, why can't drafting be too?" The fix: because each guessed word depends on the one before it — you can't guess word 2 until you've committed to word 1. Drafting is a chain; verifying is a checklist. Chains must be sequential; checklists can be done all at once. That asymmetry is the entire reason the trick works.]]

The fix for tiny-job overhead is one students already know: **fuse the launches.** DeepSeek records the whole seven-step draft as a single "CUDA graph" — one pre-recorded batch of work you press play on — so you pay the start-up cost once, not seven times. And the drafts are "greedy" (just pick the single most likely word, no fancy dice-rolling), which means no extra sampling kernels cluttering the hot path.

**Beat 2 — Verifying.** Hand the full model all seven guesses and ask, in one pass: "at each spot, what word would *you* have picked?" Because you already hold all seven inputs, this runs in parallel — the fat, tensor-core-friendly job.

[[note: confusion || The sneaky bug here is the attention mask. Each guessed word is only allowed to "see" the real text plus the guesses *before* it — never the ones after. If you let a later guess leak backward, the check silently passes garbage and the model's output drifts away from what it should say. Tell students: "the verify step must pretend it hasn't seen the future." It's the kind of bug that's invisible until your outputs are subtly wrong.]]

**Beat 3 — Accepting.** Walk the seven left to right, keep every word that matches, stop at the first mismatch, and add one bonus word. The beautiful guarantee: the final text is *exactly* what the model would have written on its own — same quality, just faster. You're not cutting corners; you're skipping redundant work.

[[note: production || This is live today. DSpark's real serving recipe sets `num_speculative_tokens: 7` with greedy drafting, exactly this pipeline. When acceptance stays high, the effective words-per-pass climbs to several, and decode latency drops by roughly that same factor. Speculative decoding is one of the biggest reasons chatbots feel snappy instead of sluggish.]]

## Why the check is cheap enough to be worth it (the two kernels)

Speculation only wins if the editor's check-pass is genuinely cheap. That's where DeepSeek's two homemade kernels come in — and each fixes one half of the model's forward pass.

### FlashMLA — attention that moves fewer bytes

Attention is the part of the model that lets each word look back at all previous words. To do that, the model keeps a "memory" of every past word, called the **KV cache**. Long conversation → giant cache → tons of bytes to drag around on every step. That cache is the single biggest thing being moved from memory during decode.

[[note: metaphor || Normal attention keeps a full, detailed **file folder** for every past word — thick, heavy, slow to carry. DeepSeek's trick (MLA) keeps only a **tiny index card** per word and reconstructs the full detail *on the spot, on the chip*, only when needed. Carrying index cards instead of folders means about **one-tenth** the weight to haul from memory. And on the chip, unfolding a card back into detail is basically free — chip-local bandwidth is enormous.]]

FlashMLA is the kernel that pulls this off: it grabs the scattered little index cards, unfolds them into full detail *inside* the chip's fast scratchpad (never writing the bulky version back to slow memory), and runs the attention math — all fused into one smooth pass so the GPU never stalls waiting.

[[note: aha || Here's the jaw-dropper: a KV cache at **one-tenth** the size means **one-tenth the bytes** to read on every single verify pass. Since decode is memory-bound — waiting on bytes — a 10× smaller cache finishes roughly **10× sooner**. That's the whole reason speculation nets out positive for DSpark: the check-pass is cheap because the cache is tiny.]]

[[fig: A hand-drawn split illustration titled "FlashMLA: index cards, not file folders". Left half labeled "normal attention": a stressed figure hauling a towering, teetering stack of thick labeled file folders down a long narrow hallway (memory), a red note "full KV per word — heavy to carry every step". Right half labeled "MLA (FlashMLA)": a relaxed figure carrying a small neat box of tiny index cards, a green note "~10% the size", with a little on-chip workbench where one card is being "unfolded" into a full sheet, a blue note "reconstruct detail ON the chip, where bandwidth is free". A dashed divider between halves. Dashed takeaway box spanning both: "move the compressed card from far memory; rebuild the detail on-chip → ~10x fewer bytes". Excalidraw style, white background, charming, hand-lettered. || FlashMLA taught as index cards versus file folders: carry the tiny compressed version, rebuild the full detail on-chip.]]

### DeepGEMM — running the experts in one cheap batch

The other half is the experts (the MoE part). Each word picks a few specialists, and each specialist is a big matrix multiply. Doing them naively means firing dozens of tiny separate jobs — the exact overhead trap.

[[note: metaphor || Imagine a post office where each parcel needs a different specialist clerk. The dumb way: line up at clerk 1, then clerk 2, then clerk 3 — a dozen separate queues, each with its own start-up shuffle. The smart way (a "grouped GEMM"): sort all parcels by clerk first, then process every clerk's pile in **one organized sweep**. Same work, but you paid the setup cost *once* instead of a dozen times, and no clerk ever sits idle.]]

DeepGEMM does two things at once: it **groups** all the experts into a single launch (the post-office sweep), and it runs the numbers in **FP8/FP4** — tiny 8-bit and 4-bit numbers instead of chunky ones. Fewer bits per number means fewer bytes to move — a near-linear speed-up in a memory-bound world. The clever bit is *scaling*: each little tile of the matrix gets its own scale factor so the tiny numbers don't lose accuracy, while the additions still happen in full precision. Half the bytes, and the model stays just as smart.

[[sn: DeepGEMM's other quiet flex: it's only a few hundred lines of readable CUDA. Production GEMM libraries are famously impenetrable walls of template code. DeepSeek's is small enough to teach from — the honest, legible answer to "how do you actually do FP8 matrix-multiply with block scaling." A gift to *learners*, not just to servers.]]

[[fig: A hand-drawn Excalidraw two-panel figure titled "DeepGEMM: one organized sweep, not a dozen queues". Panel A labeled "naive: one job per expert" — several small separate matrix-multiply boxes stacked, each with its own red "launch" tag and a little grey sleeping-SM figure beside it, orange warning "many tiny launches → overhead-bound". Panel B labeled "grouped: one launch" — a single wide blue-hatch input matrix whose rows are sorted into colored bands, each band arrowing to its expert's weight tile in a stacked green-hatch block, all feeding one pale-yellow output tile, purple note "sort tokens by expert → one masked launch". Green spec note "FP8/FP4 weights, full-precision add, per-tile scale". Numbered circles (1) sort by expert (2) one launch (3) tiny-number matmul. Dashed takeaway box: "half the bytes (FP8) + one launch instead of N → kills the overhead tax". Excalidraw style, white background, hand-lettered. || DeepGEMM packs all the experts into a single grouped, low-precision matrix multiply instead of a swarm of tiny launches.]]

## How it all clicks together on real hardware

Zoom out to the full serving picture — this is the "and this is where it earns money" slide. DSpark's reference setup runs on a **single node of four GB300 GPUs**. The serving framework (vLLM) owns the outer loop — scheduling, batching, recording the CUDA graph — and hands the two hottest inner jobs to DeepSeek's kernels: attention to FlashMLA, experts to DeepGEMM.

[[note: teach || Draw this as a layered stack, top to bottom: (1) vLLM engine at the top — "the manager who schedules everyone." (2) The paged memory below it. (3) A split middle layer: FlashMLA on the left, DeepGEMM on the right. (4) The four-GPU hardware at the bottom. Then trace one request with your finger: comes in the top, attention handled by FlashMLA, experts by DeepGEMM, word comes out. Say: "the framework schedules; the lab's kernels do the two hard inner loops."]]

The one honest wrinkle to mention: speculation makes batches *ragged* — every sequence keeps a different number of words each step, so the work is lumpy. The grouped mega-MoE kernel and the pre-recorded CUDA graph are exactly what keep the GPUs busy through that lumpiness instead of stalling.

[[fig: A hand-drawn Excalidraw layered stack titled "Where the kernels live in a real server". Four horizontal layers top to bottom. Top layer (black outline) "vLLM engine — scheduler · batching · CUDA graph", red margin note "framework owns the loop". Second layer (green) "paged KV memory (block-size 256)". Third layer split by a dashed vertical line: left blue box "Attention → FlashMLA", right yellow box "Experts → DeepGEMM (grouped, FP8/FP4)", red margin note "lab owns the hot kernels". Bottom layer (green) a wide box "4× GB300 node" with green specs "FP4/FP8 tensor cores · fast memory". Numbered red circles trace a request: (1) request in at top, (2) attention via FlashMLA, (3) experts via DeepGEMM, (4) word out. Dashed takeaway box: "the framework schedules; the open kernels do the two hot inner loops". Excalidraw style, white background, hand-lettered. || The full serving stack: vLLM manages the loop and delegates attention and experts to DeepSeek's two open kernels running on four GB300 GPUs.]]

## The lesson to send them home with

Tie the bow. Everything in this final chapter is the same three levers from day one, just aimed at a real model:

- **Move fewer bytes** — the 10%-size KV cache (FlashMLA), FP8/FP4 numbers (DeepGEMM).
- **Fuse the launches** — greedy drafting, one acceptance step, the whole cycle recorded as one CUDA graph.
- **Keep the cooks fed** — grouped expert GEMMs so no GPU sits idle on ragged work.

[[note: production || Leave them with this: DeepSeek didn't just publish a model. They published their *homework* — the actual kernels that make it fast — because for a model this weird, the kernel **is** the product. Everything your students learned on the GEMM ladder is exactly the vocabulary needed to read FlashMLA and DeepGEMM today, and to write the next one tomorrow. That's not academic. That's the job.]]

## You can now teach

- **Speculative decoding** as the intern-and-editor story — guess several words cheap, check them all in one glance, keep the matching prefix plus a bonus — and why it turns a starving job into a well-fed one.
- **Mixture of Experts (DSpark)** as a huge hospital with a few doctors per patient: 1.6T total parameters, 49B active, huge brain and small bill.
- **FlashMLA** as index cards instead of file folders — a ~10× smaller KV cache rebuilt on-chip — and why that makes the verify pass cheap enough for speculation to pay off.
- **DeepGEMM** as one organized post-office sweep instead of a dozen queues — grouped, low-precision expert matmuls that halve the bytes and pay launch cost once.
- The **full serving picture**: vLLM owns the loop; the lab's two open kernels do the hot inner work on a four-GPU node.
- The **closing lesson**: why a frontier lab writes and open-sources its own kernels — for a weird-shaped model, the kernel *is* the product, and it's built from the exact three levers the whole workshop taught.
