By the end of this chapter you'll be able to stand at a whiteboard and teach *why the whole craft of kernel engineering is one thing: keeping the data you need close by.* No electronics. No CUDA yet. Just a story about a workshop with a desk, some drawers, a shelf across the room, and a warehouse across town — and how far you have to walk to fetch a tool. Once a student feels those distances in their legs, every optimization we teach for the next four weeks becomes obvious. Let's build it.

## The one idea, in plain words

A GPU does not have "memory." It has *memories* — several separate places to keep numbers, each a different distance away from where the actual math happens. The closest ones are tiny but instant. The far ones are huge but slow. And here is the whole game: **the math units are so fast that they spend most of their time waiting for numbers to arrive.** So the job — the entire job — is to keep the numbers you're about to use as close as possible.

That's it. That's the chapter. Everything else is putting distances on it.

[[note: say || "The chip can multiply numbers almost infinitely fast. What it cannot do is *fetch* numbers fast. So a kernel engineer is not really a mathematician. A kernel engineer is a logistics person. The question is never 'how do I compute this?' — it's 'how do I keep the ingredients within arm's reach?'"]]

## The metaphor: your workshop and how far you walk

Picture yourself as a craftsperson building something at a workbench. When you need a tool, how long it takes to grab it depends entirely on *where it is*.

- On your **desk**, right under your hands: you grab it instantly. But your desk is tiny — a few tools fit, no more.
- In the **drawer** of your workbench: a step and a reach. A little slower, but the drawer holds a lot more than the desktop.
- On a **shelf across the room**: you get up and walk. Much slower. But the shelf is big — most of your tools live there.
- In the **warehouse across town**: you drive there and back. Painfully slow. But the warehouse is enormous — it holds *everything*, and it's the only place big enough to.

You'd never drive to the warehouse for a screwdriver you use every ten seconds. You'd walk it to your desk once and keep it there. That single instinct — *fetch the far thing once, keep it near* — is exactly what a good GPU kernel does.

[[note: metaphor || Four distances for one tool: **desk** (registers), **drawer** (shared memory / L1), **shelf across the room** (L2 cache), **warehouse across town** (HBM, the big global memory). Same tool, wildly different fetch times depending on where it sits. A brilliant worker who keeps sprinting to the warehouse gets nothing done. A mediocre worker with everything on the desk flies.]]

[[fig: A warm hand-drawn illustration titled "How far do you walk for a tool?". A single craftsperson at a workbench in the center-left. Four zones drawn at increasing distance, connected by a winding hand-drawn path that gets longer and longer. Zone 1, right at the hands: a small desktop labeled in orange "DESK — registers", green note "instant · holds almost nothing". Zone 2, a drawer in the bench, a short step away, labeled in yellow "DRAWER — shared memory", green note "a reach · holds a tile". Zone 3, a shelf across the room with a little figure walking toward it, labeled in blue "SHELF — L2 cache", green note "get up and walk · holds a lot". Zone 4, far right, a tiny warehouse building with a road and a delivery truck, labeled in blue "WAREHOUSE — HBM global memory", green note "drive across town · holds EVERYTHING". A big red arrow along the path labeled "farther = slower, but bigger". A dashed takeaway box at the bottom: "the craft = fetch from the warehouse ONCE, keep it on the desk". Excalidraw style, white background, charming, handwritten labels. || The core metaphor: four distances you walk for one tool. Close is fast but tiny; far is vast but slow.]]

## Put real numbers on the distances

Metaphors are for feeling; numbers are for believing. Here are the honest figures for an NVIDIA H100 — the GPU that's serving models right now. Don't make students memorize these. Make them *feel the ratios*.

- **Registers (the desk):** about 1 cycle to read. Effectively free. But an SM's whole register file is only ~256 KB, split across tens of thousands of threads — each thread gets a handful.
- **Shared memory (the drawer):** roughly 20–30 cycles. About 31 TB/s of bandwidth. Up to 228 KiB per SM. Big enough to hold a working tile.
- **L2 cache (the shelf):** around 50 MiB, shared by all the SMs on the chip.
- **HBM / global memory (the warehouse):** ~500 cycles of latency, 80 GB of capacity, 3.35 TB/s of bandwidth. It's the only place big enough to hold a real model's weights, and it's the slowest.

[[note: example || Do the ratio on the board and let it land. The desk is ~1 cycle. The warehouse is ~500 cycles. That's **500× farther** for the same number. Say it as time a human can feel: "if grabbing a tool off your desk takes 1 second, driving to the warehouse takes over 8 minutes." Now ask: "how many times would you drive to the warehouse before you decide to just keep the thing on your desk?" That question *is* kernel engineering.]]

[[note: aha || The jaw-drop line: **"The chip can do math about 300 times faster than the warehouse can deliver the numbers to do math on."** On an H100 the tensor cores can do ~989 trillion operations per second, but HBM only delivers 3.35 trillion bytes per second. So for most kernels, the math units sit *idle*, tapping their feet, waiting on the truck from the warehouse. We are not compute-starved. We are delivery-starved.]]

[[fig: A hand-drawn memory pyramid titled "The four distances, with numbers (H100)", four horizontal bands stacked narrow-top to wide-bottom. TOP band (orange fill, narrowest) "REGISTERS — the desk", green note "~1 cycle · ~256 KB/SM · per-thread". SECOND band (yellow hatch) "SHARED MEMORY — the drawer", green note "~20-30 cycles · 31 TB/s · up to 228 KiB/SM · per-block". THIRD band (blue hatch) "L2 CACHE — the shelf", green note "~50 MiB · shared by all SMs". BOTTOM band (widest, blue hatch) "HBM GLOBAL MEMORY — the warehouse", green note "~500 cycles · 80 GB · 3.35 TB/s · whole grid". A red arrow down the left side labeled "farther, slower, BIGGER". A red arrow up the right side labeled "closer, faster, tinier". Small orange sticky pointing at the top two bands: "these are on the chip". Small blue sticky at the bottom band: "this is off the chip, across the interposer". Dashed takeaway box: "500x range in fetch time — the engineer's job is to climb UP this pyramid and stay there". Excalidraw style, white background, handwritten. || The same picture as the workshop, now with numbers. A 500x span from desk to warehouse.]]

## Where these places actually are on the chip

You don't need physics to teach this, but one honest sentence makes it real for students who ask "but where *is* the warehouse?"

The warehouse — HBM — is a set of memory chips stacked into little towers, sitting *right next to* the main compute chip on one shared slab of silicon. "Across town" is a metaphor for *time*, not distance: physically it's millimeters away, but ~500 cycles slow because it's a separate chip you send a request to and wait for. The desk, drawer, and shelf (registers, shared memory, L2) all live *on the compute chip itself* — that's why they're fast. The moment your data leaves the compute chip to ask the warehouse, you pay the 500-cycle toll.

[[fig: A hand-drawn "on-chip vs off-chip" figure titled "Why crossing off the chip is the toll". A large rounded rectangle labeled in blue "THE COMPUTE CHIP (the die)" on the left. Inside it, three nested boxes drawn close together: a tiny orange box "registers (desk)", a yellow-hatch box "shared mem (drawer)", a blue-hatch box "L2 (shelf)", with a green note "all on-chip = fast, no toll". To the right, OUTSIDE the rectangle, a separate stack of little hatched tower-boxes labeled "HBM (warehouse)", connected by a single blue dashed arrow crossing a red dashed line labeled "chip boundary = ~500 cycle toll". Purple sticky near HBM: "'local memory' secretly lives out HERE too". Dashed takeaway box: "fast = stays on the chip; slow = has to leave and ask the warehouse". Excalidraw style, white background, handwritten. || The real reason for the 500-cycle toll: fast memories live on the compute die; the warehouse is a separate chip.]]

[[note: teach || When a student asks "why is HBM slow if it's only millimeters away?", don't dive into physics. Say: "Slow isn't about distance in centimeters, it's about crossing off the chip. Anything that lives *on* the compute die is fast. HBM lives on a *different* die and you have to send a request and wait for the answer to come back — like texting a warehouse versus opening your own drawer." That reframing settles it every time.]]

There's also a **trap** worth planting early, because it wastes students an entire afternoon later. There's a thing called "local memory" that *sounds* close and fast. It is a lie. "Local" describes who can see it (one thread), not where it lives. Physically, local memory sits out in the warehouse — the same slow HBM — and costs the full 500 cycles. You never ask for it; the compiler puts things there when a thread tries to keep too much stuff on its tiny desk and overflows. We'll name that spill later; for now just flag it.

[[note: confusion || "Local memory must be the fast, close one, right? It has 'local' in the name." No — and this is the single most common mystery-slowdown for beginners. **Local memory is not local. It lives in the warehouse (HBM) and is dead slow.** The fix line: "local' means private to one thread, not physically near it. When your desk overflows, the extra stuff gets shipped to the warehouse — that's local memory, and it's the slowest thing in your kernel."]]

[[fig: A hand-drawn "the trap" illustration titled "Local memory: the name is a lie". Center, a worker at a desk with too many tools — the desk is overflowing, tools spilling off the edge, a red note "desk (registers) FULL". A red dashed 'spill' arrow curves from the overflowing desk all the way to a distant warehouse box labeled "HBM", where the spilled tools land in a box labeled in orange "'LOCAL' memory (actually in the warehouse!)". A big cartoon signpost near the warehouse reads "LOCAL" with a hand-drawn arrow, but a second red scribbled note crosses it out: "NOT local — ~500 cycles!". Green note by the desk: "you never ask for this — the compiler ships your overflow here". Dashed takeaway box: "when the desk overflows, your 'local' variable is secretly the slowest memory on the chip". Excalidraw style, white background, charming, handwritten. || The trap drawn: an overflowing desk spills to the warehouse, and the warehouse box is mislabeled "local".]]

## A tiny by-hand example: the naive way vs. the close-by way

Let's make "keep it close" arithmetic, not vibes. Imagine a block of 4 workers all building parts of the same answer, and they all need the same small tile of numbers — say a row of 4 values from A and a column of 4 values from B.

**The naive way (everyone drives to the warehouse):** each of the 4 workers, independently, drives to the warehouse to fetch the row of A and the column of B. That's 4 workers × (4 + 4) values = **32 warehouse trips**, and most of them fetch the *exact same numbers* the person next to them just fetched.

**The close-by way (fetch once, share the drawer):** the 4 workers cooperate. Together they make *one* trip to the warehouse, drop the row of A and the column of B into the shared **drawer** (shared memory), and then everyone reads from the drawer. That's **8 warehouse trips instead of 32** — a 4× cut in the slow part — and every later read is a fast drawer-reach instead of a warehouse drive.

[[note: example || On the board: "4 workers, each grabbing 8 numbers from the warehouse = 32 slow trips. But they're grabbing the *same* numbers as each other! Fetch it *once* into the drawer, and it's 8 trips plus a bunch of instant drawer-reaches. Scale the block to 32 workers and each warehouse number gets reused ~32 times — you cut the slow traffic by 32×." Write "32 trips → 8 trips" big, circle it. That circle is the entire GEMM ladder in miniature.]]

That reuse number — how many times you use each warehouse byte before throwing it away — is the master dial of this whole workshop. Naive kernels use each byte once (drive, use, forget). Good kernels use it dozens of times. Same math, same answer.

[[fig: A hand-drawn two-panel figure titled "Fetch once, or fetch it 32 times?". LEFT panel labeled "(A) naive — everyone drives", showing 4 little worker figures each with their own separate road to a warehouse box labeled "HBM", 4 trucks on 4 roads, a big red note "4 workers x 8 numbers = 32 warehouse trips (all fetching the SAME numbers!)". RIGHT panel labeled "(B) tiled — fetch once, share the drawer", showing the same 4 workers but now ONE shared road to the warehouse bringing back one truck into a central drawer box (yellow hatch) labeled "shared memory tile", from which 4 short blue dashed arrows fan out to the 4 workers, green note "8 trips once, then instant drawer-reaches". A purple label between panels "this is TILING". Dashed takeaway box: "reuse each warehouse byte many times = cut the slow trips = the whole ladder". Excalidraw style, white background, handwritten. || The naive kernel makes everyone drive for the same box; the tiled kernel fetches once and shares the drawer.]]

## The real math: arithmetic intensity and the wall

Now name the thing gently. The ratio at the heart of it is **arithmetic intensity**: how much math you do per byte you fetch from the warehouse. Math is cheap; fetching is expensive. So you want lots of math per fetched byte.

The chip has a break-even point called the **ridge**. On an H100 it's about **295 flops per byte**: if your kernel does *more* than 295 units of math per byte it hauls from HBM, the math units are the bottleneck (good — they're the fast part). If it does *fewer*, you're **memory-bound** — stuck waiting on the warehouse truck, math units idle. Almost every naive kernel is wildly memory-bound: the naive matrix-multiply does barely ~1 flop per byte, roughly *300 times* below the ridge. That's why it reaches a humiliating 1.3% of what the tuned library does.

[[note: production || This isn't a classroom ratio — it's where the money is. When DeepSeek or Meta serve a model to millions of people, the GPUs cost hundreds of thousands of dollars each, and a badly-fed one runs at 10% of what you paid for while a well-fed one runs at 90%. The famous **FlashAttention** kernel got adopted across the entire industry in months for exactly one reason: it stopped shuttling the attention scores out to the warehouse and back, and kept them in the drawer. Same math, far fewer warehouse trips. That's it. That reuse instinct on this whiteboard is what the whole serving stack — vLLM, every H100 and B200 cluster — is built around.]]

[[fig: A hand-drawn roofline figure titled "The wall you're fighting", a simple log-log plot with hand-lettered axes. X-axis in red "math per byte fetched (arithmetic intensity)". Y-axis in red "actual speed (FLOP/s)". A rising blue diagonal line labeled "memory-bound: limited by the warehouse truck (3.35 TB/s)". A flat green horizontal ceiling labeled "compute roof: 989 TFLOP/s". Where they meet, a red vertical dashed line and a circled label "ridge ~ 295 flops/byte". A low orange dot far down-left labeled "naive matmul (~1 flop/byte) = 1.3% of the library". A high dot near the roof labeled green "well-tiled = 90%+". A thick blue dashed arrow curving from the low dot up-and-right toward the roof, labeled "the whole workshop = more reuse, fewer warehouse trips". Dashed takeaway box: "below the ridge you're waiting on the truck, not doing math". Excalidraw style, white background, handwritten. || The naive kernel lives far down the memory-bound slope; every optimization drags it right, toward the compute roof.]]

[[sn: The ridge slides *right* every GPU generation, because compute grows faster than bandwidth. The A100's ridge sat around 13 flops/byte; the H100's BF16 ridge is ~295. So each generation, more kernels are "automatically" memory-bound and the reuse game matters *more*, not less.]]

## Teaching notes: how to actually deliver this

Here's the order that lands, tested against the way students think.

1. **Open with legs, not chips.** Draw the workshop — desk, drawer, shelf, warehouse — before you ever say "register" or "HBM." Walk the room while you talk: stand at the "desk," take a step to the "drawer," walk across to the "shelf," mime driving to the "warehouse." Physical distance makes the latency real.
2. **Then hang the numbers on it.** Only after the picture lands do you write ~1 / ~30 / ~500 cycles on the four zones. The 500× ratio is your first jaw-drop.
3. **Do the 32-trips-vs-8-trips arithmetic by hand.** This is the demo. It converts "keep it close" from a slogan into a number they computed themselves.
4. **Name the wall last.** Arithmetic intensity and the ridge come *after* they already feel that fetching is the expensive part. Now the roofline is just a picture of something they already believe.
5. **Close with the money.** FlashAttention, vLLM, dollar-per-token. "This exact instinct is what the industry pays kernel engineers for."

[[note: demo || The one live demo: run the naive matrix-multiply kernel and the tiled one back to back on a real GPU and show the wall-clock times. The tiled one is often 5–10× faster on identical math. Then say the line: "I didn't change the math by a single operation. I only changed *how far the numbers had to travel.* That's the whole job." Nothing sells the chapter like the same answer arriving 8× sooner.]]

[[note: confusion || Second common trip-up: "so shared memory is just a faster version of global memory — the chip is being nice to me?" No. The drawer (shared memory) is a *scratchpad you fill by hand*, not an automatic cache. Nothing lands in it unless you, the programmer, walk to the warehouse and put it there. The fix line: "The drawer starts empty every time. The GPU won't stock it for you — *you* decide what tile to fetch and keep. That deciding is the skill." ]]

## The mental spine to leave them with

Say this at the end and let it sit: *A GPU is a genius mathematician chained to a slow delivery truck. It computes almost anything instantly, but only on numbers it already holds. So making it fast has almost nothing to do with math and almost everything to do with logistics — fetching each number from the warehouse as few times as possible and keeping it on the desk while you use it.* Every rung of the GEMM ladder, every trick with tiles and threads over the next four weeks, answers one question: **how do I keep the data close?**

## You can now teach

- The **memory hierarchy** as four walking distances — desk (registers), drawer (shared memory), shelf (L2), warehouse (HBM) — close-and-tiny vs. far-and-vast.
- The **honest numbers** for an H100 (~1 / ~30 / ~500 cycles; 80 GB at 3.35 TB/s) and the 500× ratio that makes the metaphor land.
- Why **local memory is a trap** — it sounds close but lives in the warehouse — and why students must never trust the name.
- The **fetch-once-reuse-many** idea by hand (32 trips → 8 trips), which is tiling in miniature and the master dial of the whole course.
- **Arithmetic intensity and the ridge**: why most naive kernels are memory-bound, waiting on the truck while the math units idle.
- The **production hook**: FlashAttention, vLLM, and dollar-per-token — that "keep the data close" is exactly what kernel engineers are paid to do.
