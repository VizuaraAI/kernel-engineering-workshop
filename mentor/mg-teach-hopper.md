By the end of this chapter you'll be able to stand at a whiteboard and teach *why the H100 has a special copy machine and a special matmul instruction* — and why the fastest attention kernel in the world, FlashAttention-3, cannot exist without both. You do not need to have written a single line of CUDA to teach this well. You need two good metaphors, one honest number, and the patience to reveal the pieces in the right order. Let's build it.

## Where we are in the story

By now your students have climbed a ladder of matmul kernels. They can chant the lesson from the CPU-vs-GPU chapter: *a GPU is almost never limited by how fast it can do math — it's limited by how fast it can be fed data.* Every optimization so far has been a better way to feed the cooks.

This chapter is about the H100 (codename **Hopper**), the chip that took feeding the cooks so seriously that NVIDIA built *two brand-new pieces of hardware* just for it. One is a dedicated delivery truck that moves data so the cooks don't have to. The other makes a hundred cooks do one enormous chunk of math on a single command. Their names are **TMA** and **WGMMA**. Scary letters, simple ideas.

[[note: say || "Everything you've learned so far, we did with ordinary workers doing double duty — they cooked AND they ran to the pantry. Today we meet the chip that finally hires a delivery driver and a head chef who commands the whole line at once. Same kitchen. Better division of labor. That's the entire chapter."]]

## The old way: every cook is also a delivery boy

Remember how a tiled matmul loads data. A block of threads — a crew of cooks — needs a tile of matrix `A` and a tile of `B` sitting in fast **shared memory** (the counter beside the stove) before it can multiply anything. On the older chips, *the cooks themselves* had to go get it. Each of the 128 or 256 cooks computes an address, walks to the far pantry, grabs a few crumbs, walks back, and only *then* cooks.

[[note: metaphor || Picture a busy kitchen where every cook has to keep dropping their knife, run to the warehouse across the street, carry back a handful of onions, and only then chop. Hundreds of trips, all the cooks doing it, all of them wasting their skilled hands on hauling boxes. It works — but it's a shameful use of trained chefs. That's a pre-Hopper tile load.]]

There's a second, uglier problem. The head chef (the tensor core, the real matmul unit) is fussy about *how* the ingredients sit on the counter. It wants them in a special shuffled arrangement — a **swizzle** — so it can grab from every part of the counter at once without two hands colliding. On the old way, the cooks did that shuffling *by hand*, element by element, every load. It's finicky code that stays quietly wrong for a week before anyone notices.

[[fig: A two-panel warm hand-drawn kitchen comparison titled "Who fetches the ingredients?". LEFT panel labeled "(A) THE OLD WAY" in orange: a busy kitchen with 8 small cook figures, each drawn mid-run carrying a tiny box, a long winding path leading to a distant warehouse labeled in green "GMEM pantry (far away)"; each cook has a thought bubble "compute address + carry crumbs + shuffle by hand"; a red scrawl over the chaos "hundreds of trips — trained chefs hauling boxes". RIGHT panel labeled "(B) THE TMA WAY" in orange: the same 8 cooks now standing calmly at their stations with little "chopping" motion lines, while ONE cook circled in orange (labeled "elected leader") hands a small clipboard to a friendly green delivery-truck figure labeled "TMA engine"; the truck drives one big pallet of ingredients from the pantry straight onto the counter labeled in blue hatch "shared memory"; a green note "1 order slip · truck does the shuffling". A dashed takeaway box spanning both panels: "TMA turns hundreds of cook-trips into ONE truck delivery — the cooks just cook." Excalidraw style, white background, charming, handwritten labels. || Left: every cook runs to the pantry. Right: one cook hands a slip to a delivery truck, and the cooks go back to cooking.]]

## Meet TMA: the delivery truck

**TMA** stands for **Tensor Memory Accelerator**. Forget the name; here's the picture. It is a dedicated piece of silicon whose *only* job is to move rectangular tiles of data from the far pantry (global memory, called **GMEM** or **HBM**) into the counter (shared memory) — and to do the fussy swizzle-shuffle for free on the way.

The magic is in *how you order the delivery*. One cook — an elected leader, conventionally the thread numbered zero — fills out a tiny order slip and hands it to the truck: "Here's a matrix in the pantry with this shape. Carve out the tile at *these coordinates*, drop it *here* on my counter, shuffled the way the head chef likes." Then that cook walks away. The truck delivers *in the background* while all the cooks — including the leader — go do useful work.

[[note: aha || Here is the sentence that makes it land: **the cooks are free the instant the delivery is *ordered*, not when it *arrives*.** On the old way, a cook was stuck walking to the pantry and back — dead time. With TMA, ordering takes one instant, and then the truck handles the whole trip on its own while the kitchen keeps cooking. That overlap — deliver-while-you-cook — is the entire reason TMA exists.]]

That order slip has a real name: a **descriptor**, or `CUtensorMap`. It's a 128-byte blob describing the matrix's shape, its stride (how far apart the rows sit in memory), the tile size to carve, and the swizzle mode. And the beautiful part: **you build it once, on the host CPU, before the kernel even runs.** The knowledge of "how is this matrix laid out" leaves the hot loop entirely and lives in that little slip.

[[note: example || Do the counting on the board so it's concrete. Old way: to load one tile with 256 cooks, that's 256 separate load instructions, each cook computing its own address — hundreds of instructions, and address math burning precious register space. TMA way: **one** instruction. One cook, one order slip, one truck. 256 → 1. Write "256 loads → 1 descriptor" on the board and circle it.]]

[[fig: A hand-drawn "anatomy of an order slip" figure titled "The descriptor (CUtensorMap, 128 bytes)". Center: a rounded rectangle drawn as a stack of labeled slots, each hand-lettered: "data type = BF16", "matrix shape = M × K", "stride (bytes between rows)", "tile to carve = 128 × 64", "swizzle = 128B". To the LEFT, a big red-hatched matrix labeled "A in the pantry (GMEM)" with red dimension arrows for M and K, and a small pale-yellow sub-rectangle inside it circled and labeled "the tile"; a blue dashed arrow runs from that sub-rectangle to the "tile to carve" slot. To the RIGHT a green note "built ONCE on the host CPU, before the kernel runs" with a green arrow into the block. Dashed takeaway box bottom: "the descriptor is the matrix's address book — write it once, the truck reads it every delivery." Excalidraw style, white background, handwritten. || The descriptor is a 128-byte order slip: shape, stride, tile size, swizzle — written once, read on every delivery.]]

## How do the cooks know the delivery arrived?

This is the one genuinely subtle part, so slow down — it's where every engineer gets tangled the first time.

On the old way, a cook knew its onions had arrived because it carried them itself. But the TMA truck runs in the background. So the cooks need to be told "the pallet has fully landed — start cooking." That signal is an **mbarrier**, a little shared counter everyone waits at.

And here's the twist: the mbarrier does **not** count cooks. It counts **bytes**. When the leader orders the delivery, it *also* announces "expect exactly this many bytes" (for a 128×64 tile of 2-byte numbers, that's 128 × 64 × 2 = 16,384 bytes). The truck counts those bytes down as it unloads: 16,384 → … → 0. Only when the counter hits zero does the barrier flip and release the waiting cooks.

[[note: confusion || The #1 TMA bug, guaranteed: students think completion is measured like the old way — "my thread is done." It isn't. TMA completion is measured in **bytes landed**, not in per-thread loads finished. If you announce the wrong byte count, one of two disasters: the barrier releases too early (cooks grab a half-empty pallet — garbage), or it never releases (cooks wait forever — a hang). The fix, said out loud: "You're not waiting for the cooks. You're waiting for the *bytes*." Make them repeat it.]]

[[fig: A hand-drawn timeline titled "The mbarrier handshake (waiting on BYTES)". A horizontal time axis with a red arrow "time →". Three stacked lanes. TOP lane "leader cook (thread 0)": a box "order delivery" then a box "announce: expect 16,384 bytes". MIDDLE lane "TMA truck" in green: a long green box "unloading tile onto counter (swizzled)" with a shrinking counter scrawl "16384 → 8000 → … → 0", little blue arrows dripping into a blue-hatched box "shared memory". BOTTOM lane "all other cooks": a short box "check in" then a long hatched bar "WAIT — blocked". A bold orange vertical dashed line drops exactly where the counter hits 0, labeled "barrier flips: all checked in AND bytes == 0", and all three lanes resume together into a shared box "→ start cooking (matmul)". Dashed takeaway box: "completion = bytes landed, not cooks finished." Excalidraw style, white background, handwritten. || The leader announces a byte count, the truck counts it down, and the barrier releases everyone only when the bytes have landed.]]

## Meet WGMMA: one command, a hundred cooks

Now the second new piece. TMA fixed *feeding*. WGMMA fixes *cooking*.

**WGMMA** stands for **Warpgroup Matrix Multiply-Accumulate**. On the old chips, the smallest team of cooks was a **warp** — 32 threads moving in lockstep — and each cook did one tiny multiply-add per command. To get near the H100's rated **989 trillion** BF16 operations per second, you'd issue commands so fast the manager (the warp scheduler) would collapse before the cooks did. One command per multiply-add is simply too many commands.

WGMMA's answer: make the command *enormous*. A **warpgroup** is four warps stuck together — exactly **128 cooks** — and WGMMA gives all 128 a single order that multiplies a whole tile at once.

[[note: metaphor || Old way: the head chef shouts a separate order for every single potato — "chop this one! now this one!" — thousands of shouts. WGMMA is the head chef shouting *once*: "the whole crew, together, prepare this entire tray." One command, 128 cooks, a giant slab of work done. Fewer shouts, more food.]]

The canonical WGMMA does a `64 × 16` tile of `A` times a `16 × N` tile of `B` (where `N` can be 64, up to 256), landing in a `64 × N` result. Let's count the work in one command.

[[note: example || A single `m64n64k16` WGMMA computes roughly 2 × 64 × 64 × 16 ≈ **131,000 floating-point operations from ONE instruction**. Write that on the board next to "old way: 1 multiply-add per instruction." One command doing the work of a hundred thousand. *That* is the leverage Hopper was built to give — and the jaw-drop number for this block.]]

Two more things break students' old intuition, and both are worth saying slowly.

**First: the answer lives in the cooks' pockets.** The `64 × 64` result — 4,096 numbers — is too big for any one cook. So WGMMA spreads it across all 128: each holds a 32-number fragment in its own registers. And it *stays* there through the whole multiplication — you never re-copy or re-zero it. A flag on the command says "add onto what you already have," so accumulating across many tiles is free.

**Second: the ingredients are read straight off the counter.** The old CUDA-core way copied ingredients from the shared counter into the cooks' hands (registers) *before* multiplying. WGMMA skips that — the head chef reads `A` and `B` **directly out of shared memory**. That's exactly why TMA's swizzle mattered: TMA lays the ingredients out in precisely the shuffled pattern WGMMA wants to read. **TMA and WGMMA are a matched pair** — one produces the layout the other consumes.

[[fig: A hand-drawn "one command, 128 cooks" figure titled "WGMMA". Center: a rounded rectangle labeled "warpgroup = 4 warps = 128 cooks" holding four small boxes "warp0..warp3". To its left, a blue-hatched tile labeled in red "A: 64 × 16" and a green-hatched tile "B: 16 × N". A fat blue dashed arrow from both feeds a pale-yellow-hatched output tile labeled in red "C: 64 × N accumulator". A purple code sliver under it: "wgmma.mma_async ... m64n64k16". A red callout on C: "4096 numbers ÷ 128 cooks = 32 each, kept in pockets (registers)". An orange note by A and B: "read STRAIGHT off the counter (SMEM) — no copying into hands!". A green spec note: "1 command ≈ 131,000 FLOPs (vs 1 per old instruction)". Dashed takeaway box: "the answer stays in the cooks' pockets the whole time; ingredients stream from shared memory." Excalidraw style, white background, handwritten. || One warpgroup-wide command multiplies shared-memory tiles into a result that lives in the cooks' registers.]]

## The catch, one level up: who fetches while the chef cooks?

Here's the honest failure that ties it all together — and it's the same disease from day one, just one floor higher.

Write the obvious kitchen: one crew that loops — order a tile, wait, cook, order the next, wait, cook. Profile it. The head chef (tensor core) is idle most of the time! It finishes its giant tray in a flash, then sits there while *the same cooks* trudge off to order and wait for the next delivery. We've re-created the naive problem one floor up: the cooking unit waits on the fetching, because the *same crew does both*.

There's a hint at the fix already hiding in WGMMA. Remember, WGMMA is *async* — the command returns almost immediately and the head chef cooks in the background. Between issuing a batch and needing the result, there's a window of free time. The whole game is filling that window with the *next* delivery.

[[fig: A hand-drawn "async window" technical figure titled "WGMMA runs in the background". On the left, a handwritten assembly-style column in purple, top to bottom: "wgmma.fence", "wgmma.mma_async ×N", "wgmma.commit_group", "wgmma.wait_group". A blue curly brace spans the two middle lines, annotated in blue "issue & return instantly — tensor core works in the background". A red bracket spans commit → wait with a bold orange label "← THIS window is FREE: do other work here". On the right, a small diagram: a green-hatched box "SMEM (counter)" feeding via a blue dashed arrow labeled "read straight" into a box labeled "TENSOR CORE (head chef)", which writes into a pale-yellow-hatched box labeled in red "acc (registers / pockets)". A green note near the fence: "fence = tell the chef the ingredients are ready to read". Dashed takeaway box: "commit → wait is the free window — fill it with the next tile's delivery." Excalidraw style, white background, handwritten. || The async window: everything between commit and wait is free time the crew must spend fetching the next tile.]]

[[note: teach || This is the emotional pivot of the whole chapter — build the anticipation. Draw the single-crew loop and point at the idle chef. Ask the room: "The truck is fast. The chef is fast. Why is the chef standing around?" Let them stew. Then deliver the answer: because one crew can only do one thing at a time. The fix is not a faster truck or a faster chef — it's *dividing the labor*.]]

## Warp specialization: split the crew by job

The fix is called **warp specialization**, and it's simple once the metaphor is in place. Stop making every cook do both jobs. Split the crew:

- **Producer cooks** do *nothing but order deliveries.* Their loop: find an empty counter spot, fire a TMA truck to fill it with the next tile, mark it "full," repeat. They never touch the stove.
- **Consumer cooks** do *nothing but cook.* Their loop: wait for a full spot, fire WGMMA on it, mark it "empty" so producers can refill it, accumulate the result. They never run to the pantry.

The two crews meet at a **ring of counter slots** in shared memory — usually three or four deep. Producers run *ahead*, filling slots 2, 3, 4 while consumers cook slot 1. As long as they stay ahead, the head chef *never waits*: the next tray is always already on the counter. The tensor cores stay lit.

[[note: production || This is not a diagram in a textbook — it is *literally* the shape of the fastest kernels running today. **FlashAttention-3**, the attention kernel powering fast inference on H100s across the industry, is built exactly this way: producer warps firing TMA loads, consumer warps firing WGMMA, overlapping through a shared-memory ring. It's how vLLM, DeepSeek, and every serious H100 serving stack squeeze attention. When your students draw this producer/consumer picture, they're drawing the real machine that decides whether a model costs a dollar or a penny per million tokens.]]

[[fig: A warm hand-drawn kitchen illustration titled "Split the crew: producers and consumers". LEFT: a group of small cook figures labeled in blue "PRODUCER cooks — only order deliveries", each waving down a little green TMA truck; blue dashed arrows carry pallets to a central counter. CENTER: a counter drawn as a ring of 4 slots (green-hatched boxes) labeled "shared-memory ring (4 slots)", two slots marked "full", two "empty". RIGHT: a group of cooks labeled in orange "CONSUMER cooks — only cook (WGMMA)", pulling ingredients from full slots (purple dashed arrows) and working a stove labeled "tensor core". Two little red circles mark the signals: "① full" (producer → consumer) and "② empty" (consumer → producer). An orange callout over the stove: "chef never waits — next tray already on the counter". Dashed takeaway box: "divide labor by JOB, not by time — fetching and cooking happen at once." Excalidraw style, white background, charming, handwritten. || Producers only order deliveries, consumers only cook, and a ring of counter slots keeps the chef from ever waiting.]]

[[note: confusion || Students confuse this with plain double-buffering ("just prefetch the next tile"). The difference worth stating: double-buffering has *one* crew doing both jobs but planning ahead. Warp specialization has *two different crews*, each doing only one job. The second is cleaner because a producer and a consumer literally cannot get in each other's way — they're different threads running different code. Different people, different jobs, no collision.]]

Two grown-up details, if the room is hungry — but don't lead with them. One: producers and consumers **don't use a normal block-wide barrier** (that would force everyone to stop together and kill the overlap). They coordinate through cheap mbarriers, one "full" and one "empty" per slot: the classic bounded-buffer handshake, in silicon. Two: because consumers hold the whole result in their pockets, they need far more register space than producers. Hopper lets you **hand registers over** at runtime — give consumers ~240 each and producers ~24 — so the pockets go to the cooks who carry the load.

[[sn: A very common real layout is *two* consumer warpgroups fed by *one* producer warpgroup. One producer can comfortably keep two cooking crews supplied, because the truck's limit is delivery bandwidth, not how fast it takes orders — so one order-taker keeps two stoves blazing.]]

## The payoff number

Put the reward on the board. On the CUDA-core ladder — cooks doing double duty, no tensor cores — the best FP32 kernels topped out near **40 TFLOP/s**. Bolt on WGMMA, TMA, and a warp-specialized producer/consumer pipeline, and a clean BF16 kernel lands near **500 TFLOP/s** — *roughly half the chip's realistic peak* — before you've even started the chip-wide tricks.

[[note: aha || Say it plainly: "The tensor cores were sitting inside the chip the whole time — 989 TFLOP/s of horsepower. The old kernels just couldn't talk to them. The moment we built a delivery truck (TMA) to feed a head chef who commands the whole crew (WGMMA), and split the crew so fetching and cooking happen at once — the chip woke up. Forty to five hundred. That's the H100 finally being the H100."]]

## Teaching notes: the board sequence

Reveal it in this exact order and it lands every time:

1. **Recall the mantra** — "feeding beats math." Two minutes. Everything hangs off this.
2. **The old kitchen** — every cook runs to the pantry AND shuffles by hand. Draw the chaos. Establish the pain.
3. **TMA the truck** — one order slip, delivers in the background, cooks go free the instant it's *ordered*. Draw the truck. Circle "256 loads → 1."
4. **The byte-counting barrier** — the one subtle bit. Draw the countdown timeline. Drill "you wait on bytes, not cooks."
5. **WGMMA the mega-command** — one shout, 128 cooks, 131,000 FLOPs. Answer stays in pockets; ingredients read straight off the counter (this is *why* the swizzle mattered — callback to TMA).
6. **The idle-chef puzzle** — draw the single crew, point at the bored chef, make them find the answer.
7. **Warp specialization** — split the crew, the ring of slots, producers run ahead. Then reveal: *this is FlashAttention-3*.
8. **The number** — 40 → 500 TFLOP/s. Land the plane.

The one live demo, if you have an H100: profile the naive single-crew tensor-core kernel in Nsight and show the tensor cores idle; then profile the warp-specialized version and show them pinned near-busy. No GPU? The byte-countdown timeline drawn live is the demo — nothing makes the mbarrier click like watching you count 16,384 down to 0 on the board.

## You can now teach

- **TMA** as a dedicated delivery truck: one cook hands over an order slip (a descriptor built once on the host), and the truck moves a whole tile — swizzled — in the background while the cooks cook.
- The **byte-counting mbarrier**: completion is measured in *bytes landed*, not cooks finished — and why getting the byte count wrong is the classic hang-or-garbage bug.
- **WGMMA** as one giant command for a 128-cook warpgroup: ~131,000 FLOPs per instruction, the result kept in the cooks' registers, ingredients read straight off shared memory.
- Why **TMA and WGMMA are a matched pair** — the truck lays out ingredients in exactly the swizzle the chef reads.
- The **idle-chef problem** and its fix, **warp specialization**: split the crew into producers (only fetch) and consumers (only cook), rendezvousing through a shared-memory ring so the tensor cores never wait.
- The **production hook and the number**: this exact pipeline is FlashAttention-3, and it's what takes an H100 from ~40 TFLOP/s of double-duty kernels to ~500 TFLOP/s of a chip finally being fed.
