By the end of this chapter you'll be able to stand at a whiteboard and teach the most surprising fact about how chatbots work: *making a model chat is not a math problem, it's a memory problem.* You'll explain what the KV cache is, why every new word forces the machine to re-read a growing pile of memory, and how one trick borrowed from operating systems — PagedAttention — stopped that pile from wasting half your GPU. You need no prior knowledge of transformers. You need one good metaphor, one honest number, and the patience to reveal it in the right order.

Write one sentence at the top of the board and leave it up the whole time: **training a model is a compute problem; serving one is a memory problem.** The rest of this chapter is that sentence, made concrete.

## Two jobs hiding inside one chatbot

When you hit send, the machine does two very different jobs, back to back. Students think it's one smooth process. It isn't. Teach the split first, because everything else hangs off it.

Job one is **prefill**. The model reads your *whole* prompt at once — every word you typed — and works out the first word of its reply. All your words go in together, in one big gulp.

Job two is **decode**. Now the model writes its reply one word at a time: writes a word, looks at everything so far, writes the next word, looks again. Word by word, until it's done. Every word after the first is a decode step.

[[note: metaphor || Think of a chef reading a recipe. **Prefill** is reading the entire recipe once, start to finish, in a single sitting — a big burst of concentrated reading. **Decode** is then *cooking*, one step at a time: chop, glance back at everything you've done so far, stir, glance back again, plate, glance back again. The reading happens once, fast and dense. The cooking happens step by step, and each step you re-check the whole progress so far. Same recipe, two totally different rhythms.]]

[[fig: A warm hand-drawn two-panel illustration titled "Two jobs inside a chatbot". Left panel labeled "PREFILL: read the whole prompt at once" — a friendly chef figure reading a long scroll/recipe in one big gulp, several word-boxes ("tell", "me", "a", "story") all flowing into the chef's head together through one fat arrow, a green handwritten note "all your words go in TOGETHER — one big burst". Right panel labeled "DECODE: cook one step at a time" — the same chef at a stove producing one small dish (one word "Once") at a time, with a little dashed loop arrow labeled "repeat for every word", and a red note "one word out per step". A dashed divider between panels. Dashed takeaway box spanning both: "prompt in = one gulp (prefill). reply out = one word at a time (decode)." Excalidraw style, white background, charming, hand-lettered. || The chatbot does two jobs: read the whole prompt at once, then write the reply one word at a time.]]

Hold that picture. The whole chapter is about why the *second* job — the slow word-by-word cooking — is where all the pain lives.

## The naive way is insane: re-reading the whole book every word

Here's what makes chat expensive. To write its next word, the model looks back at *every* word that came before — your prompt plus everything it has said. That "looking back" is the attention mechanism; you don't need its details today, only one fact: **each new word depends on all previous words.**

Now imagine doing that the naive way. To write word 100, you re-read words 1–99. To write word 500, you re-read 499 words. The re-reading gets longer every step. The total work grows like the *square* of the length — a 1000-word reply is about half a million word-reads. That's hopeless.

[[note: example || Put tiny numbers on the board. For a 5-word reply, naive re-reading costs 0+1+2+3+4 = **10** look-backs. For 100 words: 0+1+...+99 = **4,950**. Double the length and the cost roughly *quadruples*. That exploding curve is why nobody does it the naive way.]]

## The KV cache: don't recompute, remember

The fix is the obvious human one: **don't re-read — keep notes.**

When the model processes a word, it produces two little summary vectors for it: a **key** and a **value** — "what this word offers to future words." The trick: once computed, a word's key and value *never change*. Word 7's key is word 7's key forever. So compute them once and stash them in a running list. That list is the **KV cache** — the K (keys) and V (values) for every word so far.

[[note: metaphor || The KV cache is a **stack of index cards**. Every time the model finishes a word, it writes one card for that word — a summary of what that word means — and drops it on the pile. To write the next word, the model doesn't re-read the whole book; it just flips through the existing pile of cards and adds one fresh card at the end. The book stays closed. The pile only ever grows by one card per word. That pile of index cards *is* the KV cache.]]

[[fig: A warm hand-drawn illustration titled "The KV cache = a growing pile of index cards". A friendly robot/model figure at a desk. To one side, a closed book labeled "the actual words (never re-read)". In front of the robot, a visible stack of index cards, each card labeled with a word and two little tags "K" (blue) and "V" (green). The robot is dropping ONE new card on top, arrow labeled "one new card per word". A red curved arrow shows the robot's eyes flipping through the WHOLE existing stack labeled "reads every card, every step". A green note "cards never change once written". Dashed takeaway box: "remember, don't recompute — but you must re-read the whole pile every single word." Excalidraw style, white background, charming, hand-lettered. || The KV cache turns re-reading the book into flipping through a pile of cards that grows by one per word.]]

This turns the exploding square-cost back into something linear: each step, a little fresh work plus one read of the growing list. But look at the price you just paid — it's the crux of the whole chapter.

[[note: aha || The twist that makes students sit up: **you fixed the compute, but created a memory monster.** You no longer recompute old words — but every decode step you still *read the entire pile of cards* out of memory, all of it, to produce one new word. The pile grows every step. The machine spends its life heaving a bigger and bigger stack of cards across memory, doing almost no math on them. The tensor cores you paid a fortune for sit idle. This is why decode is **memory-bound**: the bottleneck isn't thinking, it's *fetching*.]]

## A real number, so it lands with weight

Vague "it's a lot of memory" won't move a room. Give them the actual size.

Take a real-ish model: an 8192-word context, 8 KV heads, head dimension 128, 2-byte numbers. Per layer the keys take `8192 × 8 × 128 × 2 bytes = 16 MiB`, and the values another 16 MiB — 32 MiB per layer. A real model stacks about 64 layers. So:

```
per layer:  K = 16 MiB,  V = 16 MiB    →  32 MiB
× 64 layers                            ≈  2 GiB
```

**Two gigabytes of cache — for one conversation.** Every decode step reads the whole thing.

[[note: aha || Now the jaw-drop. An H100 GPU moves memory at about 3.35 terabytes per second. Streaming 2 GiB takes roughly **600 microseconds** — and that's the *floor* on how fast you can produce one word. It doesn't matter that the same chip can do 989 trillion math operations per second; during decode almost all of that math power is asleep. Say it out loud: "This chip is a Ferrari engine, and chat forces it to spend its whole life carrying boxes." That line reframes the entire session.]]

[[note: production || This is exactly what's happening right now when you chat with ChatGPT, Claude, DeepSeek, or a Llama model. The reason your reply streams in word by word at a readable pace — and not instantly — is this memory floor. The reason serving a long conversation costs more than a short one is that the KV pile is bigger, so each word costs more memory-reads. Companies serving these models to millions of people live and die on this number. Cutting the bytes moved per word is, directly, cutting the electricity bill and the number of GPUs you need.]]

[[fig: A hand-drawn technical diagram titled "What decode reads to make ONE word". A tall stacked-DRAM block on the left (blue rounded rectangles, HBM stacks) with a green spec label "80 GB HBM @ 3.35 TB/s". A fat orange-outlined arrow with pale-yellow hatch pours from the DRAM into a small yellow output box on the right labeled "1 new word out". The fat arrow is labeled in red "KV cache: ~2 GiB, grows every step ↑" with an orange emphasis note "read the WHOLE pile, every word". Below it a much thinner blue arrow labeled "tiny bit of fresh math". Off to the side, greyed-out idle tensor cores labeled in red "989 TFLOP/s — asleep". Numbered red circles: (1) stream whole KV cache, (2) do a sliver of math, (3) emit one word, (4) pile grows, repeat. A dashed takeaway box: "decode moves gigabytes to make ONE word → memory-bound, and the pile only grows." Excalidraw style, white background, hand-lettered. || Every decode step drags the entire KV cache across memory to make a single word, while the math units sit idle.]]

[[note: teach || Board sequence that works: (1) write "training = compute, serving = memory" and leave it up. (2) Draw the two-jobs chef picture. (3) Do the naive cost by hand (0+1+2+3+4=10) so they *feel* the square. (4) Introduce the KV cache as index cards — "remember, don't recompute." (5) Spring the twist: you still read the whole pile every word. (6) Land the 2 GiB / 600 μs number. That order matters — the twist only lands if they first believe the cache was a pure win.]]

## Where the naive cache bleeds memory

The cache is big and unavoidable. But the *first* way anyone stores it is also wasteful, and the waste sets up the fix. Two problems, both about not knowing how long the reply will be.

**Problem one: you reserve for the worst case.** You don't know if a reply will be 10 words or 4000, so the naive code reserves one giant contiguous slab sized for the maximum, up front, for every request. A request that stops after 10 words has paid for thousands of empty slots. This reserved-but-never-used space is **internal fragmentation**.

**Problem two: leftover holes.** Conversations start and finish at different times, freeing oddly-sized gaps. A new long conversation arrives, and even with plenty of *total* free memory, there's no single contiguous run big enough. That's **external fragmentation**.

[[note: metaphor || Picture a parking lot where every car must reserve a whole *row* of spaces on arrival, in case it turns out to be a bus. Most cars are tiny. Rows sit 90% empty (internal waste). And as cars leave, you get scattered single empty spots but no full row free for the next bus (external waste). The lot looks full but is mostly air. That's the naive KV cache: it looks like you're out of memory, but 60–80% of it is wasted on reservations and holes.]]

[[note: production || This isn't a small effect. The original vLLM paper measured real serving systems wasting **60–80% of their KV memory** to exactly these two problems. And here's why it hurts: the KV cache is what caps how many conversations you can serve at once. Waste 70% of it and you serve roughly a third as many users on the same GPU. That's throughput — and money — left on the floor.]]

[[fig: A hand-drawn "parking lot" metaphor titled "Why the naive cache is mostly air". A parking lot drawn from above. Each car (small) has reserved a whole long row of spaces (dashed outline) labeled "reserved for max length". Most of each row is empty, shaded grey and labeled in red "internal waste — reserved, never used". Scattered single empty spots between departed cars are circled and labeled in red "external waste — holes, no full row free". A frustrated bus at the entrance labeled "new long request — nowhere to park despite free spots". An orange bracket over all the grey: "60–80% wasted". Dashed takeaway box: "the lot looks full but is mostly air — same for the naive KV cache." Excalidraw style, white background, charming, hand-lettered. || The naive cache reserves whole rows for the worst case and leaves unusable holes — most of the memory is wasted air.]]

## PagedAttention: give the cache a page table

Now the payoff — a delightful idea, *stolen from operating systems*, and students love that. You don't need to teach virtual memory; the metaphor carries it.

The fix: **stop demanding that a conversation's cache live in one contiguous slab.** Chop all the KV memory up front into a big pool of small, identical **blocks** — each holding the cards for a fixed handful of words (commonly 16). Give each conversation a little lookup list, a **block table**: "my first 16 words live in block #7, my next 16 in block #2, my next 16 in block #9." Those blocks can be scattered anywhere. The conversation *thinks* its cards are in a neat row; physically they're sprinkled all over. This trick is **PagedAttention**.

[[note: metaphor || It's a **coat check**. You hand over your coats and get a numbered ticket for each — ticket 7, ticket 2, ticket 9. Your coats hang scattered across the whole cloakroom, wherever there was a free hook, but your fistful of tickets lets you find them instantly and *in order*. Nobody reserves you a private closet sized for a hundred coats you might bring. You take one hook per coat, wherever it's free, and hold the tickets. The block table is your fistful of tickets.]]

[[fig: A hand-drawn "coat check" illustration titled "PagedAttention = a coat-check ticket system". Left: a person (one conversation) holding a fan of numbered tickets labeled "block table: 0→#7, 1→#2, 2→#9" (purple). Right: a big cloakroom wall of many identical hooks/cubbies (the pool), with coats (KV cards, blue K + green V tags) hung on scattered non-adjacent hooks #7, #2, #9 — the rest empty and available. Long thin dashed arrows curve from each ticket to its hook, deliberately crossing to show the coats are NOT next to each other. A green note "every hook same size — any coat fits any hook". An orange callout "coats scattered, but tickets keep them in order". Dashed takeaway box: "no private closet, no reserved rows — one hook per coat, wherever it's free." Excalidraw style, white background, charming, hand-lettered. || Each conversation holds a fistful of tickets (the block table) pointing at coats (KV blocks) scattered across a shared cloakroom.]]

[[fig: A hand-drawn technical diagram titled "PagedAttention block mapping". Left: a vertical LOGICAL view of one sequence — a column of small numbered word-cells grouped into blocks of 16, labeled in red "logical block 0 (words 0-15)", "logical block 1 (words 16-31)", "logical block 2 (words 32-47)". Middle: a small array drawn as a stack of boxes labeled purple "block_table[]" with entries "0 → #7", "1 → #2", "2 → #9". Right: a big grid of many identical empty physical blocks (the POOL), with blocks #7, #2, #9 filled with hatch (blue=K, green=V) and NOT adjacent; long thin dashed blue arrows curve from each block_table entry to its physical block, crossing to show non-contiguity. Green spec note "block = 16 words · all same size". Orange callout "blocks need not be contiguous!". Dashed takeaway box: "logical order + physical scatter = almost no waste; only the LAST block of a sequence is partly empty." Excalidraw style, white background, hand-lettered. || The technical translation of the coat check: a block table maps a conversation's logical blocks to scattered physical blocks in a shared pool.]]

The waste almost vanishes. No reserving for the worst case — you grab a new block only when the conversation grows into it. No unusable holes — every block is the same size, so *any* free block fits *any* conversation. The only leftover waste is each conversation's last, partly-filled block. That 60–80% you were bleeding? You get most of it back, and reclaimed KV memory turns almost directly into more users served.

[[note: aha || The number that sells it: recovering the wasted KV memory can roughly **triple** how many conversations one GPU serves at the same time — no new hardware, just a smarter way to store the same cards. vLLM, the most widely used open-source serving engine, is built around exactly this. When you hear a company say they "cut serving costs 3× by switching to vLLM," this block-table trick is the heart of it.]]

[[note: production || A bonus falls right out of the coat-check design: **sharing**. If two conversations start with the same system prompt, their first few blocks are identical — so both block tables point at the *same* physical blocks. Store the shared prefix once, not twice. This "prefix caching" is standard in production today and is why repeated system prompts are nearly free to re-serve. It's copy-on-write, straight from the OS playbook.]]

[[note: confusion || The #1 place students get lost: they think PagedAttention makes decode *faster per word*. It doesn't — each word still reads the whole KV pile, still memory-bound. Fix it in one sentence: "Paging doesn't make each word cheaper; it lets you fit *more conversations* in memory at once, so the GPU is never idle." Paging is about **capacity and throughput**, not per-word latency.]]

[[note: say || The clean closing line for the board: "We turned re-reading the whole book every word into flipping through a pile of index cards — that's the KV cache. Then we stopped reserving a private closet for every pile and used a coat-check ticket system instead — that's PagedAttention. The math never got faster. We just stopped wasting memory, and memory was the whole bottleneck."]]

[[sn: If a student asks "can we make the cards themselves smaller?" — yes, and it's the next frontier. Storing the cache in 8-bit numbers (FP8) instead of 16-bit halves the bytes, which halves the memory-read time *and* doubles how many conversations fit. Architectural tricks like grouped-query attention (that "8 KV heads" above) and DeepSeek's compressed attention cut the number of cards you keep at all. Paging, FP8, and compression stack — they each attack a different chunk of the same enemy: bytes moved.]]

## You can now teach

- The **two jobs** inside a chatbot — prefill (read the whole prompt at once) and decode (write the reply one word at a time) — as a chef reading a recipe, then cooking step by step.
- Why the naive "re-read everything each word" approach explodes as the *square* of the length, demonstrated by hand (0+1+2+3+4 = 10).
- The **KV cache** as a growing pile of index cards — "remember, don't recompute" — and the twist that you must still re-read the whole pile every single word.
- Why decode is **memory-bound**, with the number that lands it: ~2 GiB of cache per conversation, a ~600-microsecond floor per word, tensor cores asleep.
- Where the naive cache **bleeds** (internal + external fragmentation, 60–80% wasted) as a parking lot that reserves whole rows.
- **PagedAttention** as a coat-check ticket system — scattered blocks, a block table of tickets — and why it recovers the waste to serve ~3× more users, plus the confusion-fix that it's about capacity, not per-word speed.
