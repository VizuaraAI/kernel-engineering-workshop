By the end of this chapter you'll be able to stand at a whiteboard and teach the most magical moment in the whole workshop: the kernel where you change *one line*, don't remove a single calculation or a single byte, and the code runs about **6 times faster**. Students won't believe it at first. That disbelief is the whole point — you're going to build the suspense, then reveal the trick, then draw the picture that makes it obvious. Let's learn it from zero.

This is Kernel 2 on the GEMM ladder. Kernel 1 was the naive matmul — one thread per output cell — and it ran at a humiliating **1.3% of what NVIDIA's own library (cuBLAS) can do.** This chapter takes it to **8.5%**, and almost nothing changes. That's what makes it unforgettable.

## The setup: same math, we only change *labels*

Say this out loud at the start so nobody gets confused later: **we are not changing the math.** Same one-thread-per-output-cell. Same additions and multiplications. Same numbers loaded from memory. All we change is *which thread computes which output cell* — a relabeling of workers onto jobs. That relabeling is the entire optimization.

[[note: say || "I'm going to show you a change so small it looks like a typo. We won't delete one multiply. We won't skip one byte of memory. We'll just renumber our workers. And the kernel will run six times faster. Watch me build up why — because when you understand *why*, you'll understand the single most important rule about GPU memory."]]

To teach this you need exactly one new fact about the hardware. Let's get it.

## The one fact: the GPU reads memory in fixed-size chunks

Here's the thing students never guess on their own: **a GPU does not fetch one number at a time.** When it goes to memory, it always grabs a whole fixed-size block — even if you only wanted one number out of it.

[[note: metaphor || The delivery truck. Imagine your data lives in a giant warehouse far away, and the only way to get it is a delivery truck. The truck has 32 slots. Every trip, it goes to the warehouse and comes back — and it *always* comes back with a full shelf of 32 boxes, whether you needed 32 boxes or just 1. The truck can't carry a single box. It carries a shelf. So the question that decides everything is: **when you send the truck, do you fill all 32 slots with boxes you actually wanted — or do you waste the trip bringing back 31 boxes you'll throw away?**]]

That "shelf of 32" is real. On a modern GPU (Hopper, the H100 generation), memory moves in **128-byte lines** — and a 128-byte line is exactly **32 floating-point numbers** side by side (each float is 4 bytes; 32 × 4 = 128). One trip to memory brings back one 128-byte line, no matter what.

[[fig: A warm hand-drawn illustration titled "The memory truck always carries 32 boxes". Center: a friendly cartoon delivery truck driving along a road, its cargo bed drawn as a row of exactly 32 small numbered slots (0..31). A green handwritten label on the truck reads "1 trip = 1 shelf = 128 bytes = 32 floats". On the far left, a big warehouse labeled in green "MEMORY (HBM, far away)". Two versions of the truck are shown returning: the top truck has all 32 slots filled with happy orange boxes labeled "all wanted!" with a green note "GOOD trip, 100% useful"; the bottom truck has only 1 box in slot 0 and 31 empty/greyed slots with sad faces, a red note "WASTED trip, 1/32 useful". A dashed takeaway box at the bottom: "the truck always carries 32 — the only question is how many you actually use." Excalidraw style, white background, charming, handwritten labels. || The one hardware fact the whole lesson rests on: memory always arrives in shelves of 32, so a wasted shelf is wasted bandwidth.]]

Why a *whole shelf* at once? Because 32 threads run together. The GPU schedules threads in fixed groups of exactly **32, called a warp.** A warp runs in lockstep — all 32 threads run the same instruction at the same instant. So on a "load from memory" instruction, the hardware looks at all 32 addresses the threads want and tries to serve them together.

[[note: example || Do this by hand on the board with just 4 threads instead of 32 (the idea is identical, the drawing is cleaner). Draw 4 workers. Case A: they want memory addresses 100, 101, 102, 103 — right next to each other. The hardware says "those all fit in one shelf!" and does **one trip.** Case B: they want addresses 100, 200, 300, 400 — far apart. Now each one needs its *own* shelf. **Four trips**, and each trip hauls back a full shelf to deliver just one number. Same 4 numbers wanted. One trip versus four. That gap is the whole lesson.]]

## The magic word: coalescing

Now — and only now, after the truck picture has landed — introduce the real term.

When the 32 threads of a warp ask for 32 numbers that sit *right next to each other* in memory, the hardware fuses all 32 requests into **one single trip.** Every box on the shelf is used. That's called a **coalesced** access — "coalesce" just means "merge into one." One warp, one instruction, one full 128-byte trip, zero waste.

When the 32 threads instead ask for numbers scattered far apart, the hardware can't merge anything. It makes **up to 32 separate trips**, and each trip drags back a whole shelf to deliver a single number. You use 1 out of every 32 boxes. You throw away the other 31. This is a **strided** (or "scattered") access — the bandwidth killer.

[[fig: Two side-by-side hand-drawn panels titled "Coalesced vs Scattered — the bus", drawn as a friendly bus metaphor. LEFT panel labeled "COALESCED" in orange: a single bus with 32 seats, all 32 seats filled with smiling passenger-boxes sitting together, one blue arrow from a row of 32 numbered workers (t0..t31) boarding the one bus; green note "everyone rides ONE bus — 1 trip, every seat used". RIGHT panel labeled "SCATTERED" in orange: 32 separate tiny buses each driving off in a different direction, each bus carrying just ONE passenger and 31 empty seats drawn greyed/hatched; red note "each worker takes their OWN bus — up to 32 trips, 31 empty seats each". A red banner across the right: "7/8 of the road wasted". Dashed takeaway box spanning both: "same 32 passengers — one full bus, or 32 near-empty ones. The seating chart is the whole game." Excalidraw style, white background, warm and charming, handwritten labels. || The emotional core of the chapter drawn as buses: 32 riders either share one full bus (coalesced) or splinter into 32 near-empty ones (scattered).]]

[[note: aha || Here's the number that stops the room. On the naive kernel, memory runs at about **15 GB/s.** The hardware is *capable* of **3,350 GB/s** (3.35 TB/s). So the naive kernel is using less than **half of one percent** of the memory the GPU can move — not because it reads too much data, but because it reads it in the wrong order and throws 7/8 of every trip away. After the one-line fix, memory jumps to about **110 GB/s** — over 7× more bandwidth, for free. Write "15 → 110 GB/s" on the board and circle it. That's the miracle you're about to explain.]]

## Now the by-hand piece: why does a matrix stride the wrong way?

To see how the naive kernel splinters into scattered buses, students need one more small fact: **how a matrix is laid out in memory.**

A matrix is a 2D grid, but memory is a 1D line. So we flatten the grid into a line, one row after another. This is called **row-major** layout. Element `A[row][col]` lives at position `row × N + col` in the flat line. The consequence to burn into their heads:

- Numbers **across a row** (same row, next column) sit **right next to each other** in memory. Contiguous. Coalesces.
- Numbers **down a column** (next row, same column) sit **N apart** in memory. Scattered. Does not coalesce.

[[fig: A hand-drawn figure titled "Row-major: rows are neighbors, columns are far apart". Top: a 4x4 matrix drawn as a labeled grid (blue diagonal hatch), cells labeled with their [row][col]. Below it, a long horizontal "memory strip" of 16 boxes showing the flattened order: row 0's four cells first, then row 1's four, etc., with a green bracket under the first four labeled "row 0 lives together — contiguous". Two colored arrows: a green arrow tracing three cells ACROSS a row of the grid down to three ADJACENT boxes on the strip, labeled "across a row -> neighbors (stride 1)"; a red arrow tracing three cells DOWN a column of the grid down to three boxes that are each N=4 apart on the strip, labeled "down a column -> far apart (stride N)". Dashed takeaway box: "same row = coalesces. same column = scatters. Memory is a 1D line pretending to be a grid." Excalidraw style, white background, handwritten labels. || Why direction matters: a matrix is stored one row at a time, so walking across a row is contiguous but walking down a column jumps by N.]]

Now here's the trap in the naive kernel. In Kernel 1, the fast-changing thread index (the one that sweeps across a warp) was wired to the **row** of the output. So as the 32 threads of a warp step forward, they march *down a column* of the data — stride `N`, scattered buses, wasted trips. The lanes were pointed the wrong way relative to how memory is stored.

[[note: confusion || The confusion that trips up literally everyone the first time: "a 32×32 block of threads — isn't a warp just one column of that block?" No. And getting this backwards inverts the whole analysis. The GPU builds a warp by flattening the block with **x fastest**: a warp is 32 threads that share the same `threadIdx.y` and sweep `threadIdx.x = 0..31`. So a warp is a *horizontal row* of the thread block, not a vertical column. The fix-sentence: "warps run sideways — x first, always." Make them repeat it.]]

[[fig: A hand-drawn technical figure titled "How a warp is really formed". A 32x32 grid of tiny thread squares (blue diagonal hatch) labeled "one thread block". One horizontal strip along the top row is boxed in bold orange and labeled in red "WARP 0 = threadIdx.x 0->31, threadIdx.y = 0". A red X is drawn over a vertical column with a note "NOT this — warps are not columns". Small numbered circles: (1) points at the flattening rule written in purple "linear id = threadIdx.y * 32 + threadIdx.x  (x fastest)"; (2) points at the boxed row. To the right, a stack of 32 horizontal strips labeled "32 warps, one per row of y". Dashed takeaway box: "warps run sideways — the fast axis is x, always." Excalidraw style, white background, handwritten labels. || The technical translation of the common confusion: the hardware flattens a block x-first, so each warp is a horizontal row of constant threadIdx.y.]]

## The one-line remap — the reveal

Build suspense before you show it. Tell them: "the naive kernel's warp runs *down a column* of memory — the scattered-bus disaster. All we have to do is turn the warp *sideways* so it runs *across a row*. And the way we do that is..." — then reveal the change.

Kernel 1 took the row and column from a 2D thread index. Kernel 2 flattens the block to 1D and splits the index *by hand*:

```cpp
const uint BLOCKSIZE = 32;
const uint row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const uint col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
```

That's it. The whole optimization is `row = threadIdx.x / 32` and `col = threadIdx.x % 32`. The `%` (remainder) is the operation that cycles fastest as `threadIdx.x` counts up: 0 gives 0, 1 gives 1, up to 31 gives 31, then it wraps. The `/` (integer divide) changes only every 32 steps. So within one warp — `threadIdx.x` running 0 to 31 — the `row` stays **constant** (all divide to the same value) and the `col` sweeps `0, 1, 2, …, 31`. The warp now runs *across a row*. Sideways. Coalesced.

Take a breath here and make sure the students feel how little we did. We didn't add shared memory. We didn't tile anything. We didn't touch the loop that does the actual multiplying and adding. We changed *two arithmetic expressions* that compute a pair of indices — and those two expressions decide which direction 32 threads march through memory. That is the entire lever. The naive kernel chose its warp-to-data mapping *by accident*; Kernel 2 chooses it *on purpose*.

[[note: teach || Reveal it in this exact order for maximum drama. (1) Write the naive line, circle "warp runs down a column," draw the 32 scattered buses. Let it feel bad. (2) Say "one line." Pause. (3) Write the new line and *underline the `%` and the `/`*. (4) Trace `threadIdx.x = 0,1,2,3` out loud: "row is 0, 0, 0, 0 — constant! col is 0, 1, 2, 3 — sweeping!" (5) Redraw the buses: now one full bus. (6) *Then* show the number. Don't show the speedup before the picture — the picture is what earns the gasp.]]

Follow the change through all three arrays and it's the exact mirror of the naive kernel:

- **B**, `B[k*N + col]`: `col` sweeps across the warp, so 32 threads read 32 **adjacent** numbers in a row of B. One full 128-byte bus. **Coalesced.**
- **A**, `A[row*N + k]`: all 32 threads share the same `row` and `k`, so they want the **exact same number**. The hardware broadcasts one value to all 32 — cheap, no waste.
- **C**, `C[row*N + col]`: writes 32 adjacent cells per warp. **Coalesced** where it used to scatter.

Every access the warp makes is now either one full bus or one broadcast. Nothing splintered.

[[fig: A hand-drawn "before and after" figure titled "Turn the warp sideways", two stacked panels sharing a 32x32 output tile. TOP panel labeled "NAIVE (Kernel 1)" in red: a 32x32 grid (pale-yellow hatch) with one warp drawn as 32 numbered lanes stacked VERTICALLY down a column; a red arrow shows them diving into memory as 32 scattered buses off to the side, red note "warp runs down a column -> stride N -> 32 near-empty buses". BOTTOM panel labeled "COALESCED (Kernel 2)" in green: the same 32x32 grid with the 32 lanes now laid out HORIZONTALLY along one row; a purple code box shows "row = tid/32 (constant), col = tid%32 (sweeps 0->31)"; a green arrow shows them boarding ONE full bus into a contiguous row of matrix B (green hatch), green note "warp runs across a row -> contiguous -> 1 full bus". A big orange curved arrow between the panels labeled "the one-line remap — same threads, same math, just relabeled". Dashed takeaway box: "rotate the warp from vertical to horizontal and every bus fills up." Excalidraw style, white background, handwritten labels. || The whole trick in one image: the remap rotates each warp from running down a column (scattered) to running across a row (coalesced). Nothing else changes.]]

[[note: confusion || A sharp student will object: "if the math and the bytes are identical, the compiled instructions must be different — where's the speed hiding?" Beautiful question. Show them: the inner loop's assembly (SASS) is *byte-for-byte the same* in both kernels — same `LDG` loads, same `FFMA` multiply-add. The only difference is the *address arithmetic outside the loop*. The speed isn't in doing less work. It's in the memory system finally filling every bus instead of running them near-empty. "Same instructions, different addresses" is the sentence that closes the loop.]]

## The measurement — the payoff

Now you can drop the numbers and they'll *mean* something, because the students already understand why.

- Memory bandwidth: **~15 GB/s → ~110 GB/s.** The buses are full instead of 1/8 full. Over 7× more useful data per second.
- Overall speed: **~300 GFLOP/s → ~1990 GFLOP/s.** From **1.3% → 8.5% of cuBLAS.**
- The headline: a **6.4× speedup** from relabeling threads. No new memory. No new instructions. No fewer bytes.

[[note: aha || The jaw-dropper to say last: "We didn't make the GPU do less. We made it stop wasting 7 out of every 8 delivery trips. The naive kernel was leaving 87% of its memory bandwidth on the floor — and the hardware handed it all back the instant we pointed the warp in the right direction. One line." Let that sit. This is the moment the whole idea of 'access pattern' becomes real to them.]]

[[fig: A hand-drawn "results" figure titled "One line, 6.4x — the reveal". Two big thermometer/bar gauges side by side. LEFT gauge labeled "memory bandwidth": a short red bar filled to "15 GB/s" against a tall faint outline labeled in green "3.35 TB/s = full scale", next to it a taller green bar filled to "110 GB/s", orange arrow between them "7x". RIGHT gauge labeled "% of cuBLAS": a tiny grey bar "1.3% (naive)" and a taller orange bar "8.5% (coalesced)", labeled "6.4x faster". Between the two gauges a purple box shows the single changed line "col = threadIdx.x % 32" circled in orange with note "the entire diff". Dashed takeaway box: "no new math, no fewer bytes — just full buses instead of near-empty ones." Excalidraw style, white background, handwritten labels. || The payoff on two gauges: bandwidth 15 to 110 GB/s and 1.3 to 8.5 percent of cuBLAS, all from one relabeling line.]]

## In production, right now

This isn't a classroom curiosity — coalescing is the *first thing* every real kernel engineer checks, on every kernel, forever. Before anyone reaches for a clever algorithm, they ask the boring question: are my warps reading contiguous, aligned memory? If the answer is no, no amount of cleverness downstream will save the kernel, because the memory system is quietly throwing most of its bandwidth away.

[[note: production || The famous **FlashAttention** kernel that runs inside virtually every LLM served today (ChatGPT, Llama, DeepSeek, Claude) is, at its heart, a masterclass in feeding the GPU coalesced, contiguous data. **vLLM's** PagedAttention lays out its key-value cache in memory blocks specifically so that warps read them coalesced. When DeepSeek hand-tunes kernels for their H100 and H800 clusters, "are these loads coalesced?" is question number one. On the newest **B200 (Blackwell)** chips the memory lines and rules are the same shape — bigger, faster, same law. A serving team that gets coalescing wrong ships a model that uses a fraction of a multi-million-dollar GPU cluster. This one rule, on this one whiteboard, is worth real money at scale.]]

[[fig: A hand-drawn "production" figure titled "Coalescing is the first check, everywhere". A friendly checklist on a clipboard. Three rows, each a green checkmark next to a logo-styled label: "FlashAttention — feeds warps contiguous tiles" ; "vLLM PagedAttention — KV cache laid out for coalesced reads" ; "DeepSeek / H100 & B200 kernels — question #1: are loads coalesced?". To the right, a small stack of GPU chips drawn as green rectangles labeled "$$$ multi-million cluster", with a red note "un-coalesced -> only a fraction of it used". Orange banner across the top: "same law on Hopper and Blackwell — bigger, faster, same 128 B line". Dashed takeaway box: "this whiteboard rule is worth real money at serving scale." Excalidraw style, white background, handwritten labels. || Where the one rule lives in production: FlashAttention, vLLM, and every hand-tuned DeepSeek H100/B200 kernel start by checking that loads coalesce.]]

And here's the honest caveat that sets up the next chapter. Coalescing made each trip *full* — but it did nothing about the fact that we take *far too many trips.* We still re-read the same numbers from far-away memory over and over (`N` times each). Kernel 2 fixed *how* we read; Kernel 3 (shared memory) fixes *how often.* That's the next rung, where the real climb begins.

[[sn: If you keep a 2D thread block and just read `threadIdx.y` as the column and `threadIdx.x` as the row — the "swap x and y" trick — you get the identical coalesced layout. The explicit `/` and `%` on a 1D block are used here only because they make the warp-to-address mapping impossible to misread on a whiteboard.]]

## Teaching notes: the board plan

Here's a clean 12-minute sequence for this block:

1. **(2 min) The truck fact.** Draw the delivery truck with 32 slots. "The GPU always brings back a full shelf." Don't say "coalescing" yet.
2. **(2 min) By-hand 4 threads.** Case A (addresses 100,101,102,103 → 1 trip) vs Case B (100,200,300,400 → 4 trips). This is the whole idea in miniature.
3. **(2 min) Row-major.** Rows are neighbors, columns are far apart. Draw the grid flattening into a line.
4. **(2 min) The trap.** Naive warp runs *down a column* → scattered buses. Draw them sad and empty.
5. **(2 min) The reveal.** One line. `/32` and `%32`. Trace threadIdx 0,1,2,3 out loud. Redraw as one full bus.
6. **(2 min) The number.** 15 → 110 GB/s, 1.3% → 8.5%, 6.4× faster. Then the FlashAttention/vLLM production tie.

[[note: demo || The one live demo: if you have a GPU handy, run the naive kernel and the coalesced kernel back to back and let students watch the wall-clock time drop by ~6×. If no GPU, show the two source files side by side in an editor with the single changed line highlighted — the visual of "that tiny diff caused *that* speedup" is almost as powerful. Best of all, run Nsight Compute and show `l1tex__t_sectors_per_request` dropping toward the ideal floor of 4 — that's the profiler literally showing the buses filling up.]]

**Checkpoint questions** to confirm it landed: (1) "Why does reading down a column waste memory but reading across a row doesn't?" (2) "How many threads are in a warp, and which direction does a warp run — sideways or down?" (3) "Did we change the math or the number of bytes read? Then where did the speedup come from?" If they can answer all three, they own it.

## You can now teach

- The **one hardware fact** — the GPU always fetches a full 128-byte line (a shelf of 32 floats) — using the delivery-truck / bus metaphor.
- **Coalesced vs scattered** access: 32 threads share one full bus, or splinter into 32 near-empty ones — the seating chart is the whole game.
- **Row-major layout** and why walking across a row coalesces but walking down a column strides by `N` and scatters.
- The **one-line remap** (`row = tid/32`, `col = tid%32`), traced by hand so students *see* the warp rotate from vertical to horizontal.
- The **payoff and the reveal choreography**: 15 → 110 GB/s, 1.3% → 8.5%, a 6.4× speedup with no new math — and the "same instructions, different addresses" punchline.
- The **production stakes**: coalescing is the first thing checked in FlashAttention, vLLM, and every DeepSeek/H100/B200 kernel — worth real money at scale.
