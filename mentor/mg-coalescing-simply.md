By the end of this chapter you'll be able to stand at a whiteboard and teach *memory coalescing* — why 32 threads reading neighbouring addresses fetch their data in a single trip, why 32 threads reading scattered addresses waste almost the whole trip, and why fixing this is the cheapest big speedup in the entire workshop. You need no electronics. You need one bus, one metaphor, and one honest number.

This is the first real optimization students will meet after the naive kernel. And it's the sweetest one to teach, because it changes *nothing* about the math. Same multiplies. Same adds. Same bytes we care about. We just read them in the right order — and the kernel runs several times faster. That contrast is the whole lesson: *the pattern of access, not the amount, decides your speed.*

## The one idea: memory comes in busloads, not by the person

Start with the fact that reorganizes everything. A GPU **does not fetch one number at a time from memory.** When you ask for a float, the memory system doesn't hand you back four lonely bytes. It hands back a whole fixed-size chunk — a **128-byte line**, which happens to be exactly 32 consecutive floats — whether you wanted all of them or just one.

So memory doesn't run like a taxi that carries one passenger to one door. It runs like a **bus**: it always makes the trip with a full 32-seat vehicle, and the only question that matters for speed is *how many buses you have to dispatch.*

[[note: metaphor || The neighbourhood bus. Picture a warp — a group of 32 threads that move in lockstep — as 32 kids who all need a ride to school at the same moment. The city runs one kind of vehicle: a 32-seat bus. If all 32 kids live on the *same street*, one bus swings down that street, everyone climbs aboard, and it's a single trip. If the 32 kids are scattered one-per-street across 32 different streets, the city has to send *32 separate buses*, each driving all the way out to pick up a single child and coming back 31 seats empty. Same 32 kids. Same destination. Thirty-two times the fuel. Coalescing is just: *make the kids live on the same street.*]]

[[fig: A warm hand-drawn illustration titled "One warp, one busload". Top half labeled "COALESCED — everyone on one street": a single cheerful yellow school bus driving down one street lined with 32 little houses in a row, all 32 kids (small stick figures numbered t0..t31) climbing aboard the one bus, a green handwritten note "1 bus · 32 seats all full · one trip". Bottom half labeled "STRIDED — one kid per street": eight separate identical buses each driving down its own faraway street to pick up a SINGLE child, the other seats drawn empty with little "empty" tags, a red handwritten note "up to 32 buses · 31 empty seats each · wasted fuel". A dashed takeaway box spanning the bottom: "same 32 kids, same school — scattered means many near-empty buses." Excalidraw style, white background, charming, handwritten labels. || The core metaphor: a warp is a busload. Neighbours ride together in one trip; scattered riders each need their own near-empty bus.]]

That's the entire chapter in one picture. Everything from here is making "same street" precise and showing students where a real kernel accidentally scatters the kids.

## The tiny by-hand number: 1 bus vs 32 buses

Put concrete numbers on the board so it stops being a vibe and becomes arithmetic.

A warp is **32 threads**. A memory line is **128 bytes**. A float is **4 bytes**. So one line = 128 / 4 = **32 floats** — one line holds *exactly one warp's worth* of floats. That coincidence is the whole reason this works out so cleanly, and it's worth pausing on.

[[note: example || Do both cases on the board with the same 32 threads. **Case A — contiguous.** Thread `t` reads the float at position `t`: thread 0 reads float 0, thread 1 reads float 1, … thread 31 reads float 31. All 32 addresses fall inside one 128-byte line. **One trip.** 128 bytes fetched, 128 bytes used. 100% useful. **Case B — strided.** Thread `t` reads the float at position `t × 1000`: thread 0 reads float 0, thread 1 reads float 1000, thread 2 reads float 2000… Every address lands in a *different* line. **32 trips.** You fetch 32 × 128 = 4096 bytes and use 32 × 4 = 128 of them. That's about **3% useful; 97% thrown away.**]]

Say that last number out loud and let it land. Same 32 floats wanted in both cases. Case A costs one trip; Case B costs thirty-two. The bytes you *care about* are identical — the bytes you're *charged for* differ by 32×.

[[note: aha || Here's the sentence that flips the room: **"Coalescing doesn't read fewer bytes you want. It stops you paying for bytes you don't."** Students assume a fast kernel must be doing less work. This one does the exact same math and requests the exact same useful data — it just stops dragging home 31 empty seats on every bus. The speedup is pure waste-removal, which is why it's free.]]

## The real mechanism, built up gently

Now name the machinery, one term at a time, each defined as it arrives.

A **warp** is a group of 32 threads that execute the same instruction at the same time. When a warp hits a load instruction, the memory system does *not* see 32 independent requests. It gathers the 32 addresses those threads want and asks: how many fixed-size lines do these addresses touch?

The line is **128 bytes**, aligned to a 128-byte boundary — 32 consecutive floats.[[sn: On Hopper (H100) the 128-byte line is split into four 32-byte **sectors**, and the hardware can fetch just the sectors it needs. So a partial access wastes down to 32-byte granularity, not the full 128. For teaching, "128-byte busload" is the right first picture; mention sectors only if a sharp student asks why the waste isn't always exactly 32×.]] If the warp's 32 addresses all fall inside one line, the hardware runs **one memory transaction** — one bus. If they scatter across 32 lines, it runs **up to 32 transactions** — 32 buses.[[sn: "Up to" because if two threads happen to land in the same 128-byte line, they share a bus. Worst case — a large stride — gives every thread its own line and hits the full 32× penalty squarely.]]

That's it. That's the whole rule: **for one load by one warp, coalescing asks how many lines the 32 addresses touch.** One line is the dream. Thirty-two is the disaster. Nothing about the *amount* of useful data changed — only the *pattern*.

Say a word about the road the data travels, because it explains *why* the line is the unit. A load walks a fixed path — from HBM (the far, huge memory) up through the L2 cache, into the fast on-chip memory near the cores, and finally into registers. **Every rung of that path is priced in whole lines, never in single floats.** So you pay per line touched, not per byte used. That single sentence is why a strided load is so ruinous: it touches many lines to use one float from each.

[[fig: A technical Excalidraw diagram titled "One warp, one load: how many transactions?". Top panel labeled (A) COALESCED in orange: 32 small numbered thread boxes t0..t31 (blue) in a row, each with a thin blue arrow pointing DOWN into a single long horizontal bar drawn as one 128-byte line (green label "128 B line = 32 floats, aligned"), arrows landing on adjacent cells left-to-right, every cell shaded. A green note: "1 transaction · 128 B fetched · 100% used". Bottom panel labeled (B) STRIDED in orange: the same 32 thread boxes, but each blue arrow jumps a large gap to a DIFFERENT line — draw 8 separate line-bars, each with exactly one cell shaded pale-yellow and the other 31 cells hatched grey and labeled "wasted". Red dimension note "stride = big" and a red warning "→ up to 32 transactions, ~3% used". A dashed takeaway box: "same useful floats requested, up to 32× the buses." Excalidraw style, blue=threads, green=specs, red=dims, orange=labels, white background, handwritten. || The technical translation of the bus picture: whether one load costs one transaction or thirty-two depends entirely on how thread indices map to addresses.]]

[[fig: A technical Excalidraw "memory pyramid" figure titled "You pay per line, not per byte". A vertical stack of layered boxes, widest at the bottom, narrowing upward. Bottom box "HBM3" (green spec "80 GB · 3.35 TB/s · fetched in 128 B lines"). Above it a wider box "L2 cache" (green "≈50 MiB · 128 B line"). Above that a narrower box "L1 / shared memory per SM" (green). At the very top, 32 small numbered thread boxes t0..t31 (blue) labeled "one warp". A single fat blue arrow runs top-to-bottom labeled "1 coalesced load = 1 × 128 B line". Beside it a thin red squiggly arrow crossing many lines labeled "strided load = up to 32 lines". Orange emphasis note pointing at the L2 box: "granularity is the LINE, not the float". Dashed takeaway box: "every level charges you per line touched, not per byte used." Excalidraw style, blue=mechanism, green=specs, red=labels, orange=emphasis, white background, handwritten. || The path a load travels: HBM → L2 → L1/shared → registers, priced in fixed-size lines at every rung — which is exactly why the pattern beats the volume.]]

## Why does anyone ever scatter? Because of layout

The cruel part is that scattering happens *by accident*, in code that looks perfectly reasonable and computes the correct answer. To see why, students need one fact about how matrices sit in memory.

C and CUDA store a 2-D array in **row-major** order: the element `A[i][j]` lives at linear position `i × N + j`. Read that carefully with students. Elements *along a row* (increasing `j`) are right next to each other — positions `…, i·N+j, i·N+j+1, …`, one float apart. Elements *down a column* (increasing `i`) are `N` floats apart — a giant jump.

[[note: metaphor || A matrix in memory is a book with no chapter breaks — just one long ribbon of numbers, row after row after row, taped end to end. Reading *along a row* is reading the ribbon left to right: the next number is right under your finger. Reading *down a column* means reading one word, then leaping a whole row's width to the next, then leaping again — same book, but you're pole-vaulting across the page instead of sliding along it. The ribbon (memory) rewards sliding and punishes leaping.]]

[[fig: A warm hand-drawn illustration titled "Row-major: the matrix is one long ribbon". Top: a small 3×4 matrix drawn as a grid with cells labeled row by row, arrows showing it being 'unrolled' into a single long horizontal ribbon/tape below, the ribbon showing the cells in order row0(4 cells), row1(4 cells), row2(4 cells) taped end to end. A green sliding-finger icon under three adjacent ribbon cells labeled "read ALONG a row → neighbours → coalesces". A red pole-vault / big-hop arrow icon leaping N cells down the ribbon labeled "read DOWN a column → stride N → scatters". Dashed takeaway box: "memory is a ribbon: sliding is cheap, leaping is expensive." Excalidraw style, white background, charming, handwritten labels. || Row-major layout drawn as a ribbon: neighbouring columns are adjacent on the tape (coalesce), neighbouring rows are N floats apart (scatter).]]

So the entire coalescing verdict for any matrix access collapses to a single question you can teach students to ask every time:

> As the thread number increases across a warp, does the **column** index change, or the **row** index?

Column-changing means neighbours on the ribbon — coalesced, one bus. Row-changing means leaping by `N` — scattered, many buses. That one question is the whole diagnostic skill, and it's the thing students should carry out of this workshop even if they forget every number: *look at what moves as the thread index rises, and ask whether that motion slides along memory or leaps across it.*

## Where it bites the naive kernel

Now connect it to the kernel from the matmul chapters, because this is where the abstract idea becomes a real 6× on a real GPU.

In the naive kernel the two loads inside the inner loop are `A[row·N + k]` and `B[k·N + col]`. The threads of a warp differ in `col` (the fast axis) while sharing the same `row`. Trace it:

- **`B[k·N + col]`** — as we step across the warp, `col` increases by 1 each thread, so we walk `B[k][col], B[k][col+1], …` along a *row* of B. Neighbours on the ribbon. **Coalesced — one bus.** Good.
- **`A[row·N + k]`** — `row` and `k` are the *same* for all 32 threads, so all 32 read the identical address. That's a **broadcast**, which the hardware also handles cheaply — one fetch, shared by all.

[[note: teach || The board sequence that makes this click: (1) draw the ribbon and the "column changes → slide / row changes → leap" rule. (2) Write just the two load expressions, big. (3) Ask the room, for each load: *"as the thread number goes up, which index moves — row or column?"* Make them answer before you do. When they say "column" for B, cheer — that's the coalesced one. This turns a passive derivation into an active diagnosis, which is exactly the skill you want them to own. Don't front-load the SASS; the ribbon question is the transferable tool.]]

The honest point: the naive kernel isn't the *worst* case — B coalesces and A broadcasts. The trouble is that this happened **by accident**, from CUDA's default thread numbering, and it's fragile. The next kernel makes the good mapping *on purpose*, so you can reason about it and build on it.

## The fix: one line, chosen deliberately

The remap is almost anticlimactically small — and that's the punchline you want students to feel. We flatten the thread block to 1-D and compute the 2-D position ourselves, so the fast-moving thread axis is *guaranteed* to land on the contiguous (column) axis:

```cpp
const uint BLOCKSIZE = 32;
const uint row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const uint col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

if (row < N && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}
```

The whole change is `col = threadIdx.x % 32` and `row = threadIdx.x / 32`. Now a warp is exactly `threadIdx.x = 0..31`, so **`col` runs 0..31 and `row` stays constant** across the warp. Consecutive threads → consecutive columns → the `B` load is one clean 128-byte transaction per warp, and the `A` load is a tidy broadcast. We *designed* the good mapping instead of inheriting it by luck.

[[note: confusion || The number-one confusion here: students think warps are built from `threadIdx.y` — "one warp per row of the block, right?" No. CUDA flattens threads with **x fastest**: the linear index is `threadIdx.x + threadIdx.y·blockDim.x`. So a warp is 32 threads with consecutive `threadIdx.x`. Get this backwards and your *entire* coalescing analysis inverts — you'll call the coalesced load strided and vice versa. The fix: on the board, physically number the threads 0..31 by walking left-to-right along x FIRST, only wrapping to the next y-row after 32. "x fills up before y moves." Make them chant it.]]

[[fig: A technical Excalidraw diagram titled "Kernel 2: remap the warp onto the output tile", two numbered panels. Panel 1 (circle ①): a 32×32 output tile of C drawn as a grid, one horizontal strip of 32 cells highlighted pale-yellow and labeled in red "one warp = 32 threads". A purple code box shows "row = tid/32 ; col = tid%32" with an orange arrow to the strip: "col varies → contiguous". Panel 2 (circle ②): matrix B drawn to the right as a green-hatched square; a blue dashed arrow runs from the warp strip to a single contiguous ROW segment of B labeled "B[k][col..col+31] → 1 transaction · 128 B". Below, matrix A as a blue-hatched square with a single cell circled, blue note "A[row][k] same for all 32 threads → broadcast". Green spec note: "block = 1024 threads (1-D) · warp = 32". Dashed takeaway box: "consecutive threads → consecutive columns → coalesced load of B." Excalidraw style, blue=mechanism, green=specs, red=labels, purple=code, orange=emphasis, white background, handwritten. || The whole change: deciding on purpose that the warp's fast axis lands on B's contiguous axis.]]

## The number that makes jaws drop

Here's the payoff to run as a live demo. The naive kernel sat at about **1.3% of cuBLAS** (cuBLAS is NVIDIA's hand-tuned reference library — the speed everyone measures against). After the remap, the kernel jumps to about **8.5% of cuBLAS** — roughly a **6.4× speedup**. Write those two percentages on the board *before* you show the code change, so the size of the win is fixed in their minds when they see how small the change is.

[[note: demo || Run both kernels back to back under `nvidia-smi`/Nsight and show two numbers on screen: the naive time and the coalesced time. Then show a diff of the source — it's *two lines*. The room's reaction to "6× faster, two lines, zero fewer multiplies" is the emotional peak of the memory section. If you can, open Nsight Compute and point at the metric `l1tex__t_sectors_per_request` (sectors fetched per load): the naive number is bloated; the coalesced one drops toward the ideal floor of 4 sectors per warp. That single metric *is* "how full were the buses."]]

[[fig: A hand-drawn "what the profiler sees" figure. Left: a small bar chart with two bars — a short grey bar labeled "naive 1.3%" and a much taller orange bar labeled "coalesced 8.5%", y-axis hand-labeled "% of cuBLAS", with a big handwritten "≈6.4×" and an up-arrow between them. Right: two stacked metric readouts in monospace hand-lettering — top "NAIVE: sectors/request ▓▓▓▓▓▓▓ HIGH (scattered)" in red, bottom "COALESCED: sectors/request ▓ → 4 (one 128 B line)" in green. A blue dashed arrow connects the low sectors/request to the tall bar. Dashed takeaway box: "same FLOPs, same useful bytes — every bus now full → ~6.4× faster." Excalidraw style, white background, handwritten, red=bad/green=good/orange=emphasis. || The change is invisible in the FLOP count and loud in the memory metrics: fuller buses, fewer trips, a 6× win.]]

[[note: production || This is not a lab curiosity — it's the first thing any serious kernel gets right, everywhere money is spent on inference. Every production matmul and attention kernel in vLLM, in FlashAttention, in the kernels DeepSeek and Meta run to serve models to millions, is built so that consecutive threads read consecutive addresses. On an H100, HBM3 can move about **3.35 TB/s** — but *only* if your accesses are coalesced. Scatter them and you might see a third of that. That gap — running your multi-million-dollar cluster at 30% vs 90% of its rated memory bandwidth — is decided by whether the kids ride one bus or thirty-two. Coalescing is the price of admission to fast; you can't skip it and be clever later.]]

## Be honest about what it did *not* fix

Leave students with the right sense of proportion, or they'll think this was the whole game. Coalescing made each bus *full* — but it did nothing about the fact that we send *far too many buses*. The kernel still re-reads the same element of A from slow memory once for every thread that needs it. We're still hauling `O(N³)` floats from HBM to do `O(N³)` math — dreadful reuse.

[[note: say || "Coalescing filled every bus. It did not stop us running the same route over and over. We're still fetching the same rice from the far pantry a thousand times because a thousand cooks each walked out to get it themselves. The next win is to fetch it *once* and share it — that's shared memory, kernel 3, and that's where the real climb begins." This frames coalescing correctly: not the summit, but the tax you must pay before any cleverness is worth attempting.]]

That's the frame to end on. Coalescing is the cheapest big win — ~6×, zero math removed — precisely because it fixes *waste*, not *work*. And it's mandatory first: there's no point staging data into fast memory if the loads that fill it are scattered. Get the warp's fast axis onto contiguous memory. Then you've earned the right to be clever.

## You can now teach

- **The bus metaphor**: a warp is a 32-seat busload, memory comes in 128-byte lines, and speed is just "how many buses did we dispatch?" — one for neighbours, up to 32 for scattered riders.
- **The by-hand number**: contiguous = 1 transaction, 100% used; strided = up to 32 transactions, ~3% used — same useful floats either way.
- **The ribbon of row-major memory** and the one diagnostic question: *as the thread number rises, does the column index move (slide, coalesced) or the row index (leap, scattered)?*
- **Where it bites the naive matmul** (B coalesces, A broadcasts — but only by accident) and **the two-line remap** that makes the good mapping deliberate.
- **The warp-numbering trap** (x fills before y moves) that flips students' whole analysis if they get it wrong — and how to drill it.
- **The jaw-drop demo and production stakes**: ~6.4× from two lines and zero fewer multiplies, and why every real kernel — vLLM, FlashAttention, DeepSeek — is designed around coalescing to hit 3.35 TB/s instead of a third of it.
