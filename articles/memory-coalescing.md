The naive GEMM kernel left us at a humiliating **1.3% of cuBLAS** — a measured **309 GFLOP/s** on a card that can do tens of thousands — and the profiler was blunt about why: we are memory-bound, and the memory system is barely awake. It reported **15 GB/s** of global-memory throughput on hardware rated for **3.35 TB/s**. That is not "a bit slow." That is running the firehose at about half a percent of what it can deliver.

Here is the surprising part, and the reason this article exists: the fix costs **one line**, removes **zero** floating-point operations, reads the **exact same bytes** the naive kernel read — and roughly **quadruples throughput to 110 GB/s**, lifting us to **8.5% of cuBLAS**, a **6.4× speedup**. It doesn't read *less*. It reads the same amount, in the *right order*. That reordering is called **memory coalescing**, and it is the single most important habit in all of GPU programming. Almost every kernel you will ever write lives or dies by it.

So the question this article answers is exactly this: **why does the order in which 32 threads ask for memory change how fast the memory arrives — when the total amount of memory is identical?** To answer it we have to stop thinking about a "load" as one thread grabbing one number, and start thinking about what the hardware physically does when a whole warp asks for memory at the same instant. Let me build that up from nothing.

## First, the one prerequisite: what a warp is

If you have read [Threads, warps, blocks, grids](threads-warps-blocks-grids.html) you can skim this; if not, here is the whole idea in one paragraph, because everything downstream depends on it.

When you launch a CUDA kernel you get thousands of **threads**. But threads do not execute one at a time, and they do not execute fully independently. The hardware bundles them into groups of exactly **32**, called a **warp**, and the 32 threads of a warp march in lockstep: at each step they all execute the *same instruction* at the *same time*. When that instruction is a memory load, all 32 threads issue their load *together*, in the same clock. That simultaneity is the whole story. The memory system does not receive 32 requests one after another that it can think about separately — it receives 32 addresses *at once* and has to decide how to service them as a batch.

Which 32 threads form a warp? CUDA flattens the `(x, y, z)` thread index into a single linear number with `x` moving fastest: `tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)`. Warp 0 is `tid` 0–31, warp 1 is `tid` 32–63, and so on. [[sn: This is the detail almost everyone gets wrong the first time. Warps are formed by flattening `(x, y, z)` with `x` fastest — NOT by `threadIdx.y`. In a `32×32` block, one warp is a full row of constant `y` spanning `threadIdx.x = 0..31`. Get this backwards and your entire coalescing analysis inverts, because you'll reason about the wrong 32 addresses.]] For a simple 1-D block, that just means a warp is `threadIdx.x = 0..31`, then `32..63`, and so on. Hold onto that: **the threads of one warp have consecutive `threadIdx.x`.** We will use it constantly.

[[fig: A hand-drawn intuition figure titled "32 threads, one instant". Center: a warp drawn as 32 small numbered boxes t0..t31 in a single row, all connected to one thick black bracket above them labeled in orange "one warp = 32 threads, lockstep". A blue handwritten note: "when the instruction is LOAD, all 32 ask for memory in the SAME clock". To the left, a tiny analogy sketch: 32 people at a counter handing 32 order slips to a SINGLE waiter simultaneously (label the waiter "memory system" in green). A red note under the people: "the waiter batches the slips — he does NOT walk 32 times if the orders are next to each other". Dashed takeaway box bottom-right: "the hardware sees 32 addresses AT ONCE and services them as a batch, not one by one". || A warp is 32 threads that issue the same load in the same clock. The memory system must serve all 32 addresses as one batch — and how it batches them is everything.]]

## The unit of a memory access is not a float

Now the fact that reorganizes everything: **the GPU does not fetch one float at a time.** It cannot. Global memory (HBM) and the [L2 cache](l2-cache.html) that sits in front of it are organized into fixed-size chunks called **lines** (or **transactions**, or **bursts** — same idea). On Hopper the natural line is **128 bytes**, aligned to a 128-byte boundary. And 128 bytes is exactly **32 FP32 floats** (32 × 4 B). That number is not a coincidence you should ignore — 32 floats per line, 32 threads per warp — it is the hinge the whole design turns on.

Internally that 128-byte line is four **32-byte sectors**, and the hardware can fetch just the sectors it needs. [[sn: The 128 B line splits into four 32 B sectors, and L2 tracks presence per-sector. So the real granularity of a *partial* access is 32 B, not 128 B — a strided pattern that touches one float per line still drags a full 32 B sector, wasting 28 of every 32 bytes rather than the full 124 of 128. The GPU supports 32 B, 64 B, and 128 B transactions; the scheduler picks the smallest set of them that covers the addresses.]] So the real number to track is **sectors fetched per warp-load**, but the mental model is simpler if you first think in whole 128-byte lines and refine later.

Here is the whole coalescing question, stated precisely, and it is the only question that matters for the rest of this article:

> *For one load instruction issued by one warp, how many memory transactions does the hardware have to run to satisfy all 32 addresses?*

Everything else is a consequence of that count. Fewer transactions per load = more bandwidth actually used. So let's compute that count by hand for the two cases that matter.

## The good case: 32 contiguous floats = one transaction

Suppose the 32 threads of a warp ask for 32 **contiguous, aligned** floats: thread `t` wants the address `base + t*4` bytes. So t0 wants `base`, t1 wants `base+4`, t2 wants `base+8`, … t31 wants `base+124`. Add it up: the addresses span `base` to `base+127`. That is exactly **128 bytes**. Exactly **one aligned line.** [[sn: The word "aligned" is doing real work here. If `base` is not a multiple of 128 B, those same 32 contiguous floats straddle a 128 B boundary and spill into a *second* line — so the warp needs 2 transactions instead of 1. That's why libraries pad rows to nice alignments (`cudaMallocPitch`, 128 B leading dimensions): contiguity gets you close, but only alignment gets you the clean single-line load.]]

So the hardware runs **one** transaction. It fetches 128 bytes, and all 128 bytes are wanted — every single byte lands in some thread's register. **100% of the fetched bytes are used.** This is a *fully coalesced* access, and it is the only pattern that lets you approach the **3.35 TB/s** HBM3 is rated for. One warp, one load, one line, zero waste.

Let me put a number on "zero waste," because it's the number that comes back at the end. 32 threads wanted 32 floats = 128 bytes of *useful* data. The hardware moved 128 bytes. Useful-fraction = 128/128 = **100%**.

[[fig: A hand-drawn figure titled "One warp, one load: coalesced". 32 small numbered thread boxes t0..t31 in a row across the top. Below them, ONE long horizontal bar drawn as a single 128-byte line, divided into 32 equal cells, green label under it "128 B = 32 floats, one aligned line". Each thread has a thin straight blue arrow pointing straight DOWN into its own adjacent cell, left-to-right in order (t0→cell0, t1→cell1, …, t31→cell31), so the arrows form a clean non-crossing comb. A red dimension arrow across the bar "↔ 128 B". A big green handwritten callout: "1 transaction · 128 B fetched · 128 B used · 100%". A purple code snippet floats to the side: "addr = base + threadIdx.x * 4  ← contiguous". Dashed takeaway box bottom-right: "the ONLY pattern that hits full HBM bandwidth". || When consecutive threads want consecutive floats, all 32 addresses fall in one aligned 128-byte line. One transaction serves the whole warp, and nothing is wasted.]]

## The bad case: strided access = the same reads, thrown away

Now break it, and watch the count explode. Suppose consecutive threads ask for addresses that are `N` floats apart — thread `t` wants `base + t*N*4` bytes. This is not a contrived pattern; it is what you get, by accident, the moment you walk down a *column* of a row-major matrix. We'll see exactly that in a minute.

With a stride of `N` floats and a reasonably large `N` (say `N = 4096`, a common matrix size), the 32 addresses are `base`, `base + 16384`, `base + 32768`, … each **16 KB apart**. No two of them are within 128 bytes of each other. So each address needs its **own** line.

Count it: **32 addresses, 32 different lines, up to 32 transactions** for one load instruction. [[sn: "Up to" 32 because if the stride is small enough that two threads land in the same 128 B line, they share a transaction. At stride `N` with a large `N`, no sharing happens and you hit the worst case squarely. This is why a stride of 2 or 3 hurts far less than a stride of 4096 — it's about how many *distinct lines* the 32 addresses touch, not the stride number itself.]]

And here is the cruelty. Each of those 32 transactions drags in a whole line (at minimum a 32-byte sector), of which the thread uses exactly **one** 4-byte float. So the useful-fraction, in the worst case with 128-byte lines, is 128 bytes wanted out of 32 × 128 = 4096 bytes moved = **3.1%**. Even counting the finer 32-byte sector granularity, you use 4 bytes of every 32 = **12.5%**. You paid for the whole warehouse and took home one item from each aisle.

Stop and notice how surprising this is. The FLOPs are *identical*. The bytes you *logically requested* — 128 bytes of actual data — are *identical*. The source code looks *reasonable* and produces *correct answers*. Nothing errors. And yet effective bandwidth collapses by up to **32×**, because 31 out of every 32 bytes the hardware moves get discarded on the floor. This is the trap: coalescing failures are silent. The kernel is correct. It is just secretly running the memory system at a third — or a thirtieth — of its rated speed.

[[fig: A before/after side-by-side figure titled "Same bytes requested, up to 32× the transactions", two panels. LEFT panel (A) COALESCED in orange: the clean comb from before — 32 threads, one 128 B line, blue arrows straight down, green note "1 transaction, 100% used". RIGHT panel (B) STRIDED in orange: the same 32 thread boxes t0..t31 across the top, but now draw 8 separate 128 B line-bars spread far apart across the panel, and each blue arrow jumps a big diagonal gap to land in a DIFFERENT bar; in each bar shade ONE cell pale-yellow ("used") and hatch the other 31 cells grey ("wasted, 31/32 thrown away"). Red note "stride = N floats (e.g. 4096) → each thread its own line". Red warning in a box "→ up to 32 transactions · ~3% of bytes used". A purple code snippet between the panels: "addr = base + threadIdx.x * N*4  ← the bug". Dashed takeaway box bottom: "same FLOPs, same data requested, same correct answer — up to 32× slower, silently". || The naive-vs-good comparison at the heart of the article: strided access moves the same wanted bytes but wraps each one in a whole wasted line.]]

## The path a load actually travels

Before we connect this to GEMM, one zoom-out, because it explains *why* the line — not the float — is the unit you're charged for. When a warp issues a load, the request walks a fixed path down the memory hierarchy, and **every rung of that path is priced in fixed-size lines.**

The request starts at the SM, misses the small on-chip caches, and travels down to the [L2 cache](l2-cache.html) (about **50 MiB** on Hopper, shared by all SMs), which is organized in 128-byte lines of four 32-byte sectors. If L2 misses, the request goes all the way to [HBM3](hbm-global-memory.html) (**80 GB at 3.35 TB/s**), which is *also* read in bursts, not bytes. At no point on this path can the hardware fetch "just one float." The smallest thing that moves is a sector. So the count of *lines you touch* is literally the count of *work the memory system does* — and that is why lining your 32 threads up under one line, instead of 32, is worth up to 32×.

[[fig: A memory-pyramid figure titled "You pay per line touched, not per byte used". A vertical stack of layered boxes, widest at the bottom, narrowing upward. Bottom box "HBM3" (green spec "80 GB · 3.35 TB/s · read in bursts, not bytes"). Above it a wide box "L2 cache" (green "≈50 MiB · shared by all SMs · 128 B line = 4×32 B sectors"). Above that a narrower box "L1 / SMEM per SM" (green "up to 256 KiB · fast, on-chip"). At the very top, 32 small numbered thread boxes t0..t31 labeled "one warp". Draw a single fat blue arrow running top-to-bottom labeled "coalesced: 1 line touched = 1 transaction". Beside it a thin red squiggly arrow that forks into MANY thin lines crossing every layer, labeled "strided: up to 32 lines touched = 32 transactions". An orange emphasis note pointing at the L2 box: "the unit here is the LINE, never the float". Dashed takeaway box bottom-right: "count the LINES your 32 addresses touch — that count IS your cost". || Every load walks SM → L2 → HBM, and each level moves fixed-size lines. Your cost is the number of distinct lines your warp touches, not the bytes you actually use.]]

## Why this bites GEMM specifically: row-major layout

Now we can connect the abstract stride to real matrix code, and it comes down to **memory layout**. C and CUDA store 2-D arrays in **row-major** order: the element `A[i][j]` lives at linear offset `i * N + j`. Read that carefully, because the whole GEMM story falls out of it:

- Consecutive elements **within a row** (`A[i][j]`, `A[i][j+1]`, …) are **contiguous** — 4 bytes apart.
- Consecutive elements **down a column** (`A[i][j]`, `A[i+1][j]`, …) are **`N` floats apart** — a stride of `N`.

So the entire coalescing verdict for any matrix access collapses to one question you can answer at a glance:

> **As `threadIdx.x` increments across the warp, does the *column* index of the accessed element increment, or does the *row* index?**

Column-index-varying is contiguous → coalesced → one line. Row-index-varying strides by `N` → up to 32 lines → the disaster we just computed. That's it. That single question is the lens for every load in every GEMM kernel you'll ever profile.

[[fig: A zoom-in figure titled "Row-major: the same warp, two very different loads", drawn as one small 8×8 matrix so the reader can trace by hand (scale the '32' idea down to 8 for legibility, label it "shown at 8 for clarity — real warp is 32"). Draw matrix A as an 8×8 grid, cells numbered with their LINEAR offset i*N+j (row 0: 0,1,2,...,7; row 1: 8,9,...; use N=8). Highlight ROW 0's eight cells in pale-yellow and annotate in green "walking a ROW: offsets 0,1,2,...,7 → contiguous → 1 line → COALESCED". Separately highlight COLUMN 0's eight cells (offsets 0,8,16,...,56) in red hatch and annotate in red "walking a COLUMN: offsets 0,8,16,...,56 → stride N → 8 lines → STRIDED". A blue note between them: "consecutive threads (threadIdx.x) should map to consecutive COLUMNS". Dashed takeaway box: "in row-major, coalescing = 'does the column index track threadIdx.x?'". || The by-hand version. In a row-major matrix, walking along a row is contiguous and walking down a column strides by N — so a warp coalesces only when its fast axis lands on the column index.]]

## Diagnosing the naive kernel

Look back at [GEMM kernel 1](gemm-kernel-1-naive.html). We mapped threads to output elements the obvious way:

```cpp
const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // column of C
const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // row of C
```

with a `32 × 32` block. Now apply our lens. Recall a warp is 32 threads of consecutive `tid`, and CUDA flattens with `x` fastest, so a warp shares one `threadIdx.y` and spans `threadIdx.x = 0..31`. Within one warp, then, `x` (the column) varies 0..31 and `y` (the row) is constant. So far, so good — that sounds coalesced. And for one of the two loads, it is. Let's trace both loads in the inner `k` loop, `acc += A[y*N + k] * B[k*N + x]`:

- **`B[k*N + x]`** — as we step across the warp's threads, `x` increments while `k` and `y` are shared. So we walk `B[k][x]`, `B[k][x+1]`, … along a **row** of `B`: offsets `k*N+x`, `k*N+x+1`, … contiguous. **Coalesced.** One line. Good.
- **`A[y*N + k]`** — here `y` and `k` are both constant across the warp (all 32 threads share the same row `y` and the same `k`). So all 32 threads read the **exact same address**. That's a **broadcast**, which the hardware actually handles well — it fetches once and hands the value to all 32. [[sn: A broadcast (all 32 threads → one address) and a coalesced load (32 threads → 32 adjacent addresses) are both *one transaction* — the two happy cases. The unhappy case is everything in between and beyond: 32 threads scattered across many lines. So "coalesced" is slightly loose shorthand; the real goal is "few distinct lines touched," and a broadcast is the degenerate best case of that.]]

So the naive `32×32` kernel is *not* the pure 32× worst case — its dominant `B` load already coalesces and its `A` load broadcasts. Then why is it stuck at **15 GB/s** and 1.3%? [[sn: The headline 15 GB/s figure and the 6.4× win come from the reference build in [the CUDA-MMM worklog](https://siboehm.com/articles/22/CUDA-MMM). The naive kernel's exact bandwidth depends on how the compiler schedules the two loads and the tiny amount of L2 reuse it accidentally gets; the point isn't the precise number but that a 2-D `32×32` mapping leaves the warp-to-data assignment to chance instead of choosing it.]] Because the mapping was chosen *by accident*, not designed. The 2-D indexing makes the warp-to-data assignment fragile: it depends on the block being exactly `32×32` and on the compiler laying warps out the way you hoped. Change the block shape and your coalescing quietly inverts. The problem isn't that this specific mapping is catastrophic — it's that nobody *decided* where the warp's fast axis should land. Kernel 2 decides it, on purpose.

## The remap: one line, chosen deliberately

The fix is to stop letting the 2-D block layout choose our warps for us, and instead assign the flattened thread index to `(row, col)` by hand — so that the fastest-moving axis of the warp (`threadIdx.x`) maps to the **column** of the output, which is the contiguous axis of both `C` and the dominant `B` load. We keep 1024 threads per block, but declare the block as **1-D** and compute the 2-D position ourselves:

```cpp
const uint BLOCKSIZE = 32;
// block is now 1-D: blockDim = 32*32 = 1024, threadIdx.y == 0
const uint row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const uint col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

if (row < N && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}
```

launched with a flat block:

```cpp
dim3 block(BLOCKSIZE * BLOCKSIZE);          // 1024 threads, 1-D
dim3 grid(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
sgemm_coalesced<<<grid, block>>>(N, A, B, C);
```

The entire change is those two arithmetic expressions: `col = threadIdx.x % 32` and `row = threadIdx.x / 32`. Let's verify it with our lens. A warp is `threadIdx.x = 0..31`. Feed that through the map:

- `row = threadIdx.x / 32` = `0/32, 1/32, …, 31/32` = **0 for all 32 threads**. Row is constant across the warp.
- `col = threadIdx.x % 32` = `0, 1, 2, …, 31`. Column runs **0..31**, one per thread.

So consecutive threads in a warp map to **consecutive columns** of the output. Now re-trace the two loads:

- **`B[k*N + col]`** — `col` runs 0..31, `k` is shared. Offsets `k*N+col` step by 1: contiguous, one 128-byte line, **fully coalesced** for every warp. This is the load that dominates, and it's now perfect.
- **`A[row*N + k]`** — `row` and `k` both constant across the warp → all 32 threads hit the same address → clean **broadcast**.

Every warp's `B` load is now exactly **one** 128-byte transaction with **zero** waste, by design and not by luck. And notice what we did *not* do: we did not change a single FLOP, we did not stage anything in shared memory, we did not read fewer elements. We only decided, deliberately, where the warp's fast axis lands in memory.

[[fig: A before/after side-by-side figure titled "Kernel 2: choosing where the warp lands", two labeled panels. LEFT panel "NAIVE (2-D block, by accident)": a 32×32 output tile of C; highlight one warp's cells but draw them AMBIGUOUSLY placed with a red note "warp shape depends on block layout — fragile"; show the mapping code in purple "x=...threadIdx.x ; y=...threadIdx.y". RIGHT panel "COALESCED (1-D block, by design)": the same 32×32 C tile with ONE horizontal strip of 32 cells highlighted pale-yellow, red label "one warp = 32 threads", purple code box "row = tid/32 ; col = tid%32", orange arrow on the strip "col varies 0..31 → contiguous". To the far right draw matrix B green-hatched with a blue dashed arrow from the warp strip to one contiguous ROW segment labeled "B[k][col..col+31] → 1 line, 128 B", and matrix A blue-hatched with one cell circled and a blue note "A[row][k] same addr for all 32 → broadcast". Green spec top-left "block = 1024 threads (1-D) · warp = 32". Dashed takeaway box: "consecutive threads → consecutive columns → one coalesced B load per warp". || The whole optimization in one picture: stop letting the 2-D block choose your warps, and deliberately land the warp's fast axis on B's contiguous column axis.]]

## The profile, and the bold number

Now we do what the worklog always does: form the hypothesis (coalescing the `B` load will collapse transactions-per-load and free up bandwidth), make the change, and let the profiler judge.

Compile and inspect the SASS: the global load servicing `B` compiles to an `LDG.E` that Nsight Compute reports as one sector-efficient transaction per warp instead of a scatter. The metric to watch in the memory workload section is `l1tex__t_sectors_per_request` — sectors fetched per load request. The naive mapping bloats this above the ideal; the remapped kernel drives the `B` access down toward the floor of **4 sectors** (one 128-byte line) per warp. And the top-line number the profiler cares about moves exactly as predicted: global-memory throughput jumps from **15 GB/s to 110 GB/s** — about **7.3×** more bandwidth actually used.

[[fig: A SASS-plus-barchart figure titled "What the profiler sees", two columns. LEFT column: a handwritten SASS listing in mono-style hand lettering, two labeled blocks. Top block "NAIVE" shows a load line "LDG.E R4, [R6]" with a red margin note "sectors/req: high — the B load scatters across lines"; bottom block "COALESCED" shows "LDG.E R4, [R6]" with a green margin note "sectors/req → 4 (one 128 B line) · broadcast on A". RIGHT column: a hand-drawn bar chart with two pairs of bars. Pair 1 labeled "GMEM throughput": short grey bar "15 GB/s" next to tall green bar "110 GB/s", tiny note "of 3350 GB/s peak". Pair 2 labeled "% of cuBLAS": short grey bar "1.3%" next to taller orange bar "8.5%". A blue dashed arrow connects the COALESCED SASS block to the tall bars. Dashed takeaway box bottom: "same FLOPs, same bytes requested — every transaction now fully used → 6.4× faster". || The change is invisible in the FLOP count and loud everywhere in the memory metrics: bandwidth used goes 15→110 GB/s and speed goes 1.3%→8.5% of cuBLAS.]]

The result: **8.5% of cuBLAS** (about **1986 GFLOP/s**), up from 1.3%. That is a **6.4× speedup** from a change that touched two arithmetic expressions and removed *not one* floating-point operation and *not one* logically-required byte. We simply stopped throwing away most of every memory transaction. When I first internalized this, it genuinely bothered me — 6.4× for free, hidden behind `%` versus `/`? — until I did the by-hand count above and saw that the naive version was, on its worst loads, moving up to 32× the lines it needed. The speedup isn't magic. It's the wasted lines we stopped fetching.

## Being honest: coalescing is the floor, not the ceiling

It's worth stating plainly what coalescing did **not** fix, because the next kernel exists precisely to fix it.

We are *still* reading `O(N³)` floats from HBM to do `O(N³)` FLOPs. The [arithmetic intensity](arithmetic-intensity.html) is still about **1 FLOP per element loaded** — hundreds of times below the H100's ridge point of ~295 FLOPs/byte from [the three regimes](the-three-regimes.html). Coalescing made each transaction *fully useful*, but it did nothing about the fact that we issue *far too many* of them: the element `A[m][k]` is still re-fetched from global memory by every one of the `N` threads that need it. We fixed the *efficiency* of each load; we did nothing about the *number* of loads.

So the ceiling here is low by construction — single digits of cuBLAS. And that ordering is deliberate: coalescing is the tax you pay *before* the next optimization is even worth attempting. There is no point staging tiles into fast on-chip memory if the loads that *fill* those tiles are themselves strided and wasteful — you'd just be caching garbage-efficiency reads. With the access pattern fixed, the obvious lever is to **stop reading the same data over and over**, by staging blocks of `A` and `B` into [shared memory](shared-memory-l1.html) and reusing them across a whole block of threads. That is [kernel 3](gemm-kernel-3-shared-memory.html), where we finally attack *reuse*, and where the real climb begins — from single-digit percentages up toward the **68.7%** the 2-D block-tiled kernel eventually reaches, and beyond.

[[fig: A ladder/roadmap figure titled "Where coalescing sits on the climb". Draw a staircase of steps rising left-to-right, each a box with a % label. Step 1 (grey, low) "Naive — 1.3%" with a red note "memory-bound, transactions wasted". Step 2 (orange, highlighted, slightly higher) "Coalesced — 8.5%" with a green note "every transaction fully used". Step 3 (blue, higher) "Shared memory — reuse tiles" with a blue note "stop re-reading the same data". Step 4 (blue, higher still) "2-D block-tiling — 68.7%". A dashed arrow labeled orange "next" points from step 2 to step 3. A recurring mental-model banner across the top in blue: "coalescing fixes HOW each load is served · tiling fixes HOW MANY loads we issue". Dashed takeaway box bottom-right: "coalescing is the floor you must reach before caching is worth it". || Coalescing is step two of a long ladder. It fixes the efficiency of each load so that the next optimization — reusing data in shared memory — is even worth doing.]]

The pattern to carry forward is the one this whole change embodies, and it will govern every kernel from here on: **the memory system rewards you for lining threads up under contiguous, aligned addresses, and punishes you — silently, without an error — for anything else.** So the first question you ask of any new kernel is never "how many FLOPs" and never even "how many bytes." It is: *where does the warp's fast axis land in memory?* Get that right, and every fetch pays full freight. Get it wrong, and no amount of cleverness downstream will save you — you'll just be efficiently orchestrating loads that were wasting 90% of their bandwidth the whole time. Coalesce first. Earn the right to be clever second.
