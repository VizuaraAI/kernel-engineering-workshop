Here is a question that sounds too simple to be interesting, but turns out to decide almost everything about how fast your kernel runs: **when you write `float x = A[i];` inside a CUDA kernel, where does `A` actually live?**

You might answer "in memory." But a GPU does not have *memory* the way a laptop has memory — one big pool you allocate from and forget about. A GPU has *memories*, plural. Half a dozen distinct address spaces, each with its own latency, its own bandwidth, its own scope (who is allowed to see it), and its own rules. And the single most consequential fact about that line of code is *which* of those spaces `A` lives in — because that one choice can swing your kernel's throughput by **two orders of magnitude**.

That is not an exaggeration. The whole GEMM optimization ladder on this site — from the [naive kernel that reaches 1.3% of cuBLAS](gemm-kernel-1-naive.html) to the [warptiled one that reaches 93.7%](gemm-kernel-8-warptiling.html) — is, underneath all the tiling tricks and vectorization, **one long argument about moving data out of the slow spaces into the fast ones and keeping it there**. If you understand the memory spaces, the entire ladder stops feeling like a bag of tricks and starts feeling inevitable.

So before we write another kernel, we need the map. This article answers three questions, in order. **What are the memory spaces?** **Where do they physically live** (this is where it gets surprising)? And **how does that map drive every optimization** you will ever do? A newcomer can start right here — I will build the whole picture from the ground up.

## First, the only two memories that physically exist

Let me give you the honest physical picture before any CUDA vocabulary, because the vocabulary hides it.

Strip away the names, and an H100 has exactly **two** kinds of memory that matter:

1. **Off-chip DRAM.** This is the 80 GB of **High-Bandwidth Memory** (HBM3) — stacks of DRAM chips sitting *next to* the GPU die on the same package, connected by a very wide bus. It is large and comparatively slow. A read takes roughly **500 clock cycles** to come back.[[sn: "500 cycles" is a round number for the *latency* — the time from asking for a byte to getting it. Bandwidth is a separate axis: HBM3 delivers ~3.35 TB/s in aggregate. High latency and high bandwidth coexist because the memory system is deeply pipelined — many requests are in flight at once. This is exactly why GPUs oversubscribe threads: while one warp waits 500 cycles, others run.]]

2. **On-chip SRAM.** This is small, fast memory etched into the GPU die itself, right next to the compute units. A read can come back in as little as **one cycle**. There is not much of it — a few hundred kilobytes per **Streaming Multiprocessor** (SM), the GPU's basic compute tile.

That is the whole hardware story. Two memories: far-and-vast, or near-and-tiny. **Every "memory space" CUDA gives you is a naming convention layered on top of those two physical things.**[[sn: There is arguably a third physical tier: the register file is also SRAM, but it is a fundamentally different structure from a cache — a multi-ported, statically-addressed array wired directly into the datapath. Calling it "just SRAM" undersells how special it is. We give it its own row below.]]

Hold onto this, because it is the source of every surprise in this article. Two of the six logical spaces have names that lie about where they physically are.

[[fig: A hand-drawn intuition figure titled "Two memories, wearing six hats". LEFT half: a large blue-hatched box labeled "OFF-CHIP DRAM (HBM3)", green note "80 GB · ~500 cycles · 3.35 TB/s", drawn as fat stacked chips beside a die outline. RIGHT half: a small orange-outlined box labeled "ON-CHIP SRAM", green note "few hundred KB/SM · ~1-30 cycles", drawn tucked inside the die. Above them, six small paper luggage-tags on strings hang down: "global", "local", "constant", "texture" pointing (dashed blue arrows) toward the DRAM box; "shared", "register" pointing toward the SRAM box. One orange sticky by the "local" tag: "wait — local points to DRAM?!". Dashed takeaway box bottom: "6 logical spaces, 2 physical memories. The names are hats, not homes." || The mental model to keep for the whole article. There are only two real memories. The six CUDA "spaces" are labels — and two of the labels are misleading on purpose.]]

## An analogy to carry the whole way through

Before the technicalities, here is the picture I keep in my head. It will do a lot of work.

Imagine you are cooking at a stove.

- The **giant walk-in pantry** down the hall is **global memory** (HBM). Everything is in there. But every trip costs you a long walk.
- The **shared countertop** by your stove is **shared memory** (on-chip SRAM). You stage the ingredients for the current dish there so you and your fellow cooks stop running to the pantry.
- The **knife in your hand** and the **pinch of salt between your fingers** are your **registers** — whatever you are working on *this instant*. Instant access, but you can only hold a few things.
- **Constant memory** is the **recipe card taped above the stove**: read-only, and every cook glances at the *same* line at the same time.

The entire craft of kernel engineering is: **stop walking to the pantry.** Move your working ingredients onto the countertop, keep the active one in your hand, and touch the pantry as few times as you possibly can. Every optimization in the GEMM ladder is a version of this one move. Keep the kitchen in mind; I will point back to it.

[[fig: A hand-drawn kitchen analogy figure titled "The kernel engineer's kitchen". A simple side-view scene, hand-inked. Far LEFT, down a long hallway (drawn as a receding corridor), a big box "PANTRY = GLOBAL MEM (HBM)", green note "vast · long walk · ~500 cycles". A little cook figure walks the long hallway with a red dashed path labeled "expensive round-trip". CENTER, at a stove, a "COUNTERTOP = SHARED MEM", blue note "stage ingredients once, all cooks reuse", yellow-hatch fill. In the cook's hand a tiny "KNIFE + PINCH OF SALT = REGISTERS", orange note "what I'm using RIGHT NOW · ~1 cycle". Above the stove a taped card "RECIPE = CONSTANT MEM", green note "read-only · everyone reads the same line". Numbered circles (1) pantry (2) countertop (3) hand (4) recipe. Dashed takeaway box: "the whole craft = stop walking to the pantry." || The mental model as a kitchen. Global is the far pantry; shared is the countertop; registers are your hands; constant is the recipe on the wall. Optimizing = fewer pantry trips.]]

## The six spaces, one at a time

Now let me walk each of the six spaces CUDA exposes — **global**, **shared**, **register**, **local**, **constant**, **texture** — and say plainly which of the two physical memories it lives in.

**Global memory** *is* HBM. This is the big pool. `cudaMalloc` hands you a pointer into it; every block and every thread in the grid can read and write it; it persists for the life of the allocation. It is the space your input data arrives in and the space your answer must be written back to — you cannot avoid it. It is also the slowest (hundreds of cycles) and the one whose **3.35 TB/s** aggregate bandwidth you will spend your career fighting to saturate. Because it is the countertop's opposite — the far pantry — it gets its own deep-dive in [HBM and global memory](hbm-global-memory.html), and reads from it go through the [L2 cache](l2-cache.html) on the way.

**Shared memory** is on-chip SRAM, private to a single thread block, and here is the first genuinely important fact: it is *physically the same array* as the **L1 data cache**. On H100 the two share a **256 KiB** pool per SM, and you can carve out up to **228 KiB** of it as explicitly-managed shared memory, leaving the rest as L1.[[sn: Those 228 KiB are not free-for-the-asking. You opt in past the 48 KiB default with `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`, and asking for the max forces a single block per SM, which can crush your occupancy. Also, the pool cannot be handed out to shared memory *entirely* — a slice (roughly a KiB per block) is reserved for the driver. Whatever you do not claim stays available as L1.]] Its latency is roughly **20–30 cycles** — an order of magnitude below global — and its bandwidth is enormous, around **31 TB/s**, because it is split into **32 banks** you can read in parallel. This is the workhorse of tiling, the countertop in our kitchen, and it gets its own article: [shared memory and L1](shared-memory-l1.html).

**Registers** are the fastest storage on the chip — roughly **one cycle**, no addressing, wired straight into the arithmetic units, running at something like **124 TB/s**. Each SM has a **256 KB register file** (that is `65536` 32-bit registers), partitioned among all its resident threads. A single thread may use at most **255** of them. Registers are private to one thread and vanish when the thread exits — the knife in your hand. See [the register file](register-file.html).[[sn: "Private to one thread" has one blessed exception: warp shuffle instructions (`__shfl_sync`) let threads in the same warp read each other's registers directly, without going through shared memory. It is the fastest way to move a value between lanes, and FlashAttention leans on it for the softmax reduction.]]

**Local memory** is the trap. It gets its own section below, because it is the number-one source of mystery slowdowns. Spoiler from the physical picture above: despite the name, it lives in HBM.

**Constant memory** is a small (**64 KB**) read-only region of global memory, but backed by a dedicated per-SM **constant cache**. It is optimized for exactly one pattern — every thread in a warp reading the *same* address — and for that pattern it is startlingly fast. The recipe card on the wall.

**Texture memory** is another read-only view onto global memory, routed through the **texture cache** with hardware interpolation and 2D-spatial locality — a holdover from graphics that still earns its keep in a few niches.

## The table you should memorize

Here is the whole map in one grid. Read it as one sentence: **the closer to the thread, the faster and smaller and more private.**

| Space | Physical location | Scope | Latency | R/W | Cached |
|---|---|---|---|---|---|
| Register | SM register file (SRAM) | one thread | ~1 cycle | R/W | n/a |
| Shared | SM L1 SRAM (up to 228 KiB) | one block | ~20–30 cyc | R/W | n/a (is the cache) |
| Constant | HBM + constant cache | grid (read-only) | ~1 cyc on hit | R | yes, broadcast |
| Texture | HBM + texture cache | grid (read-only) | tens on hit | R | yes, spatial |
| Local | HBM (global) | one thread | hundreds | R/W | L1/L2 |
| Global | HBM (DRAM) | whole grid | hundreds | R/W | L2 (~50 MiB) |

Two rows in that table are lies of a sort — the two where the *logical* name and the *physical* reality diverge. **Local** memory says "local" but lives in the farthest, slowest DRAM. **Constant** memory says "global-sized and slow" but can beat shared memory for the right access pattern. These two surprises are worth their own sections, so let me take them one at a time. If you understand *why* these two are surprising, you understand the whole map.

[[fig: A hand-drawn memory pyramid titled "The CUDA memory spaces (H100)". Four stacked bands, narrow-top to wide-bottom. TOP band (orange fill) "REGISTERS", green note "256 KB/SM · 65536 × 32-bit · ~1 cycle · ~124 TB/s · per-thread". SECOND band (yellow hatch) split into "SHARED MEM" and "L1 CACHE" joined by a purple brace "same 256 KiB SRAM · up to 228 KiB SMEM · 32 banks · ~31 TB/s · per-block"; a small orange sticky beside it "CONSTANT $ + TEXTURE $ caches live on-chip too (read-only)". THIRD band "L2 CACHE", green note "~50 MiB · shared by all SMs · 128 B lines". BOTTOM band (widest, blue hatch) "GLOBAL MEMORY (HBM3)", green note "80 GB · 3.35 TB/s · ~500 cycles · grid scope". Left red arrow top-to-bottom "latency ↑, capacity ↑"; right red arrow bottom-to-top "bandwidth ↑, scope ↓". A red dashed ghost-arrow drops from a small "LOCAL" tag near the top ALL the way down into the HBM band, labeled "local *pretends* to be near — it's really here". Dashed takeaway box: "6 logical spaces, 2 physical memories: on-chip SRAM vs off-chip HBM." || The memory pyramid. Fast and small at the top, slow and vast at the bottom. The engineer's job is to move the working set up — and to notice when "local" quietly drops it to the bottom.]]

## Local memory is really global memory

This is the single most common source of mystery slowdowns for people learning kernels, so let me be very precise.

**Local memory** is per-thread private storage. But read the word "local" carefully: it describes the *scope* — who can see it — not the *location*. Physically, local memory is carved out of global HBM. **A local-memory access is a global-memory access wearing a friendlier name, and it costs exactly what a global access costs: hundreds of cycles.**[[sn: It is at least cached in L1 and L2 on the way down, so a hot local variable that stays resident in cache is not as catastrophic as a fully uncached HBM round-trip. But you are now betting on the cache to rescue you from a problem you could have avoided entirely — and under register pressure the cache is usually already busy.]]

Here is the thing that trips everyone up: **you never explicitly ask for local memory.** There is no `__local__` keyword you type by mistake. The compiler decides to use it, silently, in two situations.

The first is **register spilling.** Registers are a fixed budget — 255 per thread, and often far fewer if you want good occupancy. If your kernel needs more live values at once than that budget allows, the compiler has nowhere to put the overflow *on-chip*, so it *spills* the extra values to local memory. Your fast registers quietly become slow HBM. In the kitchen: you ran out of hands, so you keep setting things down and walking back to the pantry to fetch them again.

The second is **indexed private arrays.** This one is subtle. Registers have no addresses — you cannot compute `register[i]` at runtime, because there is no `i`-th register to index. So if you declare a small private array and index it with a value the compiler *cannot* figure out at compile time, it cannot keep that array in registers. The whole array falls to local memory.

```cpp
__global__ void danger(const float* in, float* out, int n) {
    float acc[32];                 // wants to be registers...
    for (int i = 0; i < n; ++i)    // ...but n is a runtime value,
        acc[i % 32] += in[i];      // so acc[] is indexed dynamically ->
    // acc[] gets SPILLED to LOCAL memory = HBM. Every access is a global load.
    out[threadIdx.x] = acc[0];
}
```

Let me put a number on how bad this is, from basics. Say that inner line runs `n = 1024` times. Ideally `acc[i % 32]` is a register: **~1 cycle** per access. Spilled to local memory, each access is a load and a store into HBM at roughly **500 cycles** each. That is a **~500× penalty per access**, hidden behind an innocent-looking `+=`. The arithmetic is trivial; the memory is the entire cost. This is what "a variable you thought was free is the slowest thing in your kernel" means, concretely.

How do you *catch* it? Two tells, and you should learn to look for both reflexively.

Ask `nvcc` for `-Xptxas -v` and it prints, per kernel, a line like `N bytes stack frame`. **A non-zero stack frame *is* your local-memory footprint** — the compiler is telling you, in plain text, that it had to spill. Then inspect the SASS (the actual machine code — see [PTX vs SASS](ptx-vs-sass.html)): you will find `LDL` and `STL` instructions (load-local, store-local) sitting exactly where you expected pure arithmetic. Nsight Compute counts that local traffic against your HBM budget so you can see it eating your bandwidth.

The fix is almost always one of three moves: **reduce register pressure** so nothing spills; **unroll loops** (with `#pragma unroll`) so array indices become compile-time constants and the array can live in registers; or **shrink the tile** so the whole working set fits in the budget. All three are versions of "keep the ingredients in your hands, don't set them down."

[[fig: A two-panel before/after SASS-plus-diagram titled "The spill". LEFT panel "(A) what you wrote": a purple code block "float acc[32];  acc[i%32] += ...", green note "you think: 32 registers · ~1 cycle each". A blue dashed arrow curves rightward labeled "compiler:". RIGHT panel "(B) what the compiler did": top, a small orange box "REGISTER FILE — FULL" with tiny slots hatched solid and a red X; a long blue dashed arrow labeled "SPILL ↓" drops to a wide blue-hatched box "LOCAL MEMORY = HBM" holding a purple SASS listing "STL [R1], R4  // store local / LDL R4, [R1]  // load local"; a red note "~500 cycles each — a global round-trip → ~500× slower". Numbered circles (1) register file full, (2) spill arrow, (3) HBM box. Dashed takeaway box: "non-zero 'stack frame' in -Xptxas -v = you are in HBM. Fix: fewer live regs, #pragma unroll, smaller tile." || Register spilling, side by side. What you wrote (left) versus what the hardware did (right). Exceed the register budget and your "local" array silently becomes the slowest memory on the chip.]]

## Constant memory and the broadcast trick

Now the opposite surprise. Constant memory looks unpromising on paper — it lives in HBM, it is tiny (64 KB), it is read-only. And yet, used correctly, it behaves like the *fastest* thing you have. Why?

You declare it at file scope with `__constant__`, fill it from the host before the launch, and the device may only read it. What makes it special is a feature called **broadcast**, and to see why broadcast matters we have to think about what a warp actually is.

A **warp** is a group of 32 threads that execute the same instruction in lockstep (see [threads, warps, blocks, grids](threads-warps-blocks-grids.html)). When those 32 threads all hit a memory instruction at once, the hardware has to service 32 requests. Normally that means 32 addresses to look up. But suppose all 32 threads read the *exact same* constant address — say every thread wants `kFilter[j]` for the same `j`. The constant cache notices they are all asking for one value, **fetches it once, and broadcasts that single value to all 32 lanes** in one shot. One fetch, 32 answers, register-fast.

That is the recipe card taped above the stove: every cook glances at the same line at the same time, so one card serves the whole kitchen. It is exactly the right structure for coefficients that every thread needs identically — a scaling factor, filter weights, a shared dimension. Put those in `__constant__` and you get register-speed reads for free.

But there is a sharp flip side, and it is the whole reason constant memory is a *specialized* tool and not a default. If the 32 threads read *different* constant addresses, the cache cannot broadcast — there is no single value to hand out. It **serializes**: it services distinct address 1, then address 2, then address 3, one after another. A fully-divergent warp, where all 32 lanes want different addresses, costs up to **32× a single read.** Constant memory rewards uniform access and punishes divergent access, hard. That one property is the entire decision rule: *use it only when the whole warp reads the same address.*

[[fig: A hand-drawn two-panel before/after figure titled "The broadcast". Panel (A) labeled "uniform read — FAST": a warp drawn as 32 small circles (lanes 0..31) in a green-hatched row, each with a blue dashed arrow all converging onto ONE cell of a box labeled "CONSTANT $" pointing at address "kFilter[j]". A single orange lightning bolt and a green note "1 fetch → broadcast to all 32 lanes · ~1 cycle". Panel (B) labeled "divergent read — SLOW": the same 32 lanes but each blue arrow points at a DIFFERENT cell of the "CONSTANT $" box, fanning out. A red note "32 distinct addresses → serialized, one at a time · up to 32× slower". A purple code sticky spanning both: "s += in[t+j] * kFilter[j];  // same j for all lanes ✓". Numbered circles (1) on panel A, (2) on panel B. Dashed takeaway box: "constant memory is register-fast ONLY when the whole warp reads the same address." || The broadcast, before and after. One shared address for the whole warp is a single cheap fetch (A); 32 different addresses serialize into 32 fetches (B). This is the only thing you need to remember about constant memory.]]

```cpp
__constant__ float kFilter[64];   // fits in the 64 KB constant space

__global__ void conv(const float* in, float* out) {
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; ++j)
        s += in[threadIdx.x + j] * kFilter[j];  // every lane reads kFilter[j]
    // all 32 lanes hit the SAME address each iteration -> one broadcast, ~1 cycle
    out[threadIdx.x] = s;
}
```

Notice what makes this work: the index `j` is the *same for every thread in the warp*. The thread index only varies the `in[...]` part, which is ordinary global memory. So `kFilter[j]` is a textbook broadcast, and this convolution gets its filter weights at register speed while paying nothing extra.[[sn: If you flipped the roles — indexed the constant array by `threadIdx.x` so each lane read a different weight — you would hit the serialized path and it would be *slower* than just putting the array in shared memory. The tool is only fast for its one intended pattern. That is a recurring theme in kernel work: specialized hardware paths are gifts, but only if your access shape matches them.]]

## Texture memory, briefly

**Texture memory** is the graphics inheritance. It is another read-only path into global memory, routed through the **texture cache**, which is tuned not for 1D contiguity but for **2D spatial locality** — the idea that if a thread reads pixel `(x, y)`, its neighbors will soon want `(x±1, y±1)`. It also throws in free boundary clamping and hardware linear interpolation, both leftovers from sampling images.

For GEMM and most deep-learning kernels you will rarely reach for it — your access patterns are 1D-contiguous rows and columns, which the ordinary path handles fine. Texture memory earns its keep in stencils, image resampling, and irregular 2D gathers, where the texture cache's 2D-locality model actually matches how you touch memory. Know it exists; move on.

## Where this is going: L2 and distributed shared memory

Two spaces sit *between* the six classic ones, and they matter more every hardware generation, so I want to name them rather than leave them as a mystery when you meet them later.

The **L2 cache** is a single ~**50 MiB** pool of SRAM shared by *all* the SMs, sitting between them and HBM (details in [the L2 cache](l2-cache.html)). Every global and local access passes through it. You do not manage it directly — it is a hardware cache — but it is the reason a value re-read soon after its first read can come back far faster than the "500 cycles" headline suggests. In our kitchen it is a shared prep-station in the middle of the room: not your countertop, but much closer than the pantry.

The newer one is **distributed shared memory** (DSMEM), introduced on Hopper. It lets a thread block read the shared memory of *another* block in the same cluster, over a fast on-chip network, without bouncing through global memory. It is slower than your own block's shared memory but far faster than HBM — a way for neighboring cooks to reach across to each other's countertops. It shows up in the newest GEMM and attention kernels ([Hopper TMA](hopper-tma.html) leans on the same cluster machinery), and it is the reason the "one block, one countertop" story is starting to blur at the top of the ladder.

Put the whole hierarchy in one line and it reads like a relay: data flows **HBM → L2 → shared → registers** on the way in, computes at the top, and flows **registers → global** on the way out. The good kernels do the leftward-to-rightward move *once per tile* and then stream compute; the naive kernel does the full HBM round-trip *per operand, per thread*.

[[fig: A hand-drawn horizontal pipeline/dataflow figure titled "The relay: data flowing up the hierarchy". Five boxes left-to-right connected by fat blue arrows: (1) blue-hatch "GLOBAL / HBM3" green note "80 GB · ~500 cyc"; arrow labeled "L2 on the way"; (2) grey "L2 CACHE" green note "~50 MiB · shared by all SMs"; arrow labeled "cp.async / TMA (bulk, per tile)"; (3) yellow-hatch "SHARED MEM" green note "~20-30 cyc · per block · 32 banks"; arrow labeled "load fragment"; (4) orange "REGISTERS" green note "~1 cyc · per thread"; then a compute burst icon "c += a*b" and a single thin arrow curving all the way back to the GLOBAL box labeled in red "write result — ONCE, at the end". Above the whole relay, two contrasting timeline bars: a green bar "GOOD: 1 HBM trip per TILE, then stream compute" and a red bar (much longer, striped) "NAIVE: 1 HBM trip per OPERAND per THREAD". Numbered circles (1)-(4) along the relay. Dashed takeaway box: "move it up once, reuse many times, write down once." || The memory hierarchy as a relay/pipeline. Data climbs HBM → L2 → shared → registers, computes at the top, and is written back once. Good kernels pay the long climb per tile (green); naive kernels pay it per operand, per thread (red).]]

## Zoom in: one thread, one FMA, and the whole hierarchy in miniature

Let me make all of this concrete by shrinking the picture down to a single thread doing a single multiply-add — the atom of GEMM — and watching every memory space light up.

The thread's job is `c += a * b`. Walk it:

- The operands `a` and `b` have to be in **registers** for the arithmetic unit to touch them. Nothing computes out of any other space. So the real question is only ever: *how did `a` and `b` get into registers, and how expensive was that trip?*
- In the naive kernel, `a` came straight from **global memory** — a ~500-cycle walk to the pantry — and so did `b`, and the very next thread over fetched almost the same row of `A` again, and the one after that again. Same ingredients, fetched from the far pantry over and over.
- In the tiled kernel, the block first cooperatively stages a tile of `A` and a tile of `B` into **shared memory** — one set of pantry trips, shared by everyone. Now each thread's `a` and `b` come from the **countertop** at ~20–30 cycles, and each byte gets *reused* by many threads. The accumulator `c` lives in a **register** the whole time and only touches global once, at the very end, when the answer is written back.

That is the entire ladder in one FMA. Same three FLOPs either way. The only thing that changed is *which memory space each operand came from* — and that is the difference between 1.3% and 93.7% of cuBLAS.

[[fig: A hand-drawn zoom-in figure titled "One FMA, all the spaces". Left: the big pyramid shrunk to a thumbnail with a magnifier icon over one thread. Right: the magnified view — a single thread box in the center performing "c += a * b" (purple code, big). Three inputs feed it: "a" arrives via a blue dashed arrow from a yellow-hatch "SHARED (tile of A)" box, green note "~20-30 cyc, reused by 8+ threads"; "b" arrives from a yellow-hatch "SHARED (tile of B)" box likewise; "c" sits in an orange "REGISTER (accumulator)" box, note "stays here the whole loop · ~1 cycle". Below, a faded red ghost-path labeled "the naive way" shows a and b coming instead straight from a blue-hatch "GLOBAL (HBM)" box with a red "~500 cyc EACH, re-fetched per thread". A green by-hand tally: "tiled: 1 HBM trip per tile, shared by 64 threads → ~1/64 the pantry walks". Numbered circles (1) load tile to shared (2) read a,b from shared to regs (3) accumulate in reg (4) write c to global once. Dashed takeaway box: "same 3 FLOPs. only the SOURCE of a and b changed. that's the whole ladder." || The whole hierarchy in one multiply-add. Zoomed to a single thread, the optimization is literally just: change where a and b come from. Shared-and-register (top path) versus global-every-time (faded bottom path).]]

## The mental map, and how it drives the ladder

Step back, and the six spaces collapse into one short decision procedure. Ask, for each piece of your working set, one question: *how high up the pyramid can this live?*

- **Global memory** is where your data starts and where the answer must end up. You cannot avoid it — only minimize how many times you touch it. The far pantry.
- **Registers** hold the values a thread is computing on *right now*. The knife in your hand.
- **Shared memory** holds the tile a whole block cooperates on, so `N` threads reuse each byte instead of each re-fetching it from global. *This is the entire idea of tiling.* The countertop. It is why [kernel 3 jumps to 12.8%](gemm-kernel-3-shared-memory.html) the moment it introduces a shared-memory tile, and why the tiled kernels past it climb into the 60s and 70s.
- **Constant** memory holds the small, uniform, read-only coefficients every thread shares. The recipe on the wall.
- **Texture** memory is for the day your access pattern is genuinely 2D.
- **Local** memory is what you get when you weren't paying attention — a spill, dropping your ingredient back into the far pantry.

[[fig: A hand-drawn Excalidraw decision-flow diagram on pure white, titled "Which memory space?". A black rounded start box "I need to store some data" at the left; five long thin curved dashed arrows fan out to five outcome boxes, each arrow carrying a BLUE handwritten decision predicate. Branch 1, blue "one thread, computing right now?" -> orange-fill box "REGISTERS", green spec note "~1 cycle · per-thread". Branch 2, blue "a block reuses it many times?" -> yellow-hatch box "SHARED MEM", blue mechanism note "stage a tile once, reuse — this IS tiling". Branch 3, blue "read-only, same value across the warp?" -> plain box "CONSTANT", orange emphasis note "broadcast = register speed!". Branch 4, blue "read-only, 2D neighbour pattern?" -> plain box "TEXTURE", green note "spatial cache". Branch 5, blue "everything else / the final answer" -> blue-hatch box "GLOBAL (HBM3)", green spec note "80 GB · 3.35 TB/s". A red-outlined warning cloud off to the side, red handwriting: "LOCAL — you never choose this. spills land here = HBM. avoid!". Numbered circles (1)(2)(3)(4)(5) marking each branch. Dashed takeaway box: "every optimization = move the working set UP the pyramid and keep it there." || The decision procedure. Kernel engineering as one flowchart: what is this data, and how high up the pyramid can it live?]]

The predict-then-measure habit from [the three regimes](the-three-regimes.html) plugs in directly here. Before you ever run the profiler, name the space each piece of your working set lives in, and ask whether it could live higher. When the naive GEMM kernel does `2N` FLOPs but issues `2N` global loads per thread, the diagnosis is not "the math is slow" — the math is nothing. It is a **memory-space failure**: the row of `A` and the column of `B` are being re-fetched from HBM by *every* thread, when they could be staged once in shared memory and reused by the whole block. Every cook running to the pantry for the same onion.

That single reframing — *stop touching global, stage it on-chip* — is the engine behind almost every rung of the ladder that follows, and it is why the memory spaces are the first thing to learn, before any specific kernel. Next we make the on-chip pool concrete in [shared memory and L1](shared-memory-l1.html), where those 228 KiB and 32 banks stop being trivia and become the thing you actually tune. And once you are moving data through shared memory in bulk, the very next question — *how do you get it there without wasting bandwidth?* — is [memory coalescing](memory-coalescing.html).
