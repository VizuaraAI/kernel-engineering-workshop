A GPU does not have "memory" the way a laptop has memory. It has *memories* — half a dozen distinct address spaces, each with its own latency, its own bandwidth, its own scope, and its own rules about who is allowed to see what. When you write `float x = A[i];` in a CUDA kernel, the single most consequential thing about that line is which of those spaces `A` lives in, because that choice can swing your kernel's throughput by two orders of magnitude. The whole GEMM ladder — from the naive [kernel that hits 1.3% of cuBLAS](gemm-kernel-1-naive.html) to the warptiled one that hits 93.7% — is, underneath, one long argument about moving data from the slow spaces into the fast ones and keeping it there.

So before we write another kernel, we need the map. This article walks the six memory spaces CUDA exposes — **global**, **shared**, **local**, **constant**, **register**, and **texture** — and builds the model of which to reach for and when. The non-obvious idea to leave with is that these six *logical* spaces do not map onto six *physical* memories. Several are the same silicon wearing different hats, and one — local memory — is a trap.

## The six spaces, and where they actually live

Let me give the honest physical picture first, because the CUDA vocabulary hides it. The H100 has exactly two kinds of memory that matter: **off-chip DRAM** — the 80 GB of **High-Bandwidth Memory** (HBM3) stacked next to the die — and **on-chip SRAM**, small, close, and fast. Everything else is a naming convention layered on those two.[[sn: There is a third tier people forget: the register file is technically SRAM too, but it is a fundamentally different structure — a multi-ported, statically-addressed array wired directly into the datapath, not a cache. Calling it "just SRAM" undersells how special it is.]]

- **Global memory** is HBM. It is the big pool: `cudaMalloc` hands you a pointer into it, every block and every thread in the grid can read and write it, and it persists for the life of the allocation. It is also the *slowest* space — hundreds of cycles of latency — and the one whose bandwidth (**3.35 TB/s** aggregate on H100) you will spend most of your career fighting to saturate. We cover it in depth in [HBM and global memory](hbm-global-memory.html).

- **Shared memory** is on-chip SRAM, private to a single thread block, physically the same array as the **L1 data cache**. On H100 the two share a **256 KiB** pool per **Streaming Multiprocessor** (SM), and you can carve out up to **228 KiB** of it as explicitly-managed shared memory.[[sn: Those 228 KiB are not exactly free either — you must opt in past the 48 KiB default with `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`, and asking for the max forces one block per SM, which can cost you occupancy. The pool can't be handed out to shared memory in full: a small slice (roughly a KiB per block) is held back for system/driver use, and whatever you don't claim as shared stays available as L1.]] Latency is roughly an order of magnitude below global; bandwidth is enormous because it is banked into **32 banks** you can hit in parallel. This is the workhorse of tiling, and it gets its own article: [shared memory and L1](shared-memory-l1.html).

- **Registers** are the fastest storage on the chip — single-cycle, no addressing, wired straight into the arithmetic units. Each SM has a **256 KB register file** (`65536` 32-bit registers), partitioned among its resident threads; a thread can use at most **255**. Registers are private to one thread and vanish when it exits. See [the register file](register-file.html).

- **Local memory** is the trap, and it gets its own section below. Spoiler: it is not local and it is not fast.

- **Constant memory** is a small (64 KB) read-only region of global memory backed by a dedicated per-SM **constant cache**, optimized for one specific pattern: every thread in a warp reading the *same* address.

- **Texture memory** is another read-only view onto global memory, routed through the texture cache with hardware interpolation and 2D-spatial locality — a holdover from graphics that still has niche uses.

[[fig: A hand-drawn memory pyramid titled "The CUDA memory spaces (H100)". Four stacked bands, narrow-top to wide-bottom. TOP band (orange fill) "REGISTERS", green note "256 KB/SM · 65536 × 32-bit · ~1 cycle · per-thread". SECOND band (yellow hatch) split into "SHARED MEM" and "L1 CACHE" joined by a purple brace "same 256 KiB SRAM · up to 228 KiB SMEM · 32 banks · per-block"; a small orange sticky beside it "CONSTANT $ + TEXTURE $ live here too (on-chip, read-only)". THIRD band "L2 CACHE", green note "~50 MiB · shared by all SMs · 128 B lines". BOTTOM band (widest, blue hatch) "GLOBAL MEMORY (HBM3)", green note "80 GB · 3.35 TB/s · hundreds of cycles · grid scope". Left red arrow top-to-bottom "latency ↑, capacity ↑"; right red arrow bottom-to-top "bandwidth ↑, scope ↓". Dashed takeaway box: "6 logical spaces, 2 physical memories: on-chip SRAM vs off-chip HBM". || The memory pyramid. Fast and small at the top, slow and vast at the bottom; the engineer's job is to move working data up.]]

## The table you should memorize

Here is the whole map in one grid. Read it as: *the closer to the thread, the faster and smaller and more private.*

| Space | Physical location | Scope | Latency | Read/write | Cached |
|---|---|---|---|---|---|
| Register | SM register file (SRAM) | one thread | ~1 cycle | R/W | n/a |
| Shared | SM L1 SRAM (228 KiB) | one block | ~20–30 cycles | R/W | n/a (is the cache) |
| Local | HBM (global) | one thread | hundreds | R/W | L1/L2 |
| Constant | HBM + constant cache | grid (read-only) | ~1 cycle on hit | R only | yes, broadcast |
| Texture | HBM + texture cache | grid (read-only) | tens on hit | R only | yes, spatial |
| Global | HBM (DRAM) | whole grid | hundreds | R/W | L2 (~50 MiB) |

Two rows in that table are lies of a sort — the two where the *logical* name and the *physical* reality diverge. Local memory says "local" but lives in the farthest, slowest DRAM. Constant memory says "global-sized and slow" but can beat shared memory for the right access pattern. Let me take them one at a time.

## Local memory is really global memory

This is the single most common source of mystery slowdowns for people learning kernels, so I want to be very precise about it. **Local memory** is per-thread private storage — but "private" describes its *scope*, not its *location*. Physically, local memory is carved out of global HBM. A local-memory access is a global-memory access wearing a friendlier name, and it costs exactly what a global-memory access costs: hundreds of cycles.[[sn: It is at least cached in L1 and L2 on the way, so a hot local variable that fits in cache is not as catastrophic as an uncached HBM round-trip. But you are now depending on the cache to save you from a problem you could have avoided.]]

You never explicitly ask for local memory; the compiler puts things there in two situations. First, **register spilling**: when your kernel needs more live registers than allowed (the 255-per-thread ceiling, or a lower limit forced by your occupancy target), the compiler *spills* the overflow to local memory. Second, **indexed private arrays**: if you declare `float tmp[16];` and index it with a value the compiler cannot resolve at compile time, it cannot keep that array in registers — registers are not addressable — so the whole thing lands in local memory.

```cpp
__global__ void danger(const float* in, float* out, int n) {
    float acc[32];                 // wants to be registers...
    for (int i = 0; i < n; ++i)    // ...but n is a runtime value,
        acc[i % 32] += in[i];      // so acc[] is indexed dynamically ->
    // acc[] gets SPILLED to LOCAL memory = HBM. Every access is a global load.
    out[threadIdx.x] = acc[0];
}
```

The tell is in the compiler and the profiler. Ask `nvcc` for `-Xptxas -v` and it prints, per kernel, a line like `N bytes stack frame` — a non-zero stack frame *is* your local-memory footprint. In the SASS you will see `LDL`/`STL` (load/store local) instructions where you expected pure arithmetic, and Nsight Compute will flag local-memory traffic against your HBM budget. The fix is almost always to reduce register pressure, unroll loops so array indices become compile-time constants, or shrink the tile so the working set fits. The lesson: **a variable you thought was free can quietly be the slowest thing in your kernel.**

[[fig: A two-panel SASS-plus-diagram titled "The spill". LEFT panel "(A) what you wrote": a purple code block "float acc[32];  acc[i%32] += ...", with a blue dashed arrow curving to the right. RIGHT panel "(B) what the compiler did": top, a small orange box "register file — FULL" with tiny slots hatched solid and a red X; a long blue dashed arrow labeled "SPILL" drops to a wide blue-hatched box "LOCAL MEMORY = HBM" holding a purple SASS listing "STL [R1], R4 //store local / LDL R4, [R1] //load local"; a green note "hundreds of cycles each — a global round-trip". Numbered circles (1) register file, (2) spill arrow, (3) HBM box. Dashed takeaway box: "non-zero 'stack frame' in -Xptxas -v = you are in HBM. Fix: fewer live regs, unroll, smaller tile." || Register spilling. Exceed the register budget and your "local" array silently becomes the slowest memory on the chip.]]

## Constant memory and the broadcast trick

Constant memory is the opposite surprise: a slice of HBM that, used correctly, behaves like the fastest thing you have. You declare it at file scope with `__constant__`, fill it from the host before launch, and it is read-only to the device — small (**64 KB** total) and backed by a dedicated **constant cache** on each SM.

What makes it special is **broadcast**. When all 32 threads of a warp read the *same* constant address in one instruction, the constant cache services the whole warp in a single access — one fetch, broadcast to all lanes — as fast as reading a register. That is exactly the pattern for coefficients every thread needs identically: a scaling factor, filter weights, a dimension lookup. Put those in `__constant__` and you get register-speed reads for free.

The flip side is the failure mode: if the threads in a warp read *different* constant addresses, the constant cache cannot broadcast. It **serializes** — it services each distinct address in turn, so a fully-divergent warp costs up to 32× a single read. Constant memory rewards uniform access and punishes divergent access, hard. That single property tells you exactly when to use it.

[[fig: A hand-drawn two-panel figure titled "The broadcast". Panel (A) labeled "uniform read — FAST": a warp drawn as 32 small circles (lanes 0..31) in a green-hatched row, each with a blue dashed arrow all converging onto ONE cell of a box labeled "CONSTANT $" and pointing at address "kFilter[j]". A single orange bolt icon and a green note "1 fetch → broadcast to all 32 lanes · ~1 cycle". Panel (B) labeled "divergent read — SLOW": the same 32 lanes but each blue arrow pointing at a DIFFERENT cell of the "CONSTANT $" box, drawn fanning out. A red note "32 distinct addresses → serialized, one at a time · up to 32×". A purple code sticky "s += in[t+j] * kFilter[j];  // same j for all lanes ✓". Numbered circles (1) on panel A, (2) on panel B. Dashed takeaway box bottom: "constant memory is register-fast ONLY when the whole warp reads the same address". || The broadcast. One address for the whole warp is a single cheap fetch; 32 different addresses serialize into 32 fetches.]]

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

## Texture memory, briefly

**Texture memory** is the graphics inheritance: another read-only path into global memory, routed through the texture cache, which is optimized for **2D spatial locality** — neighbors in a 2D grid, not in a 1D address range — plus free boundary clamping and linear interpolation. For GEMM and deep-learning work you will rarely reach for it; it earns its keep in stencils, image resampling, and irregular gathers where the 2D-locality cache actually matches your access shape. Know it exists; move on.

## The mental map, and how it drives the ladder

Step back and the six spaces collapse into one decision procedure. **Global memory** is where your data starts and where your answer must end up — you cannot avoid it, only minimize how many times you touch it. Everything else is a strategy for *not* going back to HBM. **Registers** hold the values a thread is computing on *right now*. **Shared memory** holds the tile a whole block cooperates on, so `N` threads reuse each byte instead of each re-fetching it from global — that is the entire idea of tiling, and why kernel 3 jumps to 12.8% and the tiled kernels past it climb into the 60s and 70s. **Constant** memory holds the small, uniform, read-only coefficients every thread shares. **Local** memory is what you get when you weren't paying attention, and **texture** memory is for the day your access pattern is genuinely 2D.

[[fig: A hand-drawn Excalidraw decision-flow diagram on pure white, titled in black handwriting "Which memory space?". A black rounded start box "I need to store some data" sits at the left; five long thin curved dashed arrows fan out to five outcome boxes, each arrow carrying a BLUE handwritten decision predicate (blue = the mechanism/question being asked). Branch 1, blue "one thread, computing right now?" -> orange-fill box "REGISTERS", green spec note "~1 cycle · per-thread". Branch 2, blue "a block reuses it many times?" -> yellow-hatch box "SHARED MEM", blue mechanism note "stage a tile once, reuse — this IS tiling". Branch 3, blue "read-only, same value across the warp?" -> plain box "CONSTANT", orange emphasis note "broadcast = register speed!". Branch 4, blue "read-only, 2D neighbour pattern?" -> plain box "TEXTURE", green note "spatial cache". Branch 5, blue "everything else / the final answer" -> blue-hatch box "GLOBAL (HBM3)", green spec note "80 GB · 3.35 TB/s". A red-outlined warning cloud off to the side, red handwriting: "LOCAL — you never choose this. spills land here = HBM. avoid!". Hand-drawn numbered circles (1)(2)(3)(4)(5) marking each branch in reading order. Dashed rounded takeaway box bottom, black text: "every optimization = move the working set UP the pyramid and keep it there". Flat, no shadows, no gradients, generous white space. || The decision procedure. Kernel engineering as one flowchart: what is this data, and how high up the pyramid can it live?]]

The predict-then-measure habit from [the three regimes](the-three-regimes.html) applies directly. Before you profile, name the space each piece of your working set lives in and ask whether it could live higher. When the naive GEMM kernel does `2N` flops but issues `2N` global loads per thread, the diagnosis is a memory-space failure: the row of `A` and column of `B` are re-fetched from HBM by every thread, when they could be staged once in shared memory and reused by the whole block. That single reframing — *stop touching global, stage it on-chip* — is the engine behind almost every rung of the ladder that follows. Next we make the on-chip pool concrete in [shared memory and L1](shared-memory-l1.html), where those 228 KiB and 32 banks stop being trivia and become the thing you tune.
