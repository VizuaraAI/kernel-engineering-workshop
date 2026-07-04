Every warp on a GPU spends most of its life waiting. It issues a load, and then it waits hundreds of cycles for the bytes to arrive from HBM. If nothing else ran during that wait, the **Streaming Multiprocessor** (SM) would sit almost completely idle, and the 989 TFLOP/s of tensor throughput you paid for would be a very expensive space heater. The whole reason a GPU is fast is that it does *not* wait: while one warp stalls on memory, the scheduler runs another. This trick is called **latency hiding**, and the number that governs how much of it you can do is called **occupancy**. I spent an embarrassingly long time on my first kernels treating occupancy as a score to maximize, so this article is the note I wish I'd had: what occupancy actually is, how the hardware computes it for a kernel, and the counterintuitive part I learned the hard way — why chasing it to 100% is often the wrong move.

## What occupancy is

**Occupancy** is a ratio: the number of **active warps** resident on an SM divided by the maximum number of warps that SM can hold. A warp is 32 threads that execute in lockstep. On an H100 an SM can hold up to **64 warps** at once — that is 2048 threads, since `64 × 32 = 2048`.[[sn: The 64-warp / 2048-thread ceiling is an architectural constant of the `sm_90` SM, not something you configure. Older cards differ: the Ampere A6000 in Simon Boehm's worklog tops out at 48 warps (1536 threads) per SM, which is why his occupancy percentages look different from ours.]] If your kernel manages to keep all 64 resident, you are at **100% occupancy**. If the hardware can only fit 32 of them, you are at 50%.

Why does having more warps resident help? Because the SM's four **warp schedulers** pick, every cycle, an *eligible* warp — one whose next instruction has all its operands ready — and issue it. When a warp fires off a global load and then needs the result, it becomes ineligible for the ~400+ cycles the load takes. If there are 30 other warps resident, the schedulers simply issue from those instead, and the load latency is hidden completely. If there are only two other warps resident, the schedulers run out of eligible work and the SM stalls with its execution units idle. Low occupancy shows up in Nsight Compute as low *issue efficiency*: cycles where no warp was ready to go.

[[fig: A hand-drawn architecture map titled "Occupancy = active warps / max warps". Center: a large rounded rectangle labeled "SM (sm_90)" containing a 4-across grid of small hatched warp boxes; 32 of them drawn solid blue-hatch (labeled in blue "resident / active"), and 32 drawn as faint dashed empty slots (labeled in red "empty — could hold more"). To the left, four small boxes labeled "warp scheduler ×4" with blue dashed arrows into the warp grid and a blue handwritten note "each cycle: pick an ELIGIBLE warp, issue it". A green margin note on the right reads "max = 64 warps = 2048 threads per SM". A red callout points at one blue warp mid-stall with the label "waiting ~400 cyc on HBM load". An orange emphasis note near the schedulers: "30 other warps hide that wait". Dashed takeaway box at the bottom: "32 / 64 resident → 50% occupancy". || Occupancy is just how full the SM's warp slots are. The schedulers hide latency by hopping between whatever warps are resident.]]

## The three things that limit it

You almost never hit 64 warps, because three finite resources on the SM get partitioned among your blocks, and whichever runs out first sets the cap. The hardware runs this calculation — the "occupancy calculator" — every time it schedules a block.

The three limits are:

1. **Registers per thread.** The SM has one **register file** of `65536` 32-bit registers (256 KB per SM). Every thread claims some of them for its live variables, and no thread may exceed `255` registers. Total registers needed = registers-per-thread × threads-per-block × blocks. Once that exceeds 65536, no more blocks fit.
2. **Shared memory per block.** The SMEM and L1 share a 256 KiB pool per SM, of which up to `228 KiB` can be carved out as **shared memory** (SMEM).[[sn: The usable SMEM figure is a per-kernel opt-in via `cudaFuncSetAttribute` — the default carve-out is much smaller (around 48 KB) and you have to explicitly ask for the big 228 KiB configuration. There is also a small per-block CUDA runtime reservation, so your *usable* budget is a little under whatever you request.]] Each block reserves its full SMEM footprint for its entire lifetime, so SMEM-per-block × blocks cannot exceed the pool.
3. **Threads (and blocks) per block.** A block can have at most `1024` threads, and an SM can hold a bounded number of blocks and 2048 threads total. A block of 1024 threads is 32 warps, so at most two such blocks fit before you hit the thread ceiling regardless of registers or SMEM.

The occupancy for your kernel is set by the *tightest* of these three. This is the single most important mental model in the section: it is a `min()`, not a sum.

[[fig: A hand-drawn diagram titled "The occupancy calculator is a min()". Three vertical bar-gauges drawn side by side, each a tall rounded rectangle filling from the bottom. Gauge 1 labeled "REGISTERS" in green with a green cap note "65536 / SM", filled to a marked line labeled in purple "37 regs/thread × 1024 = 37888 → fits 1 block". Gauge 2 labeled "SMEM" in green with cap note "up to 228 KiB", filled low with a purple note "~9 KiB/block → 25 blocks would fit". Gauge 3 labeled "THREADS" in green with cap note "2048 / SM", filled to the top with red note "1024/block → only 2 blocks". A big orange bracket under all three points to the SHORTEST effective winner with the handwritten label "the LIMITING resource wins". Numbered circles (1)(2)(3) over the gauges. Dashed takeaway box: "occupancy = min(reg limit, smem limit, thread limit), not the average". || Three gauges, one verdict. Whichever resource is scarcest for your block dimensions caps how many blocks — and therefore warps — go resident.]]

## Working the calculation on a real kernel

Let us do the arithmetic the way the hardware does, on the shared-memory GEMM kernel from the ladder — kernel 3, the [shared memory](shared-memory-l1.html) cache-blocking version. It launches `32 × 32 = 1024` threads per block, and after compilation `nvcc` reports it uses **37 registers per thread** and **8 KiB of SMEM per block** (two `32 × 32` FP32 tiles, `2 × 32 × 32 × 4 B = 8192 B`, plus a little runtime overhead).

Now run the three limits against an H100 SM:

```python
# H100 SM budgets
MAX_REGS    = 65536     # 32-bit registers per SM
MAX_SMEM    = 228 * 1024
MAX_THREADS = 2048      # = 64 warps
MAX_WARPS   = 64

regs_per_thread = 37
smem_per_block  = 8192 + 1024   # + runtime reservation
threads_per_blk = 1024          # = 32 warps

by_regs    = MAX_REGS    // (regs_per_thread * threads_per_blk)  # -> 1
by_smem    = MAX_SMEM    // smem_per_block                        # -> 25
by_threads = MAX_THREADS // threads_per_blk                       # -> 2

blocks_per_sm = min(by_regs, by_smem, by_threads)                # -> 1
warps         = blocks_per_sm * (threads_per_blk // 32)           # -> 32
occupancy     = warps / MAX_WARPS                                 # -> 0.50
```

The register limit is the binding constraint: `37 × 1024 = 37888` registers for one block, and a second block would need `75776`, well past `65536`. SMEM could accommodate two dozen blocks and the thread ceiling allows two, but registers permit exactly **one block per SM**. One block is 32 warps, so this kernel runs at `32 / 64 =` **50% occupancy** on an H100.[[sn: On Boehm's 48-warp A6000 the *same* kernel lands at 32/48 ≈ 66%. Same code, same register count, different denominator — a reminder that "66% occupancy" is a statement about a specific chip, not a portable property of the kernel.]]

Notice the cliff hiding in that arithmetic. If a later optimization pushed register usage from 37 up to just 33 or below — `33 × 1024 = 33792`, still one block, no change — nothing happens. But hardware allocates registers in granular chunks, and crossing certain thresholds can flip you from one block to two, doubling occupancy in a single step; crossing back drops you off a cliff. This is why you sometimes see `__launch_bounds__` or `maxrregcount` in tuned kernels: they cap the register count to *force* the compiler under a threshold and keep two blocks resident.

## The counterintuitive part: more is not always faster

Here is the trap that caught me once. Having established that occupancy hides latency, my instinct was to maximize it — shrink registers, shrink SMEM, cram in warps until I hit 100%. That instinct is wrong, and understanding why is what separated me-who-had-read-about-occupancy from me-who-had-actually-profiled-a-kernel.

The reason is that latency hiding has a *sufficiency point*, not a linear payoff. **Little's Law** tells you roughly how much parallelism you need in flight to cover a given latency: if a memory load takes ~400 cycles and a warp scheduler can only re-issue from a given warp every handful of cycles, you need on the order of a dozen warps per scheduler to keep it busy through the stall.[[sn: The napkin version: ~416 cycles of latency divided by an issue cadence of ~32 cycles per warp gives ⌈416/32⌉ ≈ 13 warps needed in flight to fully hide one outstanding load each. This is deliberately rough — the real figure depends on how many independent memory operations each warp has outstanding at once, which is exactly where ILP re-enters the story.]] Once you have enough warps to cover the longest stall your kernel actually incurs, adding a 14th, or a 40th, buys you *nothing* — the schedulers were already never idle. You have satisfied the constraint; further occupancy is spent latency-hiding capacity with no one waiting to be hidden.

And it is not free. The occupancy you bought by shrinking registers came *out of the registers*. A GEMM kernel gets fast by accumulating a large tile of output — say `8 × 8` results — in each thread's registers, so that on-chip data gets reused dozens of times without touching SMEM or HBM again. That is **instruction-level parallelism** (ILP): each thread has many independent multiply-accumulates in flight at once, which hides latency *within* a single warp rather than across warps. But a big register tile means high registers-per-thread, which means fewer blocks fit, which means *lower* occupancy. The two levers fight each other.

[[fig: A hand-drawn pipeline/tradeoff timeline titled "Two ways to hide latency". Top track labeled in blue "HIGH OCCUPANCY, low ILP": a long row of many thin warp bars, each with one small compute segment, the scheduler arrow hopping across them; blue note "latency hidden ACROSS warps — needs many resident". Bottom track labeled in orange "LOW OCCUPANCY, high ILP": only a few fat warp bars, each packed with many independent MACs drawn as stacked purple ticks labeled "8×8 reg tile, all in flight"; orange note "latency hidden WITHIN one warp — needs many registers". A red double-headed arrow between the two tracks labeled "these fight over the register file". A small green side note: "past the sufficiency point, extra warps do nothing". Dashed takeaway box: "peak throughput can live at single-digit occupancy". || Occupancy across warps and ILP within a warp are two routes to the same goal. Fast GEMM kernels deliberately trade the first for the second.]]

This is not a rare edge case. The best modern GEMM and attention kernels — the ones actually shipping in `cuBLAS` and FlashAttention — frequently run at **single-digit occupancy** on purpose. They hand each thread an enormous register tile, saturate the tensor cores with a handful of warps carrying deep independent instruction streams, and leave most of the SM's warp slots empty because they simply do not need them. Their bottleneck was never latency hiding; it was arithmetic throughput, and they solved it with ILP and data reuse instead. On our own ladder, the [2D block-tiling](gemm-kernel-6-2d-tile.html) kernel that reaches **68.7% of cuBLAS** and the warp-tiling kernel at **93.7%** both *lower* occupancy relative to the simple SMEM kernel while getting dramatically faster — exactly because they spend registers on reuse instead of on resident warps.

## The habit to take away

Occupancy is a means, never an end. The correct workflow is the same predict-then-measure loop from [the three regimes](the-three-regimes.html): find out what your kernel is actually waiting on. If Nsight Compute shows low issue efficiency and long memory stalls with nothing to hide them — you are *under*-occupied, and raising occupancy is the fix. If issue efficiency is already high, or you are compute-bound and saturating the math units, then occupancy is a solved problem and spending registers to raise it further will only make you slower.

Compute the three limits by hand for any kernel you write; it takes thirty seconds and tells you which resource you are actually spending. Then let the profiler, not your intuition, decide whether more warps or bigger register tiles is the next move. Next we put this to work: we start the GEMM ladder trading occupancy for register reuse deliberately, and watch the percentage of `cuBLAS` climb even as the occupancy number falls.
