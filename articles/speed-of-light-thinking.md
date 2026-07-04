Before I write a single line of a kernel, I want to know the fastest it could possibly run. Not a guess — a number, computed on the back of a napkin from two hardware constants and one property of the workload. If my best imaginable kernel tops out at 400 microseconds, and I profile my first attempt at 2 milliseconds, I know there is 5× left on the table and roughly where it lives. If the ceiling *is* 400 microseconds and I am already at 450, I should stop, close the profiler, and go do something more useful with my afternoon. This habit — computing the **speed-of-light** limit first — is the difference between optimizing with a map and optimizing by superstition.

The [three regimes](the-three-regimes.html) article gave us the diagnostic: every kernel is bottlenecked by compute, memory bandwidth, or overhead. This article is the quantitative companion. It puts the actual numbers behind that framing, draws the picture they live in — the **roofline** — and shows how to read your regime off a plot *before* the kernel exists.[[sn: The mental model here is lifted from Horace He's *"Making Deep Learning Go Brrrr"* and Williams, Waterman & Patterson's original 2009 roofline paper. I have re-derived it for the H100 rather than the A100 those sources use.]]

## Two speeds, and only two

A GPU has exactly two throughputs that matter for the speed-of-light estimate, and they are wildly out of balance.

The first is **peak compute** — how many floating-point operations per second the arithmetic units can retire. On an H100 SXM5, the BF16 tensor cores do about **989 TFLOP/s** in the realistic, sparsity-free regime.[[sn: Marketing decks quote ~1979 TFLOP/s for the same chip, but that assumes 2:4 structured sparsity you almost never have in a dense GEMM. Use the dense number for honest estimates; halve the marketing one whenever you see it.]] The ordinary CUDA-core FP32 path is an order of magnitude slower — the tensor cores are the only thing on this chip that hits a teraflop count worth bragging about.

The second is **peak memory bandwidth** — how many bytes per second you can pull from **High Bandwidth Memory** (HBM). The H100's HBM3 delivers about **3.35 TB/s**. That sounds enormous until you divide it against the compute number.

Do the division. In BF16, every element is 2 bytes, so 3.35 TB/s is about `1.68e12` elements per second the machine can *read*. In that same second the tensor cores can perform `989e12` operations. The ratio is the whole ballgame:

```
989e12 FLOP/s  ÷  1.68e12 elem/s  ≈  590 FLOPs per element
              (or  989e12 ÷ 3.35e12  ≈  295 FLOPs per byte)
```

Read that number slowly. For every single byte you drag across the memory bus, the tensor cores can do roughly **295 floating-point operations** in the time it takes to arrive. If your kernel does *fewer* than 295 FLOPs per byte it touches, the arithmetic units will finish early and sit idle, tapping their feet, waiting on the memory system. You are memory-bound and no faster math unit can save you.

[[fig: A two-column hand-drawn "speed comparison" scene titled "Two speeds, wildly out of balance". LEFT column labeled COMPUTE in orange: a small rounded box labeled "H100 tensor cores" packed with tiny green squares, a green handwritten spec "≈ 989 TFLOP/s BF16 (dense, no sparsity)". RIGHT column labeled MEMORY in orange: a rounded box labeled "HBM3" with a fat blue arrow leaving it carrying hatched 2-byte data blocks, green spec "≈ 3.35 TB/s". Between them a big red "÷" and a red handwritten result in a circle: "≈ 295 FLOPs / byte". A blue dashed arrow curves to a note "this is the RIDGE POINT — the break-even arithmetic intensity". Bottom dashed takeaway box: "the machine can compute ~295× faster than it can feed itself." || The two hardware constants and the single ratio they produce. Compute vastly outruns bandwidth, and the gap defines the ridge point.]]

## Arithmetic intensity: the one number your kernel owns

The ridge point (~295 FLOPs/byte for H100 BF16) is a property of the *hardware*. The matching property of your *workload* is its **arithmetic intensity** (AI): the total FLOPs the kernel performs divided by the total bytes it moves between HBM and the chip.[[sn: "Bytes moved" specifically means DRAM traffic — reads and writes to global memory. Data that stays resident in registers, shared memory, or L2 across the kernel does not count against AI. This is exactly why tiling and fusion raise arithmetic intensity: they convert would-be HBM round-trips into on-chip reuse.]]

```
arithmetic intensity  =  total FLOPs  /  total bytes read+written from HBM
```

Compare the workload's AI against the machine's ridge point and you have your regime, decided before you write anything:

- **AI < ridge** → memory-bound. The bytes are the wall. Optimize movement.
- **AI > ridge** → compute-bound. The math is the wall. Optimize the math units.
- **AI ≈ ridge** → balanced, and rare; you are threading a needle.

Two worked examples make this concrete. Take an element-wise activation over an `N × N` BF16 tensor — a `GELU`, say. It reads `N²` elements, writes `N²` elements (4 bytes of traffic per element, in and out), and does a small constant number of FLOPs each. Its arithmetic intensity is well under **1 FLOP per byte**. Against a ridge of 295, that is nearly three hundred times too low: hopelessly, structurally memory-bound. The tensor cores are irrelevant to it. You will hit maybe a percent or two of peak FLOP/s and *near-peak bandwidth*, and that is the best it can ever do.[[sn: This is the whole argument for kernel fusion. Chaining `x.gelu().gelu()` unfused reads and writes HBM four times; fused, it reads once and writes once, halving the traffic and roughly doubling throughput for zero change in the math. The FLOPs are free; the bytes are everything.]]

Now take a large square GEMM, `C = A · B`, both `N × N` in BF16. It performs `2N³` FLOPs but only has to move about `3N²` elements (the three matrices). Its arithmetic intensity is on the order of `2N³ / (3N² · 2 bytes) ≈ N/3` FLOPs per byte — it *grows with the matrix size*. For `N = 8192` that is thousands of FLOPs per byte, an order of magnitude past the ridge. **Big GEMMs are compute-bound**, which is exactly why they are the workload worth pouring tensor cores at, and exactly why the GEMM ladder on this site is a fair fight against `cuBLAS`.

Below is the two examples side by side, each reduced to its arithmetic-intensity ratio and slotted against the same ridge.

[[fig: A hand-drawn two-panel "arithmetic intensity" comparison titled "One ratio decides your fate". A vertical red RIDGE line runs down the middle labeled in red "ridge ≈ 295 FLOPs/byte (H100 BF16)"; everything left of it is bracketed blue "MEMORY BOUND", everything right bracketed orange "COMPUTE BOUND". LEFT panel labeled "element-wise GELU": a small blue-hatched N×N tile with a blue in-arrow and a blue out-arrow (both crossing the HBM boundary), red dimension label "N × N", and the AI computation in purple handwriting "AI = few FLOPs / 4 bytes ≈ 0.5"; a red dot drops this workload far to the LEFT of the ridge. RIGHT panel labeled "big GEMM C = A·B": three matrices drawn as rectangles — A blue-hatch, B green-hatch, C pale-yellow-hatch — with red matrix letters (A, B, C) and red dim label "N=8192", and the purple computation "AI = 2N³ / (3N²·2B) = N/3 ≈ 2700"; a red dot places it far to the RIGHT of the ridge. A curved orange arrow arcs from GELU toward the ridge with the orange note "tiling + fusion raise AI → walk the dot rightward". Bottom dashed takeaway box: "AI is the ONE number your kernel owns. Compare it to the ridge; the regime is decided before you type." || Arithmetic intensity as the deciding ratio. GELU sits hundreds of times below the ridge (memory-bound); a big GEMM sits an order of magnitude above it (compute-bound); reuse tricks move a dot rightward.]]

## The roofline plot: putting the ceiling on the wall

Now draw it. Put arithmetic intensity (FLOPs/byte) on the x-axis, log scale, and achievable throughput (FLOP/s) on the y-axis, log scale. The performance ceiling is the **minimum of two limits**, and that minimum traces a distinctive shape:

```
achievable FLOP/s  =  min( peak_compute ,  AI × peak_bandwidth )
```

For low arithmetic intensity, `AI × bandwidth` is the binding term — a straight diagonal line rising with slope equal to your memory bandwidth. Every kernel over here is bandwidth-limited; the only way up the diagonal is to raise AI. For high arithmetic intensity, `peak_compute` is the binding term — a flat horizontal roof at 989 TFLOP/s. Kernels over here are compute-limited; more AI buys nothing, because you have hit the ceiling of the silicon.

The two lines meet at the **ridge point**. Its x-coordinate is `peak_compute / peak_bandwidth ≈ 295`. That corner is the only place on the plot where you are simultaneously saturating both resources — the theoretical sweet spot.

[[fig: A hand-drawn roofline plot titled "The H100 BF16 roofline". Black axes: x-axis labeled "arithmetic intensity (FLOPs/byte), log scale", y-axis labeled "throughput (FLOP/s), log scale". A rising blue diagonal line on the left, its slope annotated in blue "mechanism: throughput = AI × bandwidth", with the bandwidth constant "3.35 TB/s" written in GREEN beside it and the whole diagonal zone bracketed in blue "MEMORY BOUND"; the diagonal flattens into a horizontal roof whose height "peak compute 989 TFLOP/s" is written in GREEN, with the flat zone bracketed and labeled in orange "COMPUTE BOUND". The kink where the two meet is circled in red and labeled in red "RIDGE POINT ≈ 295 FLOPs/byte". Three plotted kernels marked with hand-drawn numbered circles, all three dots the SAME color red for consistency: circle (1) a red dot low on the diagonal, red label "element-wise GELU, AI≈0.5 — pinned to the diagonal"; circle (2) a red dot near the ridge, red label "small GEMM"; circle (3) a red dot sitting just below the roof, red label "big GEMM N=8192, AI≈2700". A short orange emphasis arrow points from dot (3) straight up to the roof with the orange callout "THIS gap = your headroom", and a purple code note nearby "min(peak_compute, AI × peak_bw)". Bottom dashed takeaway box: "read your kernel's dot: the vertical distance BELOW the ceiling is what's left to win." || The roofline. A kernel's arithmetic intensity fixes which line it lives under; the vertical gap between its measured dot and the ceiling is your remaining headroom.]]

Reading the plot is a two-step move. First, find your kernel's x-position from its arithmetic intensity — that immediately tells you *which* ceiling you are under, diagonal or flat. Second, plot your *measured* throughput as a dot and look at the vertical distance up to the ceiling above it. That gap, in dB, is your remaining headroom. A dot sitting right on the line is at speed-of-light for its regime; a dot a factor of ten below the line has a factor of ten to win, and the roofline has just told you whether that win comes from moving bytes better (if you are under the diagonal) or feeding the math units better (if you are under the roof).

## Estimating the floor before you write the kernel

Here is the payoff — turning the ceiling into a wall-clock estimate. The roofline gives you a *rate*; combine it with the *volume* of your workload and you get a lower bound on runtime, the fastest the kernel could conceivably be.

The estimate is just the larger of two times: how long the compute must take, and how long the memory movement must take. Whichever is bigger is your floor, because they overlap in the best case but cannot hide behind each other beyond that.

```python
def speed_of_light_us(flops, bytes_moved,
                      peak_flops=989e12, peak_bw=3.35e12):
    t_compute = flops / peak_flops        # seconds if compute-bound
    t_memory  = bytes_moved / peak_bw     # seconds if bandwidth-bound
    return max(t_compute, t_memory) * 1e6 # microseconds, whichever wins
```

Run it on the two examples. A `4096 × 4096` BF16 GEMM does `2 · 4096³ ≈ 1.37e11` FLOPs and moves `3 · 4096² · 2 ≈ 1.0e8` bytes. The compute time is `1.37e11 / 989e12 ≈ 139 µs`; the memory time is `1.0e8 / 3.35e12 ≈ 30 µs`. The `max` is **139 µs, compute-bound**, and no honest kernel beats it. A `GELU` over the same `4096 × 4096` tensor moves `2 · 4096² · 2 ≈ 6.7e7` bytes and does a trivial amount of math, so its floor is `6.7e7 / 3.35e12 ≈ 20 µs`, **memory-bound**, set entirely by the byte count. Two lines of arithmetic, and you know the target for each kernel before writing either.

[[fig: A hand-drawn two-panel "napkin estimate" worksheet titled "Speed-of-light in two lines". Panel (A) labeled "4096² BF16 GEMM" with red dimension labels: two stacked equations in purple handwriting — "t_compute = 2·4096³ / 989e12 ≈ 139 µs" and "t_memory = 3·4096²·2B / 3.35e12 ≈ 30 µs" — with an orange bracket around the larger one and an orange note "max = 139 µs → COMPUTE BOUND". Panel (B) labeled "4096² GELU (element-wise)": purple equations "t_compute ≈ tiny" and "t_memory = 2·4096²·2B / 3.35e12 ≈ 20 µs", orange bracket on the memory term, orange note "max = 20 µs → MEMORY BOUND". A green spec strip across the top lists the two constants used: "peak 989 TFLOP/s · BW 3.35 TB/s". A blue dashed arrow links each panel's winning time down to a small roofline thumbnail showing the GEMM dot on the roof and the GELU dot on the diagonal. Bottom dashed takeaway box: "floor = max(compute time, memory time). Beat 5% of it → you have a bug." || The whole discipline on a napkin. Compute both times, take the max, and you have the fastest the kernel could ever run before writing it.]][[sn: This floor is optimistic on purpose — it assumes perfect overlap, zero launch overhead, and 100% of peak, none of which you will hit. Treat it as the speed-of-light limit, not a forecast: a kernel at 70–90% of this floor is genuinely excellent, and one at 5% has a real bug to find.]]

That is the entire discipline. Two hardware constants, one workload ratio, one `max`. It tells you your regime, your ceiling, and your remaining headroom before you have compiled anything — and it turns every profiling session from a fishing trip into a checklist.

## Where this points next

The naive GEMM kernel on this site reaches a humiliating **1.3% of `cuBLAS`**. The roofline explains *why* instantly: that kernel re-reads every element of `A` and `B` from HBM for each output, dragging its arithmetic intensity down to about 1 FLOP per element and pinning its dot to the far-left diagonal — the memory-bound basin — even though the underlying math (a big GEMM) belongs up on the compute roof. The entire optimization ladder that follows is, geometrically, one motion: **drag the dot to the right and up the diagonal until it hits the roof, then climb the roof toward cuBLAS.** Coalescing, tiling in [shared memory](shared-memory-l1.html), register blocking, vectorized `float4` loads, warptiling — each one raises arithmetic intensity by converting an HBM round-trip into on-chip reuse, walking us from 1.3% through 8.5%, 12.8%, 36.5%, 68.7%, and eventually **93.7% of cuBLAS**.

None of those steps is a guess. Each is the roofline handing us the next move: measure the dot, see the gap, close it. That is speed-of-light thinking, and from here on every worklog on this site starts by computing the ceiling before touching the keyboard.
