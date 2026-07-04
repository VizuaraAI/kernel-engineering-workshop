We are five kernels into the ladder and the machine has stopped being obviously wrong. Kernel 5 — the 2D register-blocked kernel where every thread computes an `8 × 8` micro-tile of `C` — reached **68.7% of cuBLAS**. That is a real kernel. The tensor cores are not involved yet, but the FMA pipes are close to saturated and shared memory is doing honest work. When a kernel is that good, the profiler stops shouting a single obvious cause and starts whispering about second-order effects. This kernel is about listening to one of those whispers: the *shape* of our load instructions.

The hypothesis is small and almost embarrassing. We are already loading exactly the right bytes from the right places. We are just loading them one `float` at a time, and the hardware would much rather we asked for four at once.

## The hypothesis: ask for 128 bits, not 32

Every global and shared load on an NVIDIA GPU is a memory *transaction* with a fixed cost to set up: an address computation, an instruction issue slot, a trip through the load/store unit. A scalar `float` load moves 32 bits and pays that whole cost. But the load/store unit can move up to **128 bits** — four `float`s — in a single instruction for the same overhead. The intrinsic that does this is a `float4`, and at the SASS level it compiles to `LDG.E.128` for global memory and `LDS.128` for shared memory.[[sn: The `.128` suffix is the access *width* in bits, not a count of anything. `LDG` = load-global, `LDS` = load-shared, `STS` = store-shared. The `.E` on `LDG.E` marks a generic/extended address — it goes through the unified address space rather than a special window.]]

So the plan is: wherever our kernel currently issues four consecutive scalar loads to contiguous addresses, replace them with one vector load. This is the same number of bytes crossing the same wires. What changes is the number of *instructions* we spend to move them — a quarter as many — and, on the global side, the number of memory transactions the coalescer has to coordinate. For a kernel that is starting to be limited by instruction issue rather than raw bandwidth, that is exactly the right lever to pull.

[[fig: A two-panel comparison titled "One float4 vs four floats". Left panel labeled "SCALAR" in red: four stacked hand-drawn boxes each holding one small square (a single float), each with its own thin blue arrow labeled "LDS" (scalar, 32-bit) pointing up to a load/store unit box, a red note beside them "4 instructions, 4 issue slots". Right panel labeled "VECTOR" in orange: one wide box holding four squares in a row (labeled x,y,z,w in purple), a single fat blue arrow labeled "LDS.128" to the same load/store unit, green note "128 bits, 1 instruction". Between the panels a purple code line "reinterpret_cast<float4*>(ptr)[0]". A dashed takeaway box at the bottom: "same bytes, 1/4 the instructions — this is free bandwidth if you're issue-bound". || A single 128-bit load carries four floats for the cost of one instruction issue.]]

## The obstacle: the transpose

There is a catch, and it is the interesting part of this kernel. Vector loads are only legal when the four elements you want are *contiguous in memory and the base address is 16-byte aligned*. You cannot vector-load four values that are strided apart; the hardware has no gather.

Look at how kernel 5 walks the inner `k`-loop. For each step, a thread needs a small column slice of `As` (the `A` tile in shared memory) and a small row slice of `Bs`. The `Bs` slice is a row — contiguous — so vectorizing the `B` side is trivial. But the `As` slice is a *column*. If `As` is stored row-major, a column is strided by the tile width, and a `float4` over it is illegal. The four values we want live 32 floats apart, not next to each other.

The fix is a classic: **transpose `As` as we load it into shared memory.** When we bring a tile of `A` from global memory into `As`, we write it column-major instead of row-major. The store into shared memory is where we pay for the shuffle, and after that the inner loop reads *columns* of the original `A` as *rows* of `As` — which are contiguous, aligned, and beautifully vectorizable. We move the awkwardness from the hot inner loop (executed thousands of times) to the load prologue (executed once per tile). That is almost always the right trade.[[sn: The global read of `A` that feeds this transpose is itself a `float4`: we read four contiguous elements of a row of `A` with one `LDG.E.128`, then scatter them into four different columns of `As`. So the transpose costs us four scalar `STS` stores (32-bit each), not vector ones — the one place in this kernel we deliberately give up vectorization to buy it back four times over in the loop.]]

[[fig: A tiling-walkthrough figure titled "Transpose As on load so columns become rows". Panel (1): a hatched blue matrix labeled "A tile in GMEM (row-major)" with one row highlighted, red dimension "↔ BK=8", a blue dashed arrow labeled "LDG.128 reads 4 contiguous" pointing right. Panel (2): the same four values being scattered downward into a second blue-hatched box labeled "As in SMEM (transposed)" — four small purple arrows labeled "STS ×4 (scalar 32-bit)" landing in a single column, orange note "we pay the shuffle HERE, once per tile". Panel (3): the inner-loop reading a full contiguous column of As with one wide arrow labeled "LDS.128" in blue, green note "now a column of A is a row of As → vectorizable". A numbered circle (1)(2)(3) on each panel showing order. Dashed takeaway box: "move the strided access out of the k-loop and into the prologue". || Transposing As at load time turns the strided column read into a contiguous vector read in the hot loop.]]

## The code

Concretely: reinterpret the pointers as `float4`, load a whole vector from global, and hand-place the four lanes. The shape below is the load prologue of the inner-`k` iteration, with the `A` transpose spelled out.

```cpp
// BM=128, BN=128, BK=8. Each thread owns an 8x8 tile of C.
// --- vectorized global -> shared, with A transposed on the way in ---
float4 tmp = reinterpret_cast<const float4*>(
                 &A[(innerRowA) * K + innerColA * 4])[0];
// scatter the 4 loaded floats into 4 different columns of As (transpose)
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

// B needs no transpose: a row stays a row, store the whole vector
reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
    reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
```

And the inner loop, which is where the payoff shows up, now reads its register-tile operands as vectors straight out of shared memory:

```cpp
for (uint dot = 0; dot < BK; ++dot) {
    // load an 8-wide slice of A and B into registers, 4 floats at a time
    float4 regA0 = reinterpret_cast<float4*>(&As[dot * BM + threadRow * TM])[0];
    float4 regA1 = reinterpret_cast<float4*>(&As[dot * BM + threadRow * TM + 4])[0];
    float4 regB0 = reinterpret_cast<float4*>(&Bs[dot * BN + threadCol * TN])[0];
    float4 regB1 = reinterpret_cast<float4*>(&Bs[dot * BN + threadCol * TN + 4])[0];
    // ... 8x8 FMAs into the accumulator using regA{0,1}.{x,y,z,w} ...
}
```

Nothing about the arithmetic changed. We compute the same `8 × 8` outer product per `k`-step with the same 64 FMAs. Only the loads that feed it were reshaped.

## The evidence: eight scalar loads become two vector loads

This is the moment that makes the whole kernel worth a section. Compile kernel 5 and kernel 6 and diff the SASS of the inner loop. This is a job for reading the actual machine code, not the PTX — [PTX is a virtual ISA and lies about widths](ptx-vs-sass.html) that `ptxas` only decides later.

In kernel 5, loading the eight `A` operands for one `k`-step looks like this — eight separate instructions, eight issue slots, eight address computations:

```
LDS R16, [R8]
LDS R17, [R8+0x4]
LDS R18, [R8+0x8]
LDS R19, [R8+0xc]
LDS R20, [R8+0x10]
LDS R21, [R8+0x14]
LDS R22, [R8+0x18]
LDS R23, [R8+0x1c]
```

In kernel 6, the same eight `float`s arrive in **two instructions**:

```
LDS.128 R16, [R8]
LDS.128 R20, [R8+0x10]
```

[[fig: A SASS-plus-diagram figure titled "Kernel 5 vs Kernel 6: the inner-loop load". Left column: a handwritten assembly listing in black on faint ruled lines, top block labeled in red "K5 — 8× LDS" showing eight "LDS R16..R23" lines each with a tiny blue tick, bottom block labeled in orange "K6 — 2× LDS.128" showing just two "LDS.128" lines, a big orange bracket collapsing the eight into the two with note "8 → 2". Right column: a memory diagram — a blue-hatched strip labeled "As row (contiguous, transposed)" with a single 16-byte window bracket labeled green "128 bits = 4 floats", a dashed arrow to a small register-file box (yellow) labeled "R16 R17 R18 R19". Bottom-right dashed takeaway box: "4× fewer load instructions in the hottest loop on the chip → issue pressure drops". || The signature moment: eight scalar LDS collapse into two LDS.128. Same bytes, a quarter of the instructions.]]

That 8-to-2 collapse is the entire thesis of the kernel rendered in assembly. And it is not only the `A` side; the `B` operands collapse the same way, and every global tile load becomes an `LDG.E.128`. Across the inner loop the total instruction count drops sharply, and — crucially — the drop is concentrated in the load/store issue path, which is exactly where a register-blocked kernel starts to bind once the FMA pipes are busy.

## The number

Running the benchmark, kernel 6 reaches **78.4% of cuBLAS**, up from kernel 5's **68.7%**.[[sn: The exact percentage moves a little with matrix size and driver version, but the *direction and rough magnitude* are stable: vectorization on top of a good 2D-tiled kernel is reliably worth roughly ten points of cuBLAS on Hopper. On some sizes the gain is larger because the shorter instruction stream also relieves the instruction cache.]] Roughly a ten-point jump, from a change that moved not a single byte differently and did not touch the math. We simply asked the load/store unit for its data in the width it prefers.

It is worth being honest about *why* this works, because "vectorization made it faster" is the kind of folk explanation that stops you learning. Two things happened. First, we cut the number of instructions the warp scheduler has to issue in the hot loop, so more of each cycle goes to useful FMAs instead of load bookkeeping — a pure [issue-overhead](the-three-regimes.html) win. Second, on the global side, `LDG.128` presents the coalescer with a cleaner, wider request, so a warp's worth of loads maps onto fewer, fatter memory transactions — better use of every 128-byte line we touch. The [transpose of `As`](shared-memory-l1.html) is what made both of those legal for the `A` operand, which otherwise would have been stuck issuing strided scalar loads forever.

There is one thing this kernel did *not* fix, and the profiler is now polite enough to point at it. With the load path this tight, shared-memory throughput becomes the next suspect — and the transposed layout we just introduced changes which threads touch which of the 32 banks. If two threads in a warp hit the same bank on different rows, they serialize.

## The bridge

So the next move is not another loading trick. It is to look hard at the shared-memory access pattern we just created and ask whether the 32 threads of a warp are spread cleanly across the 32 banks — or whether our tidy transpose quietly introduced a [**bank conflict**](bank-conflicts.html). That, plus a broader autotuning sweep over the tile dimensions `BM`, `BN`, `BK`, `TM`, `TN` to find the shape this particular GPU likes best, is what carries us from 78.4% toward the mid-80s and then, with warp-level tiling, into the low 90s. As always, we do not guess the shape — we let a measured sweep pick it, and we read the SASS to confirm the kernel we *think* we wrote is the kernel `ptxas` actually built.
