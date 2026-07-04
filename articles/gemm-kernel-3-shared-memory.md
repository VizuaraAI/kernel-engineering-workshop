Kernel 2 fixed the access pattern and quadrupled us to **8.5% of cuBLAS** with a one-line change, but the profiler was unimpressed for a reason we already knew was coming: we are still fetching the same numbers from global memory over and over again. Coalescing made each `HBM` transaction efficient; it did nothing to reduce *how many* transactions we issue. This kernel attacks that directly. It is the first optimization on the ladder that changes the *algorithm* rather than the memory layout, and it is where the climb properly begins.[[sn: Everything below follows the shared-memory cache-blocking kernel from Simon Boehm's *"How to Optimize a CUDA Matmul Kernel"*, cross-checked against salykova's H100 GEMM worklog. The GFLOP/s and occupancy figures are Boehm's A6000 run; the ladder and the percentages of cuBLAS carry over, and the hardware specs quoted (SMEM ceiling, L2, HBM3) are H100's — we rebuild the kernel in our own voice with our own figures.]]

## The hypothesis: stop re-reading, start caching

Recall the diagnosis from [kernel 1](gemm-kernel-1-naive.html): the arithmetic intensity of naive GEMM is about **1 flop per element loaded**, hundreds of times below the H100 ridge point. The culprit is *reuse we throw away*. Element `A[m][k]` is genuinely needed by every thread in row `m` of the output — all `N` of them — but the naive and coalesced kernels each re-fetch it from `HBM` every single time. The data is reusable; we just never keep it anywhere fast enough to reuse.

The GPU gives us exactly the right piece of hardware for this. Each **Streaming Multiprocessor** (SM) has a slab of on-chip **shared memory** (SMEM) — programmer-managed scratch that a whole thread block can read and write at roughly the latency of L1, one to two orders of magnitude faster than `HBM`.[[sn: On H100 the L1 and SMEM live in the same 256 KiB physical array per SM, split at launch; up to 228 KiB of it can be carved out as SMEM. Those 228 KiB are not an exact hardware constant — 8 KiB is reserved — but the figure you configure against.]] The plan is the classic **cache-blocking** move: cooperatively load a tile of `A` and a tile of `B` into `SMEM` once, let every thread in the block compute against that cached tile many times, then move on. Read from `HBM` a little; compute from `SMEM` a lot.

## The shape of the blocking

Give each block an output tile of `C` of size `BM × BN` — say `32 × 32`. To produce that tile, block `(blockIdx.y, blockIdx.x)` needs the corresponding `BM` rows of `A` and `BN` columns of `B`. But those are full-length strips: `BM × K` and `K × BN`. At any realistic `K` they do not fit in `SMEM` at once, and that constraint — not a stylistic choice — is what forces us to **block over K**.

So we walk the `K` dimension in chunks of width `BK`. On each step we stage only a `BM × BK` slab of `A` and a `BK × BN` slab of `B`, accumulate every product those two slabs contribute to our output tile, then slide `BK` further along `K` and do it again. The partial sums live in a per-thread register (`tmp`) and survive across chunks; the `SMEM` tiles are overwritten each chunk.

[[fig: A three-panel hand-drawn "tiling walkthrough" titled "Kernel 3: block over K". Panel (1): three matrices A (blue diagonal hatch, N×K), B (green diagonal hatch, K×N), C (pale-yellow hatch, N×N), red dimension labels N, K on the axes. On C, one BM×BN output tile is boxed in orange labeled "this block's tile, 32×32". Blue dashed arrows point from the C tile to a full horizontal strip of A (BM rows) and a full vertical strip of B (BN cols), red note "these strips are BM×K and K×BN — too big for SMEM". Panel (2): the same strips chopped into vertical/horizontal chunks of width BK=32, one chunk pair highlighted, orange note "load ONE BK-wide chunk pair into SMEM". A box drawn as SMEM holding "As: 32×32" (blue) and "Bs: 32×32" (green), green spec "2·32·32·4B = 8 KiB". Panel (3): a numbered loop — circle (1) "cooperative load to SMEM", circle (2) "__syncthreads()", circle (3) "each thread: BK madds from SMEM into tmp", circle (4) "__syncthreads(), slide BK along K". Purple note "tmp is a register, survives all K/BK chunks". Dashed takeaway box bottom: "read BK-wide slabs from HBM, reuse each loaded value BM (or BN) times → traffic cut by the tile width". || Kernel 3. The K loop is chunked into BK-wide steps; each chunk pair is staged in SMEM once and reused across the whole block.]]

## The code, and the two barriers

The kernel body is a loop over `K` in `BK`-sized steps. Inside each step there are three phases: a cooperative load, a compute pass, and — crucially — a synchronization barrier on either side of the compute.

```cpp
template <const int BLOCKSIZE>
__global__ void sgemm_smem(int M, int N, int K,
                           const float* A, const float* B, float* C) {
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const uint cRow = blockIdx.y, cCol = blockIdx.x;
  const uint threadRow = threadIdx.x / BLOCKSIZE;
  const uint threadCol = threadIdx.x % BLOCKSIZE;

  // advance pointers to this block's row-of-A and col-of-B
  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float tmp = 0.0f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // 1. every thread loads one element of each tile
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();                    // tile fully populated before anyone reads

    A += BLOCKSIZE;                      // slide along K
    B += BLOCKSIZE * N;

    // 2. accumulate this chunk's contribution from SMEM
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    __syncthreads();                    // don't overwrite tiles others still need
  }
  C[threadRow * N + threadCol] = tmp;
}
```

The two `__syncthreads()` are the whole game, and getting them wrong is the classic first bug. The **first barrier** stands between the load and the compute: a thread must not read `As`/`Bs` until *every* thread in the block has finished writing its element, or it will multiply against stale or uninitialized memory. The **second barrier** stands at the bottom of the loop: a fast warp that races ahead to the next chunk must not overwrite `As`/`Bs` while a slower warp is still reading the current chunk.[[sn: `__syncthreads()` is a block-wide barrier — it synchronizes threads within one block, never across blocks, and every thread must reach it or the kernel deadlocks. It is also a compiler memory fence for shared memory, which is why the reads on the far side actually see the writes.]] With `BLOCKSIZE = 32` the block is `1024` threads, exactly the [max per block](the-three-regimes.html), and the load pattern is trivially one element per thread.

## Why the traffic drops — the napkin math

Here is the payoff, counted in bytes. In the coalesced kernel, to compute the `BM × BN = 32 × 32` output tile, the threads collectively read all of `A`'s `BM` rows and all of `B`'s `BN` columns, *per k-step*, straight from `HBM` — and each `A` element gets pulled once for every column it multiplies against, each `B` element once for every row.

With blocking, we load each `BK`-chunk of the tiles into `SMEM` exactly **once**, then read it back from `SMEM` `BLOCKSIZE` times during the inner `dotIdx` loop. Every value fetched from `HBM` is now amortized across `BM` reuses (for `B`) or `BN` reuses (for `A`) inside the block. Concretely: an element of the `A` tile is loaded from global memory once and then consumed by all `BN = 32` threads sharing its row; an element of the `B` tile once, consumed by all `BM = 32` threads sharing its column. **Global-memory traffic falls by a factor of the tile width** — roughly `32×` for this configuration — because that is exactly how many times each loaded value is now reused before we evict it.

The arithmetic intensity climbs from `~1` flop/byte toward the tens. We have not touched the flop count; we have simply stopped fetching the same numbers repeatedly.

## Why the tile can't just be bigger

If a `32`-wide tile cuts traffic `32×`, why not a `128`-wide tile for a `128×` cut? Capacity. Two `BM × BK` and `BK × BN` tiles of FP32 cost `2 × BLOCKSIZE² × 4` bytes of `SMEM`. At `32` that is a comfortable **8 KiB** per block. But `SMEM` is a *per-SM* resource, and it is also what gates **occupancy** — how many blocks the SM can host at once. Push the tile to `128 × 128` and a single block wants `128 KiB` of `SMEM`, which not only crowds out concurrent blocks but eventually collides with the `228 KiB` ceiling.

[[fig: A hand-drawn "memory pyramid" titled "The SMEM budget forces the tile size". A vertical stack of boxes, widest at the bottom. Bottom box: "HBM3 — 80 GB" green spec "3.35 TB/s, but far away". Middle box: "L2 — ~50 MiB" green "shared across all SMs". Top box drawn largest and highlighted orange: "SMEM + L1 — 256 KiB / SM, up to 228 KiB usable as SMEM", green "32 banks, ~L1 latency". To the right, a small table in purple handwriting: "tile 32×32 → 8 KiB/block ✓ many blocks/SM", "tile 64×64 → 32 KiB/block", "tile 128×128 → 128 KiB/block ✗ occupancy collapses". A red dashed arrow from the SMEM box to the table labeled "SMEM is per-SM → it caps how many blocks run at once". Dashed takeaway box: "bigger tile = less HBM traffic BUT less occupancy — the real ladder is finding that balance". || The shared-memory capacity per SM is the constraint that sets an upper bound on tile size, and therefore on how far this trick alone can take us.]]

There is a second, subtler cost hiding in the `SMEM` array itself. `SMEM` is organized into `32` banks, and if the `32` threads of a warp hit the same bank on different addresses, the accesses **serialize** — a bank conflict. Our simple `As[threadRow * BLOCKSIZE + dotIdx]` access has all threads in a warp reading the *same* `As` element (a broadcast, which is free) but the `Bs` column access can conflict depending on stride.[[sn: Later kernels pad the leading dimension of the transposed `A` tile — e.g. `128 → 132` floats — so that consecutive rows land in different banks. At this stage the conflicts are minor and not yet the bottleneck; we leave them for now and fix them when the profiler says they matter.]] We note it and move on; it is not yet the wall.

## The measurement

Compile, run, and the number moves the right way but not dramatically: about **2980 GFLOP/s**, which is **12.8% of cuBLAS** — up from the coalesced kernel's `~1990 GFLOP/s` and 8.5%. A **~50% speedup** from a genuinely more efficient algorithm.[[sn: The exact figure drifts a few tenths of a percent run to run and with matrix size; the *jump* from ~8.5% to ~12.8% is stable. Anything in that band means the SMEM tiling is doing its job.]]

Fifty percent is real money, but it is also a warning: if cutting `HBM` traffic `32×` only bought us `1.5×`, then `HBM` traffic was *not* our sole bottleneck any more. Point Nsight Compute at it and the story is clear. Occupancy sits around **66%** — with a fat `1024`-thread block the limiter is *blocks per SM*, not warps: only a couple of these giant blocks fit at once, so a chunk of the SM's warp-slot and register budget sits idle (one more reason the next kernels shrink the block) — and the instruction mix is still dominated by memory operations, but now they are `LDS` (load-from-shared) instructions, not global loads. We traded a global-memory bottleneck for a *shared-memory-and-issue* bottleneck.

Look at the inner loop and the reason jumps out. For every single `LDS` from `As` and `Bs`, we do exactly one `FFMA` (fused multiply-add). The ratio of memory instructions to math instructions in the hot loop is roughly `2:1` the wrong way — we spend more instruction slots *fetching operands from SMEM* than doing arithmetic on them. The tensor cores, and even the plain FP32 pipes, are starved not for bytes from `HBM` but for a better ratio of compute-to-load *inside the block*.

[[fig: A hand-drawn "SASS listing + diagram" figure titled "The inner loop is load-bound". On the left, a handwritten purple assembly column for one `dotIdx` iteration: "LDS R4, [As + ...]", "LDS R5, [Bs + ...]", "FFMA R0, R4, R5, R0", repeated ×BK with a red brace labeled "2 loads : 1 madd — the wrong ratio". On the right, a small diagram: an SM box containing SMEM (blue hatch) with two thin blue arrows (`LDS`) feeding a single FFMA unit (pale-yellow box), the FFMA unit drawn small and mostly idle with a red note "starved for operands". A green spec on the SM box "66% occupancy · fat 1024-thread blocks leave warp slots idle". Orange emphasis arrow pointing at the FFMA unit: "one flop per two shared loads". Dashed takeaway box bottom-right: "not HBM-bound any more — bound by SMEM issue. Fix = reuse each loaded value across MANY accumulators in registers." || The Nsight profile: the hot loop issues two shared-memory loads for every multiply-add. The next kernel raises the flops-per-load ratio with register tiling.]]

## The bridge to kernel 4

The profiler has handed us the next hypothesis, exactly as it did in kernel 1. We are no longer `HBM`-bound; we are bound by doing too few flops per `SMEM` load. The fix is not more caching — it is **arithmetic reuse in registers**. Instead of each thread computing one output element (one `FFMA` per two `LDS`), we make each thread responsible for a small *column* of outputs, load a value from `SMEM` once into a register, and reuse it across several accumulators before touching `SMEM` again.

That is **1D block-tiling**, kernel 4 on the ladder, and it is where the numbers finally start to leap: a single register-reuse trick takes us from `12.8%` to **36.5% of cuBLAS**. The pattern holds — state the hypothesis the profiler implied, write the smallest kernel that tests it, measure, and let the new bottleneck pick the next move.
