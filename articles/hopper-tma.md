Every kernel on the GEMM ladder so far has quietly paid the same tax: the threads that do the math also have to do the *copying*. Before a warp can multiply anything, thirty-two lanes spend their time computing addresses, issuing loads, and staging a tile of `A` and `B` from global memory into shared memory — and only *then* do they get to touch the tensor cores. On an H100 that is a shameful arrangement, because the chip ships with a dedicated piece of silicon whose entire job is to move multidimensional tiles around so the compute threads never have to. It is called the **Tensor Memory Accelerator** (TMA), it is new in Hopper, and wiring it in is the moment the kernel stops looking like a textbook and starts looking like `cuBLAS`.

This article is a worklog entry, not a full kernel. By the time we reach it we already have shared-memory tiling, register blocking, and warp tiles from the [warptile kernel](gemm-kernel-warptile.html) sitting at roughly **72% of cuBLAS** in FP32. The frontier kernels leave FP32 behind, switch to BF16 through the tensor cores, and the first thing they need is a fundamentally better way to feed those cores. TMA is that way. Everything here is `sm_90a` — the architecture-specific Hopper target — and none of it compiles on anything older.[[sn: The `a` suffix matters. `sm_90` is the portable Hopper target; `sm_90a` unlocks the architecture-*accelerated* instructions — `wgmma`, `cp.async.bulk.tensor`, the tensor-map machinery — that are not guaranteed to exist on future architectures. You compile the frontier kernels with `-arch=sm_90a` and accept that they are H100-only.]]

## What we were doing by hand

Recall the shape of every load in the classic tiled kernels. Each thread in the block owns a slice of the tile and copies it element by element, computing its own source and destination index:

```cpp
// hand-rolled staging: every thread copies its own elements
sharedA[ty * TILE_K + tx] = A[(blockRow + ty) * N + (kIter + tx)];
sharedB[ty * TILE_N + tx] = B[(kIter + ty) * N + (blockCol + tx)];
__syncthreads();
```

Later kernels upgraded this to `cp.async` (the `ca`/`cg` variants), which at least let the load happen in the background while threads did other work, and double-buffered the tiles so copy and compute overlapped. But `cp.async` is still a *per-thread* instruction. Every one of the 128 or 256 threads in the block issues its own asynchronous load, computes its own addresses, and participates in the `commit_group` / `wait_group` bookkeeping. The address arithmetic alone burns registers and issue slots that we would rather spend on `wgmma`. And there is a second, uglier problem: the tensor cores want their operands in shared memory laid out in a **swizzled** pattern — a deliberate permutation that spreads a tile across the 32 banks so that a `wgmma` read hits no bank conflicts. Reproducing that swizzle by hand, index by index, is possible but miserable, and it is exactly the kind of code that is wrong for a week before you notice.[[sn: The swizzle is not cosmetic. `wgmma` reads its shared-memory operand in a fixed access pattern; if the tile is stored row-major the reads collide on banks and the instruction stalls. The 128-byte swizzle interleaves the tile so the read pattern lands one 4-byte word per bank. Doing this in scalar code means a per-element XOR of the address bits — correct, but noise.]]

[[fig: A two-panel "before/after" hand-drawn comparison titled "Who moves the tile?". LEFT panel labeled (A) THE OLD WAY in orange: a block of 8 little stick-figure threads each drawn holding a tiny data crumb, each with its own thin blue arrow reaching up into a big red-labeled GMEM matrix "A (N×N)" and pulling one element down into a shared-memory box "sharedA" drawn with blue diagonal hatch; a purple note beside the threads reads "each thread: compute index + issue cp.async + swizzle by hand"; a red warning scrawl "128–256 loads, address math burns registers". RIGHT panel labeled (B) THE TMA WAY in orange: the same 8 threads all standing idle with little "zzz" marks, ONE thread circled in orange labeled "elected leader" issuing a single fat blue arrow labeled "cp.async.bulk.tensor.2d" that lifts an ENTIRE hatched tile at once from GMEM into "sharedA"; a green box off to the side labeled "TMA engine (fixed-function copy HW)" sits on that arrow; purple note "1 instruction · HW does the swizzle". A dashed takeaway box at the bottom spans both panels: "TMA turns 256 per-thread loads into ONE descriptor-driven bulk copy — threads are freed to compute." || Left: the hand-rolled cp.async world, every thread its own courier. Right: TMA, one elected thread hands a descriptor to a copy engine.]]

## The hypothesis

The whole idea in one sentence: **describe the copy once, then let dedicated hardware perform it while every thread goes and computes.** Instead of 256 threads each moving a few elements, a single elected thread hands the TMA a small **descriptor** — a `CUtensorMap` — that says "here is a 2-D array in global memory of this shape and stride; go fetch the tile at coordinate `(x, y)` into this shared-memory address, and swizzle it on the way." The engine does the rest asynchronously. The threads that would have been couriers are now free the instant the copy is *launched*, not when it finishes.

That reframing changes three things at once. The address arithmetic moves out of the kernel and into a descriptor built once on the host. The swizzle becomes a hardware feature you request by name rather than code you maintain. And the copy becomes truly fire-and-forget — issued by one thread, awaited by all, overlapped with `wgmma`.

## Building the descriptor

The `CUtensorMap` is a 128-byte opaque blob that the TMA hardware reads to understand the layout of your tensor. You do not fill it in field by field; you call the driver API `cuTensorMapEncodeTiled`, which packs your metadata into the exact bit layout the engine expects. Crucially this happens **on the host, once**, before the kernel launches — the descriptor is a compile-time-ish fact about your matrix, not something you rebuild every iteration. It must be 128-byte aligned, and the clean path is to build it on the host and pass it in by `__grid_constant__ const` value.

You describe five things: the element type, the *global* dimensions of the full matrix, the global strides in bytes, the **box dimensions** (the tile shape you want the engine to carve out — e.g. `128 × 64`), and the swizzle mode. That last argument is where you ask for the 128-byte swizzle the tensor cores need, and from then on the hardware applies it for free on every load.

```cpp
// HOST side — built once, before launch
CUtensorMap tma_map_A{};
uint64_t gmem_shape[2]  = { (uint64_t)K, (uint64_t)M };   // full A, col-major view
uint64_t gmem_stride[1] = { (uint64_t)K * sizeof(bf16) }; // bytes between rows
uint32_t box_shape[2]   = { TILE_K, TILE_M };             // the tile we fetch
uint32_t elem_stride[2] = { 1, 1 };

cuTensorMapEncodeTiled(
    &tma_map_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    2, A_gmem_ptr, gmem_shape, gmem_stride, box_shape, elem_stride,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,          // <- the swizzle wgmma wants
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

[[fig: A hand-drawn "anatomy of a descriptor" figure titled "The CUtensorMap (128 B)". Center: a rounded rectangle drawn as a 128-byte block, divided into labeled slots stacked vertically, each slot hand-lettered in black: "data type = BF16", "global dims K×M", "global stride (bytes)", "box_dim = 128×64", "swizzle = 128B", "L2 promotion". To the LEFT, a big red-hatched matrix labeled "A in GMEM  (M×K)" with red dimension arrows ↔ M and ↕ K, and a small pale-yellow-hatched sub-rectangle inside it circled and labeled in red "the box: one tile". A blue dashed arrow runs from that box up to the "box_dim" slot of the descriptor. To the RIGHT, a green note "cuTensorMapEncodeTiled() — HOST side, ONCE" with a green arrow into the block. Below the descriptor a purple code sliver "__grid_constant__ const CUtensorMap map  (by value)". A dashed takeaway box bottom-right: "the descriptor is the tile's address book — built once, read by the TMA engine every copy". || The descriptor is a 128-byte address book: shape, stride, tile size, and swizzle, encoded once on the host.]]

## Issuing the copy, and the mbarrier that reports it done

Inside the kernel the copy itself is almost anticlimactic. One thread — an elected leader, conventionally `threadIdx.x == 0` — issues a single instruction naming the descriptor, the destination shared-memory address, the tile coordinates, and a **barrier** to signal on. Under the hood this is the PTX `cp.async.bulk.tensor.2d.shared::cluster.global.tile`; through the CUDA headers it looks like a plain function call:

```cpp
// DEVICE side — one thread launches the whole tile
if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        &sharedA[0], &tma_map_A, kIter * TILE_K, blockRow * TILE_M, barA);
    // tell the barrier how many bytes to expect for this transfer
    cuda::device::barrier_arrive_tx(barA, 1, TILE_M * TILE_K * sizeof(bf16));
} else {
    barA.arrive();               // everyone else just checks in
}
barA.wait(std::move(token));     // ALL threads block until TMA is done
```

The synchronization is where TMA quietly differs from `cp.async`. There is no `cp.async.wait_group`; instead completion rides on an **mbarrier** — a shared-memory barrier object with a byte-transaction count. This is the "expect-tx" protocol, and it is worth walking through slowly because it is the part everyone gets wrong first.

The barrier is initialized once, from inside the kernel, with the number of threads that will arrive on it. Then, each iteration, the leader thread does two things: it issues the TMA copy, and it calls `barrier_arrive_tx` announcing *exactly how many bytes* this transfer will deliver — `TILE_M * TILE_K * sizeof(bf16)`. The TMA engine, as it streams the tile into shared memory, decrements that byte count. The barrier only "flips" — releasing every waiting thread — when two conditions are both met: all threads have *arrived*, and the full expected byte count has *landed*.[[sn: This is why you tell the barrier a byte count and not just a thread count. `cp.async` completion is thread-relative ("my loads are done"); TMA completion is transaction-relative ("N bytes have arrived in this SMEM region"). The `expect-tx` value must match the transfer size exactly — get it wrong and the barrier either releases early (garbage tile) or never releases (hang). It is the single most common TMA bug.]] So `barA.wait()` is not waiting on the other threads so much as on the *bytes*.

[[fig: A hand-drawn pipeline-timeline figure titled "The mbarrier handshake". A horizontal time axis runs left to right with a red arrow labeled "time →". Three lanes stacked vertically. TOP lane labeled "leader thread (tid 0)": a box "issue cp.async.bulk.tensor" then immediately a box "arrive_tx(bytes = 128·64·2)" with a purple note "announces expected bytes". MIDDLE lane labeled "TMA engine" in green: a long green box spanning right labeled "streaming tile GMEM→SMEM (swizzled)" with a green counter scrawl "bytes remaining: 16384 → … → 0", and small blue arrows dripping data into a blue-hatched "sharedA" box drawn below. BOTTOM lane labeled "all other threads": a short box "arrive()" then a long hatched WAIT bar labeled "barA.wait() — blocked". A big orange vertical dashed line drops where the green counter hits 0, labeled in orange "barrier flips: threads-arrived AND bytes==0", and all three lanes resume together to the right into a shared box "→ wgmma on the tile". A dashed takeaway box: "completion is measured in BYTES, not in per-thread loads". || The expect-tx handshake: the leader announces a byte count, the engine counts it down, and the barrier releases everyone only when the bytes have landed.]]

## The measurement

Dropping TMA into the tensor-core kernel is the single biggest jump on the frontier. The worklog kernel that first replaces hand-rolled `cp.async` staging with a TMA descriptor plus `wgmma` on the loaded tile leaps from roughly **32 TFLOP/s to about 317 TFLOP/s** in BF16 — an order-of-magnitude step, and the point at which the kernel first feels like real hardware rather than a demo.[[sn: That 317 number is a snapshot of one intermediate kernel, not the ceiling — it is the reward for merely getting the copy engine and tensor cores talking, before double-buffering, register tuning, or clusters. Treat it as the *shape* of the win, not a fixed benchmark; the exact figure shifts with tile sizes and toolkit version.]] Nsight Compute tells the story cleanly: the memory-pipe pressure that used to come from thousands of per-thread `LDGSTS` (the `cp.async` SASS) collapses, replaced by a handful of `UTMALDG` bulk-tensor instructions, and the warp stalls that were "waiting on shared-memory writes" turn into "waiting on `wgmma`" — which is exactly the stall you *want*, because it means the tensor cores are now the bottleneck.

We have, in other words, moved back into the [compute-bound regime](the-three-regimes.html). The copy is no longer on the critical path; it happens in the background on dedicated silicon while the tensor cores chew through the previous tile. That is the whole point of the exercise, and it is why TMA is a prerequisite for everything past it rather than an optimization you sprinkle on at the end.

## Why the descriptor model is the real win

It is tempting to read TMA as "a faster memcpy," but that undersells it. The deeper shift is that **the layout knowledge left the kernel.** In the hand-rolled world, every thread carried the matrix's shape, stride, and swizzle in its own registers and recomputed addresses on every iteration. With TMA that knowledge lives in a 128-byte descriptor built once, and the hot loop shrinks to "issue copy, announce bytes, wait, compute." Registers that were spent on address math are now available for accumulators, which directly raises how big a tile each warp group can hold — and bigger tiles mean more `wgmma` per byte, which means higher arithmetic intensity, which is the whole game.

The descriptor model also unlocks the two moves that carry the kernel the rest of the way to and past `cuBLAS`, both of which we take up next. The first is **multicast**: when several SMs in a **thread-block cluster** all need the same tile of `A`, one TMA copy can fan it out to all of them through a single trip out of L2, instead of each SM paying for its own load.[[sn: TMA multicast rides on Hopper's thread-block *clusters* and distributed shared memory — a group of SMs on the same GPC that can address each other's SMEM. One `cp.async.bulk.tensor` with a cluster CTA mask writes the tile into every participating SM's shared memory at once, turning N redundant L2 reads into one. It is only worth it when SMs genuinely share operands — which, for the `A` matrix in a tiled GEMM, they do.]] The second is turning the copy engine and the tensor cores into a proper producer-consumer pipeline — **warp specialization**, where a small set of warps does nothing but issue TMA loads into a ring of shared-memory buffers while the rest do nothing but `wgmma` — so that copy and compute overlap perfectly and the tensor cores never once wait on memory.

That pipeline is where the last few points of `cuBLAS` hide. But it only becomes possible because TMA made the copy something a single thread can launch and forget. We got the courier off the compute threads' backs; next we give the courier its own dedicated warps and a queue to work.
