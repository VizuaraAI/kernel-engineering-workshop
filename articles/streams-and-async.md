By default, everything I launch on a GPU runs in a single line. I copy my inputs up, I launch a kernel, I copy the results back, and each step politely waits for the one before it to finish. That is correct, and it is also slow in a very specific way: while bytes are crawling across the PCIe or NVLink boundary, the roughly 989 dense FP16 TFLOP/s of the H100's tensor cores sitting one inch away are doing absolutely nothing. This article is about getting those two things to happen at the same time — and, at the very end, about pushing the same idea *inside* the kernel, where overlapping the global-memory load with the math is the whole reason double buffering exists.

The abstraction that buys us this overlap is the **CUDA stream**, and it is worth understanding precisely before we reach for it.

## What a stream actually is

A **stream** is an ordered queue of work for the GPU. Operations enqueued on the same stream execute in issue order, one after another; operations on *different* streams have no ordering relationship at all and the hardware is free to run them concurrently. That is the entire contract. When you write plain CUDA and never mention a stream, everything you do lands on the **default stream** (stream 0), which is why naive code is serial — you have been using one queue the whole time.[[sn: The legacy default stream is also *synchronizing*: work on it implicitly blocks work on other streams unless you compile with `--default-stream per-thread`, which gives each host thread its own independent default stream. This bites people who add streams and see no overlap.]]

The mechanical picture: the GPU has a small number of hardware queues feeding its work distributor. Each stream you create maps onto one of those queues. If two queues hold work that touches independent parts of the machine — say, a **copy engine** moving bytes over the bus while the **Streaming Multiprocessors** (SMs) grind on a kernel — the scheduler runs them at once. An H100 has dedicated copy engines precisely so that host-to-device transfer and compute are not competing for the same silicon.

[[fig: An architecture map titled "Streams = ordered queues into the GPU". On the left, host code as a purple column with three handwritten lines "cudaMemcpyAsync(H→D)", "kernel<<<...,stream1>>>", "cudaMemcpyAsync(D→H)". Two rounded queue boxes labeled in black "stream 1" and "stream 2" stacked vertically, each holding little cards in issue order with a blue arrow "in-order within a stream". A big red curved arrow between the two streams crossed out, labeled in red "no ordering ACROSS streams → free to overlap". On the right, the GPU as a large rounded rectangle containing two green sub-boxes: "Copy Engine (DMA)" and "132 SMs". Blue dashed arrows connect stream boxes to the two engines. Green note bottom-right "H100: dedicated copy engines → transfer + compute run concurrently". Dashed takeaway box: "same stream = serial · different streams = concurrent". || A stream is just an ordered queue. Concurrency comes from having more than one.]]

## The serial baseline, and why it wastes half the machine

Take a workload big enough to matter: copy a chunk of input to the GPU, run a kernel over it, copy the result back, and repeat over many chunks. Written the obvious way, it is three blocking calls in a loop.

```cpp
for (int i = 0; i < num_chunks; ++i) {
    cudaMemcpy(d_in,  h_in  + i*sz, bytes, cudaMemcpyHostToDevice);
    process<<<grid, block>>>(d_in, d_out, sz);
    cudaMemcpy(h_out + i*sz, d_out, bytes, cudaMemcpyDeviceToHost);
}
```

If a single chunk spends roughly equal time in transfer and in compute, this loop is running the copy engines and the SMs at about 50% duty each — they take turns. The wall-clock time is `sum(copy_up + compute + copy_down)` when it *could* be closer to the max of the three streams of work overlapped. On a bandwidth-heavy pipeline that is a factor of two left on the floor, and it is invisible in a FLOP/s number because the FLOPs are fine — it is the *gaps between* them that hurt. This is squarely an overhead/scheduling problem, the third of the [three regimes](the-three-regimes.html), and streams are the fix.

## Pinned memory: the precondition everyone forgets

Before overlap will work at all, the host memory has to be **pinned** (page-locked). Ordinary memory from `malloc` is pageable — the OS can move or swap it — so the GPU's DMA engine cannot safely read it directly. A `cudaMemcpyAsync` from pageable memory silently falls back to a staging copy through a driver bounce buffer, which is synchronous in disguise and destroys the overlap you were trying to buy.[[sn: `cudaMemcpyAsync` from *pageable* host memory is one of the great silent performance traps: the call returns immediately in your code but the driver still serializes the transfer through an internal pinned buffer. The stream trick then appears to "do nothing".]]

Allocate host buffers with `cudaMallocHost` (or register existing ones with `cudaHostRegister`) and two things improve at once: the transfer becomes a true async DMA, and raw H2D/D2H bandwidth climbs noticeably because the bounce copy is gone.

```cpp
float *h_in, *h_out;
cudaMallocHost(&h_in,  total_bytes);   // page-locked, DMA-able
cudaMallocHost(&h_out, total_bytes);
```

The cost is real — pinned pages cannot be swapped, so you are spending physical RAM and stressing the OS if you pin gigabytes — but for the staging buffers of a streaming pipeline it is exactly the right trade.

## Overlapping copy and compute

Now the actual technique. We create a handful of streams and round-robin our chunks across them. While stream 0 is running its kernel, stream 1 can be copying the *next* chunk up, and stream 2 can be copying a *previous* result down — all three engines busy at once.

```cpp
const int NS = 3;
cudaStream_t s[NS];
for (int i = 0; i < NS; ++i) cudaStreamCreate(&s[i]);

for (int i = 0; i < num_chunks; ++i) {
    cudaStream_t st = s[i % NS];
    cudaMemcpyAsync(d_in[i%NS],  h_in  + i*sz, bytes,
                    cudaMemcpyHostToDevice, st);
    process<<<grid, block, 0, st>>>(d_in[i%NS], d_out[i%NS], sz);
    cudaMemcpyAsync(h_out + i*sz, d_out[i%NS], bytes,
                    cudaMemcpyDeviceToHost, st);
}
cudaDeviceSynchronize();
```

The ordering *within* each stream is still guaranteed — a chunk's kernel will not start until its own upload finishes, and its download will not start until its kernel finishes. What we have removed is the *cross-chunk* serialization. Chunk `i+1`'s upload no longer waits for chunk `i`'s download.

[[fig: A pipeline timeline (Excalidraw hand-drawn, fine black ink on pure white) titled in black "Overlapping copy + compute across streams". A horizontal time axis drawn as a red arrow left→right labeled in red "t". Three horizontal lanes stacked, each a wobbly rounded box labeled in black on the left edge: "Copy Engine ↑ (H→D)", "132 SMs (compute)", "Copy Engine ↓ (D→H)". Colored task blocks tiled along the lanes with red chunk labels: the upload lane holds blue-hatched blocks "c0 c1 c2 c3", the compute lane holds pale-yellow-hatched blocks "k0 k1 k2 k3" each shifted one slot to the right, the download lane holds green-hatched blocks "d0 d1 d2 d3" shifted one slot further — forming a diagonal staircase / brick-laid pattern. A blue dashed annotation on the left "in-order WITHIN a chunk: kᵢ waits for cᵢ, dᵢ waits for kᵢ". An orange curly brace bracketing the steady-state middle columns where all three lanes overlap, labeled in orange "all 3 engines lit at once". Below, a second thin grey lane labeled in black "SERIAL (default stream)" showing the same c/k/d blocks strung end-to-end in a single row, visibly ~2× longer, with a red note "≈2× wall-clock". Green spec note bottom-left "H100: separate copy engines for ↑ and ↓ → both buses run with compute". Dashed rounded takeaway box bottom-right in black "staircase, not single-file → wall-clock ≈ max(engine), not sum". Flat, no shadows, generous white space. || The staircase. Once the pipeline fills, upload, compute, and download all run in the same instant.]]

There is a subtlety that trips up first attempts: how you *issue* the calls matters. Issuing them depth-first (all three calls for chunk 0, then all three for chunk 1) can, on some driver/hardware combinations, serialize more than issuing breadth-first (all uploads, then all kernels, then all downloads), because of how the false dependencies fill the hardware queues. On Hopper's scheduler the depth-first loop above generally overlaps fine, but if a profile shows gaps, restructuring the issue order is the first thing to try.

## Events: measuring the overlap you just bought

You cannot claim a speedup you did not measure, and CPU-side timers are the wrong tool because the async calls return before the GPU has done anything. The right tool is a **CUDA event** — a lightweight marker you drop into a stream that the GPU timestamps when it reaches that point.

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);
// ... enqueue the whole streamed pipeline ...
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);        // block host until GPU reaches `stop`

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);   // GPU-clock milliseconds
```

`cudaEventElapsedTime` reports the time the *GPU* spent between the two markers, on the GPU's own clock, which is exactly what you want for kernel timing.[[sn: Events measure device-side elapsed time with roughly microsecond resolution — far more trustworthy than wrapping the launch in a host-side `std::chrono` timer, which mostly measures launch latency and driver overhead, not the kernel.]] Events do double duty: besides timing, an event recorded on one stream can be *waited on* by another via `cudaStreamWaitEvent`, which is how you express a dependency across streams without a full `cudaDeviceSynchronize` — a producer stream signals, a consumer stream waits, and everything unrelated keeps flowing.

Run the two versions and the number is unambiguous. The cleanest evidence is a Nsight Systems (`nsys`) timeline: the serial version shows the H2D, kernel, and D2H rows lit one at a time in strict succession, whereas the streamed version shows all three rows overlapping in the steady state — the staircase made visible. A well-balanced streamed pipeline lands close to the larger of "total transfer time" and "total compute time" instead of their sum. For a workload split evenly between the bus and the SMs, that is a measured **~1.8–2× speedup** in wall-clock time, achieved without touching a single line of the kernel. The tensor cores were always fast enough; I was just starving them.

## The same idea, one level down: `cp.async`

Everything above overlaps work *across* kernel launches. The most important application of the identical principle happens *inside* a single kernel, and it is the reason the top of the GEMM ladder exists.

Recall the shared-memory GEMM pattern: each block loops over tiles of the `K` dimension, and for every tile it (1) copies a slab of `A` and `B` from global memory into shared memory, then (2) computes on that slab. Written naively those two phases are serial — the SM issues the loads, stalls waiting for HBM, and only then does the math. During the stall, the compute pipes are idle. It is the copy-then-compute problem all over again, just at the granularity of a `__syncthreads()` instead of a `cudaMemcpy`.

Pre-Hopper, hiding that latency meant issuing loads through registers and leaning on the warp scheduler to find other warps to run. Hopper gives us something better: **asynchronous copy** — `cp.async` at the PTX level, exposed as `cuda::memcpy_async` in the C++ API — which copies straight from global memory into shared memory *without* staging through registers, and, crucially, without blocking the thread that issued it.[[sn: `cp.async` also bypasses the L1/register round-trip: the data goes global → shared directly. That frees the register file and removes the load-into-register instructions entirely, which is a second, quieter win on top of the overlap.]] You fire off the copy for the tile you will need *next*, keep computing on the tile you already have, and only synchronize on the copy when you actually reach the point of consuming it.

```cpp
// pseudo-C++ for the inner K-loop
cp_async(smemA[next], gmemA + next_tile);   // fire-and-forget load
cp_async(smemB[next], gmemB + next_tile);
compute_tile(smemA[cur], smemB[cur]);        // math on the CURRENT tile
cp_async_wait();                             // now block on the load
__syncthreads();
swap(cur, next);                             // ping-pong the buffers
```

That `swap` of two shared-memory buffers is the whole trick, and it has a name: **double buffering**. You hold two tiles in shared memory — one being computed on, one being filled — and ping-pong between them so the memory system and the math pipes are never idle at the same time. It is exactly the streamed pipeline from earlier, transplanted from the host loop into the kernel's inner loop, with `cp.async` playing the role of `cudaMemcpyAsync` and the buffer swap playing the role of the round-robin over streams.

[[fig: A tiling walkthrough titled "cp.async double buffering in the K-loop", two panels. Panel (A) labeled "SERIAL (no overlap)": a matrix A drawn with red dims K and a highlighted tile (blue hatch), a fat blue arrow "global → shared" pointing to a shared-memory box, then a pale-yellow "compute" box AFTER it, with a red note "SM STALLS on HBM during load — compute idle". Panel (B) labeled "DOUBLE BUFFERED": two shared-memory boxes side by side, "buf 0 (cur)" pale-yellow with a compute unit chewing on it, and "buf 1 (next)" blue-hatched being filled by a purple-labeled arrow "cp.async fire-and-forget". A curved orange arrow labeled "ping-pong swap" loops between the two buffers. Blue note "load of NEXT tile overlaps compute of CURRENT tile". Green spec note "H100 shared mem ~228 KiB/SM (opt-in) — room for 2 tiles". Numbered circles (1) fire async load (2) compute current (3) wait+swap. Dashed takeaway box: "same staircase, now INSIDE the kernel → hides HBM latency behind the math". || Double buffering is the streamed pipeline moved into the inner loop. `cp.async` is its `cudaMemcpyAsync`.]]

## Where this lands us

Streams, events, pinned memory, and `cp.async` are four faces of one idea: **find two pieces of work that use different parts of the machine, and stop making them wait for each other.** At the host level that is copy-versus-compute overlap, worth a clean ~2× on any transfer-bound pipeline and free of kernel changes. At the kernel level it is double buffering, and it is one of the last rungs that carries the GEMM ladder from the mid-80s into the low-90s percent of `cuBLAS` — the [autotuned kernel at 84.8%](gemm-kernel-autotune.html) still stalls on HBM inside its K-loop, and hiding that stall is what a `cp.async` rewrite buys.

The mental model to keep is the staircase from the timeline figure. Whenever you see a kernel or a pipeline where one engine is busy while another sits idle, ask whether the idle one could be doing the *next* iteration's work early. On a Hopper GPU the hardware has been built to let you do exactly that — dedicated copy engines outside the kernel, `cp.async` and up to ~228 KiB of shared memory per SM inside it.[[sn: The H100's unified L1/shared block is 256 KiB per SM, of which the driver lets a kernel opt into ~227 KiB as shared memory via `cudaFuncAttributeMaxDynamicSharedMemorySize` — so "228 KiB" is the round headline figure, not an exact usable limit.]] Next we spend a full article on double buffering itself, turning this three-line sketch into a real ping-pong GEMM kernel and profiling the stall we just claimed we removed.
