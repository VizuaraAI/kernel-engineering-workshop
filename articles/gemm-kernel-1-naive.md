Matrix multiplication sits at the core of modern deep learning — every linear layer, every attention score, every MLP is a GEMM. So it is the perfect thing to learn kernels on: the algorithm is three lines of math, the naive version is trivial, and the gap between that naive version and NVIDIA's hand-tuned `cuBLAS` is a factor of **seventy**. Closing that gap, one optimization at a time, teaches you almost everything about the GPU.

This is kernel 1 of the ladder. It is deliberately the dumbest thing that works, and its only job is to give us an honest baseline and a first profile to react to.[[sn: The ladder that follows is inspired by Simon Boehm's canonical *"How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"*. We rebuild it here in our own worklog voice, kernel by kernel, with our own figures.]]

We compute `C = A · B` for square matrices, `A` and `B` both `N × N`, in FP32. The math is:

```
for m in 0..N:
  for n in 0..N:
    acc = 0
    for k in 0..N:
      acc += A[m][k] * B[k][n]
    C[m][n] = acc
```

## The hypothesis

The most natural way to parallelize this on a GPU is the one everybody writes first: **one thread per output element.** Launch an `N × N` grid of threads; thread `(m, n)` walks the inner `k` loop by itself, reads one row of `A` and one column of `B`, and writes one element of `C`. No shared memory, no tiling, no cleverness.

Here is the whole kernel:

```cpp
__global__ void sgemm_naive(int N, const float* A, const float* B, float* C) {
    const uint m = blockIdx.y * blockDim.y + threadIdx.y;
    const uint n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < N && n < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[m * N + k] * B[k * N + n];
        C[m * N + n] = acc;
    }
}
```

launched with 32×32 thread blocks:

```cpp
dim3 block(32, 32);
dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(N, 32));
sgemm_naive<<<grid, block>>>(N, A, B, C);
```

[[fig: A hand-drawn diagram titled "Kernel 1: one thread per output element". On the right, three matrices A, B, C drawn as squares with red dimension labels N×N. Matrix C has one small cell highlighted with a pale-yellow hatch and labeled in red "C[m][n]". A blue dashed arrow runs from that cell to matrix A highlighting an entire ROW (blue hatch) labeled "reads row m of A", and another blue dashed arrow to matrix B highlighting an entire COLUMN (green hatch) labeled "reads col n of B". Below, a single hand-drawn thread labeled "thread (m,n)" with a purple note "walks the k-loop alone: N mults + adds". A green handwritten note on the side: "grid = N×N threads, 32×32 per block". Bottom dashed takeaway box: "each thread does 2N flops but issues 2N global loads → intensity ≈ 1 flop/element". || Kernel 1. Every thread independently reads a full row of A and a full column of B from global memory.]]

## The measurement

On the benchmark we get a throughput of about **~300 GFLOP/s**, which comes out to roughly **1.3%** of what FP32 `cuBLAS` does on the same GPU.[[sn: Exact numbers depend on the card, but the *ratio* is remarkably stable across hardware: the naive kernel always lands in the low single digits of percent. The bottleneck is structural, not tuning.]] It works, it is correct, and it is catastrophically slow. The interesting question is *why* — and the profiler answers immediately.

Point Nsight Compute at it and the memory workload analysis lights up red. The kernel is nowhere near compute-bound; it is drowning in global-memory traffic. Two facts explain everything:

1. **No reuse.** Every one of the `N²` threads reads an entire row of `A` (`N` floats) and an entire column of `B` (`N` floats) straight from global memory. Element `A[m][k]` is re-read by all `N` threads in row `m`; element `B[k][n]` is re-read by all `N` threads in column `n`. We fetch `O(N³)` floats from HBM to do `O(N³)` flops — an arithmetic intensity of about **1 flop per element loaded**. From the [three regimes](the-three-regimes.html) we already know that puts us hundreds of times below the ridge point: hopelessly memory-bound.

2. **Bad access pattern.** Look at how threads in one warp read `B`. Adjacent threads have adjacent `n`, so on a given `k` they read `B[k][n]`, `B[k][n+1]`, … — which *is* contiguous and coalesces fine. But adjacent threads reading `A[m][k]` all share the same `m` within a row of the block and stride by `N` down the `A` column as the block's `m` varies, so the access into `A` is strided and wastes most of every memory transaction.[[sn: We will fix the access pattern *first*, in kernel 2, with nothing but a one-line change to how we assign `m` and `n` to threads. It is the highest ratio of payoff-to-effort in the entire ladder.]]

## What this tells us to do next

The profile hands us our to-do list in priority order. We are memory-bound, so — per the regime playbook — every win from here is a *memory* win, not a compute win. There are two obvious levers, and we will pull them in sequence:

- **Fix coalescing** so each memory transaction we do issue is fully used. That is kernel 2, and it alone roughly quadruples us to **8.5% of cuBLAS**.
- **Stop re-reading the same data** by staging tiles of `A` and `B` in fast on-chip shared memory and reusing them across a whole block of threads. That is kernel 3, and it is where the real climb begins.

That is the rhythm for the rest of the ladder: state a hypothesis, write the smallest kernel that tests it, profile it, read the bottleneck the profiler hands us, and let *that* — not our intuition — pick the next move. Ten kernels later we are at **93.7%** of a library NVIDIA has been tuning for fifteen years, and we will have derived every trick from a measurement rather than memorized it.
