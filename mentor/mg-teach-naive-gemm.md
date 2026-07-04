By the end of this chapter you'll be able to stand up, write the dumbest possible matrix-multiply kernel on the board, run it live, show that it hits **1.3% of the professional library** — and make the room *hungry* to fix it. This chapter is not about being fast. It's about the honest, slow start that gives every optimization afterward a reason to exist.

The teaching secret: the naive kernel is a *gift*. It is correct, simple, and catastrophically slow. That gap — between "it works" and "it's good" — is the emotional engine of the whole workshop. Your only job in this first kernel is to open the gap wide and leave students staring into it.

## Why start with something bad on purpose

Students expect the *good* kernel. Resist that. Show the fast version first and they'll memorize it and understand nothing. Instead you write the version *they* would write on day one — the obvious one — then measure it and let the measurement hurt.

[[note: say || "We are going to write matrix multiply the way you'd write it if nobody had ever told you how a GPU works. It'll be correct. It'll also be one of the slowest things you can do to a GPU while still getting the right answer. And that's exactly the point — because in four weeks we're going to make this same math seventy times faster, and today is where we take the first honest measurement."]]

The whole course is a ladder. This is rung one. Every rung after it is a *reaction to a measurement* — never a trick pulled from the air. So the discipline you teach from minute one is: write the smallest thing, measure it, let the hardware tell you what's wrong. Say it out loud. It's the spine of everything.

[[fig: A warm hand-drawn illustration titled "The ladder starts at the bottom". A friendly staircase climbing left to right, each step a little cartoon person on it. The bottom step is drawn large and labeled "Kernel 1: naive — 1.3%" with a small stick figure looking up, sweating, at the huge distance above. Faint higher steps trail up and to the right labeled "coalesce", "shared memory", "...", and a tall glowing top step labeled "cuBLAS 100%". A big blue dashed "WE ARE HERE" arrow points at the bottom step. A handwritten green sticky note reads "start slow on purpose — every step up is a reaction to a measurement". Excalidraw style, white background, charming, hand-lettered. || The mindset for the whole course: we start at the bottom on purpose, and climb one measured step at a time.]]

## The one idea: one worker per answer cell

Recall what a matrix multiply *is* (you taught this already): the answer matrix `C` is a grid, and every cell of that grid is one dot product — a row of `A` slid against a column of `B`, multiplied pair by pair and summed.

Now, how do you split that job across a GPU's thousands of tiny workers? The most natural idea — the one *everybody* writes first — is beautifully simple: **give each worker exactly one cell of the answer to fill in.**

[[note: metaphor || Picture a giant paint-by-numbers wall. The wall is the answer matrix `C` — say a thousand squares across and a thousand down, a million little squares. You hire a million painters and tell each one: "You get *one* square. Go fetch the colours you need, mix them, paint your square, go home." Every painter does the identical job — fetch, mix, fill one square — just on a different square. Nobody talks to anybody. That is the naive kernel: one painter per square, no coordination.]]

[[fig: A warm hand-drawn illustration titled "One painter per square". A large grid wall (like a mural made of blank squares) labeled "C — the answer, N×N squares". In front of it, a crowd of tiny cartoon painters, each with a beret and a brush, each standing at their OWN single square (dashed lines connect each painter to exactly one square). One painter is highlighted in orange with a speech bubble "I fetch my colours, paint MY square, leave". Off to the left, two paint-supply shelves labeled in green "Row of A (colours)" and "Column of B (colours)" with long arrows showing this one painter walking all the way over to grab a whole shelf-row and a whole shelf-column. A red handwritten note: "nobody shares — every painter walks to the pantry alone". Dashed takeaway box: "one thread = one output cell = its own trip to memory". Excalidraw style, white background, hand-lettered, friendly. || The naive kernel as a paint-by-numbers wall: one painter per square, and every painter makes their own solo trip to fetch colours.]]

Why is this the obvious move? Because a GPU is thousands of tiny cores that all want to do the *same thing to different data*. Our answer has `N²` cells, each computed the same way, so we launch an `N × N` grid of workers — one per cell — and each walks its own dot-product loop. It maps one-to-one onto the three nested loops students already know.

## The kernel, built up gently

Let's write it. In GPU code, each worker is called a **thread**, and it figures out which cell it owns from its position in the grid:

```cpp
__global__ void sgemm_naive(int N, const float* A, const float* B, float* C) {
    const uint m = blockIdx.y * blockDim.y + threadIdx.y;   // my row of C
    const uint n = blockIdx.x * blockDim.x + threadIdx.x;   // my column of C
    if (m < N && n < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)                          // walk the dot product
            acc += A[m * N + k] * B[k * N + n];
        C[m * N + n] = acc;                                  // write my one cell
    }
}
```

Walk the room through it slowly. The first two lines are the thread asking *"which cell am I?"* The `if` guard is politeness — threads launch in tidy 32×32 blocks, `N` might not divide evenly, so a few edge threads are told to sit quietly and not scribble out of bounds. The heart is the `for k` loop: fetch `A[m][k]`, fetch `B[k][n]`, multiply, accumulate. That's the dot product — the "receipt" — done by one lonely thread.

[[note: confusion || Two things trip students here. First: **why `A[m * N + k]` and not `A[m][k]`?** Because the matrix is stored as one long flat line of numbers, row after row. To reach row `m`, jump over `m` whole rows (`m * N`), then step `k` across. Draw the flat array as a ribbon and physically point. Second: **"who runs this code?"** Every thread runs the *entire* function — the code is written once but a million copies run at once, each with a different `m` and `n`. That "one program, a million runners" idea is the whole GPU model; say it plainly.]]

And the launch that fires off the million threads:

```cpp
dim3 block(32, 32);                                  // 1024 threads per block
dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(N, 32));
sgemm_naive<<<grid, block>>>(N, A, B, C);
```

That's the whole thing. It compiles, it runs, it gives the *exactly correct* answer. And it's terrible. Now comes the fun part.

[[fig: A hand-drawn technical diagram titled "Kernel 1: one thread per output cell". Right side: three squares labeled A, B, C with red dimension labels "N×N". Matrix C has one small cell highlighted pale-yellow-hatch with red label "C[m][n]". A blue dashed arrow runs from that cell to matrix A highlighting an entire ROW (blue hatch), blue label "reads whole row m of A — N floats". A second blue dashed arrow runs to matrix B highlighting an entire COLUMN (green hatch), green label "reads whole col n of B — N floats". Below, a single hand-drawn stick "thread (m,n)" with a purple note "for k: acc += A[m][k]·B[k][n]" and "walks the k-loop ALONE". A green note: "grid = N×N threads, 32×32 = 1024 per block". Numbered circles (1) on the row read, (2) on the column read, (3) on the write to C. Dashed takeaway box: "each thread: 2N flops but 2N global loads → reuses nothing". Excalidraw style, white background, hand-lettered. || The technical translation of the painter picture: each thread streams a full row of A and a full column of B from far-away memory and reuses none of it.]]

## Do the count by hand first (the napkin)

Before you run anything, count the work on the board — because the count is what makes the slowness shocking instead of abstract. Good kernel engineering starts on a napkin, not in a profiler.

[[note: example || Take a modest `N = 4092` (the size in the reference benchmark). The **work** is `2 · N³` floating-point operations — a multiply and an add for each of the `N` steps in each of the `N²` cells. That's `2 · 4092³ ≈ 137 billion` operations. The **necessary data** is just three matrices, read/written once: `3 · N² · 4 bytes ≈ 268 MB`. Divide work by data: `137 GFLOP / 268 MB ≈ 511` operations per byte. Half a thousand sums for every byte we're *forced* to move. That's a very compute-heavy job — done right, the math units should be the bottleneck, not the memory.]]

Hold that number — **511 operations per byte** — on the board. It's the promise of what GEMM *could* be. Our naive kernel is about to betray it completely, and the betrayal is the lesson.

[[fig: A hand-drawn napkin-math figure titled "What GEMM should cost". A sketched paper napkin with handwritten sums in purple and green: "work = 2·N³ = 2·4092³ ≈ 137 GFLOP" on one line; "data = 3·N²·4 = 268 MB" below it; then a big divide bar and "137 GFLOP ÷ 268 MB ≈ 511 FLOP/byte" circled in orange. A red arrow points to the 511 with the note "compute-heavy! math should be the bottleneck". A small dashed takeaway box: "on paper GEMM wants to be compute-bound — remember 511". Excalidraw style, white background, hand-lettered, drawn to look like a real napkin with a coffee-cup ring stain. || The napkin count students should see before any code runs: GEMM ought to be compute-bound at ~511 operations per byte.]]

## The live demo: run it, read the number

This is the centerpiece of the block. You run the kernel in front of them and read the number aloud.

[[note: demo || Compile and run the naive kernel on the benchmark. It reports about **309 GFLOP/s**. Pause. Let that land as a *big* number — "three hundred billion operations every second!" Then, on the same machine, run `cuBLAS` (NVIDIA's professional library) doing the identical math: about **24 TFLOP/s**. Now do the division on the board, live: `309 / 24000 ≈ 0.013`. **We are at 1.3% of the library.** Write "1.3%" huge and circle it. That single fraction is the hook for the entire four weeks.]]

[[note: aha || The jaw-drop line: "We just left **ninety-nine percent** of a machine that costs as much as a car sitting completely idle. Same math. Same chip. Same answer. One is seventy times faster than the other — and the only difference is how we arranged *who reads what, when*. That gap is what you're here to learn to close." Watch the room. This is the moment they get hungry.]]

[[fig: A hand-drawn "scoreboard" figure titled "Naive vs the pros". Two big bars side by side on a simple axis labeled "speed". A short red bar labeled "OUR naive kernel — 309 GFLOP/s" that only reaches a tiny way up. A towering green bar labeled "cuBLAS — 24,000 GFLOP/s" that shoots off the top of the chart. A huge orange handwritten callout between them: "1.3%". A red sad-face doodle on the little bar and a small trophy doodle on the tall one. Dashed takeaway box: "correct, but using 1.3% of the machine — 99% left on the floor". Excalidraw style, white background, hand-lettered, a little playful. || The scoreboard that makes the room gasp: our correct kernel reaches 1.3% of the professional library.]]

## Why is it so slow? Let the hardware tell you

Model the difference between a beginner and an engineer. The beginner shrugs and randomly changes block sizes. The engineer opens the profiler and asks the hardware what's wrong. Teach the second reflex.

Point the profiler (**Nsight Compute**, `ncu`) at the kernel and the memory section lights up red. It isn't compute-bound at all — it's *drowning in memory traffic*. The reason is one word: **reuse**, or rather the total lack of it.

Back to the painters. Painter `(m, n)` walks to the pantry, grabs the *entire* row `m` of `A` and column `n` of `B`, uses them once, throws them away. Now the painter next door — cell `(m, n+1)` — needs the *exact same row `m` of `A`*... so she walks to the pantry and fetches it *all over again*. Every element of `A` gets re-fetched by all `N` painters in its row. Nobody kept anything on their tray for a neighbor.

[[note: aha || Count the actual bytes moved and watch the second gasp. Each of the `N²` threads loads `2N + 1` floats. So the real traffic is `N² · (2N+1) · 4 bytes ≈ 548 GB`. The *minimum* was **268 MB**. The naive kernel moves about **two thousand times** more data than necessary — because every thread re-reads the same handful of matrices from scratch. That 2000× is where the 1.3% comes from.]]

[[fig: A hand-drawn "zoom-in" figure titled "The reuse we throw away". Left: a single row of the answer C with four adjacent cells C[m][0..3], each with its own tiny painter beneath it. From ALL FOUR painters, four separate dashed arrows reach back to the SAME single row m of matrix A (blue hatch), all converging on it, with a red starburst "×N re-fetches!". Orange callout: "the same row of A is walked to the pantry once PER painter". Right: a small ledger in purple and green handwriting — "minimum: 268 MB" then below in red "naive actually moves: ≈ 548 GB" then a huge orange "≈ 2000× waste". Dashed takeaway box: "nobody keeps data on-chip ⇒ everyone re-reads global memory ⇒ intensity collapses to ~1 FLOP/byte". Excalidraw style, white background, hand-lettered. || Zoom in on one row: the same data is fetched from far-away memory once per thread. That factor-of-N re-read balloons 268 MB into 548 GB.]]

Remember our napkin promised **511** operations per byte. The naive kernel, by refusing to reuse, drags that down to about **1** operation per byte — it falls off the compute roof and lands in the memory-bound basement. The GPU's beautiful math units sit idle while the memory system thrashes.

[[note: production || This isn't academic. Right now, data centres full of NVIDIA H100 and B200 GPUs — the machines serving DeepSeek, Llama, ChatGPT — spend most of their electricity multiplying matrices. A naive-style kernel that reuses nothing would waste 90%+ of hardware that costs tens of thousands of dollars per chip. The reason kernel engineers are paid so well is precisely this gap: the exact same math, arranged well, turns a 1.3% kernel into a 90% one, which turns a hundred-GPU cluster into a ten-GPU cluster. Your students are learning the operation that decides whether a model costs a dollar or a penny to run.]]

## What the profile tells us to do next (the cliffhanger)

Don't fix it in this chapter. The whole pedagogy is that the *profile* hands us the next move. So end by pointing at the two things the profiler flagged, in priority order, and leave them as a promise:

- **First, coalescing.** The reads a warp of 32 threads makes are scattered across memory instead of contiguous, so most of every memory transaction is wasted — the kernel uses about **2% of the bandwidth** it's paying for. The fix is *one line* rearranging how threads map to cells, and it roughly quadruples us to **8.5% of cuBLAS**. Best payoff-to-effort on the whole ladder.

- **Then, reuse.** Coalescing makes each read efficient, but we're still doing `N`× too many reads. To kill that, we stage tiles of `A` and `B` in fast on-chip **shared memory** and share them across a whole block of painters. That's where the real climb begins.

[[note: teach || **Board plan for a ~45-minute block.** (0–5 min) Recap: matmul is a grid of dot products. (5–12) Draw the paint-by-numbers wall; introduce "one thread per output cell"; write the kernel line by line, no jargon yet. (12–18) The napkin count — 137 GFLOP, 268 MB, 511 FLOP/byte on the board. (18–28) **Live demo**: run it → 309 GFLOP/s → run cuBLAS → 24 TFLOP/s → compute 1.3% live and circle it. (28–38) "Why?" — the painters re-fetching, the 268 MB → 548 GB blowup, ~2000× waste. (38–43) Open the profiler, show it's memory-bound, name the two fixes as next lessons. (43–45) Checkpoint questions. **The one demo** is the 1.3% reveal — everything builds to it. **The jaw-drop number** is 2000× wasted traffic. **Checkpoint questions:** "Why does thread (m, n+1) re-read the same row of A?" and "Our napkin said 511 FLOP/byte — why did the real kernel only get ~1?"]]

Notice what you modeled: you didn't guess a fix. You wrote the dumbest correct thing, measured it, and let the hardware hand you a prioritized to-do list. That rhythm — **hypothesis → smallest kernel → profile → let the bottleneck pick the next move** — is the discipline of the whole workshop. State it before you close.

## You can now teach

- **Why we start slow on purpose** — the naive kernel is a correct, honest baseline whose slowness is the emotional hook for the whole course.
- The **one-thread-per-output-cell** idea as a paint-by-numbers wall, and the actual CUDA kernel built up line by line without jargon.
- The **napkin count** — 137 GFLOP of work, 268 MB of necessary data, ~511 FLOP/byte — and why GEMM *should* be compute-bound.
- The **live 1.3%-of-cuBLAS demo**: run it, read 309 GFLOP/s, divide by cuBLAS live, and make the gap visible and painful.
- **Why it's slow** — no reuse: the same rows and columns get re-fetched `N` times, blowing 268 MB up to ~548 GB, about 2000× waste.
- The **measure-don't-guess discipline** and the two fixes the profiler hands you next (coalescing, then shared-memory reuse) — leaving the room hungry to climb.
