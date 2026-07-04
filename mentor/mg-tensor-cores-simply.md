By the end of this chapter you can stand at a whiteboard and teach what a **tensor core** is — a special little machine inside the GPU that eats whole tiny matrices in one gulp — and explain, without hand-waving, why almost all the number-crunching in modern AI happens inside it, and why every fast kernel is built *around* it like a kitchen built around one enormous oven.

You already taught students two things in earlier chapters. First: a matrix multiply is a grid of dot products, and it costs a mountain of multiply-adds. Second: a GPU is a cafeteria — thousands of simple cooks doing the same tiny sum at once. This chapter adds the plot twist. Inside that cafeteria, there is a *second*, much stranger machine. It does not scoop one number at a time. It swallows a whole small tray of numbers in a single bite. That machine is the tensor core, and it is where the AI economy actually lives.

## The one-sentence answer

A normal GPU worker — a **CUDA core** — does one multiply-and-add at a time: `a × b + c`, one number in, one number out. A **tensor core** does the same idea, but one whole dimension bigger. Instead of multiplying two *numbers*, it multiplies two tiny *matrices* and adds a third — all in a single instruction.

[[note: metaphor || The espresso machine vs. the industrial oven. A CUDA core is a barista pulling one shot of espresso at a time — quick, flexible, but one cup per pull. A tensor core is a giant pizza oven that bakes a whole tray of pizzas in one go. You can't ask the oven for a single espresso; it only knows how to do a *whole tray*. But when your job is "feed a stadium," the oven that bakes a tray at a time is a hundred times the throughput of any barista. Modern AI is a stadium-sized order for the same tray, over and over.]]

[[fig: A warm hand-drawn split illustration titled "One shot vs one tray". Left half labeled "CUDA core: the barista" — a small friendly barista figure pulling a single espresso cup, with a blue handwritten note "one multiply-add at a time: a x b + c". Right half labeled "Tensor core: the pizza oven" — a big cheerful industrial oven with a whole tray of little pizzas sliding in, each pizza labeled with a tiny number, and a green handwritten note "a whole tiny matrix x matrix, in ONE bite". A dashed divider between the halves. A dashed takeaway box spanning the bottom: "the oven can't make one espresso — but for a stadium order, the tray wins by 100x". Excalidraw style, white background, charming, handwritten labels. || The core metaphor: a CUDA core serves one cup; a tensor core bakes a whole tray at once.]]

## What the tensor core actually computes

Write this on the board and box it, because everything hangs off it:

```
D = A · B + C
```

where `A`, `B`, `C`, and `D` are all small **matrices**, not single numbers. This one operation has a name — a **Matrix Multiply-Accumulate**, or **MMA**. "Multiply" is the `A · B` part. "Accumulate" is the `+ C` part, and that plus-C is not decoration — it is the whole trick.

Here is why the `+ C` matters. One MMA almost never finishes a whole answer tile by itself. It computes a *slice* of the answer and adds it onto a running total that is kept nearby. Then the next MMA computes the next slice and adds it onto the *same* running total. The tensor core marches along, folding slice after slice into one accumulator — exactly the `k`-loop dot-product from the matmul chapter, but now done a whole tile at a time.

[[note: example || Do the shapes by hand. The classic Hopper/Ampere tensor-core shape is written `m16n8k16`. That means: `A` is a 16×16 tile, `B` is a 16×8 tile, and `C`/`D` is a 16×8 tile. Count the multiply-adds it does in one instruction: 16 × 8 × 16 = **2048**. So a single tensor-core instruction is not one multiply-add — it is *two thousand and forty-eight* of them, fired as one. Write "1 vs 2048" on the board and let it sit.]]

[[note: aha || Here is the number that makes the room go quiet. A CUDA core does **1** multiply-add per instruction. A tensor core does **2048**. That is the grain of the modern chip: the unit of work is not one flop — it is a whole tiny matmul. When students internalize "2048 at a time," the rest of the course stops feeling like magic and starts feeling like plumbing.]]

[[fig: A hand-drawn diagram titled "One MMA = D = A·B + C". Center: three hatched matrices in a row — a blue-hatch rectangle labeled in red "A (16x16)", a green-hatch rectangle labeled in red "B (16x8)", and a pale-yellow-hatch square labeled in red "C/D (16x8)" — joined by a hand-drawn dot, a plus, and an equals. Above the row a purple handwritten instruction "mma.sync ... m16n8k16". A green note on the right: "16 x 8 x 16 = 2048 multiply-adds in ONE instruction". A blue dashed arrow curving into C labeled "accumulate: D reuses C, sums over the k-loop". An orange callout bottom-left: "the grain of the whole chip: not 1 flop, 2048". A dashed takeaway box: "the tensor core does matmul, not multiply". Excalidraw style, white background, handwritten. || A single MMA multiplies two small tiles and accumulates into a third. A whole GEMM is thousands of these marching along k.]]

## Why ~95% of the FLOPs live in this tiny unit

Now the surprising part, and it's a great "wait, what?" moment for a class.

There are only **four tensor cores per SM** on an H100 — one per warp scheduler. Across the whole chip's ~132 SMs, that is only a little over five hundred tensor cores on the entire die. They are big and few — the *opposite* of the "thousands of tiny cooks" picture. And yet: roughly **95% of the H100's headline compute throughput comes from these few units.**

How can five hundred units out-muscle tens of thousands of CUDA cores? Because each one retires 2048 multiply-adds per instruction while a CUDA core retires one. NVIDIA's own phrasing: a tensor core does about **100× more floating-point operations per second** than a CUDA core.

[[note: example || Put the two ceilings side by side, because this comparison is the spine of the whole second half of the course. The H100's plain CUDA-core FP32 throughput is around **60 TFLOP/s**. Its BF16 tensor-core throughput is **989 TFLOP/s**. Write both. Then say: "If you write a matmul that never touches a tensor core, the best you can *ever* do — perfect everything, zero stalls — is that 60. cuBLAS lives up on the 989. That gap is not your skill. It's which machine you're standing on."]]

[[note: production || This is why NVIDIA is one of the most valuable companies on Earth. When you chat with Llama, DeepSeek, or ChatGPT, the words become matrices and get pushed through hundreds of layers — and almost every one of those matmuls is being fed to a tensor core in a data center right now. The tensor core is, quite literally, where the electricity bill of AI is spent. Nearly all of it flows through these few square millimeters of silicon per SM.]]

[[fig: A hand-drawn "few but mighty" illustration titled "500 ovens out-bake 10,000 baristas". Left: a dense crowd of many tiny barista figures labeled in blue "~tens of thousands of CUDA cores, 1 each", with a small total "~60 TFLOP/s FP32". Right: just a handful of big oven figures (draw about five) labeled in green "~500 tensor cores, 2048 each", with a big total "989 TFLOP/s BF16". A bold orange arrow from left to right labeled "~95% of the FLOPs live in the few ovens". A red note: "big and few, not small and many". A dashed takeaway box: "each oven does 2048 at a time -> the few win overwhelmingly". Excalidraw style, white background, charming, handwritten. || A handful of large tensor cores carry almost all the throughput, because each one does 2048 multiply-adds per instruction.]]

## The catch: you feed a whole warp, not a thread

Here is the part that reshapes how kernels are written, and the part students find genuinely weird the first time.

A tensor-core instruction is not run by one thread. It is a **warp-level** instruction: all **32 threads** of the warp must execute it *together*, in lockstep. That is what the `sync` in `mma.sync` means — the whole warp arrives, or the behavior is undefined. There is no "one thread does an MMA."

And the little `A` tile does not sit in one place. The 16×16 tile of `A` is **smeared across all 32 threads** — a few elements held in each thread's registers, in a specific, fussy layout the hardware demands. These per-thread slivers of the tile are called **fragments**. Each thread holds a few registers of `A`, a few of `B`, a few of the `C`/`D` accumulator. The tensor core reads all of those registers across all 32 threads at once, does the matmul, and writes the accumulator back. With 2048 multiply-adds shared over 32 threads, each thread contributes exactly `2048 / 32 = 64` of them.

[[note: metaphor || The tandem bicycle for 32. A CUDA core is one person on one bike — pedal your own pedals, go. A tensor core is a 32-seat tandem. Nobody rides alone, and nobody holds the whole handlebar — each rider grips one little piece. The bike only moves if all 32 push at the exact same moment, in the exact seating order the frame was built for. Sit in the wrong seat, or leave one seat empty, and it doesn't roll — it crashes. That seating chart is the fragment layout.]]

[[note: confusion || The number-one confusion here: students think "I'll load my tile into shared memory and then just multiply it." No. The tensor core will not accept a tile lying in shared memory in whatever order was convenient. It demands the data already *smeared across the 32 threads' registers* in one exact layout. The fix, and the sentence that unlocks it: "You don't hand a tile to a thread. You hand it to a whole warp — pre-arranged in the seating chart the silicon dictates." That reframing is why the next point exists.]]

[[fig: A hand-drawn "fragment layout" diagram titled "A tile is smeared across the warp". Left: a 16x16 grid labeled in red "A fragment (16x16)", cells shaded in a repeating pattern with small labels "T0", "T1", "T2" ... showing thread 0 owns one little pair of cells, thread 1 the next, and so on. A purple note pointing at two highlighted cells: "T0 holds {a0, a1}". Right: a vertical stack of 32 tiny register boxes labeled "thread 0 ... thread 31", each holding a few blue A-slivers, green B-slivers, yellow C-slivers, all feeding down through a big black funnel labeled "TENSOR CORE" into a yellow accumulator tile. A blue note: "no single thread sees a whole row — the warp cooperates". A green spec note: "64 multiply-adds per thread per instruction". A red warning with a dashed arrow: "all 32 threads or UNDEFINED". A dashed takeaway box: "you feed a warp, in the layout the silicon dictates". Excalidraw style, white background, handwritten. || Operands live as fragments smeared across all 32 threads. This exact layout, not your convenience, is what the hardware requires.]]

Because arranging that seating chart by hand with ordinary loads is miserable and slow, there is a dedicated instruction that does it for you: **`ldmatrix`**. It grabs a rectangular tile out of shared memory and shuffles the elements across all 32 threads into fragment layout in one shot — even transposing `B` on the way in if you ask. `ldmatrix` is the little machine that seats all 32 riders correctly before the tandem sets off.

[[sn: This is also why shared-memory layout and "bank conflicts" matter so intensely for tensor-core kernels specifically. `ldmatrix` reads shared memory in a fixed pattern; if you laid your tile out naively, that read pattern collides with itself and stalls. A whole genre of tensor-core bug is really an `ldmatrix` bank-conflict bug in disguise.]]

## Three ways to ask for it (mention, don't drill)

Students don't need to write this code today, but they should hear the three names, because each is a rung on a performance ladder.

- **`wmma`** — the friendly CUDA C++ API. It hides the fragment layout for you. Readable, portable, a great place to *start*. It also leaves roughly 20% of performance on the table, because it makes safe, conservative choices you can't override.
- **`mma.sync`** — the raw PTX instruction. You place the `ldmatrix` calls, you own the fragment layout, you get full control per warp. This is what fast open-source kernels are built on.
- **`wgmma`** — **Warp-Group MMA**, new on Hopper. It bumps the cooperating unit up from one warp (32 threads) to a *warp group* of four warps (128 threads) issuing one giant **asynchronous** MMA together — and it can read operand `A` straight from shared memory instead of registers. On Hopper, near-peak GEMM is a `wgmma` story.

[[note: say || "There are three ways to talk to the oven. `wmma` is the easy button — it works, but it won't win a race. `mma.sync` is manual gears — more work, full speed. `wgmma` is the new Hopper way: four warps team up and the oven runs *asynchronously*, so it bakes while the next tray is still being loaded. You'll climb this exact ladder in the back half of the course."]]

[[sn: `wgmma` being asynchronous is the whole point: you launch it, it runs on the tensor core, and you sync later. That lets Hopper kernels overlap the MMA with copying in the *next* tile — the software-pipelining trick that gets cuBLAS past 90% of peak. Blackwell pushes this further with a dedicated Tensor Memory; that's a later chapter.]]

## Precision is a throughput knob

One more thing the tensor core hands you: a choice of *input* precision, which is a direct speed dial. The accumulator stays FP32 (you want the running sum to stay accurate), but the `A` and `B` inputs can be narrower — and every step narrower roughly doubles the throughput.

- **TF32** (19-bit) — a nearly-free upgrade that lets FP32-style work run on tensor cores at several times the CUDA-core rate.
- **FP16 / BF16** (16-bit) — the workhorses. This is the **989 TFLOP/s** path. BF16 keeps FP32's exponent range for stable training.
- **FP8** (8-bit) — roughly **2× BF16** on Hopper, at the cost of precision you manage with scaling. The inference frontier.

[[note: production || This is exactly the lever DeepSeek and others pull to serve models cheaply. Training and serving in FP8 instead of BF16 roughly doubles tensor-core throughput — meaning half the GPUs, or half the time, for the same work. Blackwell adds NVFP4 (4-bit) for another doubling. Every step down the precision ladder is real money saved, which is why "quantization" is one of the hottest topics in production inference right now.]]

[[fig: A hand-drawn "precision pyramid" titled "Narrower inputs -> more FLOP/s". A vertical stack of tiered boxes, widest at the bottom. Bottom tier (widest), yellow fill, red label "TF32 (19-bit)", green note "several x FP32, nearly free". Next tier, blue-hatch, red label "BF16 / FP16 (16-bit)", orange callout "989 TFLOP/s - the workhorse". Next, green-hatch, red label "FP8 (8-bit)", green note "~2x BF16, needs scaling". Top tier (narrowest), dashed outline, red label "NVFP4 (4-bit) - Blackwell", purple note "another doubling". On the right, a blue up-arrow labeled "throughput up" and a red down-arrow labeled "precision down". A dashed takeaway box: "same tensor core, narrower inputs -> you trade bits for FLOP/s". Excalidraw style, white background, handwritten. || Every step down in input width roughly doubles tensor-core throughput. The accumulator stays FP32; the inputs are the dial.]]

## Why the whole kernel is shaped around the tensor core

Now tie the bow. The tensor core is *so* fast that it will chew through a tile in a couple of cycles — but memory (HBM3) can only deliver about **3.35 TB/s**. Do the arithmetic and you find you must reuse every byte you load *hundreds* of times, or the oven sits there empty and hot, waiting for dough. That single fact dictates the entire shape of a fast GEMM:

1. **Tile into shared memory and registers** — so the tensor core never starves.
2. **Get the fragment layout exactly right with `ldmatrix`** — so no MMA stalls on a bad tile.
3. **Overlap copy with compute** — load the next tile while the tensor core grinds the current one, so the silicon is never idle.

[[note: teach || Here's the board sequence for the whole chapter. (1) Draw the barista and the oven — one shot vs one tray. (2) Box "D = A·B + C" and count 16x8x16 = 2048 by hand. (3) Write "1 vs 2048," then "60 vs 989 TFLOP/s" — the two jaw-drop numbers, back to back. (4) Draw the 32-seat tandem / fragment smear — this is where the "kernels look weird" feeling gets explained. (5) End on the pipeline: copy and compute overlapping. If you land only two things, land "2048 at a time" and "the whole craft is keeping the oven fed."]]

[[fig: A hand-drawn pipeline timeline titled "Feeding the beast: overlap copy with compute". A horizontal time axis (black arrow labeled "time" in red). Two stacked lanes. Top lane: blue boxes labeled "copy tile 0", "copy tile 1", "copy tile 2" marching left to right. Bottom lane: yellow-hatch boxes labeled "tensor core on tile 0", "on tile 1", "on tile 2" - each offset one step right so it overlaps the NEXT copy, with grey overlap shading. Blue dashed arrows from each finished copy down to the matching compute box. A green spec note: "tensor core 989 TFLOP/s vs HBM3 3.35 TB/s". An orange callout on the overlap: "the oven bakes WHILE the next tray loads - no idle silicon". A red note under a naive single-lane strip at the bottom: "no overlap = tensor core waits on memory = stall". A dashed takeaway box: "the last 10% to cuBLAS is hiding the copy behind the compute". Excalidraw style, white background, handwritten. || Software pipelining: the asynchronous tensor core lets the next tile's copy hide entirely behind the current tile's math.]]

None of the rest of this course is about doing the math faster. **The math is already the fast part.** Every technique from here on is about arranging bytes so this little machine never waits. That is the frame to leave students with: the tensor core is the oven that does 95% of the cooking, and kernel engineering is the art of keeping it fed.

## You can now teach

- What a **tensor core** is: a unit that computes `D = A·B + C` on whole tiny *matrices*, not single numbers — the espresso-vs-oven metaphor.
- The **2048-at-a-time** grain of the chip (16×8×16 done by hand), and why it makes one tensor-core instruction worth two thousand CUDA-core ones.
- Why **~95% of the FLOPs** live in only ~500 big, few tensor cores — and the "60 vs 989 TFLOP/s" ceiling that explains why non-tensor-core kernels stall far below cuBLAS.
- The **warp-cooperation catch**: MMAs are warp-level, operands are smeared across 32 threads as fragments (the 32-seat tandem), and `ldmatrix` seats them — which is why tensor-core kernels look strange.
- The three rungs — **`wmma` → `mma.sync` → `wgmma`** — and precision as a **throughput knob** (TF32 → BF16 → FP8 → NVFP4), the lever behind cheap inference today.
- The **big frame**: the whole kernel is shaped around feeding this machine — tile, lay out fragments, overlap copy with compute — because the math is fast and the feeding is the hard part.
