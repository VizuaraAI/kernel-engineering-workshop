By the end of this chapter you can stand at a whiteboard and teach the whole tensor-core rebuild — fragments, `ldmatrix`, swizzling, and the one `mma` instruction underneath it all — so that a student who has only ever seen a "one thread, one number" GPU kernel walks away understanding why we throw that mental model out and start a brand-new ladder. And, crucially, you can teach it *without* drowning anyone in the WMMA API.

This is the second time your students climb the GEMM ladder. The first time (the SIMT ladder) they tuned a scalar kernel until it hit 93.7% of cuBLAS. That number felt like victory. Your job in this chapter is to explain, gently, why it was a *lie of omission* — and then to walk them back down to the bottom of a new, higher ladder. Let's build it the way you'll build it for them.

## Start with the punchline: the floor is above the ceiling

Here is the sentence that reframes everything. When your students hit 93.7% of cuBLAS last time, that was 93.7% of cuBLAS *running on the CUDA cores* — the ordinary scalar calculators. But cuBLAS has not seriously used the CUDA cores for matrix multiply since 2017. The real library runs on a different piece of silicon: the **tensor cores**.

[[note: aha || Say this and let it land: "Everything you built last week — every clever tile, every coalesced load — was a beautiful race car. But you were racing on a go-kart track. The real library was on the highway the whole time. Today we get on the highway. And here is the shocking part: the *dumbest possible* tensor-core kernel, with none of your tricks, is already faster than your best scalar kernel. The floor of this new ladder sits above the ceiling of the old one." That single fact is the emotional hook of the whole chapter.]]

[[fig: A warm hand-drawn illustration titled "The floor above the ceiling". Two ladders drawn side by side against a hand-sketched wall. The left ladder is short, labeled in blue "SIMT / CUDA-core ladder", with its top rung labeled in red "93.7% of cuBLAS — our best". The right ladder starts HIGHER than the left one's top rung and rises much further; its bottom rung is labeled in green "naive tensor-core kernel" and sits clearly above the left ladder's top. A little climber figure is stepping across a dashed orange bridge from the top of the left ladder to the bottom-right rung of the right ladder, with an orange note "even the floor here beats everything over there". A dashed takeaway box at the bottom: "new hardware = new ladder. Start again at the bottom — but the bottom is higher." Excalidraw style, white background, charming, hand-lettered. || The reframe students need: tensor cores are a whole new, taller ladder, and its lowest rung already beats the old ladder's peak.]]

## What a tensor core actually is (say it plainly)

A **tensor core** is a small piece of hardware, tucked inside each streaming multiprocessor, that does an *entire tiny matrix multiply* in one instruction. Not one multiply-and-add — a whole little matrix-times-matrix, accumulated, in a handful of clock cycles.

[[note: metaphor || The bricklayer vs. the wall-panel crane. A CUDA core is a bricklayer: it lays one brick (one multiply-add) at a time, very fast, but one at a time. A tensor core is a crane that lifts a whole pre-made wall panel — a 16×16 slab of the answer — and sets it in place in a single motion. Same wall at the end. Wildly different speed. The tensor core doesn't do *harder* math; it does a whole *tile* of the easy math at once.]]

Put the number on the board so the crane feels real. An H100 has around 132 SMs, four tensor cores in each. Together they are rated at about **989 TFLOP/s** in BF16 — call it a thousand trillion multiply-adds a second. The CUDA-core FP32 peak on the same chip is roughly a *tenth* of that. That factor of ten is why we bother.

[[note: production || This is not a lab curiosity. When you chat with DeepSeek, Llama, or ChatGPT, the matrix multiplies that generate every word are running on exactly these tensor cores. NVIDIA became one of the most valuable companies on Earth largely because it built the best version of this crane. Every serious serving stack — vLLM, FlashAttention — is written to keep these units fed. The tensor core is where the AI economy's electricity is actually spent.]]

## The one mental shift: the WARP owns the tile

This is the single hardest idea in the chapter, and everything else depends on it. Teach it slowly.

On the old ladder, the rule was: **one thread, one output number**. Thread 47 computes cell C[3][5], all by itself, holding its own row and column in its own registers. Simple.

The tensor core breaks that rule completely. A tensor-core instruction is issued by **all 32 threads of a warp at once**, together, as one collective act. The input tiles and the output tile are spread across the registers of the *whole warp*. No single thread holds a full row. No single thread holds a full column. The 32 threads pool their registers, the hardware reads that pool as three little matrices, multiplies them, and writes the answer back into the pool.

[[note: say || "Forget 'one thread, one number.' On the tensor core, a whole warp — all 32 threads holding hands — does one tile together. Nobody owns a row. Nobody owns a column. The *warp* owns the tile. If you try to picture a single thread doing its own dot product here, you will get lost. So don't. Zoom out: one warp, one 16×16 tile, one instruction."]]

[[fig: A hand-drawn "two mental models" figure titled "The shift: thread → warp". Left panel labeled in blue "OLD: SIMT" — a single little worker figure holding one row-strip and one column-strip, dropping one number into a single highlighted cell of a grid, red note "1 thread = 1 element". Right panel labeled in green "NEW: tensor core" — 32 tiny worker figures drawn holding hands in a ring, and an orange curved arrow wrapping all 32 into ONE big block that sets down an entire pale-yellow-hatched 16×16 tile at once onto the answer grid, red note "1 warp = 1 tile". A dashed takeaway box spanning both: "stop thinking per-element. Think per-tile, per-warp." Excalidraw style, white background, hand-lettered. || The core reframe: SIMT is one thread per element; the tensor core is one warp per tile. This is the idea the whole chapter rests on.]]

## The gentle on-ramp: WMMA and fragments

Because "the operands are scattered across 32 threads' registers in a pattern nobody can memorize" is terrifying, NVIDIA built a friendly wrapper called **WMMA** (Warp Matrix Multiply-Accumulate). WMMA's entire job is to *hide* that scattering. This is your on-ramp, and you should teach it exactly as a set of sealed boxes.

The sealed box is called a **fragment**. A fragment holds one operand tile. Here is the rule you drill into students: **you never look inside a fragment.** You never index it. You do not know which thread holds which element — and you don't need to. WMMA promises only this: if you *load* into a fragment and *hand* it to the matching multiply, the pieces line up.

[[note: metaphor || A fragment is a sealed shipping crate handled by a 32-person team. You don't repack it, you don't peek inside, you don't ask which worker is holding which screw. You trust the loading dock (`load_matrix_sync`) to pack it right, and you trust the assembly machine (`mma_sync`) to unpack it right. Your only job is to move sealed crates between three stations. The moment a student tries to open a crate, they're lost — so tell them: these crates are welded shut on purpose.]]

[[fig: A warm hand-drawn illustration titled "WMMA: three sealed crates, four moves". A friendly loading-dock scene. On the left, a labeled dock "load_matrix_sync" where a small crew loads a blue crate marked "A frag (16×16)" and a green crate marked "B frag (16×16)" from a memory shelf. In the center, an "mma_sync" machine drawn as a big press/crane that takes the A and B crates plus a pale-yellow crate marked "acc (FP32)" and stamps the accumulator crate fuller. On the right, "store_matrix_sync" sends the finished yellow crate out to a truck labeled "C in memory". Above the acc crate, a small "fill_fragment = empty it first" tag. Every crate is drawn welded shut with a red padlock and a red note "never open — layout is sealed". A dashed takeaway box: "3 crates (A, B, acc) · 4 moves (fill, load, mma, store) — that's nearly the whole API." Excalidraw style, white background, charming, hand-lettered. || The whole WMMA API as a loading dock: three sealed fragment crates and four warp-collective moves, no peeking inside.]]

There are only three kinds of crate and four things you ever do. That is nearly the whole API — say that out loud, because students expect an API to be huge.

The three crates:
- an **A fragment** — a 16×16 tile of the left matrix,
- a **B fragment** — a 16×16 tile of the right matrix,
- an **accumulator fragment** — the 16×16 running total, kept in FP32.

The four moves:
- `fill_fragment(acc, 0)` — empty the accumulator crate before you start,
- `load_matrix_sync(frag, ptr, ldm)` — the whole warp loads a 16×16 tile from memory into a crate,
- `mma_sync(acc, a, b, acc)` — the crane does `acc = a·b + acc`,
- `store_matrix_sync(ptr, acc, ldm)` — write the finished tile back out.

[[note: confusion || Every four operations here end in `_sync`, and students think that's about threads "synchronizing" like `__syncthreads()`. It isn't. `_sync` means *warp-collective*: all 32 threads must call it, together, with the same arguments. Hide one inside a divergent `if` and you don't get a compile error — you get silent garbage. The fix sentence: "these aren't functions a thread calls; they're moves the whole warp makes in lockstep. If one thread skips it, the crate is packed wrong and nobody tells you."]]

[[note: teach || Don't teach the fragment layout. Not on day one, maybe not ever for WMMA. The entire selling point of WMMA is that the layout is a sealed secret. If a student asks "but which element does thread 5 hold?", answer: "WMMA won't tell you, and that's the feature. We'll open that box later, at the very top of the ladder, when we're desperate for the last bit of speed." Naming the opacity as a *deliberate design choice* prevents an hour of confused questions.]]

### Precision: the crane eats 16-bit, sums in 32-bit

One thing to flag clearly: the tensor core does not eat FP32. Its native diet is **16-bit inputs** (FP16 or bfloat16), which it multiplies, and it keeps the running sum in **FP32**. So A and B are `half`, C is `float`. Tell students this is not a compromise you're tolerating — it's the exact shape the silicon was built for. Sixteen-bit in, thirty-two-bit accumulate.

## The tiny by-hand version, and the naive kernel

Now make it concrete with numbers small enough to hold in your head, then show the whole kernel.

[[note: example || On the board, shrink everything. Pretend the tile is 2×2 instead of 16×16 and K is 4. A warp's job: zero the accumulator `[[0,0],[0,0]]`. Step down K in chunks of 2. Load a 2×2 tile of A, load a 2×2 tile of B, do one `mma` — that adds a 2×2 partial product into the accumulator. Do it again for the next K-chunk, adding on top. After all K-chunks, the accumulator holds the finished 2×2 tile, and you store it once. Now say: "the real hardware does this with 16×16 tiles and does the whole 16×16 multiply in ONE instruction — but the *shape of the loop* is exactly what we just did by hand."]]

The naive tensor-core kernel is just the old naive SIMT kernel, promoted from elements to tiles: **one warp per 16×16 output tile**. The warp zeros its accumulator, marches down K sixteen at a time — load A tile, load B tile, `mma_sync` — and stores once at the end.

```cpp
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16,16,16, float> acc_frag;

wmma::fill_fragment(acc_frag, 0.0f);
for (int k = 0; k < K; k += WMMA_K) {          // march down K
    wmma::load_matrix_sync(a_frag, A_tile_ptr, K);
    wmma::load_matrix_sync(b_frag, B_tile_ptr, K);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
}
wmma::store_matrix_sync(C_tile_ptr, acc_frag, N, wmma::mem_row_major);
```

[[fig: A hand-drawn "one warp marches down K" walkthrough in three numbered panels. Panel (1): matrices A (M×K, blue diagonal hatch), B (K×N, green diagonal hatch), C (M×N) as rectangles with red dimension labels M, N, K; one 16×16 cell of C highlighted pale-yellow, labeled red "one 16×16 tile = one warp". Panel (2): a zoom of that C-tile fed by a horizontal blue strip of A ("16 rows") and a vertical green strip of B ("16 cols"), with numbered circles (1)(2)(3) walking small 16×16 sub-tiles left-to-right across A and down B, purple note "for k+=16: load, load, mma_sync". Panel (3): a single 16×16 box labeled orange "acc_frag lives in registers the whole loop (FP32)", a blue dashed arrow out to C labeled "store once at the end". Dashed takeaway box: "warp = tile · accumulate in registers · one write to HBM." Excalidraw style, white background. || The naive tensor-core kernel: each warp owns a 16×16 tile, accumulates across K in registers, writes to memory exactly once.]]

## The catch — same as last ladder, one level up

Here is where you show students that the ladder repeats. The naive tensor-core kernel is *fast* — low tens of TFLOP/s, several times the best scalar kernel — but it's only about **8% of cuBLAS**. The crane is mostly standing idle, waiting.

Why? The exact same reason as the naive SIMT kernel: **memory, not math.** Every warp reads its A strip and B strip straight from far-away HBM. Neighboring warps re-read hugely overlapping data. Two warps in the same tile-row of C both stream the same 16 rows of A from HBM. Nothing is staged on-chip. The 989 TFLOP/s crane spends its life waiting on loads.

[[note: aha || The number that makes the room groan: "We have a unit that can do a thousand trillion operations a second, and we're feeding it through a straw. It's like renting a Formula-1 car and towing it behind a bicycle. The crane isn't the bottleneck — the delivery of bricks is." This is the same lesson as the CPU-vs-GPU chapter: fast math is easy; fast *feeding* is the whole craft — and it's true again, one level up.]]

So the to-do list writes itself, and it mirrors the SIMT ladder rung for rung: **stage tiles in shared memory** so warps reuse on-chip copies instead of re-reading HBM; then **give each warp more than one tile** so there's enough work to hide the loads. That's the climb. And on tensor cores, staging into shared memory drags a brand-new gremlin into the light: the bank conflict.

## Opening the sealed box: fragments and `ldmatrix`

Higher up the ladder, the sealed WMMA crate stops being enough, and you *do* have to look inside. This is where you level up from WMMA to the raw instruction, and you tell students plainly why.

When we stage tiles in shared memory, we want to move them into registers in exactly the scattered pattern the tensor core expects. WMMA hides that pattern, so it also hides the *move* — and that move is where the last of the speed lives. To control it, we drop to the raw `mma.sync` PTX instruction and manage the shared-to-register hop ourselves.

The scattered pattern is real. For a small MMA shape, each of the 32 threads holds a handful of elements of each operand, in an interleaved, quadrant-based arrangement fixed by the hardware ISA — not row-major, not column-major, just *the layout the tensor core was wired to expect*. You don't design it; you feed it.

[[note: metaphor || Airplane boarding by zones. The seats (elements) are laid out in the plane in a fixed pattern, but the airline calls passengers (threads) to their seats in a scrambled zone order, not front-to-back. `ldmatrix` is the gate agent who reads everyone's boarding pass and, in one announcement, sends all 32 passengers to exactly the right scattered seats at once — instead of you manually walking each one down the aisle. The hardware does the shuffle; you just call the announcement.]]

[[fig: A hand-drawn Excalidraw-style technical diagram titled "The fragment layout is scattered — ldmatrix does the shuffle". Left: a small 8×8 grid labeled red "tile in shared memory (row-major)", blue diagonal hatch, cells numbered in reading order 0..63. Center: a purple rounded box drawn as a hardware gate labeled "ldmatrix.sync (1 instruction)" with 32 thin arrows fanning through it. Right: a scattered arrangement labeled green "32 threads' registers, in fragment layout" — a grid where cells are tagged with handwritten thread ids "T0","T0","T5","T5"… showing T0 owning two little clusters in opposite quadrants, connected by a curved orange arrow "each thread: a scattered handful". A red note across the top: "not row-major, not col-major — the ISA fixes the map (thread,reg)→(row,col)". A blue label under ldmatrix: "hardware does the cross-thread shuffle in one shot". Dashed takeaway box: "you store tiles the easy way; ldmatrix rearranges them into the exact scatter the tensor core expects." White background, hand-lettered. || The technical translation of the boarding metaphor: shared-memory tiles are stored simply, and one `ldmatrix` shuffles them into the tensor core's fixed, scattered per-thread fragment layout.]]

That gate agent is a real instruction: **`ldmatrix`**. One `ldmatrix` issued by the warp reads little 8×8 patches of FP16 out of shared memory and drops them into the 32 threads' registers *already in fragment layout* — doing all the cross-thread shuffling in hardware, in one shot. It's the bridge between "how we store tiles" and "how the tensor core reads them." And it is exactly on this bridge that our next gremlin lives.

## The bank conflict, made concrete

Shared memory is split into **32 banks**, each handing out 4 bytes per cycle. A warp's 32 lanes can read all 32 banks at once — beautifully fast — *only if* their 32 addresses land in 32 different banks. If two lanes want words in the *same* bank, the hardware serializes them: a 2-way conflict costs 2 cycles, an 8-way conflict costs 8.

Now the trap. To assemble one fragment, `ldmatrix` needs the 8 rows of an 8×8 tile. But that tile lives inside a wider slab we staged — say 64 elements wide, row-major. So consecutive rows are `64 × 2 bytes = 128 bytes` apart. Banks repeat every `32 × 4 = 128 bytes` — *exactly* the row stride. So every row starts at the same bank offset, and all 8 rows funnel into the same 4 banks. That's the **8-way conflict** the profiler screams about, and it fires on every `ldmatrix` on the hot path.

[[note: example || Draw the 8 rows and the 32 banks on the board. Row 0 → banks {0,1,2,3}. Row 1 is 128 bytes further along — which wraps exactly once around the banks — so it *also* lands on {0,1,2,3}. And rows 2 through 7, same thing. Eight rows, all piled into four banks, read one after another. "Eight lanes all reaching into the same drawer — they have to take turns. Eight turns instead of one." That physical picture is the whole conflict.]]

[[fig: A hand-drawn two-panel figure titled "Why ldmatrix conflicts (and the swizzle fix)". Panel A header in orange "NAIVE row-major SMEM": a vertical stack of 8 rows (red labels 0..7) of blue-hatched cells, thin dashed arrows from every row CONVERGING onto the same group of 4 banks (drawn thick and red) in a horizontal strip of 32 boxes labeled green "banks 0..31", red scrawl "8 rows → 4 banks = 8-way conflict", blue note "row stride 128B = one full bank cycle". Panel B header in orange "SWIZZLED SMEM": the same 8 rows, but each row's arrow bends through a small purple XOR-gate box labeled "col ^= (row bits)", and now the 8 arrows FAN OUT to 8 distinct banks (thin, green), green note "1 wavefront, conflict gone". Numbered circles (1)(2) mark read order. Dashed takeaway box: "same bytes, permuted addresses — free at compile time; the conflict evaporates." Excalidraw style, white background, hand-lettered. || The 8-way conflict and its cure: the naive layout collapses 8 rows into 4 banks; an XOR permutation scatters them across distinct banks.]]

## The swizzle: one XOR that costs zero bytes

The classic fix is **padding** — store each row a little wider so successive rows fall into different banks. It works, but it wastes precious shared memory. The better fix costs *nothing*.

The insight: a bank is chosen by the *low bits* of the address, and XOR is a permutation. Before we store each element, we perturb its column index by XOR-ing in a few bits of its row index. That shuffles each row's elements into a *different* set of banks — without moving any element to a different row, without using one extra byte. We apply the *same* XOR when we store and when we compute `ldmatrix` addresses, so the data comes back correct; only the bank assignment changes.

```cpp
// same permutation on store and on ldmatrix-address computation
uint swizzle(uint row, uint col) {
    return col ^ ((row & 0b1100) >> 2);   // XOR row bits into the bank field
}
```

[[note: metaphor || The theater usher who shuffles the coat-check tickets. Every coat still hangs on its own hook (no coat is lost, no row changes), but the usher relabels which *drawer* each ticket points to so that eight people arriving together reach into eight different drawers instead of all clawing at one. And because the usher uses the same relabeling scheme going in and coming out, everyone gets their own coat back. XOR is that reversible relabeling — it's its own inverse, so store and load stay in perfect agreement.]]

Row 0 is untouched, row 1's elements shift by one bank-group, row 2 by two, and so on — the eight rows that piled into four banks now spread across distinct banks. The code change is almost insultingly small: everywhere you compute a shared-memory address, route the column through `swizzle()`. No new buffers, no padding, one XOR the compiler folds into the address math.

[[note: aha || Predict, then measure — this is the moment to model good engineering. The falsifiable claim: the bank-conflict counter on the `ldmatrix` lines should drop from an 8× wavefront ratio to 1×. Run Nsight Compute. Before: 8×, shared pipe is the top stall. After: the counter reads literally **0**, ratio is 1.0. The throughput roughly doubles — this kernel jumps from about a quarter of cuBLAS to about **50%**. "Half of cuBLAS, from a permutation that costs zero bytes and one XOR." That is one of the best trades on the whole ladder, and it lands because you *predicted* it first.]]

[[note: production || This exact bank-conflict-then-swizzle dance is not academic. Every production GEMM library — cuBLAS, CUTLASS, the kernels inside FlashAttention and DeepSeek's stack — swizzles its shared-memory layouts for precisely this reason. When your students understand this XOR, they understand a technique that is running, right now, in every tensor-core kernel serving every large model on the planet.]]

## Teaching notes: the board sequence

Deliver it in this order, and it never collapses:

1. **The reframe** (5 min). "93.7% was a lie of omission." Draw the two ladders, floor-above-ceiling. Get the groan.
2. **The crane** (5 min). Tensor core = does a whole tile per instruction. The 10× number. Bricklayer vs. crane.
3. **The one shift** (8 min). Thread → warp. Draw the 32 threads holding hands owning one tile. Repeat it three times; it's the hardest idea.
4. **Sealed crates** (10 min). Three fragments, four `_sync` moves. Do NOT open the crate. The tiny 2×2-tile by-hand loop.
5. **The catch** (5 min). 8% of cuBLAS. Formula-1 towed by a bicycle. Same memory lesson, one level up.
6. **Open the box** (10 min). Only now: fragments really are scattered; `ldmatrix` is the gate agent; the bank conflict; the XOR swizzle. Predict-then-measure to 50%.

[[note: demo || The one live demo: run the naive WMMA kernel and the swizzled kernel back to back, with Nsight Compute open on the bank-conflict counter. Show `8×` become `0`, and the TFLOP/s roughly double, in real time. If you can only show one number all lecture, show that counter flipping to zero — it makes the abstract XOR viscerally real.]]

[[note: confusion || The deepest confusion is students trying to carry "one thread, one element" onto the tensor core and asking "so what does thread 3 compute?" Head it off before it starts: draw the warp-owns-the-tile picture *first*, and every time the old model resurfaces, point back at the 32 hands holding one tile. On the tensor core, the thread is not the unit of thought. The warp is.]]

## You can now teach

- Why the 93.7%-of-cuBLAS victory was a **lie of omission**, and why tensor cores are a whole new, taller ladder whose floor beats the old ceiling.
- What a **tensor core** is in plain words — the crane that lays a whole 16×16 tile per instruction — and the ~10× / 989-TFLOP/s number that motivates it.
- The one hard mental shift: **the warp owns the tile**, not the thread — and how to keep students from carrying the old SIMT model across.
- **WMMA as sealed crates**: three fragments, four `_sync` moves, and why you deliberately never look inside — plus the FP16-in / FP32-accumulate precision shape.
- The naive kernel's **catch** (8% of cuBLAS, starved by HBM) and the to-do list that mirrors the SIMT ladder: stage in shared memory, then give each warp more tiles.
- The top-of-ladder trio — **`ldmatrix`, bank conflicts, and the XOR swizzle** — taught predict-then-measure, ending on the counter flipping from 8× to 0 and the jump to ~50% of cuBLAS.
