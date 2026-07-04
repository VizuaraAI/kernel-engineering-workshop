By the end of this chapter you'll be able to stand at a whiteboard and teach the single most famous kernel in modern AI — FlashAttention — as one clean idea a student can hold in their head: *never write the big scores matrix down.* No CUDA required. You need one honest picture of why attention wastes memory, one very good "grading papers" metaphor for streaming, and the courage to do a tiny running-max by hand. Let's build it so it's yours.

## First, what is attention doing? (the two-minute recap)

You don't need to re-teach the whole transformer here — the earlier chapters did that. You only need the shape of one attention head. Say it plainly. Attention takes three grids of numbers, all the same size: **Q** (the questions), **K** (the keys), and **V** (the values). Each is `N × d` — `N` rows, one per word in the sentence, and `d` columns (the "head dimension," usually 64 or 128).

The math is three steps:

```python
S = Q @ K.T            # (N, N)  every word scores every other word
P = softmax(S, dim=-1) # (N, N)  turn scores into weights that sum to 1
O = P @ V              # (N, d)  each word = a weighted blend of the values
```

[[note: say || "Every word looks at every other word and asks 'how much should I pay attention to you?' That produces a big square grid of scores — N words by N words. Softmax turns each row of scores into a set of weights that add up to 1. Then each word becomes a weighted blend of everybody's values. That middle grid — N by N — is the villain of this whole story."]]

The thing to burn into the room is the **size of that middle grid, S**. It is `N × N`. For a sequence of 8,192 words, that is 67 million numbers — **256 MiB** — for a *single head* in a *single layer*. And a model has dozens of heads and dozens of layers.

[[note: metaphor || The scores grid is a giant seating chart for a party of N guests: for every pair of guests, one number saying how interested guest A is in guest B. If N doubles, the chart doesn't double — it *quadruples*. That quadratic blow-up is why long context is so painful, and why this one grid is worth building a whole kernel to avoid.]]

## Why naive attention is a memory disaster

Now the part students always get wrong, so *you* must get it right. Everyone assumes the two matrix multiplies are the expensive part. They are not. The expensive part is the little softmax in the middle — not because it does much math, but because of where it forces the data to travel.

Walk through the trips the naive version makes:

1. Compute `S` (`N × N`) and **write all of it** out to the GPU's main memory (HBM).
2. **Read all of S back** to find each row's max and sum, so softmax can normalize it.
3. **Write P** (`N × N`) back out.
4. **Read P back** a third time to multiply by V.

[[note: aha || Four full trips across the slowest road on the chip, dragging a quarter-gigabyte matrix each way — and for what? The softmax itself does almost no arithmetic: one exponent and a couple of adds per number. So this is nearly *pure waiting on memory*, with no real math to hide the wait behind. Say the punchline: "The GPU's fast math units sit idle while a matrix that we didn't even want to keep gets shoved back and forth four times." That is the disaster, in one sentence.]]

This connects straight to the course's spine (the "feeding the cooks" chapter): the score matrix is a memory-bound intermediate. FlashAttention's entire pitch is one word — **fusion** — glue the two matmuls and the softmax into a *single* kernel so that S is computed, used, and thrown away without ever touching main memory.

[[fig: A warm hand-drawn illustration titled "The pointless warehouse trips". A tiny office desk on the right labeled "the chip (fast, tiny)" and a huge distant warehouse on the left labeled "HBM main memory (far away)", connected by a long winding road. A little delivery truck is drawn making FOUR labeled round-trips down the road hauling an enormous crate labeled "S = the N x N scores": trip 1 "drop off S", trip 2 "fetch S back", trip 3 "drop off P", trip 4 "fetch P back". The office worker looks exhausted with a thought bubble "...and I throw the crate away at the end anyway!". A red note over the road: "four long hauls for a crate we don't even keep". Dashed takeaway box: "the whole cost is the driving, not the work at the desk". Excalidraw style, white background, charming, handwritten labels. || Naive attention as a truck making four pointless warehouse trips with a giant crate it discards at the end. The driving is the cost.]]

[[fig: A hand-drawn "before vs after" memory-traffic diagram titled "Attention: the intermediate that shouldn't exist". LEFT panel labeled "(A) NAIVE". A tall green box on the far left labeled "HBM main memory". Three matrices as squares: small Q and K (red dims "N x d"), and a big square labeled "S = N x N scores" with red hatch and an orange callout "256 MiB at N=8192!". Blue dashed round-trip arrows numbered (1)(2)(3)(4): "write S", "read S (softmax)", "write P", "read P (times V)". Blue note "4 trips over an N x N matrix". RIGHT panel labeled "(B) FLASHATTENTION". The same Q/K/V but now inside a small orange box labeled "on-chip SRAM", with the big S square crossed out in red and annotated "never written down". A single blue arrow "stream K,V tiles in" and one arrow out "write O once (N x d)". Green note "traffic grows like N x d, not N squared". Dashed takeaway box: "fuse the whole thing so the N x N scores never reach main memory". || Naive attention makes four trips over an N-by-N matrix. FlashAttention makes none — the scores are born and die on-chip.]]

## The obstacle: softmax wants to see the whole row

Here is why nobody had just done this obvious fusion for years, and it's worth pausing on because it's the intellectual heart of the chapter.

Softmax is **global**. To normalize one row of scores you need two things that depend on *every* number in that row: the row's **maximum** (you subtract it before exponentiating, so the numbers don't overflow) and the row's **sum of exponentials** (the denominator). You cannot divide by a total you haven't finished adding up. You cannot subtract a max you haven't finished searching for.

So streaming seems impossible. If you only look at one tile of the row at a time, you don't yet know the max or the sum for the whole row.

[[note: confusion || This is *the* moment students seize up: "You can't do softmax without the whole row — so how can you tile it?" Don't wave it away. Name their objection out loud and promise the resolution: "You're right that you need the whole row *eventually*. The trick is you can carry a running guess and *correct it* as new numbers arrive — like updating a running average." Then show them the metaphor before any algebra.]]

## The metaphor: grading a stack of exam papers

This is the metaphor to draw and act out. It makes online softmax feel obvious.

Imagine you're grading a huge stack of exams and you want two things at the end: the **highest score** in the stack, and a running tally that lets you compute everyone's grade *relative to that highest score*. But the stack is too tall to lay out on your desk at once. So you grade one small pile at a time.

You keep three sticky notes on your desk:
- **m** — the highest score you've seen *so far*.
- **ℓ** — a running total (of scores measured relative to that highest-so-far).
- **O** — a running blended answer you're building up.

You grade a pile. If the top score in this new pile is *higher* than your old `m`, then your old running total was measured against a max that was too low — every number in it is now a little too big. So you **rescale**: shrink the old total by exactly the right factor, then add in the new pile. Update your sticky note `m` to the new high. Move to the next pile.

At the very end — and *only* at the end — you divide by the final total to get everyone's proper grade.

[[note: metaphor || The whole of FlashAttention's softmax is: "grade the exams one pile at a time, keep three sticky notes (highest-so-far, running total, running blend), and whenever a new pile beats your old high score, gently shrink your old totals to match before adding the new pile in." That "gently shrink the past to match the present" step is the one and only clever line in the algorithm.]]

[[fig: A warm hand-drawn illustration titled "Online softmax = grading exams one pile at a time". A friendly teacher figure at a desk with a tall stack of exam papers labeled "N scores (too tall for the desk)". On the desk, three big sticky notes drawn as post-its: a yellow one "m = highest so far", a green one "L = running total", an orange one "O = running blend". An arrow shows one small pile being pulled off the stack labeled "grade one pile (a tile)". A speech bubble from the teacher: "this pile has a higher score than my old best! shrink my old totals to match, then add this pile in." A small purple note near the sticky notes: "shrink factor = exp(old max - new max)". A dashed takeaway box: "carry running stats, correct them when a new pile beats your best, divide only at the very end". Excalidraw style, white background, charming, handwritten labels. || The core trick as a human task: you never lay out the whole stack; you keep three sticky notes and fix them up pile by pile.]]

## Do it by hand: a tiny running softmax

Numbers make it real. Do this slowly on the board. Take one row of scores — just four numbers — and pretend they arrive in two piles of two.

Row of scores: `[1, 3, 2, 5]`. Piles: `[1, 3]` then `[2, 5]`.

**Pile 1 = [1, 3].**
- New max `m = 3`.
- Exponentials relative to the max: `exp(1-3)=0.135`, `exp(3-3)=1`.
- Running sum `ℓ = 0.135 + 1 = 1.135`.

**Pile 2 = [2, 5].** The top of this pile is 5 — *bigger than our old max of 3!*
- New max `m_new = 5`.
- Shrink factor `α = exp(m_old − m_new) = exp(3 − 5) = exp(−2) = 0.135`.
- Rescale the old sum: `0.135 × 1.135 = 0.153`.
- New pile's exponentials: `exp(2-5)=0.050`, `exp(5-5)=1`.
- Running sum `ℓ = 0.153 + 0.050 + 1 = 1.203`.

Now check it against the honest, all-at-once answer. The true denominator is `exp(1-5)+exp(3-5)+exp(2-5)+exp(5-5) = 0.018 + 0.135 + 0.050 + 1 = 1.203`. **Identical.** Not an approximation — *exactly* the same number, computed without ever holding all four scores at once.

[[note: example || Put both totals on the board side by side: the streamed `ℓ = 1.203` and the all-at-once `ℓ = 1.203`. Circle them. Say: "Same answer. We never had the whole row on the desk, and softmax came out exact. *That* is online softmax — and it's the entire reason we can throw the big grid away." This is the jaw-drop moment; let it land before moving on.]]

[[fig: A hand-drawn worked-example figure titled "Online softmax, two piles, by hand". Top: a row of four score boxes "[1, 3, 2, 5]" split by a dashed line into "pile 1 = [1,3]" and "pile 2 = [2,5]". Middle-left panel "after pile 1": three sticky notes "m=3", "L=1.135", drawn in blue. Middle-right panel "after pile 2" with an orange starburst "5 > 3, new max!": a purple box showing the rescale "shrink factor = exp(3-5)=0.135", then "L = 0.135 x 1.135 + 0.050 + 1 = 1.203" in blue. Bottom: a green box "all-at-once check: exp(1-5)+exp(3-5)+exp(2-5)+exp(5-5) = 1.203" with a big orange "SAME!" between the two totals. Dashed takeaway box: "exact softmax, streamed, no full row ever held". Excalidraw style, white background, handwritten. || The by-hand proof: streaming the softmax in two piles gives the exact same denominator as computing it all at once.]]

## The real recurrence, built from the sticky notes

Now generalize the by-hand example into the actual loop. One block of the GPU owns one block of query rows (`Q_i`, say 64 of them) and keeps its three sticky notes on-chip the whole time. It walks across the key/value blocks one at a time. Here is the whole kernel, in plain form:

```python
# One block owns query rows Q_i (B_r x d), resident on-chip.
m = -inf            # running max   (sticky note 1)
l = 0               # running sum   (sticky note 2)
O = 0               # running blend (sticky note 3), shape (B_r, d)

for j in range(num_k_blocks):        # stream the key/value blocks
    K_j, V_j = load(K, j), load(V, j)     # small tiles -> on-chip
    S = Q_i @ K_j.T                       # (B_r, B_c) scores, stays on-chip
    m_new = maximum(m, rowmax(S))         # did this tile raise the max?
    P = exp(S - m_new)                    # exponentials vs the new max
    alpha = exp(m - m_new)                # the shrink factor
    l = alpha * l + rowsum(P)             # shrink old sum, add new
    O = alpha * O + P @ V_j               # shrink old blend, add new
    m = m_new

O = O / l                                 # divide ONCE, at the very end
store(O_i, O)                             # write O (N x d) -- the only write!
```

Point at each line and match it to a sticky note. The `S` tile is small (`B_r × B_c`, say 64×64), lives on-chip, and is consumed instantly — it is *never* written to main memory. The `alpha` line is the "shrink the past to match the present" move — the only line that would look strange to someone who's only written a normal softmax. And the division by `ℓ` happens exactly once, at the end, because only then do we know the true total.

[[fig: A tiling-walkthrough figure in three numbered panels titled "The FlashAttention inner loop". Setup note top-left: "Q block fixed (blue hatch, B_r x d). Stream K/V blocks (green hatch, B_c x d). Keep m, L, O per row." Panel (1): one blue-hatched Q block on the left; a row of green K/V blocks j=1,2,3,4 with a red arrow "outer loop over key blocks". Panel (2): zoom on block j -- a small pale-yellow S tile with purple code "S = Q_i @ K_j.T", then blue "m_new = max(m, rowmax(S))", "P = exp(S - m_new)". Orange callout "found a bigger max!". Panel (3): the RESCALE drawn as two beakers labeled old-L and old-O, each multiplied by a purple factor "alpha = exp(m_old - m_new)", then blue "L = alpha*L + rowsum(P)" and "O = alpha*O + P @ V_j". Red note "correct the past, then add the present". Dashed takeaway box: "one streaming pass, memory grows like N, exact softmax". Excalidraw style, white background. || The recurrence in three panels: scores are tiled, each new key block may raise the max, and we rescale the running sum and output before folding it in.]]

## What we actually bought

Be honest with students about what the win is and isn't. FlashAttention does the **same matrix math** as naive attention — in fact a few *extra* multiplies for the rescales. It does not save FLOPs. The entire win is in **memory traffic**.

Naive attention moves the `N × N` matrix across main memory four times: traffic grows like `N²`. FlashAttention moves only Q, K, V, and O — each `N × d` — once: traffic grows like `N × d`. The ratio is roughly `N / d`. At `N = 8192` and `d = 128`, that is about **64× less** traffic across the slowest road on the chip.

[[note: aha || The number to write huge on the board: at 8k context, FlashAttention moves roughly **64× fewer bytes** across main memory than naive attention — for the exact same answer. And because the memory bottleneck is gone, the fast math units finally run near their ceiling (an H100's tensor cores can do ~989 trillion operations per second) instead of idling. A memory-bound layer becomes a compute-bound one. That flip is the whole prize.]]

[[fig: A traffic-ledger figure titled "Where the bytes go". LEFT: a stacked memory pyramid top-to-bottom -- "Registers" (green, narrow top), "SRAM on-chip ~228 KiB/SM" (green), "L2" (green), "HBM main memory ~3.35 TB/s" (green, wide base). A blue bracket spans the top two labeled "S and P live and die here". RIGHT: a two-row hand-written table. Row "NAIVE": red entries "write S, read S, write P, read P (all N x N)" with orange total "about 4 x N squared". Row "FLASH": blue entries "read Q,K,V once, write O once" with orange total "about 4 x N x d". A red arrow between them labeled "ratio about N/d = 64x less main-memory traffic at N=8192, d=128". Dashed takeaway box: "same FLOPs, far fewer bytes -> a memory-bound layer turns compute-bound". Excalidraw style, white background, handwritten. || The ledger: traffic drops from roughly 4N-squared to 4Nd, dozens of times fewer bytes for long sequences.]]

[[sn: The multiplier depends heavily on `N`. At short sequence lengths the `N²` term is small and fusion barely helps; the gain grows with context length. That's exactly why FlashAttention arrived the same moment models started chasing long context — the two needs met.]]

## FlashAttention-2, in one breath

You don't need to teach FA2 in depth to a first-time audience, but you should be able to say what it added, because someone will ask. FA1 answered *where the bytes live*. FA2 answered *how the work is split up* — and it roughly **doubled** the speed again, taking the forward pass from around 35% of peak to about 70%.

The one-liner for each of its three moves:
- **Do the slow softmax work less often.** The `exp` and rescale run on the GPU's *slow* math units (~16× slower per operation than the tensor cores). FA2 keeps the output un-normalized and divides by `ℓ` just **once at the end** instead of every inner step. Fewer slow instructions blocking the fast matmuls.
- **Parallelize over sequence length.** FA1 gave one block to each (batch, head) pair. With `batch=1` long-context inference — 32 heads on a 132-SM machine — three-quarters of the GPU sits idle. FA2 hands each *query block* its own SM, filling the machine.
- **Skip the masked upper triangle.** In causal attention each word only looks *backward*, so half the score grid is thrown away anyway. FA2, with tiles as explicit loop indices, simply never computes it — roughly **2× less matmul work** for free.

[[note: production || This is not a paper on a shelf. FlashAttention is in essentially every serving stack shipping today — vLLM, the reference transformer libraries, every long-context model you've used. When DeepSeek or Meta serve a model to millions, the fused-attention kernel is a direct line item on the electricity bill and the GPU count. Your students are learning the exact kernel that made long context affordable — and the Hopper rewrite (with TMA bulk copies and wgmma async tensor-core instructions) is where the frontier keeps pushing on H100 and B200 today.]]

## Teaching notes: the board plan

Here's the order that works. Don't deviate — each step sets up the next.

[[note: teach || Board sequence: (1) Draw naive attention's three lines and the big N x N S in the middle. (2) Draw the four round-trips to main memory and say "four trips, a quarter-gig each, for a matrix we don't even keep." That's the *problem*. (3) State the obstacle: "softmax needs the whole row" — let it feel impossible for a beat. (4) Introduce the grading-exams metaphor with three sticky notes and *act it out* — pull piles off a stack. (5) ONLY THEN do the four-number by-hand example and show the streamed total equals the all-at-once total. That equality is the payoff; pause on it. (6) Generalize to the recurrence. (7) Write the 64x traffic number huge and tie to production. Metaphor and by-hand number come BEFORE the algebra, always.]]

[[note: demo || The one live demo: in a notebook, run naive attention and FlashAttention (torch's `scaled_dot_product_attention` picks the fused kernel automatically) on a long sequence, and print peak GPU memory for each. Naive will allocate the giant `N × N` buffer; fused won't. Watch the peak-memory number drop by orders of magnitude while the outputs match to floating-point noise. Same answer, a fraction of the memory — that's the whole chapter in one cell.]]

[[note: confusion || Two confusions to pre-empt. First: "Isn't the streamed softmax just an approximation?" No — walk them back to the by-hand example where both totals were 1.203 *exactly*. It is algebraically identical. Second: "So we made the math faster?" No — same math, even a few extra multiplies. We made the *memory movement* smaller. Keep hammering: FlashAttention is a logistics win, not an arithmetic one. That distinction is the mark of a student who truly gets it.]]

## You can now teach

- **Why naive attention is a memory disaster** — the `N × N` score grid written and re-read four times, quadratic in `N`, while the softmax does almost no real math.
- **The obstacle** — softmax needs the whole row (its max and its sum) — and why that makes fusion look impossible.
- **Online softmax as grading exams pile by pile** — three sticky notes (running max, running sum, running blend) and the one clever "shrink the past to match the present" rescale.
- **The by-hand proof** — a four-number, two-pile softmax that comes out *exactly* equal to the all-at-once answer without ever holding the whole row.
- **What was actually bought** — same FLOPs, roughly `N/d` (~64×) less main-memory traffic, flipping a memory-bound layer into a compute-bound one.
- **The production stakes and FA2 in one breath** — where FlashAttention runs today, and the three FA2 moves (defer the divide, parallelize over sequence length, skip the causal upper triangle).
