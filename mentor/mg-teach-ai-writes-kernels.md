By the end of this chapter you can stand at a whiteboard and teach a student — honestly, without hype and without doom — what happens when you point a language model at kernel writing: how we score it fairly, why drawing a hundred samples beats drawing one, exactly where the AI wins and where it falls flat on its face, and why the human with a profiler still has a job. This is the chapter that ties the whole workshop together, because everything the students learned to do by hand is about to be handed to a machine — and the machine turns out to be a *search*, not a genius.

Let's build it slowly. No hype. The honesty is the point.

## The question, in plain words

For four weeks your students have been the search algorithm. They looked at a slow kernel, had an idea, wrote it, timed it, read the profiler, kept the good version, tried again. That loop *is* the job. It is also slow and tiring.

So the natural question is: can we drop a language model — the same kind of model behind ChatGPT — into that loop instead of a human? Give it a PyTorch layer, ask it for fast CUDA, and let it grind?

The answer is a careful, honest *yes and no*. And teaching the exact shape of that "yes and no" is the most valuable thing you can give a student, because it inoculates them against both the hype ("AI writes all our kernels now, why are we here") and the despair ("then my skills are worthless"). Neither is true. Let's earn the real answer.

## First you need a fair judge: KernelBench

You cannot ask "can AI write good kernels" until you can *grade* a kernel fairly. So the first thing to teach is the benchmark. It is called **KernelBench**, and its design is beautiful in how simple it is.

Each problem hands the model a small PyTorch layer — a `nn.Module`, the *reference*. The model's job: give back a module that does the exact same thing, but with the forward pass written as raw CUDA. Same inputs in, same numbers out, but faster.

[[note: metaphor || Think of a **cooking contest** with a strict twist. The judge hands you a finished dish — say, a bowl of soup — and says: "Reproduce this exactly, but faster than my recipe did." You don't get a vague brief like 'make something tasty.' You get *the actual soup* to match. Your dish is only valid if a taster can't tell it apart from the judge's, and you only win if you plated it quicker. PyTorch is the judge's soup: it is the recipe, the taste-test, and the clock, all in one.]]

That triple role is the clever part. Using PyTorch *as the spec* quietly solves three headaches at once. There is no **ambiguity** — the layer says precisely what to compute, no English left to misread. There is a free **correctness check** — run both modules on random numbers and compare the outputs. And there is a real **baseline to beat** — PyTorch is not a strawman; it dispatches to genuinely well-tuned libraries. Beating it is honest work.

[[fig: A warm hand-drawn "cooking contest" illustration titled "KernelBench: match the judge's dish, but faster". On the LEFT, a smiling judge chef holding a bowl of soup labeled in green "the reference (PyTorch layer)", with a little scroll beside it reading "this IS the recipe AND the taste-test AND the clock". A blue dashed arrow labeled "your turn" points RIGHT to a contestant cook holding their own bowl labeled purple "your kernel (raw CUDA)". Between the two bowls, a taster figure with a red thought-bubble "do they taste identical?" and a stopwatch below him with an orange note "and is yours faster?". A dashed takeaway box at the bottom: "you win only if it matches AND beats the clock — two tests, not one". Excalidraw style, white background, charming, handwritten labels. || KernelBench as a cooking contest: the reference dish is simultaneously the recipe to follow, the taste-test for correctness, and the stopwatch to beat.]]

[[fig: A hand-drawn technical "transpilation contract" diagram titled "PyTorch in, CUDA out". LEFT: a box labeled green "Reference nn.Module (the spec)" with purple code "def forward(x): return self.norm(x)". A blue dashed arrow labeled "transpile" curves RIGHT to a box labeled purple "Candidate: inline .cu + wrapper" with code "__global__ void my_kernel(...)". Below, numbered circle (1) points to a blue box "run BOTH on random inputs" feeding two hatched output tiles (green hatch = reference, yellow hatch = candidate) into a red diamond "allclose(atol, rtol)?". Numbered circle (2) points to a green stopwatch "time both → speedup = t_ref / t_cand". Dashed takeaway box: "same numbers AND faster — both gates or it doesn't count". Excalidraw style, white background, hand-lettered. || The technical translation of the contest: the PyTorch module is the specification, the correctness oracle, and the performance baseline in one.]]

### Three levels, three kinds of hard

Teach students that KernelBench has three tiers, and they are not simply small-medium-large. Each one tests a *different skill*.

- **Level 1 — one operator.** A single matmul, a convolution, a softmax. Here the model goes head-to-head with a library kernel NVIDIA already tuned to death. Almost nowhere to hide.
- **Level 2 — a short sequence.** `conv → bias → scale → ReLU`. The win here is **fusion**: glue the steps into one kernel so the in-between results never leave the chip. This is the memory-movement lever, exactly what the students learned earlier in the course.
- **Level 3 — a whole block.** A full attention layer, a Vision Transformer block. Dozens of ops, many bottlenecks at once. The model has to *find* where the time goes, not just pattern-match one kernel.

[[note: teach || Draw the three levels as a pyramid, bottom to top, and say the skill next to each: L1 "match a tuned library" (brutal), L2 "delete the round-trips — fusion" (winnable), L3 "find the real bottleneck" (hard in a new way). The one line students must leave with: "Level 1 is the *hardest* to beat, not the easiest, because the opponent is already excellent." That flips their intuition, and flipping intuition is where teaching earns its keep.]]

## The fair score: fast_p (two gates, not one)

Now the metric, and this is where you make students respect the benchmark. Most coding benchmarks score one thing: did the code pass the tests? But a kernel has *two* things that both must be true. A correct-but-slow kernel is useless. A blazing-fast-but-wrong kernel is *worse* than useless — it silently corrupts your model.

So KernelBench uses **fast_p**: the fraction of problems where the kernel is *both* correct *and* at least `p` times faster than PyTorch.

[[note: metaphor || Two turnstiles, one after the other, to get into the stadium. The **first turnstile** checks your ticket is real — that's correctness (run both, compare the numbers within a tolerance). The **second turnstile** checks you ran fast enough — that's the speed gate `p`. A fake ticket stops you at turnstile one. A real ticket but a slow time stops you at turnstile two. You only get a seat if you clear *both*. There is no way to sneak in with only one.]]

The `p` is a dial you slide, and sliding it is the whole trick:

- **fast_1** — the headline — means correct *and* faster than PyTorch at all (speedup > 1×). "Did you actually beat it?"
- **fast_2** means correct *and* more than twice as fast. Now you're asking for a real win, not a coin-flip margin.

[[note: aha || Here is why this metric can't be cheated, and it's worth saying slowly: the two ways to fail are *orthogonal*. You cannot pad your score with kernels that are correct-but-slow — they fail every speed gate. And you obviously cannot cheat with fast-but-wrong kernels — they fail correctness first. To score at all you must satisfy both, independently. That's what makes the low numbers we're about to see *trustworthy* rather than gloomy.]]

[[fig: A hand-drawn illustration titled "Two turnstiles to score a point". Two turnstiles drawn in a row. FIRST turnstile has a sign in blue "CORRECT? (numbers match the reference)", a happy figure passing through and a sad figure with a fake ticket bounced back in red "wrong → out". SECOND turnstile just past it has a sign in orange "FAST ENOUGH? (≥ p× PyTorch)", with a stopwatch; a figure who cleared gate one but was slow gets bounced in red "too slow → out", and one figure who cleared both walks into a stadium seat labeled green "SCORES". A little dial on the side labeled "p = 1× … 2×" with a note "slide right to demand more speed". Dashed takeaway box: "both gates, in order — correctness then speed. Neither alone counts." Excalidraw style, white background, handwritten. || fast_p drawn as two turnstiles: a kernel must clear correctness first, then the speed bar p — and the two failure modes never overlap.]]

## The honest number: under 20%

Now deliver the number that started this whole line of work, and deliver it flat, without drama.

Point frontier language models at KernelBench. Ask for **one kernel per problem**. Score fast_1. Across all three levels, the models produced kernels that were correct *and* faster than PyTorch **less than 20% of the time.** Four out of five single attempts either failed to compile, produced wrong numbers, or produced right numbers that were *slower* than the framework they were trying to beat.

[[note: say || "Four out of five tries fail. And that's exactly the number you should have guessed. For one attempt to score, the CUDA has to *compile* — with all its `float4` loads and `__syncthreads()` and boundary guards, one typo kills it. Then it has to be *numerically correct* — every index, every reduction, every edge case right. Then it has to be *faster than a library NVIDIA hand-tuned for fifteen years.* Getting all three, first try, in one shot? Of course it's usually no. The surprise isn't that it fails 80% of the time. The surprise is coming next."]]

Teach the failure *profile*, because it's the most useful part. The models do relatively *well* on Level 2 fusion — because that win is "delete a trip to memory," which is learnable from text. They do relatively *badly* wherever the win requires driving the tensor cores through hardware-specific machinery. Hold that thought; it's the spine of the whole chapter.

## The surprise: monkeys with a checker

Here is the reframe that changed the field. Stop asking the model to be *smart*. Ask it to be *plural*.

Take DeepSeek-V3, a model with no special kernel training. Ask for a Level 2 kernel **once** — it succeeds about **4%** of the time. Dismal. Now draw **100 independent samples** for the same problem, compile and check every one, and keep the best. The success rate jumps from **4% to 37%**.

Same model. Same prompt. No feedback, no cleverness. Just a hundred rolls of the dice against a checker that cannot be fooled about whether code compiles and can barely be fooled about whether it's correct.

[[note: metaphor || The infinite monkeys — but with an editor. One monkey at a typewriter almost never types a good sentence. But put a hundred monkeys in a room, and stand an *editor* at the door who reads every page and throws out the garbage, and good pages start coming through. The magic isn't in any monkey — it's in the *editor*. In our case the editor is the compiler-plus-correctness-check. The model is a weak, cheap generator; the checker is what makes a hundred weak guesses add up to a strong answer.]]

[[fig: A warm hand-drawn illustration titled "A hundred monkeys and one editor". On the left, a big room with many small typewriter-monkeys, each holding up a page — most pages stamped red "junk", a few stamped green "good". At the door on the right, a stern editor figure at a desk labeled blue "the checker: does it COMPILE? does it MATCH?", tossing junk pages into a bin labeled red "discarded" and passing good pages through to a tray labeled green "kept — timed & ranked". A big orange callout over the whole scene: "the reliability lives in the EDITOR, not the monkeys". A small counter reads "1 try → 4% … 100 tries → 37%". Dashed takeaway box: "a trustworthy checker turns a weak generator into a strong solver". Excalidraw style, white background, charming, handwritten. || The monkeys metaphor: a hundred cheap, unreliable samples become reliable because a mechanical checker filters them — the checker carries the trust, not the model.]]

### Why this works — one tiny piece of math

Students will suspect a trick. Show them there isn't one, with a single formula they can do on the board.

If one sample solves a problem with probability `p`, then the chance that *at least one* of `k` samples solves it is `1 − (1−p)^k`. That's it. Plug in a miserable `p = 0.04` and `k = 100`:

```
1 − (1 − 0.04)^100  =  1 − 0.96^100  ≈  1 − 0.017  ≈  0.98
```

[[note: example || Do it live on the board. 0.96 to the 100th power is about 0.017 — so `1 − 0.017 ≈ 0.98`. The math says a 4%-per-try generator should clear nearly *every* problem within 100 tries. So why did the real number land at 37%, not 98%? Because for some problems the model's true `p` is exactly *zero* — it simply cannot express the idea the kernel needs, and no amount of resampling conjures an idea that isn't there. Sampling buys coverage over ideas the model *can* reach. It buys nothing over the ones it can't. That gap between 98% and 37% is the honest map of what the model doesn't know.]]

## Serial feedback: give the profiler to the model

Parallel sampling is a hundred *blind* guesses — they never learn from each other. The second knob is *serial*: run one kernel, feed its compiler errors and its profiler output back into the next prompt, and let the model react to evidence like a human worklog does.

When the Stanford group did this with a reasoning model (DeepSeek-R1) on Level 2, the score climbed from a single-shot **36%** to about **72%** — a clean doubling.

[[note: production || Sit with that 72%. A general reasoning model, handed a profiler and a few turns, correctly-and-faster solves roughly three-quarters of the fusion problems. And *this is the exact loop your students learned by hand.* A profile that says "you're memory-bound, occupancy is 25%, these loads are uncoalesced" is the same three-regimes diagnostic they read themselves — it just shifts the model's next guess toward the fix. Parallel sampling widens the net; serial feedback sharpens the aim. Real systems turn both knobs at once.]]

[[fig: A hand-drawn technical diagram titled "Two knobs of test-time scaling". HORIZONTAL axis: a row of many small purple kernel boxes labeled black "PARALLEL: k blind samples", with a green note under it "coverage = 1 − (1−p)^k" and a red number "4% → 37% at k=100". VERTICAL axis: a downward column of kernel boxes joined by blue arrows labeled "SERIAL: feed profiler + errors back in", each arrow tagged blue "condition next try on evidence", with a red number "36% → 72%". Where the axes cross, an orange starburst "do BOTH = search over programs". Dashed takeaway box: "parallel widens the net, serial sharpens the aim". Excalidraw style, white background, hand-lettered. || The two knobs of test-time scaling: parallel sampling raises coverage, serial profiler-feedback raises per-attempt quality, and strong systems use both.]]

## It's not a chatbot — it's evolution

Put the two knobs together and something clicks: this is a **search over programs**, and the whole vocabulary of search opens up.

The single most productive idea, from Ouyang and collaborators at Stanford CRFM, is to **split the move in two**. First have the model propose an *optimization idea in plain English* — "stage the B tile in shared memory," "fuse the ReLU into the epilogue so we never round-trip through memory," "turn the convolution into an implicit matmul." *Then* have it write the code for that idea.

[[note: metaphor || Decide the *play* before you run it. A team that just yells "everyone run fast!" collapses into the same messy pile every time. A team that first calls a distinct play — "left sweep," "screen pass," "long bomb" — explores genuinely different strategies. Ideas are a small, diverse space; you can list five truly different plays. Raw 200-line kernels are a huge space where greedy code-generation keeps collapsing into the same shape. So you branch on *ideas*, then realize each idea several ways.]]

Then the search structure: each idea fans out into several implementations, all compiled, checked, and timed in parallel. The fastest survivors **seed the next round**, alongside an **archive** of every correct kernel ever found. Bad branches die; good branches breed and recombine. This is evolutionary search where the mutation operator is "a language model with a hypothesis" instead of a random bit-flip.

[[note: aha || The most telling detail: they ran five rounds, and *most of the winning kernels appeared in round 4 or 5* — not early. The good kernels were compositions of earlier good ones. In one run the search seeded a convolution with a matmul kernel it had generated rounds earlier, because "conv as implicit matmul" had surfaced in the English stage. A greedy loop that kept only the single best child each round would have pruned that lineage before it ever paid off. The archive — the memory of everything that worked — is the difference between *sampling* and *evolution*.]]

[[fig: A hand-drawn evolutionary-tree diagram titled "Search with an archive". TOP: a black root box "PyTorch reference (1.0×)". Branches split downward into child kernel boxes; correct-and-fast ones filled yellow-hatch with red speedup tags "1.4×", "2.1×", "4.8×"; wrong ones grey and crossed out with a red X "won't compile / wrong". A side panel on the right, a stacked bin labeled green "ARCHIVE: every correct kernel ever found", with blue dashed arrows pulling two past winners back up to seed a new branch, noted blue "recombine a known-good ancestor". Numbered circles (1) propose idea (2) implement (3) verify (4) benchmark (5) archive trace the loop. Orange callout near a deep leaf: "best kernels appear in rounds 4–5". Dashed takeaway box: "the archive lets dead-end rounds restart from known-good DNA — that's evolution, not sampling". Excalidraw style, white background, hand-lettered. || Evolutionary kernel search: an archive of verified kernels lets the loop recombine past winners and recover from dead ends instead of re-rolling from the reference.]]

## The honest scorecard: where it wins, where it faceplants

Now the payoff, and the most important slide you'll ever show on this topic. When CRFM ran this loop on real operators, some results were *genuinely stunning* — and some were *humbling*. Show both, side by side, always.

The wins (throughput as a percentage of PyTorch; over 100% means faster):

- **LayerNorm** — **484%**, nearly 5× faster.
- **Conv2D** — **180%**, 1.8× faster.
- **Fused Conv2D + ReLU + MaxPool** — **290%**, and still **189%** even against `torch.compile`.
- **FP32 Matmul** — **101%**, essentially matching a `cuBLAS`-backed baseline.

The losses:

- **FP16 Matmul** — **52%**, roughly half speed.
- **FP16 FlashAttention** — **9%**, more than 10× *slower*.

[[note: confusion || A student will hear "484% on LayerNorm" and think the AI is doing magic — five times *less math*. Stop that immediately. It is doing the *same* math. LayerNorm is memory-bound: it reads the tensor, computes mean and variance, reads it *again* to normalize, writes it out. The win is fuse the passes so the data is read once, keep the statistics in registers, vectorize the loads with `float4`. PyTorch just left bandwidth on the table for that shape, and the search found it. It is the memory-bound playbook your students learned — executed by a machine that could try a hundred variants overnight. No magic. Just the regime, run at scale.]]

Now the losses, which are the *real* lesson. Why does the exact same loop hit **9%** on FlashAttention? Because FP16 matmul and attention on modern GPUs need the tensor cores, and feeding the tensor cores means orchestrating `wgmma` warpgroup instructions, `TMA` async copies, shared-memory staging, and a double-buffered software pipeline that hides every latency behind the next stage. That machinery is a *decade* of hand-tuned, hardware-specific engineering. A model that has seen almost no correct examples of it in its training data cannot rediscover it in five rounds. It emits a flat, un-pipelined kernel that never even reaches the tensor cores — and stalls.

[[note: aha || The rule that falls out is clean, and it's the sentence to leave students with: **the AI wins exactly where a competent human with a profiler would have won, against opponents who weren't already trying.** It closes the easy, memory-bound, FP32 gaps PyTorch left open because nobody bothered. It cannot invent the `wgmma` pipeline a compute-bound FP16 kernel demands — because that gap was already closed by people who spent years on it. The AI didn't make the kernel engineer obsolete. It made the *hard* kernels the only ones worth a human's afternoon.]]

[[fig: A hand-drawn two-column scorecard titled "Where AI kernels win — and why". LEFT column headed orange "WINS (FP32, memory-bound)": four yellow-hatched op-boxes with big orange numbers "LayerNorm 484%", "Conv2D 180%", "Conv+ReLU+MaxPool 290%", "FP32 Matmul 101%", each tagged blue "win = fuse passes + coalesce + float4; PyTorch left bandwidth on the table". RIGHT column headed red "LOSSES (FP16, compute-bound)": two red-hatched op-boxes with red numbers "FP16 Matmul 52%", "FlashAttention 9%", each tagged purple "needs wgmma + TMA + async pipeline — a decade of hand-tuning". A central vertical dashed divider. Bottom dashed takeaway box spanning both: "AI beats PyTorch only where a human with a profiler would have — the unclaimed memory-bound FP32 gaps". Excalidraw style, white background, hand-lettered. || The whole result on one card: the wins are the unclaimed memory-bound FP32 gaps; the losses are the fully-tuned compute-bound tensor-core kernels the search can't rediscover.]]

## The tie to harnesses — why this is the finale

Here is why this chapter closes the workshop. Look back at that search loop. `propose_ideas` is the *hypothesis*. `write_code` plus the two gates is *implement and verify*. `benchmark` and `profile` are *read the bottleneck*. The archive is the log of everything that worked. That is **exactly** the predict-then-measure discipline the students practiced by hand — the machine just runs it wider and never gets tired.

[[note: production || And this is live, right now. Sakana AI's "AI CUDA Engineer" evolves against an archive of good kernels. DeepMind's AlphaEvolve ran the propose-verify-keep loop and found a **32.5%** speedup on a real JAX/Pallas FlashAttention kernel. These aren't demos — they're the same loop your students built, pointed at production code. But notice what makes all of them *work*: not a smarter model, a **trustworthy checker**. The reason a hundred blind samples beat one careful one is that KernelBench's compile-and-diff gate can sift them. The center of gravity moved from the model to the *harness* around it.]]

[[note: say || "So what did you actually build in this workshop? You think you learned to write fast kernels. You did — but you learned something bigger. You learned the *loop*: hypothesize, implement, verify, profile, keep what works. That loop is now the most valuable thing in AI kernel engineering, because it's what you wrap a language model in to make it useful. The model is the cheap monkey. *You* — and the harness you build — are the editor at the door. And the editor is where the reliability lives."]]

## You can now teach

- **KernelBench** as a cooking contest: the PyTorch layer is the spec, the correctness check, and the baseline all at once — and Level 1 is the *hardest* to beat, not the easiest.
- **fast_p** as two turnstiles: correctness then speed, two orthogonal gates that make the benchmark impossible to game — and why "under 20% on fast_1" is an *honest* number, not a gloomy one.
- The **monkeys result**: 100 samples with a checker take DeepSeek-V3 from 4% to 37%, with the one-line proof `1 − (1−p)^k` — and why the trust lives in the checker, not the model.
- **Serial feedback**: handing the model the profiler doubles the score (36% → 72%), because it's the same three-regimes diagnostic a human reads.
- **Evolutionary search with an archive**: propose ideas in English, branch wide, seed the best, recombine — and why the winners show up in rounds 4–5.
- The **honest scorecard**: 484% on LayerNorm (memory-bound, no magic) vs 9% on FlashAttention (the `wgmma`/`TMA` pipeline the search can't rediscover) — the AI wins exactly where a human with a profiler would have.
- The **tie to harnesses**: the search loop *is* the predict-then-measure discipline, and the human's real product is now the harness — the trustworthy editor around a cheap generator.
