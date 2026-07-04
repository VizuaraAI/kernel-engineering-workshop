By the end of this chapter you'll be able to stand at a whiteboard and teach softmax three times over — the naive version, the *stable* version that doesn't blow up, and the beautiful *online* version that computes everything in a single walk — so clearly that your students will see, with their own eyes, the exact trick that makes FlashAttention possible. We start from zero. Just a short list of numbers, a pencil, and a running tally.

This chapter ends with a genuine "wait, that's *clever*" moment. Your job is to earn it slowly — one number at a time.

## What softmax is, in plain words

Softmax takes a list of numbers — some big, some small, some negative — and turns them into **probabilities** that add up to 1. Bigger inputs get bigger shares. That's the whole job. It's the last step of a classifier ("cat vs dog vs bird?") and the heart of every attention head ("how much should this word attend to each other word?").

The recipe is two steps. First, **exponentiate** every number — run each through `e^x`, which makes big numbers enormous and keeps everything positive. Then **divide** each result by the total, so the set sums to 1.

[[note: metaphor || Think of a **school election** where students vote by shouting. Each candidate has a raw "loudness" score. Softmax first turns loudness into cheering — `e^x` — so a slightly louder candidate gets *way* more cheers (exponentials exaggerate gaps). Then you divide each candidate's cheers by the total cheers in the room to get their **share of the vote**. Everyone's shares add to 100%. Louder wins, but everyone gets some slice.]]

[[fig: A warm hand-drawn illustration of the "school election" metaphor for softmax. Three candidate figures stand on a stage, each with a raw loudness score in a red box above their head: "2", "1", "3". Below each, a big cheering crowd bubble sized by e^x, hand-labeled in purple "e^2 ≈ 7.4", "e^1 ≈ 2.7", "e^3 ≈ 20.1". A green handwritten arrow points to a pie chart on the right split into three slices labeled "0.24", "0.09", "0.67" with a note "shares add to 1". A dashed takeaway box at the bottom reads "softmax = exaggerate with e^x, then split the vote into shares". Excalidraw style, white background, charming, handwritten labels. || Softmax as a school election: exponentiate the scores into cheers, then divide into shares of the vote.]]

## The tiny by-hand number

Do this one on the board first, before any formula. Take three logits: `[2, 1, 3]`.

- Exponentiate: `e^2 = 7.39`, `e^1 = 2.72`, `e^3 = 20.09`.
- Add them up: `7.39 + 2.72 + 20.09 = 30.20`.
- Divide each by the total: `7.39/30.20 = 0.24`, `2.72/30.20 = 0.09`, `20.09/30.20 = 0.67`.

Answer: `[0.24, 0.09, 0.67]`. They sum to 1. The biggest input (3) got the biggest share (0.67). Done.

[[note: example || On the board, write the three logits in a row, the three exponentials underneath, the single sum off to the side in a circle, and the three final shares in a bottom row. Students *see* the shape of it: a row in, exponentiate down, one sum, divide, a row out. Keep the numbers rounded to two decimals — precision isn't the lesson yet.]]

## Why the naive version explodes (the first plot twist)

Here's the naive code, exactly as a student would write it from the recipe:

```python
def softmax_naive(x):
    e = np.exp(x)          # exponentiate everything
    return e / e.sum()     # divide by the total
```

Correct on paper, a time bomb in practice. The problem is `e^x` grows *insanely* fast. In the standard 32-bit float a computer uses, `e^x` becomes `inf` — "infinity, too big to store" — the moment `x` passes about **88.7**, and attention scores in real models routinely fly past that.

Once one number is `inf`, the sum is `inf`, and every share becomes `inf / inf`, which the computer reports as `NaN` — "not a number." One overgrown value poisons the entire row.

[[note: confusion || Students assume "the math is right, so the code is right." This is the moment to break that assumption gently. Say: "The formula is perfect. The *computer* is the problem — it can only hold numbers up to a certain size, and `e^x` blows past that ceiling fast." Draw a thermometer that maxes out at 88.7 with `e^x` shooting off the top into a red `inf`. The bug isn't in the idea; it's in the hardware's finite drawer of storage.]]

[[fig: A hand-drawn "overflow thermometer" figure titled "Why naive softmax explodes". A vertical thermometer scale from 0 up to a red line marked "≈ 88.7  FP32 ceiling". A blue curve labeled "e^x" rises gently, then shoots vertically off the top of the thermometer into a jagged red starburst labeled "inf". Below, a row of three logit boxes "1, 2, 90" with the 90 circled orange and an arrow to the starburst. To the right, a poisoned output row "[NaN, NaN, NaN]" in red with a sad face. A dashed takeaway box: "one value over 88.7 → e^x = inf → whole row = NaN". Excalidraw style, white background, handwritten. || The naive bug drawn as a thermometer: exp overflows past ~88.7, and one infinity poisons the entire row.]]

## The fix: subtract the biggest number first

The rescue is one line, and it's exact — no precision lost, no approximation. Before exponentiating, find the biggest number in the row, call it `m`, and **subtract it from everything**. Now the largest value becomes 0, everything else is negative, and `e^(negative)` is always between 0 and 1. Overflow becomes structurally impossible.

Why is this allowed? Because subtracting a constant from every input doesn't change the answer at all. The same factor cancels out on the top and the bottom of the division. Show students the algebra once, slowly: `e^(x−m)` on top and `e^(x−m)` summed on the bottom — the hidden `e^(−m)` is in every single term, so it divides away.

[[note: say || "We're not changing the answer — we're just sliding all the numbers down so they fit in the computer's drawer. Watch: I subtract the biggest one from *everybody*. The winner becomes zero, everyone else goes negative, and `e` of a negative number is always small and safe. The shares come out *identical* — I've only moved the numbers, not their meaning."]]

[[note: example || Redo `[2, 1, 3]` the stable way on the board. Max is `m = 3`. Subtract: `[−1, −2, 0]`. Exponentiate: `[0.37, 0.14, 1.00]`. Sum: `1.51`. Divide: `[0.24, 0.09, 0.67]`. **Same answer as before** — that's the whole point. It's the identical result, just computed without ever risking an overflow.]]

Here's the stable version:

```python
def softmax_stable(x):
    m = x.max()            # pass 1: find the biggest
    e = np.exp(x - m)      # pass 2: shifted exp, and their sum
    return e / e.sum()     # pass 3: divide
```

## Now count the trips to memory (the real lesson)

Here softmax stops being a math lesson and becomes a *kernel engineering* lesson. Count how many times the stable code walks over the list `x`:

1. Once to find the max.
2. Once to exponentiate and sum.
3. Once to divide.

**Three passes over the data.** And here's the surprise: softmax does almost no arithmetic per number — just an exp and a couple of adds. It spends nearly all its time *fetching numbers from memory*, not computing. So on a GPU, the cost of softmax is basically "how many times did you walk the list?" Three walks means roughly three times the minimum time.

[[note: aha || Say this and watch it land: **"Softmax is not slow because the math is hard. It's slow because you keep re-reading the same numbers from far-away memory."** The exp is nearly free. The reading is the whole bill. Every optimization from here on is about *reading the list fewer times* — going from three walks, to two, to one. That reframe is the entire chapter, and it's the entire spirit of kernel engineering.]]

[[fig: A hand-drawn "three walks" figure titled "Softmax cost = how many times you walk the list". A row of logit cells drawn as a blue hatched strip labeled in red "N logits in slow memory". Three curved arrows loop over the whole strip left-to-right, stacked and numbered in orange circles: (1) "walk to find max", (2) "walk to exp + sum", (3) "walk to divide". A green side note points at a single cell: "tiny math per number — just e^x". An orange emphasis callout: "the walking is the cost, not the math". A dashed takeaway box: "3 walks = 3× the floor. the win is fewer walks." Excalidraw style, white background, handwritten labels. || The stable softmax walks the list three times. On a memory-bound kernel, the number of walks *is* the runtime.]]

[[note: production || This isn't academic. Every time you chat with a model, every attention head runs a softmax over its scores, and the final answer is a softmax over the vocabulary. In a real serving stack on H100 or B200 GPUs, these reductions are memory-bound — the fast tensor cores sit idle watching. Horace He's famous "Making Deep Learning Go Brrrr" note showed that softmax-and-friends are a rounding error of the *math* in a transformer, yet eat a wildly outsized share of the *clock*. That's exactly why cutting the walks matters in dollars.]]

## From three walks to two

The easiest walk to delete is the third — the divide. Dividing each number by the total is *pointwise*: it looks at no other number, so it needs no walk of its own. It piggybacks on whatever step *uses* the softmax output next. In attention, softmax feeds straight into another matrix multiply, and the divide folds into the read that matmul already does. So honestly, softmax is **two walks**: one for the max, one for the exp-and-sum. This is the "safe softmax" good libraries ship.

But two walks still bothers a kernel engineer. The first walk reads *every number* just to extract one tiny fact — the max — then throws the data away. The second reads *every number again*. We loaded every byte twice. So the door-opening question is: **can we find the max and the sum in one single walk — even though we don't know the max until we've seen the whole list?**

It sounds impossible: the sum depends on the max, and you don't know the max until the end. That tension is the crux, and cracking it is the payoff of the chapter.

## The online trick: keep a running tally and fix it up

Here's the idea, and it's the emotional peak of the lesson. Walk the list **once**, left to right. Keep two running values as you go:

- `m` — the biggest number **seen so far** (not the final max, just so far).
- `d` — the running sum of `e^(x − m)`, for everything seen so far, measured against the *current* running max.

The magic is what happens when you hit a number bigger than any before. Your running max jumps up. But every term already in your sum `d` was measured against the *old, smaller* max — they're all now slightly too big. So you **rescale** the whole sum by one correction factor, then add the newcomer:

```
m_new = max(m, x_i)                              # did the record change?
d = d * exp(m - m_new) + exp(x_i - m_new)        # rescale history, add newcomer
m = m_new
```

That `d * exp(m - m_new)` is the entire trick. When no new record is set, `m - m_new = 0`, the factor is `exp(0) = 1`, and it degenerates to a plain running sum. When a new record *is* set, the factor is less than 1, and it shrinks every previously-counted term into the new reference frame — exactly as if you'd known the new max all along.

[[note: metaphor || Teach it as **grading on a curve while the exams are still coming in**. You're scoring papers one at a time, curving each against the highest score *so far*. Suddenly a genius hands in a paper that beats everyone. The curve just shifted — so every score you already wrote down is now too generous. Instead of re-grading the whole stack, you multiply the running total by one correction factor to pull it into line with the new top score, and carry on. You never go back. One pass through the pile.]]

[[note: example || Run `[1, 3, 2]` by hand, one number at a time, so students see the rescale fire. Start `m = −∞, d = 0`.
• See `1`: `m = 1`, `d = 0·(…) + e^0 = 1`.
• See `3` (new record!): `m_new = 3`. Rescale: `d = 1·e^(1−3) + e^(3−3) = 1·0.135 + 1 = 1.135`. Set `m = 3`.
• See `2` (no record): `d = 1.135·e^(3−3) + e^(2−3) = 1.135 + 0.368 = 1.503`.
Final `m = 3, d = 1.503`. Check against the two-pass answer: `e^(−2)+e^0+e^(−1) = 0.135+1+0.368 = 1.503`. **Identical.** The rescale on the `3` is the moment to point at and say "there — it just fixed history."]]

[[fig: A hand-drawn three-panel figure titled "Online softmax: one walk, fix history when the max jumps". Panel (1): a row of logit cells [1, 3, 2] hatched blue, an orange scan cursor at the first cell, green state below "m = −∞, d = 0". Panel (2), circled (2): cursor on the "3" cell highlighted orange as a NEW RECORD, purple code beside it "m_new = max(m, xi)" and "d = d·exp(m−m_new) + exp(xi−m_new)"; a curved blue dashed arrow loops back over the already-scanned cells with a note "rescale old sum ↓ (shrink it)". Panel (3), circled (3): cursor at the end, green state "m = 3, d = 1.503", and a red stamp "= exact two-pass result". A dashed takeaway box at the bottom: "one walk. exp(m−m_new) corrects the sum whenever a new max appears." Excalidraw style, white background, handwritten labels. || The online update in three beats: carry a running max and sum, and every time the max jumps, retroactively rescale the sum you already have.]]

Here's the whole thing as a loop — the reference version to show right after the by-hand run:

```python
def softmax_online(x):
    m = -np.inf            # running max
    d = 0.0                # running sum, relative to current m
    for xi in x:                       # ONE walk
        m_new = max(m, xi)
        d = d * np.exp(m - m_new) + np.exp(xi - m_new)
        m = m_new
    return np.exp(x - m) / d           # pointwise divide, folds downstream
```

One walk to get both the max and the sum. The final divide is the pointwise tail we already agreed rides along with the next kernel. So online softmax is **one walk plus a free tail** — the fewest possible trips to memory for a stable softmax.

[[sn: This is exact, not approximate. After the whole list, `d` equals the true sum against the final max, down to the last bit — identical to the two-pass answer. The rescale just *spreads* the max-subtraction across the walk instead of doing it all at the end.]]

## Teach it as a running average

Students already know one "fix it up as you go" pattern: a **running average**. You keep a mean, and each new number nudges it — you never store the whole list. Online softmax is the *same shape of idea* — a running tally you correct as new data arrives — except the correction is a multiply (the rescale) instead of a nudge. Anchor the trick to that familiar feeling: "you already trust running averages; this is a running sum that also rescales when the reference point moves."

[[fig: A warm hand-drawn "running tally" metaphor figure titled "You already do this: a running average". A shopkeeper at a counter with a single small notepad, customers walking past one at a time dropping numbers. The notepad shows just two scribbled values "count" and "running mean", updated with an arrow as each customer passes — a green note "never stores the whole line, just fixes up the tally". Beside it, a second notepad labeled "online softmax" showing "running max m" and "running sum d" with a purple note "same idea + a rescale when the record breaks". A dashed takeaway box: "carry a tiny tally, correct it as you go — one pass, no storing everything". Excalidraw style, white background, charming, handwritten labels. || The mental anchor: online softmax is a running tally, just like a running average, with one extra rescale step.]]

[[note: teach || Board sequence that never fails: (1) naive softmax on `[2,1,3]` — it works. (2) Break it: swap in a `90`, watch it `NaN`. (3) Fix with subtract-the-max — same answer, safe. (4) Count the walks: three, then two. (5) Pose the impossible question — "one walk?" — and pause. Let them feel it's impossible. (6) Reveal the running max + rescale, run `[1,3,2]` by hand, and land on "identical answer, one walk." The pause before step 6 is what makes the reveal hit. Don't rush it.]]

## Where this lives in production

Now the punchline. This exact trick is the beating heart of **FlashAttention** — the kernel inside essentially every large model served today, from Llama to DeepSeek to ChatGPT. Attention must softmax a row of scores *far too big to hold in fast memory at once*, so it streams them through in tiles. Which means it can *never* do a two-pass softmax — it never sees the whole row before it must start accumulating results.

The online trick rescues it. FlashAttention keeps a running max, a running sum, *and* a running output — and every time a new tile pushes the max higher, it rescales the partial output by the same `exp(m − m_new)` factor you just taught. Softmax and the value-multiply fuse into one streaming pass that never writes the giant score matrix to memory at all.

[[note: production || The jaw-drop number: FlashAttention's whole speedup comes from *not walking memory extra times*, exactly the lever from this chapter. Going from softmax's three memory walks toward one approaches a **3× speedup on the softmax itself** — and when fused into attention, it's the difference that let context windows grow from a couple thousand tokens to hundreds of thousands. The industry adopted it within months. The rescale identity on your whiteboard is, quite literally, FlashAttention's inner loop minus the matmul.]]

[[fig: A hand-drawn timeline figure titled "Three walks → two → one". Three stacked horizontal timelines on a shared left-to-right time axis. Top row "naive stable" (black): three separate blue blocks "walk: max", "walk: exp+sum", "walk: divide" spanning the full width, red tag "3× memory traffic". Middle row "fused 2-pass": two blue blocks "walk: max", "walk: exp+sum", with a faded "divide" block merged into a downstream kernel, orange tag "2×". Bottom row "online" (orange): a SINGLE blue block "walk: max + sum (running rescale)" plus a faded "divide fuses downstream", green tag "1× — the floor". A vertical dashed line shows the online row finishing first. Dashed takeaway box: "same math, same stability — one third the trips to memory." Excalidraw style, white background, handwritten labels. || The whole optimization in one picture: identical output and identical stability, a third of the trips to memory.]]

## You can now teach

- **What softmax is** — exaggerate scores with `e^x`, then divide into shares that sum to 1 — with a clean by-hand `[2,1,3]` example.
- **Why the naive version explodes** — `e^x` overflows past ~88.7 and one infinity turns the whole row to `NaN` — and the exact **subtract-the-max** fix, shown to be the *identical* answer.
- **Why softmax is a memory problem, not a math problem** — that the number of walks over the list, not the arithmetic, sets the cost.
- **The online trick** — a single walk carrying a running max and a running sum, rescaling history by `exp(m − m_new)` whenever a new max appears — taught as a running average, with a by-hand `[1,3,2]` that matches the two-pass answer exactly.
- **The board sequence** — naive → break it → fix it → count the walks → pose the "one walk?" impossibility → reveal the rescale — and the pause that makes the reveal land.
- **The production hook** — that this exact rescale is FlashAttention's inner loop, the reason modern context windows are huge, and why fewer memory walks is worth a fortune.
