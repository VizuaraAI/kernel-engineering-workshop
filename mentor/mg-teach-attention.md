By the end of this chapter you can stand at a whiteboard and teach attention as three matrix multiplies — `QKᵀ`, then softmax, then `×V` — and, more importantly, teach *why the giant N×N scores in the middle are a disaster*, so that when you unveil FlashAttention next it lands like a magic trick with the misdirection already explained.

You do not need to know how attention was invented, or what queries and keys "mean" philosophically. For this chapter, attention is a machine that takes three tables of numbers and produces one table of numbers. We treat it that mechanically, because the mechanical view is the one that reveals the performance problem. Let's build it up the way you'll build it for students.

## The one-sentence version

You already taught matrix multiply. Attention is just **three matmuls in a row**, with one small squishing step wedged in the middle. That's the whole thing. If a student can multiply matrices — and after the matmul chapter, they can — they can already do attention. You are not teaching a new hard idea. You are teaching a recipe made of ingredients they already own.

[[note: say || "Attention looks scary because of the word and the Greek letters. But here is the entire operation: multiply, squish, multiply. Three steps. You already know two of them cold — they're just matmuls. The middle step, softmax, is a fancy word for 'turn a row of numbers into percentages.' That's it. That's attention."]]

## Meet the three tables: Q, K, V

Every word in your sentence gets turned into three little rows of numbers. Call them the word's **query**, its **key**, and its **value**. Stack all the queries into one table `Q`, all the keys into `K`, all the values into `V`. Each table has one row per word and `d` numbers across (`d` is the "head dimension" — think 64 or 128). So each is shaped `N × d`, where `N` is how many words are in the sentence.

[[note: metaphor || Think of a room full of people at a networking event. Each person carries three cards. The **query** card says "here's what I'm looking for." The **key** card says "here's what I offer." The **value** card is the actual thing they'd hand you if you talked to them. Attention is the process of every person comparing their query to everyone's keys, deciding who's worth listening to, and then collecting a blended handout from the people they cared about most. Q asks, K advertises, V delivers.]]

[[fig: A warm hand-drawn illustration of a networking-event metaphor for attention. A room of five friendly stick-figure people standing in a loose circle, each holding a fan of three labeled cards: a blue card "QUERY: what I'm looking for", a green card "KEY: what I offer", a yellow card "VALUE: what I'd hand you". One highlighted person in the center (orange glow) has a thought bubble reading "who here matches what I want?" with little dashed comparison arrows reaching out to the KEY cards of the other four people. A hand-drawn note points to the arrows: "compare MY query to EVERYONE's key". A dashed takeaway box at the bottom reads "Q asks · K advertises · V delivers — attention is everyone comparing queries to keys, then collecting a blend of values". Excalidraw style, white background, charming, handwritten labels. || The attention metaphor: a room of people comparing their query card to everyone's key card, then collecting a blended handout of value cards.]]

Don't explain where Q, K, V come from — that's a distraction here. Just hand students the three tables as given, and say: our job is to combine them into one output table `O`, also shaped `N × d`. Same shape in, same shape out. The magic is what happens in between.

## Step 1 — QKᵀ: everybody compares with everybody

The first step asks, for every word: *how much should I pay attention to every other word?* To measure "how much word i cares about word j," we take word i's query row and word j's key row and — you guessed it — do a **dot product**. Big dot product means "these two match, pay attention." Small or negative means "ignore."

Do that for every pair of words and you get a grid of scores. Row i, column j is "how much word i cares about word j." That grid is exactly `Q` times `K`-transposed, written `QKᵀ`. Its shape is `N × N` — one score for every pair of words.

[[note: example || Do it tiny on the board. Sentence of N=3 words, head dimension d=2. Say Q's first row is `[1, 0]` and K's three rows are `[1, 0]`, `[0, 1]`, `[1, 1]`. Word 1's scores are the dot products: `[1,0]·[1,0]=1`, `[1,0]·[0,1]=0`, `[1,0]·[1,1]=1`. So row 1 of the score grid is `[1, 0, 1]` — word 1 strongly cares about words 1 and 3, and ignores word 2. Do one row live; the grid fills in the same way row by row.]]

Notice the shape: three words in, but a **3×3** grid of scores comes out. Nine numbers from three words. Ten words would give a hundred scores. A thousand words would give a *million*. This is the seed of the whole problem, and you should plant it right now, quietly: *the score grid grows with the square of the sentence length.*

[[note: teach || Draw the score grid as a literal N×N spreadsheet with gridlines. Point at cell (i,j) and say the sentence "how much does word i care about word j" every single time — repeat it until the room is bored. The boredom is the goal: you want "row = who's asking, column = who's being looked at" burned in before softmax arrives. Circle the diagonal and note every word usually cares about itself.]]

There's a tiny footnote step: we divide every score by `√d`. Don't make a fuss about it. It just keeps the numbers from getting huge when `d` is big, so the next step behaves. Mention it, write `/√d`, move on.

## Step 2 — softmax: turn scores into percentages

A row of raw scores like `[1, 0, 1]` isn't a set of weights yet — the numbers don't add up to anything meaningful. **Softmax** fixes that. It takes a row of numbers and turns it into percentages that add up to 100%, where bigger inputs get bigger shares. It's the "turn scores into a pie chart" step.

The recipe, per row: raise `e` to the power of each number (this makes everything positive and stretches the big ones ahead), then divide each result by the total of the row. Now every number is between 0 and 1, and the row sums to 1. Those are your attention weights.

[[note: example || Take the row `[1, 0, 1]`. Exponentiate: `e^1 ≈ 2.72`, `e^0 = 1`, `e^1 ≈ 2.72`. Total = `6.44`. Divide each: `2.72/6.44 ≈ 0.42`, `1/6.44 ≈ 0.16`, `2.72/6.44 ≈ 0.42`. So word 1 puts 42% of its attention on word 1, 16% on word 2, 42% on word 3. They sum to 1.00. That's a softmax by hand, on real numbers, in thirty seconds.]]

[[fig: A hand-drawn two-panel figure titled "Softmax turns a row of scores into a pie of percentages". Left panel: a horizontal row of three bars of unequal height labeled with raw scores "1", "0", "1" in red, above a blue arrow labeled "e^x then divide by the row total". Right panel: a pie chart split into slices "42%", "16%", "42%" in yellow/green/yellow with a green note "adds up to 100%". Between the panels the exponent math is written small in purple: "e^1=2.72, e^0=1, e^1=2.72 → total 6.44". A dashed takeaway box reads "softmax = make everything positive (e^x), then share it out so the row sums to 1". Excalidraw style, white background, handwritten labels. || Softmax, drawn as turning a row of raw scores into a pie chart of attention percentages that sum to 100%.]]

Do this to every row of the `N×N` score grid and you get another `N×N` grid — call it `P`, for probabilities. Same size as the scores. Same square that grows with the sentence.

[[note: confusion || The classic trip-up: students softmax down the columns instead of across the rows. Fix it with the sentence from step 1 — "each row is one word deciding how to split its attention across everyone, so the split must live along the row." A word's attention is a pie; a pie belongs to one word; one word is one row. Make them point at a row and say "this word's pie" out loud.]]

## Step 3 — ×V: collect the blended handout

Now every word has a row of percentages saying how much to weigh each other word. The final step: use those percentages to blend the **value** rows. Word 1's output is 42% of value-row-1, plus 16% of value-row-2, plus 42% of value-row-3 — a weighted average of the value rows, using this word's attention percentages as the weights.

And a weighted-average-of-rows is exactly what a matrix multiply does. So step 3 is just `P` times `V`, giving the output `O`, shaped `N × d`. Same shape we started with. The sentence went in as three tables and came out as one, having let every word gather a custom blend of information from every other word.

[[note: aha || Say the shape story out loud and watch it click: "We started with tables of size N-by-d. We blew up to an N-by-N grid in the middle. Then we collapsed back down to N-by-d. Attention is a balloon — it inflates to a square in the middle and deflates at the end. And that fat square in the middle? Nobody wanted it. We only wanted the skinny output. The square was scratch work." That framing is the entire motivation for FlashAttention, delivered in one breath.]]

[[fig: A warm hand-drawn metaphor illustration titled "Attention is a balloon". Three cartoon stages left to right connected by arrows. Stage 1: a small deflated balloon labeled in green "N×d (skinny: Q,K,V)". Stage 2: the same balloon blown up huge and round, straining, labeled in orange "N×N — inflated, heavy, nobody wanted this big", with a little sweat-drop and a tiny person struggling to hold it. Stage 3: the balloon deflated small again labeled in yellow "N×d (skinny: output O)". Below stage 2, a hand-drawn thought bubble reads "and we hauled this whole balloon out to the garage (slow memory) and back". A dashed takeaway box reads "attention inflates to a giant square in the middle, then deflates — the giant middle is pure scratch we shouldn't have to store." Excalidraw style, white background, charming, hand-lettered. || The balloon metaphor: attention inflates to a giant N×N square in the middle and deflates back to skinny — the fat middle is scratch nobody wanted.]]

[[fig: A hand-drawn technical diagram titled "Attention is three matmuls: inflate, squish, deflate". Three stages left to right with numbered circles. Circle (1): two narrow-tall matrices Q and K (blue diagonal hatch, red labels "N×d") feeding a black box "QKᵀ / √d" that outputs a big SQUARE grid S (pale-yellow hatch, red labels "N×N" on both sides) with an orange emphasis note "the square nobody wanted". Circle (2): S passes through a box "softmax per row" producing another same-size square P (pale-yellow hatch, "N×N"), with a small blue note "each row → percentages". Circle (3): P times a narrow-tall V (green hatch, "N×d") through box "P·V" producing a narrow-tall output O (yellow hatch, red "N×d") labeled in orange "the skinny thing we actually wanted". A red bracket under the middle squares reads "N×N — grows with the SQUARE of sentence length". Dashed takeaway box: "small → SQUARE → small. attention inflates to N×N in the middle, then deflates. the square is pure scratch work." Excalidraw style, white background, hand-lettered. || The technical translation: attention inflates from N×d to a square N×N and back down, and the square in the middle is scratch nobody keeps.]]

## Where the giant square goes wrong

Here is the pivot of the chapter, and where you set the hook for FlashAttention. Everything above is *correct*. Written in three lines of PyTorch, it runs and gives the right answer:

```python
S = (Q @ K.transpose(-2, -1)) / math.sqrt(d)   # (N, N)  scores
P = softmax(S, dim=-1)                          # (N, N)  probabilities
O = P @ V                                       # (N, d)  output
```

It is also, for any real sentence, *shockingly slow* — for a reason that has nothing to do with how much math it does. The math is fine. The problem is that square.

Recall the catch from the CPU-vs-GPU chapter: the cooks are faster than the hallway that feeds them. A GPU is limited by how fast it can be *fed data*, not how fast it computes. Now look at what these three lines do to memory. The score square `S` is `N × N`. For a sequence of `N = 8192` words in FP16, that's `8192 × 8192 × 2 bytes = 128 MiB` — and that's *per attention head, per layer*. It is far too big to sit in the GPU's fast on-chip memory. So it gets written all the way out to slow, far-away main memory (HBM), then read all the way back for the next step.

[[note: production || This isn't a toy concern — it's *the* concern in every serving stack today. When you chat with Llama, DeepSeek, or ChatGPT with a long prompt, this exact N×N square is what threatens to blow up. That 128 MiB is per head; a model has dozens of heads and dozens of layers. Materializing all those squares would move terabytes across HBM for a single forward pass. On an H100, the tensor cores can do ~989 trillion math ops a second, but HBM only feeds them at 3.35 TB/s — the cooks vastly out-run the hallway.]]

Count the round-trips for that square and it's damning. `QKᵀ` **writes** the N×N scores out to HBM. Softmax **reads** them back, then **writes** the N×N probabilities out again. Then `×V` **reads** them back once more. That's at least four full N×N trips across the slow memory boundary — for a quantity the algorithm never actually wanted to keep. The three tables we started with (`Q, K, V`) and the output `O` are all "skinny": their size grows *linearly* with N. It's only that fat middle square that grows with **N²**. And the square is exactly what we're shoveling back and forth.

[[note: aha || The jaw-drop number. `Q, K, V, O` together are about `4·N·d` numbers. The score-square traffic is about `4·N²` numbers. The ratio is `N/d`. At N=8192 and d=128, that's **64× more bytes moved for the scratch work than for all the real data combined.** Write "64×" on the board and circle it. The kernel spends most of its wall-clock time not computing — just hauling a temporary square out to memory and back.]]

[[fig: A hand-drawn bar-chart figure titled "Where the time actually goes". Two horizontal bars. Top bar labeled in red "real data: Q,K,V,O traffic" is short, blue hatch, green note "~4·N·d — linear in N". Bottom bar labeled "the scratch square: S and P round-trips" is very long (about 6x the top bar), pale-yellow hatch, orange emphasis "~4·N² — quadratic". A red dimension bracket spans the long bar labeled "this is the bill". A small napkin-math note in purple: "N=8192, d=128 → ratio N/d = 64× more bytes for scratch". A blue arrow points from the long bar to a little drawing of a far-away memory cylinder labeled "HBM · 3.35 TB/s — the wall". Dashed takeaway box: "the FLOPs were fine. the TIME is spent hauling an N×N square we never needed out to slow memory and back." Excalidraw style, white background, hand-lettered. || The bytes, not the FLOPs, set the clock: the quadratic scratch square dwarfs all the real data traffic.]]

[[sn: The two matmuls (`QKᵀ` and `PV`) are genuinely efficient — real GEMMs that keep the tensor cores busy. It's the softmax between them, plus the mandatory write-then-read of the square, that stalls. Softmax reads N² numbers, does a pinch of arithmetic each, and writes N² back — the definition of memory-bound.]]

## The fix, foreshadowed

So end the chapter by naming the villain and pointing at the hero. The villain is not the math — the math is exactly what attention requires and the GPU eats it happily. The villain is the *decision to write the N×N square down in slow memory at all.* We inflated to a square, saved it to HBM, read it back, and deflated — when we only ever wanted the skinny output.

[[note: say || "Here's the question that breaks the whole thing open, and I want you to feel how impossible it sounds: what if we never write the square down at all? What if we compute a little block of scores, softmax it, multiply by the values, add it to the answer, and throw the block away — before it ever touches slow memory? Keep only running totals, never the whole square. If we could do that, all that 64× traffic just... vanishes. That trick has a name. It's called FlashAttention, and it's why long-context models are possible today. That's next."]]

[[note: teach || Board sequence for the whole talk, so you don't lose the room: (1) three cards Q/K/V and the networking-room metaphor — 4 min. (2) tiny 3-word QKᵀ by hand, the square appears — 6 min. (3) softmax one row into percentages by hand — 5 min. (4) ×V as a weighted blend, output is skinny again — 4 min. (5) the balloon: small→square→small, "nobody wanted the square" — 3 min. (6) the 64× traffic number and the HBM round-trips — 5 min. (7) tease FlashAttention: "what if we never write the square down?" — 2 min. The demo: run the three PyTorch lines, print the shapes, and print `S.numel() * 2 / 1e6` MiB for a big N so the room gasps at the megabytes.]]

## You can now teach

- **Attention as three matmuls** — `QKᵀ`, softmax, `×V` — built entirely from the dot product and matmul students already know.
- **Q, K, V as three cards** at a networking event: query asks, key advertises, value delivers; output is a blended handout.
- **Softmax** as "turn a row of scores into a pie chart of percentages," done by hand on real numbers.
- The **balloon shape story**: attention inflates from N×d to an N×N square and deflates back — and the square is scratch nobody wanted.
- **Why the N×N square is the problem**: it's too big for on-chip memory, so it round-trips through slow HBM four times, moving ~64× more bytes than all the real data — the FLOPs were never the issue.
- The **setup for FlashAttention**: "what if we never write the square down?" — the exact question the next chapter answers.
