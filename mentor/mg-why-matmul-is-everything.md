By the end of this chapter you'll be able to stand at a whiteboard and teach the single most important sentence in this whole workshop: *a neural network is almost nothing but matrix multiplies.* Not "uses" matrix multiplies. *Is* them. You'll trace a real transformer from the word going in to the word coming out, and you'll count the matmuls with your finger. And when you're done, the students will understand — in their bones — why we're about to spend four weeks making one operation fast.

This chapter assumes the students already met matrix multiply as a grid of dot products (the previous chapter). Here we don't teach *how* to multiply. We teach *where* it lives, *how much* of it there is, and *why that changes everything.*

## The one sentence to open with

Say this first, before any diagram: **"When you talk to ChatGPT, the machine is not thinking. It is multiplying matrices. That's it. Billions of times. The whole magic of AI, underneath, is one boring operation done at an unimaginable scale."**

Then let it sit for a second. It sounds too simple to be true. Making them believe it — really believe it — is the job of this chapter.

[[note: say || "I'm going to make a claim that sounds like an exaggeration, and then I'm going to prove it isn't. Here's the claim: everything a language model does — understanding your question, writing the answer — is, at the bottom, matrix multiplication. When I'm done, you'll be able to point at every single matmul in a transformer. There are surprisingly few kinds, and they run over and over."]]

## The metaphor: a factory with a few machine types

A modern AI model looks impossibly complex from the outside — trillions of parameters, hundreds of layers. But walk inside and it's like a huge factory that only owns **a few kinds of machine**, bolted together in a repeating pattern down a very long assembly line.

There's really just one master machine: the **matrix multiplier**. A word comes in as a little list of numbers. It hits a matmul machine and comes out transformed. It hits another. And another. A few helper machines sit between them — a "normalize" station, an "add" station, a softmax — but they're small. The matmul machines are the ones doing the heavy lifting, burning the electricity, and setting the pace of the whole line.

[[note: metaphor || Picture a car factory that builds every model of car using only one type of robot arm, repeated a thousand times down the line. From outside, the factory makes hundreds of different cars and seems endlessly sophisticated. Inside, it's the *same robot arm* over and over. A transformer is that factory. The robot arm is matrix multiply. Once students see that the sophistication is just *repetition of one simple thing*, the fear of "how does AI work" dissolves.]]

[[fig: A warm hand-drawn factory-line illustration titled "A transformer is a factory of one machine". A long horizontal conveyor belt runs left to right. On the left, a small rounded box labeled in green "your word -> a list of numbers" sits on the belt. The belt passes through a repeating series of identical large blue robot-arm machines, each hand-labeled "MATMUL", with two small grey helper stations between them labeled "normalize" and "add". A hand-drawn bracket spans several machines with an orange note "this block repeats ~100 times". At the far right the belt ends at a box labeled in yellow "-> the next word". A red annotation points at the big machines: "these do 99% of the work". A dashed takeaway box reads "few machine types, repeated deep -> that repetition IS the model". Excalidraw style, white background, charming, handwritten labels. || The whole model as an assembly line: one machine type — matrix multiply — repeated down a very long belt.]]

## First, how a word even becomes numbers

Before any multiplying happens, the model has to turn your text into numbers, because matrices are made of numbers. Teach this in two quick steps.

**Step one: tokens.** The model chops your sentence into pieces called **tokens** — roughly words or word-fragments. "kernels" might be one token; "unbelievable" might split into "un", "believ", "able". Each token has an ID number, like a seat number.

**Step two: embeddings.** Each token ID looks up a row in a giant table — the **embedding matrix** — and pulls out a list of numbers (say 4096 of them) that represents that word's "meaning" as a point in space. So a sentence of 10 tokens becomes a **10 × 4096 matrix**: one row per token, one column per meaning-dimension.

[[note: example || Keep the numbers tiny on the board. Say the sentence is 3 tokens and the model's width is 4. Then the input is a 3×4 grid — three rows (one per word), four columns. Draw it as a literal spreadsheet: 3 rows of 4 boxes. Now the whole rest of the model is: this 3×4 grid gets multiplied by weight grids, over and over. That grid of numbers is *the sentence*, in the only language the GPU speaks.]]

That grid — rows are tokens, columns are the model's width — is the raw material. Everything downstream is that grid meeting weight-grids in matrix multiplies. Let's count them.

## Counting the matmuls in one transformer block

A transformer is a stack of identical **blocks**. Teach one block completely, then just say "now repeat this 32 times" (or 80, or 96). Each block has two halves: **attention** and the **MLP**. Both are made of matmuls. Let's walk them.

### The attention half — four matmuls plus the score

Attention is how each word "looks at" the other words to gather context. Here's the plain version. Our input grid `X` (tokens × width) first gets turned into three new grids — **Q** (queries), **K** (keys), and **V** (values) — and each one is made by *a matrix multiply* of `X` with a learned weight matrix.

- `Q = X · W_Q` — one matmul.
- `K = X · W_K` — one matmul.
- `V = X · W_V` — one matmul.

That's three already, and we haven't even done attention yet. Now the actual attention math, straight from the grounding: `softmax(Q Kᵀ / √d) · V`.

- `Q Kᵀ` — every query dotted with every key. **A matmul.** This is the `N × N` score matrix — how much each word attends to each other word.
- softmax — a *helper station*, not a matmul (it just normalizes each row into probabilities).
- `(scores) · V` — the probabilities times the values. **A matmul.**

And then the result gets mixed back with one more weight matrix:

- `O = (attention output) · W_O` — **a matmul.**

[[note: aha || Count them out loud with the room: Q, K, V — that's three. Q times K-transpose — four. Scores times V — five. The output projection — six. **Six matrix multiplies in the attention half of one block.** The famous, mysterious "attention mechanism" that powers every chatbot on Earth is *six matmuls and one softmax.* Watch the students exhale — attention just stopped being magic and became arithmetic they can count on one hand plus one finger.]]

[[fig: A hand-drawn technical diagram titled "Attention = 6 matmuls + 1 softmax". On the left, a narrow blue-hatched grid labeled in red "X (tokens × width) — the input". Three blue arrows fan out to three matmul boxes numbered in circles (1)(2)(3), each labeled purple "X · W_Q", "X · W_K", "X · W_V", producing three grids labeled red "Q", "K", "V". A fourth matmul box (4) "Q · Kᵀ" takes Q and K and produces a pale-yellow SQUARE grid labeled red "scores N×N" with an orange note "the whole-sentence look-at-each-other". A small grey box labeled "softmax (not a matmul)" sits after it. A fifth matmul box (5) "scores · V" produces a grid labeled "attn out". A sixth matmul box (6) "· W_O" produces yellow output "O". A dashed takeaway box reads "6 matmuls per attention half — Q,K,V, QKᵀ, ·V, ·W_O". Excalidraw style, white background, hand-lettered, semantic colors. || The attention half of a block, drawn as what it actually is: six matrix multiplies with a softmax wedged in the middle.]]

### The MLP half — two big matmuls

After attention, the grid flows into the **MLP** (multi-layer perceptron), also called the feed-forward network. This part is even simpler. It's two matmuls with an activation squashed between them:

- `H = X · W_1` — blow the width *up*, usually 4× wider. **A matmul.** (This is often the single biggest matmul in the whole model.)
- an activation function (GELU/SiLU) — a *helper station*, applied to each number, no matmul.
- `Y = H · W_2` — bring the width back *down*. **A matmul.**

[[note: example || Give it size so students feel the weight. If the model width is 4096, the MLP first matmul multiplies by a 4096 × 16384 weight matrix — over 67 million numbers in *one* weight grid, in *one* block. And there are dozens of blocks. This is why the "parameters" of a model number in the billions: they're mostly these weight grids, and every one of them is used in a matmul on every single word.]]

[[note: metaphor || The MLP is a "think it over" station. Attention gathers information from the other words; the MLP is where each word privately mulls over what it gathered. The first matmul spreads the thought out wide (16384 scratch dimensions to reason in), the activation adds a little non-linear "judgment," and the second matmul folds the conclusion back down to normal width. Expand, judge, compress — two matmuls bracketing one squash.]]

### Add them up

So one transformer block is: **6 matmuls (attention) + 2 matmuls (MLP) = 8 matmuls.** Plus a couple of tiny helper stations. Now the punchline that makes the room go quiet:

[[note: aha || A model like Llama has around **80 of these blocks stacked on top of each other.** Eight matmuls per block times 80 blocks is **640 matrix multiplies** — to process the input *once*. And then, to generate each new word, the machine runs the whole stack again. Write on the board: 8 × 80 = 640 matmuls per pass. A paragraph of 200 words is 200 passes — over **one hundred thousand matrix multiplies** for a few sentences of reply. This is the number that justifies the entire course.]]

[[fig: A hand-drawn stacked-blocks figure titled "Count the matmuls in a whole model". A tall vertical stack of ~6 identical rounded blocks drawn receding upward with "×80" written beside a curly brace spanning them in orange. One block is exploded out to the side into two labeled halves: a blue box "ATTENTION = 6 matmuls" and a green box "MLP = 2 matmuls", with a red sum "= 8 matmuls / block". Below the stack an arithmetic line in bold: "8 × 80 = 640 matmuls to read your prompt ONCE". Below that, a second line: "× 200 words generated = ~128,000 matmuls for one reply". A small yellow box at the top labeled "-> next word out". A dashed takeaway box reads "the model is DEEP repetition of ONE operation". Excalidraw style, white background, hand-lettered, semantic colors. || Stacking it up: eight matmuls per block, eighty blocks, run once per generated word — the count explodes into six figures fast.]]

## The real math, built up gently

Now put the shapes on it, so a curious student can verify the cost. Take one MLP matmul: input `X` is `(N × d)` — N tokens, width d. Weight `W_1` is `(d × 4d)`. From the shape rule (inner dims cancel), the result is `(N × 4d)`.

The cost of a matmul, in multiply-adds, is **rows × cols × inner** = `N × 4d × d`. For `N = 200` tokens and `d = 4096`, that's `200 × 16384 × 4096 ≈ 13 billion` multiply-adds — for *one* of the 640 matmuls. Multiply out across the whole model and you land in the **trillions of operations per generated word**, exactly the figure the grounding article quotes.

[[note: confusion || The number-one confusion: "if it's just multiplication, why is it slow / why does it need a supercomputer?" The fix is to separate *simple* from *few*. Each operation is trivially simple — a multiply and an add, the thing a calculator does. But there are *trillions* of them per word. Say it plainly: "The operation is easy. There are just an insane number of them." A GPU isn't for hard math; it's for *a mountain of easy math.* That reframe is the bridge to the CPU-vs-GPU chapter.]]

## Why the kernels matter — the whole point

Here's where you close the loop and tell them why they're here. Every one of those 640-plus matmuls is executed on the GPU by a small program called a **kernel**. The model just says *which* matmuls to do and in what order. The kernel decides *how fast* each one runs. And two kernels computing the *exact same* matmul can differ by 50× or more in speed depending purely on how they move data around the chip.

[[note: production || This is live and it's where the money is. Right now, in data centres full of NVIDIA H100 and B200 GPUs, essentially all the electricity spent on AI is spent inside these matmul kernels. An H100 can do about **989 TFLOP/s** of BF16 math — but a naive matmul kernel reaches only about **1.3% of cuBLAS**, NVIDIA's hand-tuned library. That means you can rent a top-tier chip and run it at a few percent of what you paid for. The GEMM ladder in this workshop climbs from that 1.3% up to **93.7% of cuBLAS** — a 70× win — one measured step at a time. When your students make matmul faster, they change whether a model costs a dollar or a penny to run.]]

[[note: teach || Board plan for this whole chapter, in order: (1) write the opening claim and let it sit; (2) draw the factory belt; (3) turn a 3-word sentence into a 3×4 grid; (4) walk the attention half and tally 6 matmuls on the side of the board; (5) walk the MLP and tally 2; (6) circle the running total "8", then write "× 80 blocks = 640", then "× 200 words ≈ 128,000". (7) Only now reveal that each matmul is a kernel, and that a naive kernel wastes 98% of the chip. Keep the tally visible the whole time — the growing number is the drama.]]

[[note: demo || The one live demo: open a Python shell, load any small model (or just make random matrices of the model's real shapes), and run `torch.matmul` on the MLP-sized matrices in a timing loop. Show the raw FLOP/s. Then show that the *same* multiply written as a naive triple-loop in pure Python would take longer than the class. The gap between "PyTorch's tuned kernel" and "the obvious loop" — on the identical math — is the entire reason kernel engineers have jobs.]]

[[sn: A subtlety worth a mentor knowing but not necessarily leading with: during *generation*, most passes process only one new token at a time, so those matmuls become skinny matrix-times-vector operations that are memory-bound, not compute-bound. That's why serving is often bandwidth-limited while training is compute-limited — same matmuls, different shapes. It sets up the "three regimes" chapter beautifully.]]

## The frame to leave them with

Send them out with this: *a language model is a very deep stack of the same handful of matrix multiplies, run once for every word it says.* The intelligence is in the *weights* — the numbers inside those grids, learned from the internet. But the *doing* — the actual work the machine performs, the thing that costs money and time and electricity — is matrix multiplication, hundreds of times per word, trillions of operations deep. Everything else in this workshop is in service of making that one operation fly.

## You can now teach

- The **opening claim** — "a model isn't thinking, it's multiplying matrices" — and how to prove it isn't hyperbole.
- How a **word becomes a grid of numbers** (tokens → embeddings → a tokens×width matrix), the raw material every matmul feeds on.
- The **attention half** of a block as exactly **six matmuls plus a softmax** (Q, K, V, QKᵀ, ·V, ·W_O), demystifying "the attention mechanism."
- The **MLP half** as **two matmuls** bracketing an activation — expand, judge, compress — and why the weights number in the billions.
- The **count that lands the point**: 8 matmuls × 80 blocks = 640 per pass, ×200 words ≈ 128,000 matmuls for one reply, trillions of operations deep.
- The **production hook**: every matmul is a kernel, a naive kernel wastes 98% of an H100, and the whole workshop is about closing that 70× gap.
