By the end of this chapter you will be able to stand at a whiteboard and teach matrix multiplication so clearly that a student who has never seen it will not only compute it, but *feel* why it is the single most important operation in all of modern AI. We start from nothing. No GPUs yet. Just numbers, a pencil, and a story.

Everything in this workshop — every kernel we optimize, every trick with memory and threads — exists to make *this one operation* fast. So you have to own it completely. Let's build it up the way you'll build it up for students: slowly, with pictures, until it feels obvious.

## Start even smaller than a matrix: the dot product

Before matrices, there is the **dot product**. Take two lists of numbers of the same length. Multiply them position by position, then add up all the products. That single number is the dot product.

[[note: metaphor || Think of a shopping receipt. One list is *how many* of each item you bought: `[2, 1, 3]`. The other is the *price* of each item: `[5, 10, 2]`. The dot product — 2×5 + 1×10 + 3×2 = **26** — is your total bill. A dot product is just "multiply the matching pairs and total it up." Students already do this every time they check a receipt.]]

[[note: example || On the board, do it by hand, slowly: `[2, 1, 3] · [5, 10, 2]` = (2·5) + (1·10) + (3·2) = 10 + 10 + 6 = **26**. Write each product under its pair, then sum. Three multiplies, two adds. That is the whole operation.]]

[[fig: A warm hand-drawn illustration of a shopping receipt metaphor for the dot product. On the left, two horizontal lists of three rounded boxes each: a blue-hatched row labeled "quantities [2, 1, 3]" and a green-hatched row labeled "prices [5, 10, 2]". Thin blue dashed arrows pair box 1 with box 1, box 2 with box 2, box 3 with box 3, each pair annotated in purple with its product "2x5=10", "1x10=10", "3x2=6". On the right, a hand-drawn receipt with these three numbers added in a column and a big orange total "= 26". A dashed takeaway box at the bottom reads "dot product = multiply matching pairs, then add them all up". Excalidraw style, white background, handwritten labels. || The dot product, taught as a shopping receipt: pair, multiply, total.]]

Hold onto that word — **dot product** — because a matrix multiplication is nothing more than a big, organized grid of dot products. That is the whole secret. Once students believe that, everything else is bookkeeping.

## A matrix is just a table of numbers

A **matrix** is a rectangle of numbers arranged in rows and columns. That's it. A 2-by-3 matrix has 2 rows and 3 columns. We say its **shape** is 2×3.

[[note: teach || Draw a matrix as a literal grid with gridlines, like a spreadsheet or a chessboard — never as bare numbers floating in brackets. Students who see the grid can point at "row 1" and "column 2" with a finger. Always say the shape out loud as "rows by columns" and write it under the grid. The row-then-column order is a convention they must never mix up, so drill it early.]]

In an AI model, these grids are everywhere. The numbers coming into a layer are a matrix (one row per word, say). The **weights** the model learned are a matrix. To push the data through the layer, you multiply them. So "running a neural network" is, almost entirely, "multiplying matrices." We'll make that concrete in the next chapter; for now, just plant the flag: matrices are the nouns, and multiplication is the verb.

## Multiplying two matrices: the grid of dot products

Here is the rule, and here is how to teach it without anyone getting lost.

To multiply matrix `A` by matrix `B` and get matrix `C`, **every cell of the answer `C` is one dot product**: the dot product of a *row of A* with a *column of B*. The cell in row `i`, column `j` of `C` is the dot product of row `i` of `A` with column `j` of `B`.

[[note: say || "To fill in *this* box of the answer" — point at C[1][2] — "I take *this* row of A, and *this* column of B, and I do our receipt trick: pair them up, multiply, and add. That number goes in the box. Then I move to the next box and do it again. That's the whole thing — one receipt per box."]]

[[fig: A hand-drawn diagram titled "Every cell of C is one dot product". Three grids drawn as spreadsheets with visible gridlines: matrix A (2x3, blue diagonal hatch) on the left, matrix B (3x2, green diagonal hatch) on the top right, and matrix C (2x2, pale-yellow hatch) on the bottom right, positioned so A is left of C and B is above C (the classic matmul layout). One specific cell of C, row 1 column 2, is highlighted with a bold orange outline and labeled in red "C[1][2]". A blue dashed arrow sweeps across the entire row 1 of A (highlighted) and another blue dashed arrow sweeps down the entire column 2 of B (highlighted), both converging on the highlighted C cell. A purple handwritten note near the cell reads "dot product of row 1 of A and col 2 of B". A dashed takeaway box reads "row of A  .  column of B  ->  one cell of C". Excalidraw style, white background. || The matmul layout students should memorize: a row of A meets a column of B to make one cell of C.]]

Now do a full **2×2 times 2×2** on the board. Small enough to finish, big enough to show the pattern.

```
A = [ 1  2 ]      B = [ 5  6 ]
    [ 3  4 ]          [ 7  8 ]
```

Fill C one cell at a time, saying the receipt each time:

- `C[1][1]` = row 1 of A · column 1 of B = (1·5) + (2·7) = 5 + 14 = **19**
- `C[1][2]` = row 1 of A · column 2 of B = (1·6) + (2·8) = 6 + 16 = **22**
- `C[2][1]` = row 2 of A · column 1 of B = (3·5) + (4·7) = 15 + 28 = **43**
- `C[2][2]` = row 2 of A · column 2 of B = (3·6) + (4·8) = 18 + 32 = **50**

```
C = [ 19  22 ]
    [ 43  50 ]
```

[[note: confusion || The number-one place students get lost: they try to pair a *row of A* with a *row of B*. Fix it with a physical gesture. Sweep your hand left-to-right along A (a row), then top-to-bottom down B (a column). "A goes sideways, B goes down." Make them do the hand motion with you. Muscle memory beats the rule.]]

## The shape rule, and why it exists

There is exactly one rule about when you're *allowed* to multiply: the number of **columns in A** must equal the number of **rows in B**. A `(m × k)` times a `(k × n)` gives an `(m × n)` result. The two inner numbers — the `k` — must match and then vanish; the two outer numbers survive as the answer's shape.

[[note: metaphor || The `k` is a *handshake*. A's rows reach out with `k` fingers; B's columns must reach back with exactly `k` fingers or the hands don't meet. When they match, the handshake happens and collapses to a single number (the dot product). The two `k`s disappear into that handshake; only the outer dimensions `m` and `n` are left standing as the shape of C.]]

[[fig: A hand-drawn "shape handshake" figure. Two labeled shape tags written large in red: "(m x k)" for A and "(k x n)" for B, placed side by side with the two inner k's directly adjacent and circled together in orange, connected by a little hand-drawn handshake icon and a green note "these must match -> then vanish". The two outer letters m and n are boxed in blue and an arrow leads down to a result tag "(m x n)" in red labeled "the shape that survives". Below, a small worked example in purple: "(2x3) . (3x2) -> (2x2)  ✓" and a crossed-out red "(2x3) . (2x2)  ✗ inner 3 != 2". Dashed takeaway box: "inner dims must match and cancel; outer dims are the answer". Excalidraw style, white background, handwritten. || The shape rule as a handshake: inner dimensions must match and cancel; outer dimensions survive.]]

## Now write it as a program — three nested loops

Here is the beautiful part you'll show right after the hand example: the entire operation is **three nested loops**. Once students see the loops, they see the *cost*, and the cost is the reason this whole workshop exists.

```python
# C = A @ B,  A is (M x K),  B is (K x N),  C is (M x N)
for i in range(M):            # which row of C
    for j in range(N):        # which column of C
        total = 0
        for k in range(K):    # the dot product (the "receipt")
            total += A[i][k] * B[k][j]
        C[i][j] = total
```

The outer two loops walk over every cell of the answer. The inner loop is our receipt — the dot product. Three loops, and inside the deepest one, a single **multiply-and-add** (in hardware this fused step is called an **FMA**, a fused multiply-add).

[[note: aha || Count the work out loud and watch the room react. For square `N × N` matrices, the loops run `N × N × N = N³` times. Multiply two 1000×1000 matrices and that is **one billion** multiply-adds — for what looks like a tiny operation. Now say the punchline: "A real model does matrices far bigger than this, millions of times, for every word it generates. *That* is why we spend four weeks making this one operation fast." This number is the emotional hook of the entire course.]]

[[fig: A hand-drawn figure titled "Matmul is three nested loops". On the left, three concentric rounded rectangles nested inside each other like Russian dolls, labeled from outside in: outer loop "i: rows of C" (red), middle loop "j: cols of C" (red), inner loop "k: the dot product" (blue), with the very center holding a small purple box "total += A[i][k] * B[k][j]  (one FMA)". On the right, the C matrix drawn as a grid with an orange arrow snaking through its cells in reading order (row by row) labeled "outer 2 loops visit every cell", and a single highlighted cell with a small blue coil arrow labeled "inner loop fills it". Bottom: an orange callout "N x N x N = N^3 FMAs  ->  1000^3 = 1 billion". Dashed takeaway box: "the whole operation: 3 loops, one multiply-add at the core". Excalidraw style, white background. || The whole operation in one picture: three loops, a billion multiply-adds, and the cost that motivates the course.]]

## Why this is the whole game (the production link)

Plant this now, even before the next chapter drives it home. When you send a message to ChatGPT, DeepSeek, or a Llama model, the machine turns your words into numbers, then pushes them through hundreds of layers. **Almost every one of those layers is a matrix multiply** — your data-matrix times a learned-weight-matrix. Generating a single word can take *trillions* of these multiply-adds; a full conversation, far more.

[[note: production || This is not a textbook exercise. Right now, in data centres full of NVIDIA H100 and B200 GPUs, the overwhelming majority of the electricity and money spent on AI is spent doing exactly the operation on this whiteboard. NVIDIA became one of the most valuable companies on Earth essentially by building the best chips for this loop. When your students make matmul 70× faster over the next four weeks, they are touching the exact thing that decides whether a model costs a dollar or a penny to run.]]

That's the frame to leave them with at the end of this first piece: *matrix multiplication is small enough to do by hand on a receipt, and important enough that the entire AI economy is built on doing it quickly.* Everything else we teach is in service of that second half.

## You can now teach

- The **dot product** as a shopping receipt — pair, multiply, total — and why it's the atom of everything that follows.
- A **matrix** as a labeled grid, and the strict "rows by columns" convention.
- **Matrix multiply** as a grid of dot products (row of A meets column of B), demonstrated with a full 2×2 by hand.
- The **shape rule** as a handshake: inner dimensions match and cancel, outer dimensions survive.
- Matmul as **three nested loops** with an FMA at the core, and the `N³` cost that makes 1000×1000 a *billion* operations.
- The **production hook**: this exact loop is where the AI economy spends its money — which is why the whole workshop exists.
