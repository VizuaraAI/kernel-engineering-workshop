By the end of this chapter you can stand at a whiteboard and teach *vectorized loads* — why asking the GPU for four numbers at once instead of one makes a kernel roughly ten points faster — and then deliver the single most satisfying moment in the whole course: showing students the machine code where eight load instructions collapse into two.

This is a short idea with a big payoff. The math does not change. Not a single byte moves differently. We just change the *shape* of how we ask for the data, and the kernel speeds up. That surprises students, and surprise is the best glue there is. Your job is to set it up so the surprise lands.

## Where we are on the ladder

Remind the room, in one breath, how far they've come. We've been climbing a ladder of matmul kernels, each faster than the last. Kernel 5 — where every thread computes an 8×8 patch of the answer — already hit **68.7% of cuBLAS** (cuBLAS is NVIDIA's own hand-tuned library, our "100% is the pro" yardstick). That is a genuinely good kernel. The calculators are busy. Shared memory is doing honest work.

When a kernel gets *that* good, there's no single dumb mistake left to fix. The profiler stops shouting and starts whispering. And the whisper it makes here is about one small, almost embarrassing thing: we are loading our numbers **one at a time**, and the hardware would much rather hand them over **four at a time**.

[[note: say || "We're not going to change the math today. Not one multiply, not one add. We're going to change how we *ask* for the numbers — and the kernel gets ten percent faster for free. That should annoy you a little. Good. Let's find out why."]]

## The plain idea: carry four boxes in one trip

Every time the GPU fetches a number from memory, there's a fixed cost to *set up the trip* — figure out the address, book an instruction slot, send the request through the load unit. That setup cost is paid whether you fetch a little or a lot.

A single `float` is 32 bits. But the load unit can carry **128 bits — four floats — in one instruction, for the same setup cost.** So if you need four numbers that happen to sit next to each other in memory, you have a choice: make four separate trips, or make one trip that grabs all four. Same numbers arrive either way. One way pays the setup cost four times; the other pays it once.

[[note: metaphor || You're carrying groceries from the car to the kitchen. Each trip down the driveway costs the same walk whether your arms are empty or full. Carrying one box per trip means four walks for four boxes. Carrying four boxes in one trip means one walk. The boxes weigh the same total either way — you just stop wasting walks. A `float4` load is "grab four boxes, walk once."]]

[[fig: A warm hand-drawn illustration titled "Carry four boxes in one trip". Left half labeled in red "SCALAR — one box per trip": a little stick figure walking a long dashed driveway from a car to a house FOUR times, each time carrying one small labeled box (a single float), sweat marks and a tired face, blue handwritten note "4 walks, 4 setups — wasteful". Right half labeled in orange "VECTOR — four boxes, one trip": the same stick figure walking the driveway ONCE carrying a tall stack of four boxes labeled x,y,z,w, a happy face, green handwritten note "1 walk, same groceries". A dashed divider down the middle. Dashed takeaway box at the bottom spanning both halves: "same boxes arrive — you just stop wasting trips". Excalidraw style, white background, charming and friendly, handwritten labels. || The whole idea in one picture: four floats carried in one trip cost the same as one float, so stop making four trips.]]

[[note: teach || Draw the driveway and *act it out*. Walk the four sad little trips across the front of the room, one box each, huffing. Then do one big confident trip with an armful. The physical comedy sells it before a single number appears. Only *after* the picture lands do you say the real word: "the fancy name for the four-box trip is a **vectorized load**, or a `float4`."]]

## The tiny by-hand number

Put concrete numbers on the board. Say each thread needs eight of A's values for one step of its work.

- **The scalar way:** eight separate load instructions. `load, load, load, load, load, load, load, load`. Eight setups.
- **The vector way:** each `float4` grabs four. Eight values ÷ four-per-load = **two load instructions.** Two setups.

Eight becomes two. Write it that big on the board — **8 → 2** — and circle it. That fraction, one quarter of the instructions, *is* the entire chapter. Everything else is explaining why it's allowed and proving it really happened.

[[fig: A hand-drawn technical figure titled "8 loads -> 2 loads". Left: a red-labeled column of eight small stacked boxes, each holding one float and each with its own thin blue arrow labeled "load" pointing up to a single "load/store unit" box, red note "8 instructions, 8 setups". Right: two wide boxes, each holding four floats in a row (labeled x,y,z,w in purple), each with one fat blue arrow labeled "float4 load (128-bit)" to the same load/store unit box, orange note "2 instructions, same 8 floats". A big orange "8 -> 2" between the two sides. Dashed takeaway box: "one quarter of the load instructions — this IS the chapter". Excalidraw style, white background, hand-lettered labels. || The by-hand count as a diagram: eight scalar loads versus two 128-bit vector loads, same eight floats.]]

[[note: example || Do it live with small numbers. "Thread needs A's eight values: positions 0 through 7. Scalar: eight loads. Now `float4`: first load grabs 0,1,2,3. Second load grabs 4,5,6,7. Done — two loads. Eight into two. Same eight numbers land in the same registers. We just stopped asking one at a time."]]

## The catch: the four boxes must be neighbours

Here's the honest complication, and it's the interesting part of the lesson. You can only grab four in one trip if the four sit **right next to each other** in memory. The hardware can carry a stack of four *adjacent* boxes, but it cannot run around the warehouse gathering four boxes off four different shelves in one trip. There's no "scatter-grab."

Now look at what our kernel actually needs. In the inner loop, each thread wants a little **row** slice of matrix B and a little **column** slice of matrix A.

- **B's slice is a row.** In memory, a row's numbers sit next to each other. Neighbours. So vectorizing B is free — a row is already four-boxes-in-a-stack.
- **A's slice is a column.** And here's the problem: in memory, a column's numbers are *not* neighbours. They're spread far apart — the next value in a column lives a whole row-width away. Four values from a column live 32 boxes apart, on four different shelves. You cannot grab them in one trip. A `float4` over a column is simply illegal.

[[note: confusion || This is exactly where students get lost: "wait, why can't I `float4` the A side too?" The fix is one sentence with your hands. Sweep a hand *sideways* — "a row's numbers are neighbours in memory." Then sweep a hand *down* — "a column's numbers are far apart, one row-width between each." Vector loads need neighbours. Rows are neighbours; columns are strangers. That's the whole obstacle.]]

[[fig: A hand-drawn technical figure titled "Rows are neighbours, columns are strangers". A single blue diagonal-hatched matrix drawn as a grid with visible gridlines, labeled in red "A tile in memory (row-major)". One full ROW highlighted with a green outline and a green note "row: 4 values sit side by side -> float4 OK ✓", with a short fat blue arrow labeled "one trip grabs all 4". One full COLUMN highlighted with a red outline and a red note "column: 4 values 32 apart -> float4 illegal ✗", with four separate thin scattered arrows reaching to distant shelves labeled "4 trips, strided". Below the grid a purple strip showing how the grid is really stored as one long line of boxes, with the row's four boxes bracketed together ("adjacent") and the column's four boxes marked far apart along the line ("strided by row-width"). Dashed takeaway box: "vector loads need adjacent boxes — rows are, columns aren't". Excalidraw style, white background, handwritten labels. || Why B vectorizes for free but A does not: in memory a row's values are adjacent, a column's are spread a full row-width apart.]]

## The fix: rearrange A once so its columns become rows

We don't want to give up on vectorizing A. So we use a classic trick: **transpose A as we load it into fast memory.**

Here's the plain-words version. Before the inner loop runs its thousands of steps, we first copy a tile of A from slow memory into fast shared memory. That copy is a *one-time* setup, done once per tile. So while we're copying, we deliberately *lay the numbers down sideways* — we write A's columns as if they were rows. Now, in the hot inner loop that runs thousands of times, the thing that used to be a strided column is a nice contiguous row. Neighbours again. Vectorizable.

[[note: metaphor || Unpacking the moving truck. The truck (slow memory) is loaded row by row, but the recipe you'll cook from a thousand times reads *down the columns*. So as you unload, you don't shelve things in the same order — you re-shelve them column-wise into the pantry (fast memory). You do that annoying re-sorting **once**, while unpacking. Then every one of your thousand cooking steps just grabs a neat adjacent handful off the shelf. Move the awkward work out of the loop you repeat, into the setup you do once.]]

[[note: aha || Say the principle out loud, because it's bigger than this one kernel: **"Pay for the awkward rearrangement once, in the setup, so the loop you run ten thousand times gets to be simple and fast."** That trade — pain once vs. pain ten thousand times — is a move students will reach for again and again. This is the moment to name it.]]

[[fig: A three-panel hand-drawn walkthrough titled "Transpose A on load: columns become rows". Panel (1), numbered circle 1: a blue-hatched matrix labeled "A in slow memory (row-major)" with one row highlighted and a blue dashed arrow labeled "grab 4 neighbours in one trip (float4)" pointing right; red note "read a row — that part's easy". Panel (2), numbered circle 2: those four values being scattered DOWNWARD into a single column of a second blue-hatched box labeled "A in fast memory (laid sideways)", four small purple arrows labeled "place them down a column", orange note "we do this awkward re-sorting HERE — once per tile". Panel (3), numbered circle 3: the inner loop reading a full contiguous column-now-row out of fast memory with one wide blue arrow labeled "float4 — one trip", green note "now a column of A is a neat row -> vectorizable, thousands of times". Dashed takeaway box: "awkward rearrange once in setup -> simple fast loads forever after". Excalidraw style, white background, handwritten labels. || Transposing A while loading it moves the strided access out of the hot loop and into the one-time setup, so the inner loop gets clean vector loads.]]

[[sn: The transpose itself is the one place we *give up* vectorization on purpose. Reading A from slow memory is still one fat `float4` (four neighbours in a row), but *placing* those four into four different columns of fast memory is four separate small stores. We pay four scalar stores once per tile to buy back two-instead-of-eight loads thousands of times. Easy trade.]]

## The real payoff: reading the machine's own words

Now the centrepiece. This is where the chapter earns its title, and where you must slow all the way down.

Here is the thing students don't know yet: **the code you write is not the code that runs.** A CUDA kernel goes through translation stages, and the final honest version — the real instructions the chip runs — is called **SASS** (the machine's native assembly). There's an in-between version called PTX, but PTX is a polite fiction about widths that the final compiler decides later. So to *prove* your `float4` really became a four-box trip, you don't trust the source or PTX. You look at the SASS.

[[note: production || This "read the machine code to check what really happened" habit is not academic — it's the daily craft of the engineers tuning kernels for vLLM, FlashAttention, and DeepSeek's stack on H100 and B200 GPUs. When someone says "I vectorized that load," the follow-up in a serious shop is always "did you *check the SASS*?" Because sometimes the compiler silently ignores you and scalarizes it anyway, and the only place that shows up is the machine code. You are teaching students a real professional reflex.]]

So we compile kernel 5 and kernel 6, and we put their inner loops side by side. In kernel 5, loading A's eight values looks like **eight instructions** — eight separate `LDS` (load-from-shared) lines:

```
LDS R16, [R8]
LDS R17, [R8+0x4]
LDS R18, [R8+0x8]
LDS R19, [R8+0xc]
LDS R20, [R8+0x10]
LDS R21, [R8+0x14]
LDS R22, [R8+0x18]
LDS R23, [R8+0x1c]
```

In kernel 6, the exact same eight floats arrive in **two instructions** — two `LDS.128` (the `.128` means "128 bits wide," i.e. four floats at once):

```
LDS.128 R16, [R8]
LDS.128 R20, [R8+0x10]
```

[[note: demo || This is *the* demo of the chapter — run it live if you possibly can. Have both SASS listings ready on the same screen. Reveal kernel 5's eight lines first, let them sit. Then reveal kernel 6's two lines right beside them. Draw a big bracket collapsing eight into two and write "8 → 2" between them. Then be quiet for a beat and let the room see it. This picture — eight instructions becoming two, in the machine's own handwriting — is the most satisfying moment in the whole course. Don't rush it. Don't talk over it.]]

[[fig: A hand-drawn "SASS side by side" figure titled "Eight loads become two". LEFT card labeled in red "Kernel 5": a handwritten assembly listing on faint ruled lines showing eight lines "LDS R16 ... R23", each with a tiny blue tick, red bracket around all eight labeled "8 separate loads, 8 setups". RIGHT card labeled in orange "Kernel 6": just two handwritten lines "LDS.128 R16" and "LDS.128 R20", orange bracket labeled "2 loads, same 8 floats". A big hand-drawn orange arrow sweeps from the eight lines to the two, with a bold "8 → 2" written across it. Far right, a small memory diagram: a green-hatched strip labeled "A row in fast memory (adjacent)" with one 16-byte window bracketed and labeled green "128 bits = 4 floats", a dashed arrow into a yellow register box labeled "R16 R17 R18 R19". Dashed takeaway box: "a quarter of the load instructions in the hottest loop on the chip". Excalidraw style, white background, handwritten labels, monospace-ish ink for the code. || The signature moment rendered in machine code: eight scalar loads collapse into two 128-bit loads. Same bytes, a quarter of the instructions.]]

## The number

Run the benchmark. Kernel 6 reaches **78.4% of cuBLAS**, up from kernel 5's **68.7%** — roughly a **ten-point jump** from a change that moved not one byte differently and did not touch a single multiply.

[[sn: The exact percentage wobbles a little with matrix size and driver version, but the direction and rough size are rock-solid: vectorizing loads on top of a good tiled kernel is reliably worth about ten points of cuBLAS on modern NVIDIA hardware. The shorter instruction stream sometimes buys even more, because it also eases pressure on the instruction cache.]]

Be honest about *why* it worked. Two real things happened. First, we cut the load instructions the scheduler must issue in the hot loop, so more of each cycle goes to math instead of load bookkeeping. Second, one wide memory request is cleaner for the hardware than four narrow ones. The transpose made both legal for A. No magic — just fewer, fatter trips.

[[note: confusion || A sharp student will object: "you moved the same bytes — how is fewer instructions faster if the data is the bottleneck?" This is the right question and here's the fix. At *this* point on the ladder the kernel isn't starved for data anymore — the calculators are nearly full. What's now in short supply is *instruction slots*: every cycle spent issuing a load setup is a cycle not spent doing math. Cutting eight setups to two hands those cycles back to the math. Earlier kernels were data-starved; this one is instruction-starved. Different bottleneck, different fix.]]

## The bridge to next time

Leave them with a cliffhanger. Our tidy transpose, the thing that saved us, quietly rearranged *which* threads touch *which* parts of fast memory. And fast memory is split into lanes (called "banks"). If two threads now reach for the same lane at the same time, they have to wait in line — a "bank conflict." So the next kernel isn't another loading trick; it's going back to inspect the traffic pattern our own clever transpose just created. That honesty — every fix creates the next problem — is the rhythm of the whole workshop.

## You can now teach

- **Vectorized loads** as "carry four boxes in one trip": the setup cost is paid per-trip, so grabbing four adjacent floats in one `float4` beats four separate loads.
- The **8 → 2** by-hand count: eight scalar loads collapse into two 128-bit loads, a quarter of the instructions for the same bytes.
- The **catch and the fix**: vector loads need adjacent values, so rows vectorize free but columns don't — until you **transpose A on load**, paying the awkward rearrangement once so the hot loop stays simple.
- The **read-the-SASS reveal**: eight `LDS` lines becoming two `LDS.128` lines in the machine's own code — the most satisfying demo in the course — and why you trust SASS over your source.
- The **ten-point number** (68.7% → 78.4% of cuBLAS) and the *honest* reason it works: fewer instruction slots wasted on load bookkeeping once the kernel is instruction-bound, not data-bound.
- The **principle worth naming**: pay for awkward work once in setup so the loop you run ten thousand times gets to be fast.
