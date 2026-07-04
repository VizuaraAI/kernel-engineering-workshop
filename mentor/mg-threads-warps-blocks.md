By the end of this chapter you can stand at a whiteboard and teach the whole GPU execution hierarchy — thread, warp, block, grid — as one clear parade, so that a student who has never written a line of CUDA can tell you exactly who runs where, why the number 32 keeps showing up, and why a single `if` can cut a kernel's speed in half.

This is the map underneath everything else in the workshop. Every optimization we teach later is really a sentence about one of these four levels. So let's make the levels feel obvious first, using people, not silicon.

## The one-sentence answer

When you launch a GPU program, you don't launch *one* worker. You launch thousands of identical workers at once, and every single one runs the *exact same instructions* — just on a different slice of the data. The whole art is that the workers are organized into a strict little hierarchy, and each rung of that hierarchy lands on a specific piece of the physical chip.

[[note: metaphor || Picture a giant **marching band** on a football field. One **musician** is a single thread — one person, playing their own note, holding their own sheet music. The band moves in **rows of 32 people who step in perfect lockstep** — that row is a *warp*; when the drum major calls "left foot," all 32 left feet go down in the same instant, no exceptions. Several rows together form a **block** — a squad that shares one practice room and can shout to each other to stay in sync. And the **whole band on the field** is the *grid* — the entire performance. That's the entire chapter: musician, row-of-32, squad, whole band.]]

[[fig: A warm hand-drawn illustration titled "The GPU is a marching band". A football field seen from above. The whole field is boxed and labeled in red "GRID = the whole band (the entire launch)". On the field are several rectangular squads, each boxed and labeled "BLOCK = a squad (shares a practice room)"; one squad is highlighted pale-yellow. Inside the highlighted squad are horizontal rows of little stick-figure musicians; one row of exactly 32 figures is bracketed and labeled in green "WARP = 32 musicians in lockstep". One single figure in that row is circled orange and labeled in red "THREAD = one musician, one note". A drum major stands to the side with a speech bubble "LEFT foot!" and a blue note "one command -> all 32 step together". Dashed takeaway box: "musician -> row of 32 -> squad -> whole band". Excalidraw style, white background, charming, hand-lettered. || The core metaphor: one thread is a musician, a warp is a row of 32 marching in lockstep, a block is a squad, the grid is the whole band.]]

[[note: teach || Draw the band from the outside in, and *say each level out loud as you draw it*. Start with the big field box (grid), drop in a couple of squad rectangles (blocks), draw one row of little heads inside one squad (a warp), then circle a single head (a thread). Four boxes, nested. Do NOT introduce the words "SM" or "scheduler" yet — right now they only need people. The metal comes on the second pass.]]

## Three names you type, one the hardware forces on you

Here is the twist that makes GPUs confusing at first, and it's worth naming plainly. Of the four levels, **you personally choose three of them** when you launch the program: how big the grid is, how big each block is, and therefore how many threads exist. But the **warp — the row of 32 — you never asked for.** The hardware slices every block into rows of exactly 32 whether you like it or not.

[[note: say || "You get to design the band. You decide how many squads, and how many people per squad. But you do NOT get to decide the rows. The hardware walks in and says: I will chop every squad of yours into rows of 32, always 32, and those 32 people will step together forever. That row is called a *warp*, and it is the real unit of everything that happens on this chip."]]

That number 32 is not a rough guideline. It is baked into the scheduler, into how registers are laid out, into how memory is read. Almost every performance rule in this whole workshop is downstream of that one constant. So drill it: **a warp is exactly 32 threads, always.**

[[fig: A hand-drawn "nesting dolls" technical diagram titled "The execution hierarchy". Four concentric rounded rectangles, outside in. Outer black box labeled GRID with red note "the whole launch — every thread". Inside it a 3x3 arrangement of medium boxes each labeled "BLOCK", one highlighted pale-yellow with an orange callout "you choose this size". Inside the highlighted block, four thin horizontal strips each labeled "WARP" with a green note "= 32 threads, hardware forces this". Inside one warp strip, 32 tiny numbered squares 0..31, one hatched blue and labeled red "THREAD = private registers". A vertical bracket down the left labeled purple "you TYPE grid / block / thread" spanning the outer three, and a separate red bracket on the warp strip "the HARDWARE imposes the warp". Dashed takeaway box: "3 levels are yours to size; the warp of 32 is not". Excalidraw style, white background, hand-lettered. || The four levels: you size grid, block and thread — the hardware silently chops every block into warps of 32.]]

## A tiny number: a block of 100 wastes a whole row

Now put a small number on the board so the "always 32" rule bites. Suppose a student asks for a block of **100 threads**. How many warps is that?

The hardware can only make rows of 32. So 100 threads becomes **four rows**: 32 + 32 + 32 = 96, and then a fourth row holding the last **4 real musicians padded out with 28 empty uniforms.** That fourth row still marches. It still takes up a full row's worth of space and attention. But 28 of its 32 members are doing nothing.

[[note: example || On the board: 100 threads / 32 = 3 full warps (96 threads) + 1 ragged warp (4 real, 28 wasted). You paid for 128 threads' worth of hardware and used 100. That last warp does 4/32 = **one-eighth** useful work. Now write the fix in green: pick block sizes that are *multiples of 32* — 128, 256, 512 — and every row is full. This is the whole reason "128 or 256" is the default advice.]]

[[fig: A hand-drawn illustration titled "A block of 100 wastes a warp". A squad drawn as four horizontal rows of little musician figures. Rows 1, 2, 3 are full — 32 solid figures each, labeled green "full warp". Row 4 has only 4 solid figures on the left and 28 faded/dashed ghost figures in empty uniforms on the right, bracketed red "28 wasted — but still marches". A big orange callout: "100 threads -> 4 warps, last one 1/8 full". To the right, a clean squad of exactly 128 in four full rows with a green check "128 = 4 x 32, no waste". Dashed takeaway box: "block size = multiple of 32, or you pay for empty uniforms". Excalidraw style, white background, hand-lettered. || A block of 100 becomes four warps — the last with 4 real threads and 28 wasted lanes. Multiples of 32 pack cleanly.]]

## Where each level actually lives on the chip

Now do the second pass: land each level of the band on the real hardware. This is the mapping to memorize, because every later optimization is a statement about one of these arrows. Use an NVIDIA H100 as the concrete machine.

A **thread** maps to a **lane** — one slot in the machine's datapath, with its own tiny stash of private registers (the H100 gives each processor a `256 KB` register file, at most `255` registers per thread). One musician, one music stand.

A **warp** maps to a **warp scheduler**, and this is the beating heart of why GPUs are fast. Each H100 SM has **four** warp schedulers. Every cycle, a scheduler looks at all the rows of 32 it's holding, picks one that is *ready* (not stuck waiting), and issues its next instruction. Here's the magic: when row A is stuck waiting `~500` cycles for data to arrive from far-away memory, the scheduler doesn't sit idle — it issues row B, then C, then D. The waiting never disappears; it gets *hidden* behind other rows' work.

[[note: aha || This is the number that makes the room lean in. A far-away memory read on a GPU costs roughly **500 cycles** — an eternity. A CPU would stall and twiddle its thumbs. The GPU scheduler instead keeps *dozens* of warps resident and just marches a different row every cycle while the first one waits. So the latency is real but *invisible*. Say it out loud: "The GPU is never fast because it waits less. It's fast because it always has another row ready to march while the last one waits."]]

A **block** maps to a **Streaming Multiprocessor** (an **SM**) — one of the H100's roughly **132** SMs. The squad gets one practice room and stays in it for its entire life; it never moves to another room. That room's scratchpad (**shared memory**, up to `228 KiB`) and its registers are what the squad shares. An SM can hold *several* squads at once if their combined needs fit in the room.

A **grid** maps to the **whole GPU**. A hardware traffic-cop hands squads out to SMs as rooms free up. If there are far more squads than rooms — and there usually are — they drain through in waves.

[[fig: A hand-drawn architecture map titled "Band -> Chip", two columns joined by dashed arrows. LEFT (the band, stacked boxes): GRID / BLOCK / WARP / THREAD. RIGHT (hardware): a big rounded rectangle labeled "H100 die" holding 8 small boxes labeled "GPC"; one GPC zoomed to ~16 boxes labeled "SM" with green note "≈132 SMs total"; one SM zoomed to show four small boxes labeled "warp scheduler x4" with blue note "issues 1 ready warp/cycle -> hides the 500-cycle wait", plus a green specs strip "register file 256 KB · shared mem up to 228 KiB". One tiny slot in the SM labeled red "thread = lane". Dashed arrows: GRID->die, BLOCK->SM (orange note "squad lives in ONE room, never moves"), WARP->scheduler, THREAD->lane. Dashed takeaway box: "thread->lane, warp->scheduler, block->SM, grid->GPU". Excalidraw style, white background, hand-lettered. || The mapping to memorize: thread is a lane, warp is a scheduler, block is an SM, grid is the whole GPU.]]

## "Who am I?": every musician reads their own badge

Because all the workers run the *same* code, the very first thing every kernel does is figure out *which slice of the problem it owns*. It works out its own identity from three numbers the hardware hands it: `threadIdx` (my position inside my squad), `blockIdx` (which squad I'm in), and `blockDim` (how big a squad is).

The one line every CUDA programmer writes is this:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Read it in plain English: *skip past all the squads in front of me, then add my seat number inside my own squad.* If each squad holds 256 people, then squad 0 is people 0–255, squad 1 is people 256–511, and so on. This arithmetic hands every musician a unique global number.

[[note: example || Do it by hand. Squad 1, seat 44, squads of 256. i = blockIdx.x·blockDim.x + threadIdx.x = 1·256 + 44 = **300**. So this musician owns element 300 of the array. Walk two or three seats through it slowly. "Skip one full squad of 256, then walk to seat 44." Students who can compute this by hand can debug 90% of real CUDA bugs.]]

[[note: confusion || The single most common CUDA bug lives here, and it *compiles perfectly and runs silently wrong*. Because we usually can't split the work evenly, we round UP the number of squads — which launches slightly too many musicians, a few who fall off the end of the array. You MUST add a guard: `if (i < N) { ... }`. Miss it and the extra musicians scribble past the end of your data. The fix is one line; the bug is invisible without it. Teach the guard in the same breath as the index, never separately.]]

[[fig: A hand-drawn technical figure titled "Every thread reads its own badge". A long horizontal strip of 512 cells split into two blocks of 256: Block 0 (cells 0-255) and Block 1 (cells 256-511), each boxed and labeled red "blockIdx.x = 0" and "= 1". Inside Block 1, one cell near position 300 highlighted orange with a purple handwritten equation pointing at it: "i = 1 x 256 + 44 = 300". At the far right edge, a few cells drawn as dashed ghosts past the array end, labeled red "extra threads — round up" with a blue note "guard: if (i < N)". Dashed takeaway box: "skip the squads before me, add my seat, then check I'm in bounds". Excalidraw style, white background, hand-lettered. || Each thread computes a unique global index from blockIdx, blockDim, threadIdx — and a bounds guard kills the overflow from rounding up.]]

## The one that halves your speed: when a row disagrees

Now the payoff that makes this chapter matter for performance. Remember: the 32 musicians in a warp **share one set of sheet music** — one program counter. They can only ever be on the *same line of music* at the same time. So what happens when your code has an `if`, and half the row should do one thing and half should do another?

The row cannot split into two. There's only one music stand. So the hardware does the only thing it can: it plays **both** paths, one after the other. First it plays path A with the "true" half of the row active and the "false" half frozen (marking time, discarding their work). Then it plays path B with the halves swapped.

[[note: metaphor || The row of 32 has ONE conductor and ONE score. When an `if` splits them, the conductor can't run two songs at once. So the row plays song A while 16 people mime silently, then plays song B while the *other* 16 mime. Both songs got played; only half the musicians made real sound each time. That's **warp divergence** — and if both songs are equally long, you just took **twice as long** for the same music.]]

[[note: example || Put the ceiling on the board. An even two-way split = you pay for both halves = a hard ceiling of ~**50%** of full speed, no matter how perfect everything else is. A `switch` with 8 cases that scatters a warp 8 ways can serialize into 8 passes — down toward **1/8** speed. The jaw-dropper: this is why a student's first profiled kernel sits at "40-60% warp efficiency" and they can't figure out why. It's one innocent `if`.]]

[[note: confusion || The crucial rescue: **divergence only costs you when musicians in the SAME row disagree.** If a whole row jumps the same way, it's completely free — the row just plays one song. So `if (blockIdx.x == 0)` is usually fine (whole squads agree). But `if (threadIdx.x % 2 == 0)` is a disaster — it splits *every single row* right down the middle. The fix students remember: "make the row agree." Branch on `threadIdx.x / 32` (the row number), never on something that alternates *inside* a row.]]

[[fig: A hand-drawn technical figure titled "How an if halves a warp". Panel 1: one warp as 32 numbered boxes; boxes 0-15 blue-hatched labeled red "cond TRUE", boxes 16-31 green-hatched labeled red "cond FALSE", caption "the if splits the row". Panel 2: a timeline with two stacked bars. Top bar "PASS 1: path A" shows lanes 0-15 solid, 16-31 faded/dashed, orange note "16 mime silently". Bottom bar "PASS 2: path B" the reverse. A red bracket over both: "TOTAL = A + B, run one after the other -> 2x time". Panel 3: a small green ledger "even split -> 2x slower; 8-way switch -> up to 8x". Dashed takeaway box: "divergence bites ONLY when lanes of the same warp disagree". Excalidraw style, white background, hand-lettered. || Both sides of the branch run serially with half the lanes miming — same correct answer, twice the time.]]

[[note: production || This is not academic. **SIMT** — single instruction, multiple threads — is exactly how every GPU serving Llama, DeepSeek, or ChatGPT runs *right now*. The famous **FlashAttention** kernel that every serving stack adopted is written with obsessive care to keep whole warps on the same path. When your students later push the GEMM ladder from a humiliating 1.3% of peak toward 90%, one of the very first fixes is nothing but re-assigning which musician touches which memory so that a full row of 32 reads *contiguous* addresses. On H100 and B200 clusters costing millions, the gap between a warp-aware kernel and a warp-blind one is, directly, half the electricity bill.]]

## Picking the squad size: the packing puzzle

One last practical question students always ask: *how big should a block be?* Two hard rules and one judgment call.

**Hard rule one:** a block can hold at most **1024 threads**. Ask for more and the launch simply fails. So a 2-D squad is at most 32×32.

**Hard rule two:** make it a **multiple of 32**, or you pay for empty uniforms (the block-of-100 problem above).

The judgment call is *which* multiple of 32 — 128? 256? 512? This is the **occupancy** question, and it's just a packing puzzle. Each SM (practice room) has a fixed budget: a register file, a shared-memory scratchpad, and a cap on how many rows it can hold. More resident rows means more warps for the scheduler to juggle, which means more latency it can hide. But each thread eats registers, so fat threads mean fewer fit in the room.

[[sn: More occupancy is not automatically better. Once the schedulers always have *some* ready row, extra rows buy nothing — and cramming more threads in can force the compiler to "spill" registers to far-away memory, which is slower. The best kernels on the ladder often run at only 50-60% occupancy with fat, register-hungry threads. Occupancy is a means, not the goal — don't oversell it to students as "higher is better."]]

[[fig: A hand-drawn technical figure titled "Occupancy = packing the room". Center: a tall box labeled "one SM (practice room)" with a green budget list on the left: "register file · shared mem 228 KiB · max resident warps". Inside, three stacked blocks packed in, each labeled "block = 256 threads = 8 warps", filling ~3/4 of the box; the top quarter is hatched grey labeled orange "won't fit — out of registers". A purple note: "64 regs/thread -> ~1024 threads max". A blue dashed arrow from a scheduler icon to the resident rows: "more resident rows = more latency hidden". Dashed takeaway box: "block size = multiple of 32, ≤ 1024, chosen to PACK the register budget". Excalidraw style, white background, hand-lettered. || Block size is a packing decision: registers and shared memory per thread decide how many blocks fit, which sets how much latency the scheduler can hide.]]

The default advice — **128 or 256 threads** — exists because it packs cleanly into almost any register budget while still handing each scheduler several rows to juggle. That's the safe answer to give students on day one; the tuning comes later.

## You can now teach

- The **four-level hierarchy** as a marching band: thread = one musician, warp = a row of 32 in lockstep, block = a squad sharing a room, grid = the whole band — drawn from the outside in.
- Why you **type three levels but the hardware forces the warp of 32** on you, and why block sizes should be multiples of 32 (the block-of-100 wastes an eighth of a warp).
- The **hardware mapping** to memorize — thread→lane, warp→scheduler, block→SM, grid→GPU — and why the scheduler hiding a ~500-cycle wait behind other warps is the real source of GPU speed.
- The **"who am I" index** `blockIdx.x * blockDim.x + threadIdx.x`, computed by hand, plus the bounds-guard that fixes the invisible off-the-end bug.
- **Warp divergence**: why one `if` that splits a row runs both paths serially and halves throughput, and the rescue — "make the whole row agree" — that keeps it free.
- The **occupancy packing puzzle** behind choosing a block size, and the safe default of 128 or 256 threads.
