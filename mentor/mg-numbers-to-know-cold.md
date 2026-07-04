By the end of this chapter you'll be able to stand at a whiteboard and rattle off the five numbers that describe an H100 — and, more importantly, *use* them to answer real questions on the spot: "will this be fast?", "why is this slow?", "how much of the chip am I wasting?" These five numbers turn vague hand-waving into confident arithmetic. Once a mentor owns them, they stop guessing and start estimating.

There are exactly five. Write them at the top corner of the board on day one and leave them there for the whole workshop. Everything you teach afterward is a story about one of these numbers.

```
989 TFLOP/s   — how fast it can do math
3.35 TB/s     — how fast it can move data
132 SMs       — how many worker-teams there are
228 KiB       — the on-chip scratchpad per team
32 threads    — the size of one lockstep squad (a "warp")
```

[[note: teach || Don't explain all five at once. Write them up as a list first, tell the room "by Friday every one of these will feel obvious," and then unveil them one at a time as you go. The reveal is the payoff. A number you had to *wait* for lands harder than one dumped on you.]]

## Number one: 989 TFLOP/s — the speed of the math

**TFLOP/s** means *trillion floating-point operations per second*. A "floating-point operation" is one multiply or one add on decimal numbers. So 989 TFLOP/s means the chip can do nearly **a thousand trillion** little multiply-and-adds every second. That is a 1 with fifteen zeros after it, per second.

[[note: metaphor || Picture a colossal room full of tiny calculators, and every one of them can punch in a multiply-and-add. 989 TFLOP/s is the whole room finishing a thousand-trillion sums in the time it takes you to say "one." It is not one genius doing math impossibly fast — it is an ocean of ordinary calculators all going at once. That distinction *is* the GPU.]]

Here's the catch to plant early: 989 is a special-precision number. It's the speed when the inputs are **BF16** — a 16-bit "short" number format — *and* the work goes through the chip's matrix-multiply units (the tensor cores, next chapter). Plain 32-bit math on the ordinary cores gets only about **60 TFLOP/s**. Same chip. The 989 path is roughly **16× faster** than the 60 path.

[[note: aha || Say this out loud and watch it land: "The same H100 is either a 60-TFLOP/s chip or a 989-TFLOP/s chip depending on whether you used the right units and the right number format. Most of this workshop's second half is about earning the difference between those two numbers." That gap is the whole reason tensor cores exist.]]

[[fig: A warm hand-drawn illustration titled "989 TFLOP/s = a room full of calculators". A large room drawn in friendly Excalidraw style, packed wall-to-wall with rows of tiny cartoon calculators, each with a little "×+" on its screen, all lit up at once. A big hand-lettered banner across the top reads "989,000,000,000,000 multiply-adds every second". In one corner, a single lonely calculator labeled in blue "plain 32-bit math = 60 TFLOP/s". The vast room is labeled in orange "BF16 on tensor cores = 989 TFLOP/s". A red curved arrow between them labeled "≈16× faster — same chip!". Dashed takeaway box at the bottom: "the speed depends on which units and which number-format you use". White background, hand-lettered labels. || The math ceiling drawn as a room of calculators — and the 16× gap between using the chip well and using it badly.]]

## Number two: 3.35 TB/s — the speed of the plumbing

**TB/s** means *terabytes per second* — trillions of bytes moved per second. The H100's main memory (called **HBM**, high-bandwidth memory) can pour data into the chip at **3.35 TB/s**. That's the width of the pipe feeding all those calculators.

[[note: metaphor || Number one was a room full of cooks. Number two is the hallway that brings them ingredients. 3.35 TB/s is a very wide hallway — but the cooks (989 TFLOP/s) are *even faster* than the hallway is wide. So the cooks spend most of their day tapping their feet, waiting for the next tray of rice. That mismatch — fast cooks, narrower hallway — is the single most important tension in the entire course.]]

Now put the two numbers *together* — that's where the magic is. The math chews 989 trillion operations per second; the pipe delivers 3.35 trillion bytes per second. Divide them:

```
989 TFLOP/s  ÷  3.35 TB/s  ≈  295 operations per byte
```

This number, **~295**, is called the **ridge point**. It says: for every single byte you drag in from memory, you'd better do about 295 operations on it — or the calculators starve while the pipe struggles to keep up.

[[note: example || Do this division on the board, slowly, as real arithmetic. `989 ÷ 3.35 ≈ 295`. Then say what it means in plain words: "If your code does *fewer* than 295 sums per byte it loads, the hallway is your bottleneck and the calculators sit idle. If it does *more*, the calculators are the bottleneck and the hallway keeps up. 295 is the exact tipping point of this specific chip." Students will remember a number they watched you compute.]]

[[fig: A hand-drawn "ridge point" figure titled "295 operations per byte — the tipping point", drawn as a simple balance scale. On the left pan, a stack of tiny calculators labeled in orange "MATH: 989 TFLOP/s". On the right pan, a wide pipe with water flowing, labeled in blue "PIPE: 3.35 TB/s". Above the fulcrum a big red tag reads "balance point = 989 ÷ 3.35 ≈ 295 ops/byte". A green note on the left: "do MORE than 295 ops/byte → math is the limit (good, you want this)". A blue note on the right: "do FEWER than 295 → the pipe is the limit (starving calculators)". A small purple worked line at the bottom: "reuse every byte ~300× or the room waits". Dashed takeaway box: "the whole craft = do lots of math per byte you load". White background, hand-lettered. || The two headline numbers divided into one: 295 operations per byte, the line between fast and slow.]]

[[note: production || This isn't academic. When vLLM or DeepSeek serve a model to millions of users, the engineers are fighting this exact ratio. A kernel that reuses each byte only a few times lives on the wrong side of 295 and wastes most of a machine that cost a quarter-million dollars. FlashAttention became famous precisely because it slashed how many bytes it moved — pushing attention past the ridge so the expensive tensor cores stayed busy. Every dollar of GPU efficiency in the industry is decided by which side of 295 your code lands on.]]

[[note: confusion || Students hear "3.35 terabytes per second" and think "that's enormous, memory can't possibly be the problem." Fix it with the ratio, not a speech: "Yes, the pipe is huge. The calculators are just *huger*. 3.35 is a big number and 989 is a bigger one — that's why the pipe loses." The number sounding infinite is the trap; the division is the escape.]]

## Number three: 132 SMs — the worker-teams

The chip isn't one giant brain. It's **132 near-identical little processors** bolted side by side. Each one is called a **Streaming Multiprocessor**, or **SM**. When you write GPU code, you're really describing the work for *one* team, and the machine copies that work across all 132 teams.

[[note: metaphor || Think of a huge kitchen split into 132 identical cooking stations. Each station has its own cooks, its own little countertop, its own recipe card. You write *one* recipe; the head chef stamps out 132 copies and every station cooks in parallel. When students later ask "how does one line of my code run on a whole GPU?" — this is the answer. You wrote one station's job; the hardware ran 132 of them.]]

[[note: aha || The jaw-dropper: each of those 132 SMs can keep **2048 threads** alive at once. Multiply it on the board — `132 × 2048 ≈ 270,000`. "This one chip is juggling a quarter of a *million* tiny workers at the same instant." Let that number sit. It's the concrete face of the word "parallel."]]

[[fig: A warm hand-drawn illustration titled "132 SMs = 132 cooking stations". A big kitchen floor plan seen from above, filled with a neat grid of small identical cooking stations, drawn charmingly, a "×132" hand-lettered in the corner. One station is circled in orange and enlarged in an inset, showing tiny cook figures, a little countertop, and a recipe card labeled in purple "your kernel — one recipe, copied 132×". A green note reads "each station keeps ~2048 workers busy". A red banner across the top: "132 × 2048 ≈ 270,000 workers at once". Dashed takeaway box: "you write ONE station's job — the chip runs 132 of them in parallel". White background, hand-lettered labels. || The 132 SMs as 132 identical kitchen stations, all running the one recipe you wrote.]]

## Number four: 228 KiB — the tiny scratchpad

Each of those 132 SMs has a small, blazing-fast scratchpad right next to its calculators. It's called **shared memory**, and on an H100 an SM can dedicate up to **228 KiB** of it to your program. (KiB is kibibytes — 228 KiB is about 233,000 bytes.)

Why does this tiny number matter so much? Because of number two. Main memory (3.35 TB/s) is *far* and slow. This on-chip scratchpad is *near* and roughly ten times faster. The entire art of a fast kernel is: **haul a chunk of data in from far memory once, park it in this scratchpad, reuse it many times** before throwing it away. That reuse is how you reach the good side of the 295 ridge.

[[note: metaphor || Number two was the long hallway to a distant pantry. 228 KiB is the little countertop right beside each cooking station. You send *one* runner down the long hallway, grab a big tray of ingredients, drop it on the countertop — and now every cook at the station grabs from the tray a hundred times without anyone walking the hallway again. The countertop is tiny, so you can only hold a small tray at once. That smallness is exactly why kernels are written in *tiles*.]]

[[note: example || Contrast the two speeds out loud with round numbers. Reaching into this scratchpad takes about **20–30 cycles**. Reaching all the way out to main memory takes about **500 cycles** — twenty-plus times longer. "So if you can serve a piece of data from the countertop instead of the pantry, you just made that access twenty times faster. Do that for a whole tile of numbers and you've turned a slow kernel into a fast one." The 20-vs-500 gap is the reason shared memory exists.]]

[[note: confusion || A student sees "228 KiB" and thinks it's a typo — surely a $30,000 chip has more than a couple hundred kilobytes? Fix it: "That's not the chip's memory. The chip has 80 *gigabytes*. This 228 KiB is the tiny fast *countertop* per station — deliberately small, because fast-and-near always means small-and-scarce. The whole skill is fitting your working tile into that small space." Separate 'big far memory' from 'tiny near scratchpad' and the confusion dissolves.]]

[[fig: A hand-drawn technical diagram titled "The on-chip budget of one SM", drawn as a vertical pyramid of stacked boxes, widest and slowest at the bottom. Top box (smallest), blue-hatched, labeled in red "Registers · 256 KB/SM" with a green note "fastest, private to one thread". Middle box, green-hatched, labeled in red "Shared memory · up to 228 KiB/SM" with a green note "≈20–30 cycles · the countertop" and an orange callout "reuse lives here". Bottom box (widest), faded grey, labeled in red "HBM main memory · 80 GB @ 3.35 TB/s" with a red note "≈500 cycles · the far pantry". A blue arrow up the left side labeled "faster · smaller · nearer". A purple dashed arrow from bottom to middle labeled "haul in once → reuse many times". Dashed takeaway box: "the whole game: move data UP the pyramid and squeeze it dry". White background, hand-lettered. || The memory pyramid: 228 KiB of fast near scratchpad sitting above 80 GB of slow far memory.]]

## Number five: 32 threads — the lockstep squad

Inside an SM, workers don't act as loose individuals. They march in fixed squads of exactly **32**. A squad of 32 threads is called a **warp**, and — this is the key fact — all 32 execute the *same instruction at the same time*, in perfect lockstep. Step, step, step, together.

[[note: metaphor || A warp is a **rowing crew of 32**, or a marching band row of 32 — one drummer calls the beat and all 32 oars hit the water on the same stroke. They can't row independently; if one rower wants to do something different, the other 31 have to *wait* while she does it, then everyone resumes together. That "everyone waits for the odd one out" is the source of a whole family of GPU bugs called *divergence*.]]

Why know 32 cold? Two reasons. First, it's why you never launch 30 or 100 threads — you launch **multiples of 32**, because the hardware hands out work one full squad at a time. Ask for 40 and it runs a squad of 32 plus a squad with 24 rowers sitting idle. Second, 32 is the unit that talks to memory together: if all 32 threads read 32 *neighboring* addresses, the hardware fetches them in **one** trip down the hallway instead of 32. That trick is called **coalescing**, and it's one of the biggest early speedups in the whole workshop.

[[note: aha || Here's the number that makes the whole chip snap into focus. `132 SMs × 4 squad-schedulers × 32 threads` — write it out — is over 16,000 threads that can literally be *stepping at the same instant*, and with many squads queued up to hide waiting, the chip juggles ~270,000 in flight. "Thirty-two is the atom. Everything above it — warps, SMs, the whole GPU — is just thirty-twos stacked up." When 32 clicks, the architecture clicks.]]

[[fig: A warm hand-drawn illustration titled "A warp = 32 rowers in lockstep". A long rowing boat drawn charmingly with 32 little rower figures, all with oars raised at the exact same angle, one drummer at the front labeled in blue "the scheduler calls the beat". A hand-lettered note above: "all 32 do the SAME instruction on the SAME beat". To the side, a small inset showing one rower doing something different while the other 31 sit with folded arms, labeled in red "divergence — the 31 wait for the 1". Below, a second tiny panel: 32 rowers all reaching for 32 neighboring buckets in one sweep, labeled in green "coalescing — 32 neighbors fetched in ONE trip". Dashed takeaway box: "32 is the atom of the GPU — always launch in multiples of 32". White background, hand-lettered labels. || The warp as a 32-oar rowing crew: same stroke together, and why neighbors-in-one-trip (coalescing) is so fast.]]

[[sn: You'll hear people call a warp "SIMT" — single instruction, multiple threads. Don't put that acronym on the board in the first pass. "A squad of 32 that steps together" is the whole idea; the jargon can come later once the picture is solid.]]

## Putting all five to work: the back-of-envelope demo

Now the payoff. Five numbers, and suddenly a mentor can *answer questions* instead of shrugging. Do this live.

[[note: demo || The one demo to run: pick a real matmul, say two 4096×4096 matrices. On the board, count the math: about `2 × 4096³ ≈ 137 billion` operations. Count the bytes it *must* move: about three 4096×4096 tiles of 2-byte numbers ≈ `100 million` bytes. Divide: `137e9 ÷ 100e6 ≈ 1,370` operations per byte. Compare to the ridge, 295. "1,370 is way above 295 — so this matmul *can* be compute-bound; the chip's math is the limit, not its pipe, *if* we write the kernel well." You just predicted the character of a kernel with five numbers and one division. That's the whole skill.]]

[[note: say || The line that ties the chapter together: "Five numbers. How fast it computes, how fast it's fed, how many teams, how big each team's scratchpad, how big a squad. Give me those five and I can tell you — before writing a line of code — whether your kernel will be starved or fed, and how much of this expensive machine you're actually going to use." Say it, then point back at the five numbers still sitting in the corner of the board.]]

[[fig: A hand-drawn "cheat card" titled "The five numbers to know cold", drawn as a friendly index card pinned with a thumbtack. Five hand-lettered rows, each with a tiny icon: a room of calculators "989 TFLOP/s — how fast it computes"; a wide pipe "3.35 TB/s — how fast it's fed"; a grid of stations "132 SMs — how many teams"; a small countertop "228 KiB — the near scratchpad, per team"; a row of 32 rowers "32 threads — one lockstep squad (a warp)". Below the five, one boxed derived number in orange: "989 ÷ 3.35 ≈ 295 ops/byte — the ridge". A green note in the corner: "these live on the board ALL WEEK". Dashed takeaway box: "hand-waving → arithmetic". White background, hand-lettered labels, charming. || The keeper: the five numbers plus the one you derive from them, drawn as a pin-up cheat card for the whole workshop.]]

[[note: production || Ground it before you close. These aren't trivia — they are the numbers on the spec sheet that decides billion-dollar cluster purchases. When a lab chooses H100 vs B200, when DeepSeek tunes a kernel to shave 10% off its serving bill, when a team debates FP8 vs BF16 — they are arguing about exactly these five numbers and the ridge you divide out of them. A mentor who owns these five can read any GPU announcement and immediately know what got bigger and what it means.]]

## You can now teach

- **989 TFLOP/s** as a room full of calculators — and the 16× gap between the fast (BF16 + tensor core) path and the slow (plain 32-bit) path on the *same* chip.
- **3.35 TB/s** as the feeding pipe, and why — despite sounding infinite — it's the bottleneck because the math is even faster.
- The **ridge point, ~295 ops/byte**, derived live by dividing the first two numbers, and what each side of it means for whether a kernel starves.
- **132 SMs** as 132 identical cooking stations running one recipe in parallel, juggling ~270,000 workers at once.
- **228 KiB** of shared memory as the tiny fast countertop, the 20-vs-500-cycle speed gap, and why reuse-in-scratchpad is the whole craft.
- **32 threads** as a lockstep rowing crew (a warp) — why you launch in multiples of 32, and why 32-neighbors-in-one-trip (coalescing) is an early big win.
- The **back-of-envelope demo**: using all five numbers to predict, before writing code, whether a real matmul will be compute-bound or starved.
