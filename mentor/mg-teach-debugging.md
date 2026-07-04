By the end of this chapter you can stand at a whiteboard and teach GPU debugging as a set of detective stories — the race, the misaligned load, the silent NaN — and hand students the exact tools (compute-sanitizer, core dumps, cuda-gdb) that crack each case. You don't need to have shipped a kernel yourself to teach this well. You need three good crime stories and the discipline to say, every time: *a GPU fails quietly, so we have to go looking.*

This is the chapter where students stop being scared. Everyone who writes a kernel eventually hits a bug that makes no sense — the answer is wrong, or the program just freezes forever. Left alone, that experience makes people feel stupid. Taught well, it makes them feel like detectives. Your whole job here is to flip fear into curiosity.

## The one thing that makes GPU bugs different

Start with the plain, unsettling truth. On a normal CPU program, when something breaks, it breaks *right there*. You get an error, a line number, a stack trace pointing at the exact line. GPU code does not do this. It fails **silently**, it fails **later**, and it fails **somewhere else**.

Why? Because launching a kernel is like mailing a letter. You write `kernel<<<grid, block>>>(...)`, and that line returns *immediately* — the letter is in the mailbox, your Python keeps running. The GPU does the actual work later, on its own time. So if the work goes wrong, your Python is already twenty lines further down the page. The thread that made the mistake has finished and vanished. There's no one left at the scene to question.

[[note: metaphor || GPU debugging is a **cold case**. On a CPU, you arrive while the crime is still happening — the suspect is standing there. On a GPU, by the time you notice anything's wrong (a NaN in your output, a frozen terminal), the guilty thread finished its shift, clocked out, and its desk was reassigned to someone else. You're not catching a criminal in the act. You're a detective walking into an empty room, hours later, reconstructing what happened from whatever traces got left behind.]]

[[fig: A warm hand-drawn detective illustration titled "GPU bugs are cold cases". Left panel labeled "CPU: caught in the act" — a small cartoon detective pointing at a culprit standing right next to a broken window, a blue handwritten note "crashes HERE, on this line, right now". Right panel labeled "GPU: cold case" — the same detective standing alone in an empty office with an outline of chalk on the floor, a clock on the wall reading "much later", a green handwritten note "the guilty thread already left — reconstruct from clues". A dashed divider between panels. Dashed takeaway box spanning both: "CPU fails HERE and NOW; GPU fails later and somewhere else." Excalidraw style, white background, charming, handwritten labels. || The core mental model: a CPU crash is a live scene; a GPU bug is a cold case you reconstruct after the fact.]]

[[note: teach || Do NOT open with tools. Open with this feeling. Ask the room: "Who has ever had code that gave the wrong answer but didn't crash?" Every hand goes up. Then say: "On a GPU that's the *normal* case, and today you learn how detectives handle it." You've now framed the whole chapter as detective work instead of a tedious tools lecture. Reveal the tools only once they *want* them.]]

## The two kinds of case: dead body vs. missing person

Here is the single organizing idea for the entire chapter. Write it on the board and keep pointing back at it. There are exactly two ways a kernel goes wrong, and they need two completely different detectives.

**Case one: the kernel finishes, but the answer is wrong.** The letter got delivered, but the contents are garbage. Someone scribbled in memory that wasn't theirs, or two workers fought over the same scrap of paper. The program *returns*. You just can't trust what it gives you.

**Case two: the kernel never finishes at all.** The letter went into a black hole. The GPU is spinning forever and your terminal is frozen. Nothing comes back, ever.

[[note: say || "Every GPU bug is one of two crimes. Either the program *finishes* and lies to you" — hold up one finger — "or it *hangs* and never comes back at all" — hold up two. "And here's the punchline you'll remember forever: for the second kind, hitting Ctrl-C does absolutely nothing. We'll see why. Two crimes, two detectives — always ask which one you're on before you reach for a tool."]]

[[fig: A hand-drawn Excalidraw-style decision-tree diagram, white background, fine black ink, hand-lettered labels. A black diamond at top asks "Does the kernel RETURN?". A branch labeled in blue "YES -> wrong answer" flows down-left into a blue-outlined box "compute-sanitizer" with three purple monospace sub-lines "memcheck (out-of-bounds)", "racecheck (shared-mem race)", "synccheck (bad __syncthreads)", and a green tag "catches it WITHOUT a debugger". A branch labeled in blue "NO -> HANGS forever" flows down-right into a numbered vertical stack: circle 1 "arm core-dump env vars", circle 2 "poke named pipe from 2nd terminal", circle 3 "read dump in cuda-gdb". A bold orange callout points at the hang branch: "Ctrl-C does NOTHING here". Dashed takeaway box at bottom: "wrong answer -> sanitizer.   hang -> core dump.". Excalidraw style, flat, generous white space. || The whole chapter on one slide: two failure modes, two toolkits. Always decide which branch you're on first.]]

## Story 1 — the misaligned load (an out-of-bounds crime)

Tell it as a story, not a definition. Your kernel launches a thousand threads. Each thread is told "go read the number at position `my_id` in this array." But the array only has 900 slots. Threads 900 through 999 go reach for numbers that were never theirs — they read past the end of the fence, into the neighbor's yard. That's an **out-of-bounds** access. The kernel usually still *finishes*; it just comes back holding garbage from someone else's memory.

The sibling crime is the **misaligned load**. GPUs like to grab memory in neat, aligned chunks — imagine mailboxes that only open in blocks of four. If a thread tries to read starting from mailbox 3, straddling two blocks, the hardware faults. Same family of bug: reaching for memory the wrong way.

[[note: example || Do it by hand with 8 mailboxes on the board, numbered 0–7. Say "the array only has 6 items — mailboxes 0 through 5." Now launch 8 threads. Draw thread 6 and thread 7 reaching arrows past the fence into mailboxes 6 and 7 (drawn as a neighbor's yard). "Those two threads just read garbage. The kernel finishes. Your answer is quietly wrong." Two threads out of eight — that's all it takes. Students feel how easy it is to be *slightly* off.]]

The detective for this crime is **compute-sanitizer**. You don't change your code. You just run your normal command with `compute-sanitizer` in front of it, like putting a detective's magnifying glass over the whole thing:

```bash
compute-sanitizer --tool memcheck python my_repro.py
```

The `memcheck` tool watches every memory access and, the instant a thread reaches out of bounds, it stops and tells you *which thread* and *which source line*. It's the CUDA cousin of Valgrind. Yes, it runs your program 10× slower or more — and that's completely fine, because you're not timing anything. You're asking one yes/no question: *did anyone reach over the fence?*

[[fig: A warm hand-drawn illustration titled "Out-of-bounds: reaching over the fence". A row of 8 mailboxes labeled 0–7, with mailboxes 0–5 painted green ("array = 6 items") and a hand-drawn wooden fence after mailbox 5. Eight little worker figures (threads) each reach an arrow to their numbered mailbox; workers 6 and 7 reach arrows OVER the fence into a red-hatched "neighbor's yard" holding junk symbols, red note "reads garbage — kernel still finishes!". Above the whole scene, a cartoon detective with a magnifying glass labeled in blue "compute-sanitizer --tool memcheck" spotlights worker 7 with a speech bubble "thread 7 out of bounds at line 42". Dashed takeaway box: "OOB = a thread reaches past the array end; memcheck names the thread and line." Excalidraw style, white background, charming, handwritten. || The out-of-bounds crime and its detective: threads reach past the fence, and memcheck fingers the exact thread and line.]]

[[note: production || This is not a toy problem. Real serving stacks like vLLM run kernels over batches whose sizes change every request. An index that's computed one-off-wrong sails past the end of a tensor and corrupts an output token. The vLLM team's own debugging playbook opens with exactly this move: before anything fancy, wrap the repro in `compute-sanitizer`. It's the first thing a professional reaches for, every single time.]]

## Story 2 — the race (two workers, one sheet of paper)

This is the most important story in the chapter, because races are the bugs that make grown engineers cry. Tell it carefully.

Inside a block, threads can share a tiny fast scratchpad called **shared memory**. It's a whiteboard the whole team can write on. Now imagine two workers both need to update the same cell on that whiteboard. Worker A reads the value, adds one. Worker B reads the *same old* value, adds one. Both write back. You wanted the number to go up by two — it went up by one. Nobody made a "mistake." They just stepped on each other. That's a **race condition**.

[[note: metaphor || A race is **two chefs, one shared recipe card**. The head chef writes "add salt" on the card. A second chef, at the exact same moment, erases it to write "add sugar." Depending on the precise microsecond, the dish comes out salty, sweet, or a bizarre mix — and it's *different every time you cook it*. That's the signature of a race: the answer is *usually* right, and occasionally, maddeningly, wrong, and you can never quite reproduce it. The fix is a rule that says "everybody finish reading before anybody writes" — in CUDA that traffic cop is `__syncthreads()`.]]

[[note: confusion || THE big one. Students think "wrong answer = my math is wrong." So they stare at the formula for an hour. But a race gives the *right* answer most of the time — the math is fine! The tell is *inconsistency*: run it ten times, get nine right and one wrong. Teach them this reflex: "if the same input gives different outputs on different runs, stop reading your math and suspect a race." That one sentence saves people entire afternoons.]]

The detective here is a *different* sanitizer tool:

```bash
compute-sanitizer --tool racecheck python my_repro.py
```

`racecheck` watches the shared-memory whiteboard specifically. It catches two threads writing the same spot without a `__syncthreads()` between them — the read-after-write hazard that produces those "usually right" results. There's a third cousin too, `synccheck`, which catches a subtler crime: a `__syncthreads()` that not every thread reaches, because it's hidden inside an `if` that only some threads take. We'll meet the consequences of that one in Story 3.

[[fig: A warm hand-drawn illustration titled "A race: two cooks, one recipe card". Center: a single index card on a table reading "count = 5". Two cartoon chefs labeled "thread A" (blue) and "thread B" (green) both reach for it simultaneously with arrows; A's thought bubble "read 5, write 6", B's thought bubble "read 5, write 6". Below the card, a confused result "count = 6  (wanted 7!)" circled in red with note "both read the OLD value". Off to the right, a traffic-cop figure holding a stop sign labeled in orange "__syncthreads() = everyone finish reading before anyone writes". Top corner: a detective magnifying glass labeled purple "compute-sanitizer --tool racecheck". Dashed takeaway box: "race = two threads clash on shared memory; racecheck finds it; __syncthreads fixes it." Excalidraw style, white background, charming, handwritten. || The race, dramatized: two cooks overwrite one card, the count comes out wrong, and a syncthreads traffic-cop is the fix.]]

## Story 3 — the hang (and why Ctrl-C betrays you)

Now the hard one, and the crowd-pleaser. Sometimes a kernel doesn't finish *at all*. Your terminal just sits there. You hit `Ctrl-C`. Nothing. You hit it again. Nothing. This is where students panic — so this is where you get to be the hero.

Here's the crime. `__syncthreads()` is a rule: "nobody moves until *everybody* in the block reaches this line." Now suppose some threads take an `if` branch that contains a `__syncthreads()`, and the rest don't. The threads inside the branch wait at the barrier for their teammates. The teammates already walked past — they'll never arrive. So the waiting threads wait *forever*. The GPU spins. Your program hangs. This is a **divergent barrier** deadlock.

[[note: metaphor || Picture a **group that agreed to meet at the door before leaving** — "nobody leaves till we're all here." Half the group ducks into a side room where the meeting point is, and waits by the door. The other half already left through a different exit. The waiters stand at the door forever, because the people they're waiting for are never coming. That's a divergent-barrier hang: some threads parked at a `__syncthreads()` that the rest will never reach.]]

And now the famous betrayal: **why doesn't Ctrl-C work?** Because your Python isn't running Python anymore. It's frozen deep inside the GPU driver, waiting for the GPU to say "done." `Ctrl-C` sends a polite interrupt signal, but Python only checks for that signal *between* its own instructions — and it's stuck mid-instruction inside the driver, which is waiting on a GPU that will never answer. So your interrupt sits in a queue, unread, forever. The one move everyone tries first is the one move that cannot possibly work.

[[note: aha || This is the jaw-drop moment of the chapter. Say it slowly: **"When a GPU hangs, Ctrl-C is guaranteed to do nothing — and once you understand why, you'll never waste time mashing it again."** Students have all mashed Ctrl-C at a frozen terminal. Learning that it's *structurally impossible* for it to work — Python is parked inside the driver, not in Python — genuinely lands as a revelation. This is the number-one thing they'll remember from your session.]]

[[fig: A warm hand-drawn illustration titled "Why Ctrl-C betrays you on a hang". Left: a group of little worker figures at a door labeled "__syncthreads() meet here first"; three workers wait patiently at the door (blue), while two others are drawn already walking out a different exit (red, "took the other branch — never coming"). Waiting workers have "zzz forever" marks. Right: a laptop with a frozen terminal; a hand mashing a "Ctrl-C" key with an angry orange note "SIGINT queued... never checked". A thought-bubble from the laptop shows Python asleep INSIDE a big box labeled "CUDA driver, waiting on GPU", blue note "Python isn't running Python — it's parked in the driver". Dashed takeaway box: "divergent barrier = threads wait forever; Python is stuck in the driver, so Ctrl-C can't fire." Excalidraw style, white background, charming, handwritten. || The hang and its cruel twist: threads deadlocked at a barrier, and a Ctrl-C that can never be heard because Python is asleep inside the driver.]]

## Cracking the hang: snapshot the frozen scene

So if you can't interrupt it, what do you do? You take a **photograph of the frozen GPU from the outside** — while it's still stuck. CUDA can do exactly this. It's called a **user-triggered core dump**: a snapshot of every warp's program counter — that is, exactly which instruction each group of threads is frozen on.

The workflow is a little two-terminal dance, and it's worth walking through slowly because every piece earns its keep.

**Terminal 1** — before you launch, you set a handful of environment variables that arm the camera. The two switches that matter most:

```bash
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
export CUDA_COREDUMP_PIPE="/tmp/cuda_coredump_pipe_%h.%p.%t"
export CUDA_COREDUMP_GENERATION_FLAGS='skip_global_memory,skip_shared_memory,skip_local_memory'
```

Then you run your repro normally, and it hangs, exactly as expected.

**Terminal 2** — from a *second* terminal, you poke the running process through a **named pipe** (a little mailbox in the filesystem the driver is watching). Poking it triggers the snapshot:

```bash
dd if=/dev/zero bs=1M count=1 > /tmp/cuda_coredump_pipe_...
```

[[note: confusion || Two traps here, both afternoon-eaters. First: use `dd`, NOT `echo`. A bare `echo` writes a few bytes that get stuck in the pipe's buffer and never wake the driver — you wait, nothing dumps, and you wrongly conclude the whole mechanism is broken. `dd` pushes a full megabyte and forces the driver to notice. Second: that `skip_global_memory` flag. Without it, the dump tries to save all 80 GB of the H100's memory and takes forever. With it, you save just the code and the program counters — the *only* thing you need to find a hang — and the dump takes seconds instead of minutes.]]

[[fig: A hand-drawn Excalidraw-style two-lane timeline titled "User-triggered core dump: the two-terminal dance", white background, fine black ink, hand-lettered labels. Top lane "Terminal 1" (black): a box "python my_repro.py" arrow into a red-outlined box "kernel HANGS (spinning glyph)"; orange note above "Ctrl-C useless here". Bottom lane "Terminal 2" (black): a purple monospace box "dd if=/dev/zero bs=1M count=1 > pipe" with a blue dashed arrow rising UP to the hung box, blue label "1 MB overruns the pipe buffer -> wakes the driver". From the hung box a dashed arrow drops to a yellow-hatched disk-cylinder icon labeled "core dump = code + warp PCs only". Green spec note by the disk "skip_global/shared/local mem -> seconds not minutes on 80GB". Purple note by the pipe "path = CUDA_COREDUMP_PIPE". Dashed takeaway box: "poke the pipe from OUTSIDE while it's still stuck." Excalidraw style, flat, wide, generous white space. || The rescue move drawn out: Terminal 1 hangs, Terminal 2 pokes the pipe, and the driver photographs the frozen GPU in seconds.]]

## Reading the evidence: cuda-gdb

Now you've got a photograph of the crime scene. You open it in **cuda-gdb**, the GPU debugger, which understands these dump files:

```bash
cuda-gdb
(cuda-gdb) target cudacore /tmp/cuda_coredump_...
```

It drops you right at the frozen kernel and shows you, warp by warp, which instruction each group of threads was stuck on. For a divergent-barrier hang, this is the *smoking gun*: you literally see two groups of threads parked at two different program counters — one group waiting at the barrier, the other long gone. The case solves itself the moment you can see it.

[[note: production || Template-heavy production kernels — CUTLASS, FlashAttention — are exactly where this pays off. One line of GPU assembly can be a dozen inlined functions fused together, so a raw program counter tells you *nothing* by itself. The pro fix is to compile with `NVCC_PREPEND_FLAGS='-lineinfo'`, which stamps a source-line map into the binary so cuda-gdb can turn a raw address back into "line 214 of mma.cuh." One trap the vLLM team flags: if `ccache` (a compiler cache) is on, it hands back the *old* binary with no line map and the flag silently does nothing — so set `CCACHE_DISABLE=1` for the debug rebuild. When even that isn't enough, `nvdisasm -gi` reconstructs the full chain of inlined calls that led to the crash.]]

[[sn: The silent NaN — a number that's become "not a number," often from a `0/0` or an overflow in mixed precision — is its own detective story. It doesn't crash or hang; it just poisons every later computation, because any math touching a NaN produces another NaN. The trick is to hunt for *where it first appears*: check tensors layer by layer until one comes back clean-in, NaN-out. That's your crime scene.]]

## Teaching notes: how to run the block

Here is a concrete plan for a single session.

- **Open with the cold-case framing (5 min).** Ask "who's had code that's wrong but doesn't crash?" Land the mailed-letter metaphor. Draw the two-crimes decision tree and leave it up all session.
- **Story 1, out-of-bounds (8 min).** The 8-mailbox by-hand demo. Then live-run `compute-sanitizer --tool memcheck` on a script with a deliberate off-by-one and let it name the line. That live moment — the tool pointing straight at the bug — is your first "whoa."
- **Story 2, the race (10 min).** The two-cooks-one-card drama. Hammer the "different output every run = suspect a race" reflex. Run `racecheck`.
- **Story 3, the hang (15 min).** Build the divergent-barrier meeting metaphor, then deliver the Ctrl-C betrayal as the emotional peak. Do the two-terminal core-dump dance live if you possibly can — the `dd`-through-a-pipe moment feels like magic.
- **Close (2 min).** Return to the decision tree. "Wrong answer? Sanitizer. Hang? Snapshot from outside. That's the whole map."

[[note: demo || The single demo that makes the room gasp: a kernel that hangs. Mash Ctrl-C on the projector — let it *visibly* do nothing for an uncomfortable few seconds. Then calmly open a second terminal, `dd` a megabyte into the pipe, and watch the dump appear. Going from "hopeless frozen screen" to "here's exactly where all 32 threads are stuck" in thirty seconds is the most memorable thing you'll show all day.]]

## You can now teach

- **Why GPU bugs are cold cases** — the mailed-letter model of async launch, and why failures show up silently, later, and elsewhere.
- **The two-crimes decision tree** — kernel returns wrong answer vs. kernel hangs — and which detective each one needs.
- **The out-of-bounds / misaligned-load story** with the mailbox demo, cracked by `compute-sanitizer --tool memcheck`.
- **The race condition** as two cooks fighting over one card, the "different-every-run" tell, and `racecheck` plus `__syncthreads()` as the fix.
- **The hang** as a divergent-barrier deadlock, the jaw-drop of *why Ctrl-C structurally cannot work*, and the two-terminal user-triggered core dump that snapshots the frozen GPU.
- **Reading the evidence in cuda-gdb**, plus the production reality (`-lineinfo`, the `ccache` trap, `nvdisasm -gi`) that vLLM engineers live by.
