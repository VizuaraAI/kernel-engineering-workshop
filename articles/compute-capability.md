Every kernel we write in this course eventually reaches the same fork in the road: the source is done, it compiles, and now we have to tell `nvcc` *what chip we are compiling for*. Get this wrong and one of two things happens. Either the compiler refuses an instruction you know the hardware has — `wgmma` on an H100, say — or it silently builds something that runs, but runs the *old* way, leaving half the silicon dark. This article is about the small pile of flags that decide that, and the mental model underneath them, because on Hopper the mental model has a sharp edge you can cut yourself on: the `a` in `sm_90a`.

The stakes are not academic. On this course we are chasing **93.7% of `cuBLAS`** on an H100, and the last few kernels on that ladder are built entirely out of Hopper-only instructions — the tensor-core `wgmma`, the **Tensor Memory Accelerator** (TMA), thread-block clusters. None of those instructions exist unless you target the architecture *exactly*, with the suffix. If your build line says `sm_90` instead of `sm_90a`, the fast kernels do not compile at all. So this is load-bearing plumbing, and worth understanding once, properly.

## Two architectures wearing one number

The first thing to internalize is that `nvcc` deals in **two** notions of "architecture," and they are deliberately different.

There is the **virtual architecture** — a `compute_XY` target — which is a *contract about a language*. It says: here is the set of PTX instructions and features you are allowed to assume. **PTX** (Parallel Thread eXecution) is NVIDIA's stable virtual ISA, a forward-compatible assembly-ish intermediate representation that is *not* what the hardware runs.[[sn: PTX is closer to a portable bytecode than to machine code — think of it as the JVM bytecode of GPUs. The thing the SM actually executes is SASS, and SASS is undocumented, versioned per-architecture, and can change between driver releases. We disassemble it constantly on this course, but we never write it.]]

Then there is the **real architecture** — an `sm_XY` target — which is a *contract about a chip*. It says: assemble all the way down to **SASS**, the actual binary machine code for a specific **Streaming Multiprocessor** (SM) generation, and bake it into the binary as a **cubin**. `sm_90` is the physical H100 SM; `sm_80` is the A100; `sm_75` is a Turing card.

The `XY` in both is the **compute capability**: a `major.minor` version that abstracts the physical GPU away from the instruction set. `9.0` is Hopper, `8.0`/`8.6` are Ampere, `7.5` is Turing, `10.0` is Blackwell. The major number tends to track a microarchitecture family; the minor number tracks a revision within it. When you read "the H100 is compute capability 9.0," that number *is* the virtual/real target you feed the compiler.

[[fig: A hand-drawn compilation-pipeline diagram titled "nvcc: two architectures, one number". Left to right, three rounded boxes connected by black arrows: box 1 "your .cu source" (black), box 2 labeled "PTX — virtual ISA" with a purple annotation "compute_90" and a blue handwritten note "portable, forward-compatible bytecode", box 3 labeled "SASS / cubin — real machine code" with a purple annotation "sm_90" and a green note "binary for ONE SM generation". Above box 2 a red bracket labeled "VIRTUAL arch = language contract"; above box 3 a red bracket labeled "REAL arch = chip contract". A curved dashed orange arrow loops from PTX forward to a small grey box on the far right labeled "future GPU" with an orange note "JIT at load time". A dashed takeaway box bottom-right: "-gencode carries a PTX stage (compute_) AND a SASS stage (sm_) — keep BOTH". || The compile splits into a virtual PTX stage and a real SASS stage; the flags name each one separately.]]

## The flags, decoded

There are two ways to spell the target, and it is worth seeing both because you will meet both in real build systems.

The shorthand is `-arch`:

```bash
nvcc -arch=sm_90 gemm.cu -o gemm
```

`-arch=sm_90` is really a convenience macro. It tells `nvcc` to compile through the `compute_90` virtual architecture *and* down to the `sm_90` real architecture — and, crucially, the standalone `-arch=sm_XY` shorthand *also* keeps a `compute_90` PTX stage, so it expands to something like `arch=compute_90,code=[sm_90,compute_90]`. That means the shorthand gives you native SASS for the H100 *plus* a forward-compat PTX tail for free. Fast to build, small binary, runs on the chip you named and can still JIT onto newer ones.[[sn: To get SASS with *no* PTX tail — the truly locked-to-one-chip build — you have to be explicit: `-gencode arch=compute_90,code=sm_90` with no `compute_` in the `code=` list, or `-arch=native`. `-arch=compute_90` (virtual only) does the opposite extreme: it embeds *only* PTX and no SASS, forcing a JIT compile on the first launch for every target. Useful for a maximally forward-compatible build, painful for startup latency.]]

The full form is `-gencode`, and it is the one you actually want in a shipping build because it lets you carry *several* targets in one binary — a **fat binary**:

```bash
nvcc \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \
  gemm.cu -o gemm
```

Read each line as `arch=<virtual>,code=<what to emit>`. The first two lines emit real SASS cubins for Ampere and Hopper — those run natively, at full speed, no JIT. The third line, where `code=` is itself a `compute_` target, emits a **PTX** stage for `compute_90` and keeps it in the binary. That trailing PTX is the insurance policy, and it is the entire mechanism behind forward compatibility.

[[fig: A hand-drawn "fat binary" cross-section titled "one .cubin, several arch stages". A large black rounded rectangle labeled "fat binary (kernels)" contains three stacked inner boxes: box 1 with a green hatch fill labeled "SASS cubin — sm_80" and a green note "runs native on A100", box 2 with a green hatch fill labeled "SASS cubin — sm_90" and a green note "runs native on H100", box 3 with a blue hatch fill labeled "PTX — compute_90" and a blue note "JIT fallback for newer GPUs". On the right, three little chip icons: A100, H100, and a grey "GB300 (future)" — black dashed arrows connect A100→box1, H100→box2, and an orange dashed arrow connects GB300→box3 with an orange callout "no cubin → JIT the PTX". A purple annotation at the bottom "-gencode ... code=sm_80 / sm_90 / compute_90". Dashed takeaway box: "native SASS for chips you know, one PTX tail for the ones you don't". || A fat binary carries a native SASS cubin per known chip plus one PTX stage as forward-compat insurance.]]

## Forward compatibility, and the onion that Hopper broke

Here is the promise NVIDIA has historically made, and it is a good one. PTX is forward-compatible under an **onion model**: PTX generated for `compute_X.Y` will run on any GPU with compute capability ≥ `X.Y`, because the driver ships a **JIT** compiler that translates the embedded PTX down to whatever SASS the *actual* installed GPU wants, the first time the kernel is loaded. The result is cached on disk, so you pay the translation cost once per (binary, driver, GPU) tuple rather than per launch — which is why the first kernel launch after a driver update can be mysteriously slow, and why it isn't slow again. So a binary you built years ago with a `compute_70` PTX stage will still *run* on an H100. It will not run *well* — JIT-ed old PTX can't invent tensor-core instructions that didn't exist when it was written — but it runs. That is the onion: each newer layer contains the older ones.

This is why the `-gencode` idiom above ends with a `compute_90` PTX line. Your native SASS covers today's chips; the PTX covers tomorrow's. A GB300 that ships after your binary does will find no `sm_100` cubin, fall back to the `compute_90` PTX, JIT it, and run.

And now the sharp edge. **Hopper introduced `sm_90a`**, and the `a` suffix — for **architecture-specific** — deliberately *breaks* the onion. Targets ending in `a` are not forward-compatible; their PTX is not promised to JIT onto any future architecture, and NVIDIA says so explicitly.[[sn: Blackwell added a parallel `f` suffix (e.g. `sm_100f`, "family-specific") with its own, looser compatibility rules. The taxonomy is growing: plain `sm_XY` is the portable onion, `a` is locked to one architecture, `f` is locked to one family. For this H100 course only `sm_90` vs `sm_90a` matters.]] Why would NVIDIA ship a target that throws away its own best feature? Because some instructions are too tied to the physical layout of one specific SM to pretend they are portable. On Hopper those are exactly the instructions we care about most.

[[fig: A hand-drawn "onion vs. locked" comparison, two panels side by side. LEFT panel titled "sm_90 — the onion (portable)": concentric hand-drawn rings labeled from inside out sm_70, sm_80, sm_90, with an orange dashed arrow escaping the outer ring toward a grey box "future GPU" and a green note "PTX JITs forward ✓". RIGHT panel titled "sm_90a — architecture-specific (locked)": a single solid box labeled sm_90a with a bold red X over an arrow trying to leave it toward "future GPU", red note "no forward-compat — PTX will NOT JIT ✗". Below the right panel, a purple code line "-arch=sm_90a" with an orange callout "required for wgmma · TMA · clusters". Dashed takeaway box spanning both: "the 'a' buys you the fast instructions and pays with portability". || Plain sm_90 stays inside the forward-compatible onion; sm_90a leaves it in exchange for Hopper-only instructions.]]

## Why `sm_90a` exists: the instructions that need it

The features that make the top of our GEMM ladder possible are all Hopper-native and all gated behind the `a` target:

- **`wgmma`** — warpgroup-level matrix-multiply-accumulate. Where earlier `mma` instructions operated per-warp, `wgmma` issues one asynchronous tensor-core operation across a whole **warpgroup** (four warps, 128 threads), which is how you actually feed the H100's tensor cores fast enough to approach **989 TFLOP/s** of BF16.
- **TMA** — the Tensor Memory Accelerator, a dedicated hardware engine that copies whole multidimensional tiles between HBM and shared memory with a single instruction and address descriptor, freeing threads from computing per-element addresses.
- **Thread-block clusters and DSMEM** — the ability to group thread blocks so they can directly read each other's **distributed shared memory**, a new tier between per-SM SMEM and L2.

Below is the map of what the `a` unlocks. Try to compile a kernel using any of these against `-arch=sm_90` (no `a`) and `nvcc` rejects the instruction outright: the plain virtual architecture simply does not define it. Add the `a` and it compiles.

[[fig: A hand-drawn "what sm_90a unlocks" map, one Hopper SM in the centre with three instruction call-outs. Centre: a black rounded box labeled "Hopper SM (sm_90a)" containing a small yellow-hatch box "tensor cores" and a green spec note "989 TFLOP/s BF16 (dense)". Three purple code chips point into it via blue dashed data-movement arrows: chip 1 purple "wgmma.mma_async" with a blue note "1 op across a WARPGROUP = 4 warps / 128 threads" and a red dimension tag "128×N×K"; chip 2 purple "cp.async.bulk.tensor (TMA)" with a blue note "copies a whole 2D/3D tile HBM→SMEM in one instruction" and a green note "threads freed from address math"; chip 3 purple "cluster / DSMEM" with a blue note "blocks read each other's shared memory" and a green tier label "new tier: SMEM → DSMEM → L2". Along the bottom, a red gate symbol labeled "compile-time gate: valid ONLY under .target sm_90a" with a bold red X on a fallback arrow (orange note "no runtime detect-and-fallback"). Dashed takeaway box: "these three instructions are welded to the H100 SM — that weld is why forward-compat had to go". || The three Hopper-only instructions — wgmma, TMA, clusters — are gated behind sm_90a because each is bolted to the physical SM.]][[sn: The mechanism is that `wgmma`, TMA descriptors, and cluster intrinsics are emitted as PTX that is only *valid* under `.target sm_90a`. It is a compile-time gate, not a runtime one — there is no "detect and fall back" for these; you either built the `a` cubin or you did not have the instruction.]] There is no partial credit and no graceful degradation at the instruction level — which is exactly why the compatibility guarantee had to be dropped. The instruction is welded to the H100 SM's physical behaviour.

## Practical build guidance

So what should your build line actually say? A few rules that have never bitten me:

**Target the real chip, always include a PTX stage.** For a portable release build across the datacenter GPUs you support, carry a native SASS cubin per architecture plus one trailing PTX stage for the newest, so future cards still run via JIT:

```bash
nvcc \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \
  -O3 kernels.cu -o kernels
```

**For the Hopper fast path, switch to the `a` target and accept the lock-in.** The moment a translation unit contains `wgmma`, TMA, or cluster code, it must be built `sm_90a`:

```bash
nvcc -arch=sm_90a -O3 gemm_wgmma.cu -o gemm_wgmma
```

You will not carry a forward-compatible PTX stage for that unit — there is no such thing for `a` code — so a binary built this way targets H100-class hardware and only H100-class hardware. That is the correct trade for a kernel whose entire reason to exist is a Hopper instruction. In practice you split your codebase: portable, onion-friendly kernels in fat-binary translation units, and the architecture-specific hot kernels compiled separately for `sm_90a`.

**Match the build to the deployment, not to your laptop.** The single most common failure I see is building for the machine that ran `nvcc` — often via `-arch=native`, which asks the compiler to probe the *local* GPU — and then shipping to a different card. `-arch=native` is a lovely convenience for a quick local profile run and a footgun for anything you deploy.[[sn: `-arch=native` is the dangerous one: NVIDIA's docs are explicit that it generates SASS for the detected GPUs and *no PTX at all*. Move that binary to a newer card and it can't even JIT-fall-back — there's no PTX to JIT — so it fails to load entirely rather than running slowly. (Contrast the plain `-arch=sm_90` shorthand, which *does* keep a PTX tail.) Always keep a PTX stage in anything that leaves the machine it was built on.]]

**Don't over-fatten.** Every extra `-gencode` line is another full compile of every kernel and more bytes in the binary; only include architectures you actually run on. Two or three real targets plus one PTX tail is almost always enough.

## The bridge

That is the whole contract: `compute_XY` names the virtual PTX language, `sm_XY` names the real SASS chip, `-gencode` lets you carry several of each in one fat binary, and a trailing `compute_` PTX stage is your forward-compatibility insurance via driver JIT — *except* on Hopper's `sm_90a`, where you trade that insurance away to unlock `wgmma`, TMA, and clusters.

Which means we now have the key to the door. Everything up to kernel 7 on the ladder — through the [vectorized and autotuned kernels](gemm-kernel-1-naive.html) — compiles happily against plain `sm_90` and stays inside the onion. But the warptile kernel that finally reaches **93.7% of `cuBLAS`**, and every tensor-core kernel after it, is `sm_90a` code by necessity. Before we can write a single `wgmma`, we had to know why the build line needs that one extra letter. Now we do, and we can go spend the compute the way the [three regimes](the-three-regimes.html) told us to.
