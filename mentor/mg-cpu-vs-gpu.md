By the end of this chapter you'll be able to answer, at a whiteboard and without hand-waving, the question every student silently has on day one: *what actually is a GPU, and why is it good at AI?* You don't need any electronics knowledge to teach this well. You need one very good metaphor, one honest number, and the discipline to not over-complicate it. Let's build it.

## The one-sentence answer

A **CPU** (the normal processor in your laptop) is a handful of extremely clever workers who can do complicated things quickly, one after another. A **GPU** is thousands of simpler workers who each do a small, dull task, but all at the same time. For most everyday computing, the clever few win. For AI — where the same simple sum has to be done a billion times — the thousands win, overwhelmingly.

[[note: metaphor || The restaurant. A CPU is a **fine-dining kitchen**: four master chefs, each able to cook any dish start to finish, improvising as they go. A GPU is a **school cafeteria**: a thousand line cooks, each of whom only knows how to put one scoop of rice on one tray — but a thousand trays get filled in the time it takes a master chef to plate a single dish. If your job is one intricate dish, hire the chefs. If your job is *feed ten thousand people the same meal*, the cafeteria wins by a mile. A neural network is the cafeteria's dream: the same simple multiply-add, needed a billion times.]]

[[fig: A warm hand-drawn split illustration titled "CPU vs GPU". Left half labeled "CPU: the fine-dining kitchen" — four large detailed chef figures with tall hats at workstations, each with a thought-bubble showing a complicated multi-step dish, a blue handwritten note "few, brilliant, flexible — great one-after-another". Right half labeled "GPU: the cafeteria line" — a long counter with a dense row of many small identical cook figures, each doing the identical tiny action (one scoop onto one tray), a green handwritten note "thousands, simple, identical — all at once". Between the halves a dashed divider. A dashed takeaway box at the bottom spanning both: "one hard dish -> chefs win.  a billion identical scoops -> the line wins." Excalidraw style, white background, charming and friendly, handwritten labels. || The core metaphor: a CPU is a few master chefs; a GPU is a thousand-strong cafeteria line doing the same tiny task at once.]]

[[note: teach || Draw the two kitchens side by side and *act it out*. Walk slowly like one careful chef doing four steps, then rapidly mime a whole row of cooks slapping trays at once. The physical comedy makes it stick better than any diagram. Only after the picture lands do you introduce the word "parallelism" — the fancy name for "many workers at the same time." Never lead with the jargon.]]

## The trade the GPU makes

The GPU didn't get thousands of workers for free. It made a **trade**. To fit thousands of cooks on one chip, each cook has to be small and simple — less "scratchpad" space to think, no fancy tricks for guessing what to do next. The CPU spends most of its chip area on being *clever* (predicting, reordering, big caches). The GPU spends almost all of its chip area on *raw arithmetic* — rows and rows of tiny calculators.

[[note: example || Put rough numbers on the board so it's concrete, not vague. A high-end CPU has on the order of 10–100 powerful cores. An NVIDIA H100 GPU has **132 "streaming multiprocessors,"** and each of those runs many groups of 32 workers at once — adding up to tens of thousands of simple operations in flight simultaneously. Don't sweat exact figures; the students only need the shape of it: *tens of clever workers vs. tens of thousands of simple ones.*]]

[[fig: A hand-drawn "chip floorplan" comparison. Left: a square labeled "CPU chip" mostly filled with a few big blue-hatched blocks labeled "control / prediction / big cache", and only a small green corner labeled "actual math". Right: a square labeled "GPU chip" almost entirely filled with a dense grid of tiny green squares labeled "math, math, math..." and only a thin blue sliver labeled "control". Red annotations point out the contrast: "CPU spends its area on being clever" and "GPU spends its area on doing sums". A dashed takeaway box: "the GPU trades cleverness for overwhelming arithmetic." Excalidraw style, white background, handwritten. || The trade, drawn as chip real estate: the CPU spends silicon on cleverness; the GPU spends it on raw math.]]

This is the deep idea, and it's worth saying plainly to students: **the GPU is not smarter than the CPU. It is more numerous, and AI happens to be a problem where numerous beats smart.**

## Why AI in particular loves this

Recall from the matrix-multiply chapter that pushing data through a neural network is, at bottom, a mountain of identical multiply-adds. There's no cleverness required for any single one of them — just `a × b + c`, over and over, billions of times. That is *exactly* the cafeteria's kind of job: the same trivial action, needed at enormous scale, with no improvisation.

[[note: say || "A neural network never asks the chip to do anything hard. It asks it to do something *easy* — a multiply and an add — an unimaginable number of times. The GPU is the machine that says: fine, I'll do a easy thing ten thousand times at once. That match, between what AI needs and what a GPU is, is the whole reason this technology took off when it did."]]

[[note: confusion || A student will ask: "then why not use GPUs for everything?" Answer with the metaphor, not a lecture. Opening a web browser, running your operating system, reacting to a mouse click — those are *one intricate dish at a time* jobs, full of branches and decisions. Hand a cafeteria a single complicated à-la-carte order and 999 cooks stand idle. GPUs only win when the work is massively repetitive and identical. Most of daily computing isn't; AI's core loop is.]]

## The honest catch: feeding the cooks

Here's the tension you'll return to for the entire workshop, so introduce it gently now. A thousand cooks can only work as fast as the ingredients arrive. If the pantry is far away and the hallway is narrow, your thousand cooks spend most of their time *waiting for rice*, not scooping it. A GPU has the same problem: its thousands of calculators are so fast that the real struggle is shoveling data to them quickly enough.

[[note: aha || This is the sentence that reframes the whole course: **"A GPU is almost never limited by how fast it can do math. It's limited by how fast it can be fed data."** Students arrive thinking kernel optimization is about clever arithmetic. It's the opposite — it's about *logistics*, about keeping the cooks supplied. Say this early and often; every optimization on the GEMM ladder is a better way to feed the cooks.]]

[[fig: A hand-drawn illustration titled "Feeding the cooks is the real bottleneck". A big cafeteria line of many small green cook figures on the right, mostly standing idle with little "zzz" / waiting marks, tapping their feet. On the left, a distant pantry labeled in green "HBM memory (far away)" connected by a long thin winding hallway labeled in blue "limited bandwidth" down which only a few small ingredient boxes are trickling. A red annotation over the idle cooks: "cooks faster than the hallway -> they wait". An orange callout: "the whole craft = keep the cooks fed". Dashed takeaway box: "fast math is easy; fast feeding is the hard part." Excalidraw style, white background, charming, handwritten. || The catch that motivates the whole course: the cooks are faster than the hallway that feeds them.]]

## The production link

Frame the stakes so students know this isn't a toy. The reason companies buy racks of H100 and B200 GPUs — spending hundreds of thousands of dollars each — is precisely this cafeteria bargain: for the massively-repetitive math of AI, one GPU replaces a warehouse of CPUs. And the reason kernel engineers are paid so well is the *catch*: those expensive cooks sit idle unless someone writes the code that keeps them fed. A model served on a poorly-fed GPU might use 10% of the hardware you paid for; a well-fed one, 90%. That gap — nine-tenths of a multi-million-dollar cluster — is what your students learn to close.

[[note: production || Concrete and current: when DeepSeek or Meta serve a model to millions of users, the difference between a kernel that keeps the GPU 40% fed and one that keeps it 85% fed is, directly, *half the electricity bill and half the GPU count.* The FlashAttention kernel your students will study became famous for exactly this reason — it fed the cooks far better than what came before, and the entire industry adopted it within months. Kernel engineering is where hardware money is won or lost.]]

That's the chapter. Two kitchens, one trade, and one catch. If a student leaves able to explain *why numerous beats smart for AI* and *why feeding the cooks is the real problem*, you have given them the mental spine for everything that follows.

## You can now teach

- The **CPU-vs-GPU** difference as fine-dining chefs vs. a cafeteria line — few-and-clever vs. many-and-simple.
- The **trade** the GPU makes: spending its silicon on raw arithmetic instead of cleverness — "not smarter, more numerous."
- **Why AI fits the GPU** so perfectly: the network's core work is the same trivial multiply-add at enormous scale.
- Why GPUs *don't* win at everyday, branchy computing (answer the "why not use them for everything?" question).
- The **catch** that sets up the whole course: a GPU is limited by how fast it's *fed*, not how fast it computes.
- The **production stakes**: keeping the cooks fed is worth a fortune, and it's exactly what kernel engineers are paid to do.
