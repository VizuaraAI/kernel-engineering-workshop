#!/usr/bin/env python3
"""Bulk parallel figure generation for Kernel Engineering.

Scans articles/*.md for  [[fig: <excalidraw scene prompt> || <caption>]]  blocks,
assigns figures/<slug>-<n>.png (same order as build.py), and renders each with
gemini-3-pro-image-preview via a thread pool. Idempotent (skips existing), resumable.

  export GEMINI_API_KEY=AQ....     # or it falls back to the embedded key
  python3 gen_figures.py [--only <slug>] [--workers 14]

These figures REQUIRE handwritten labels (unlike text-free Vizuara pipelines), so accept
~10% garbled-text regens; re-run to fill gaps (delete a bad file, run again).
"""
import os, re, sys, time, concurrent.futures, pathlib
from google import genai
from google.genai import types

ROOT = pathlib.Path(__file__).parent
ART = ROOT / "articles"
MENTOR = ROOT / "mentor"
OUT = ROOT / "docs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not KEY:
    sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment before running.")

BASE_STYLE = (
 "Hand-drawn technical diagram in Excalidraw style, drawn with a fine black ink pen on a pure "
 "white background. Slightly wobbly hand-drawn rounded rectangles, thin strokes, and hand-lettered "
 "handwriting-style text for ALL labels and annotations (like the Virgil / hand-drawn font). Use a "
 "strict semantic color palette for the handwritten annotations: BLUE handwriting for mechanisms and "
 "how data moves, GREEN handwriting for hardware specs / sizes / bandwidth numbers, RED for dimensions "
 "and matrix labels (A, B, C) and warnings, PURPLE for code snippets and advanced configuration tips, "
 "ORANGE for emphasis callouts. Long thin slightly-curved dashed arrows connect each margin annotation "
 "to the exact component it describes. Draw matrices as rectangles filled with light diagonal hatching "
 "(blue hatch for A / sharedA, green hatch for B / sharedB, pale-yellow hatch for output C and warp tiles). "
 "Dimension arrows with numbers. Hand-drawn numbered circles (1)(2)(3) marking reading order. List any "
 "constants/parameters top-left in handwriting. Key takeaways go inside a dashed rounded box. Flat, no "
 "shadows, no gradients, no photorealism, no typeset fonts, wide composition, generous white space, "
 "clean and legible handwriting."
)

def figs_in(md):
    """Yield (index, prompt) for each [[fig:]] in order."""
    lines = md.replace("\r\n", "\n").split("\n")
    i, n, k = 0, len(lines), 0
    out = []
    while i < n:
        if lines[i].strip().startswith("[[fig:"):
            buf = [lines[i]]
            while "]]" not in buf[-1] and i + 1 < n:
                i += 1; buf.append(lines[i])
            raw = " ".join(buf).strip()
            m = re.match(r"\[\[fig:\s*(.+?)\]\]\s*$", raw, flags=re.S)
            body = m.group(1) if m else raw[6:]
            prompt = body.split("||", 1)[0].strip()
            k += 1; out.append((k, prompt))
        i += 1
    return out

def collect():
    jobs = []
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    srcs = sorted(list(ART.glob("*.md")) + (list(MENTOR.glob("*.md")) if MENTOR.exists() else []))
    for md in srcs:
        slug = md.stem
        if only and slug != only:
            continue
        for k, prompt in figs_in(md.read_text()):
            jobs.append((f"{slug}-{k}.png", prompt))
    return jobs

def gen(name, prompt, retries=3):
    out = OUT / name
    if out.exists() and out.stat().st_size > 9000:
        return f"skip {name}"
    full = f"{BASE_STYLE}\n\nDIAGRAM TO DRAW:\n{prompt}"
    for attempt in range(retries):
        try:
            client = genai.Client(api_key=KEY)
            r = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[full],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )
            for part in r.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image"):
                    out.write_bytes(part.inline_data.data)
                    return f"ok {name}"
        except Exception as e:
            if attempt == retries - 1:
                return f"ERR {name}: {str(e)[:90]}"
            time.sleep(2 * (attempt + 1))
    return f"FAIL {name}"

def main():
    workers = 14
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    jobs = collect()
    print(f"{len(jobs)} figures to consider · {workers} workers", flush=True)
    done = fails = made = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(gen, n, p) for n, p in jobs]
        for f in concurrent.futures.as_completed(futs):
            r = f.result(); done += 1
            if r.startswith(("ERR", "FAIL")): fails += 1
            elif r.startswith("ok"): made += 1
            if done % 10 == 0 or r[0] in "EF":
                print(f"[{done}/{len(jobs)}] {r}  (new={made} fails={fails})", flush=True)
    print(f"DONE. {done} considered, {made} generated, {fails} failed.", flush=True)

if __name__ == "__main__":
    main()
