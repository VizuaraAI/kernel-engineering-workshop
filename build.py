#!/usr/bin/env python3
"""Kernel Engineering static site generator.

Two surfaces:
  - SHELL (sidebar + top bar): Modal-glossary dark terminal green, always present.
  - CANVAS: dark green for landing/section indexes; warm-white "bearblog Tufte" for articles.

Reads manifest.json + articles/<slug>.md, writes a complete static site into docs/.
Custom markdown extensions:
  [[fig: <excalidraw prompt> || <caption>]]   -> hand-drawn figure (file: figures/<slug>-<n>.png)
  [[sn: <note text>]]                          -> Tufte margin sidenote (numbered, red superscript)
Run:  python3 build.py
"""
import json, os, re, html, shutil, pathlib

ROOT = pathlib.Path(__file__).parent
DOCS = ROOT / "docs"
ART = ROOT / "articles"
MAN = json.loads((ROOT / "manifest.json").read_text())
SITE = MAN["site"]

# ----- flatten article order for prev/next + lookups -----
FLAT = []
for sec in MAN["sections"]:
    for a in sec["articles"]:
        FLAT.append({**a, "section_id": sec["id"], "section_num": sec["num"], "section_title": sec["title"]})
SLUG2IDX = {a["slug"]: i for i, a in enumerate(FLAT)}

# ============================================================ markdown
def esc(s): return html.escape(s, quote=False)

def inline(text, ctx):
    """Inline formatting. ctx carries per-article sidenote counter list."""
    # sidenotes first (may sit mid-sentence)
    def sn(m):
        ctx["sn"] += 1
        n = ctx["sn"]
        note = inline_basic(m.group(1).strip())
        return (f'<label for="sn-{ctx["slug"]}-{n}" class="sn-ref">{n}</label>'
                f'<input type="checkbox" id="sn-{ctx["slug"]}-{n}" class="sn-toggle">'
                f'<span class="sidenote"><sup>{n}</sup> {note}</span>')
    text = re.sub(r"\[\[sn:\s*(.+?)\]\]", sn, text, flags=re.S)
    return inline_basic(text)

def inline_basic(text):
    # inline code (protect), then links, bold, italic
    codes = []
    def stash(m):
        codes.append(m.group(1)); return f"\x00{len(codes)-1}\x00"
    text = re.sub(r"`([^`]+)`", stash, text)
    text = esc(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: f'<a href="{esc(m.group(2))}" target="_blank" rel="noopener">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{esc(codes[int(m.group(1))])}</code>", text)
    return text

def md_to_html(md, slug):
    ctx = {"slug": slug, "sn": 0, "fig": 0}
    lines = md.replace("\r\n", "\n").split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        # fenced code
        if line.strip().startswith("```"):
            lang = line.strip()[3:].strip()
            i += 1; buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append(f'<pre class="code" data-lang="{esc(lang)}"><code>{esc(chr(10).join(buf))}</code></pre>')
            continue
        # figure (own line/block, may span lines until closing ]])
        if line.strip().startswith("[[fig:"):
            buf = [line];
            while "]]" not in buf[-1] and i + 1 < n:
                i += 1; buf.append(lines[i])
            i += 1
            raw = " ".join(buf).strip()
            m = re.match(r"\[\[fig:\s*(.+?)\]\]\s*$", raw, flags=re.S)
            body = m.group(1) if m else raw[6:]
            if "||" in body:
                prompt, cap = body.split("||", 1)
            else:
                prompt, cap = body, ""
            ctx["fig"] += 1
            fname = f"{slug}-{ctx['fig']}.png"
            caph = inline_basic(cap.strip())
            out.append(
                f'<figure class="fig"><div class="fig-frame">'
                f'<img src="../figures/{fname}" alt="{esc(cap.strip()[:120])}" loading="lazy" '
                f'onerror="this.parentNode.classList.add(&#39;fig-missing&#39;);this.remove();" '
                f'data-fig="{fname}">'
                f'<span class="fig-ph">figure rendering &middot; {esc(cap.strip()[:70])}</span>'
                f'</div>' + (f'<figcaption>{caph}</figcaption>' if cap.strip() else '') + '</figure>')
            continue
        # heading
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1)); txt = inline(m.group(2).strip(), ctx)
            hid = re.sub(r"[^a-z0-9]+", "-", m.group(2).strip().lower()).strip("-")
            out.append(f'<h{lvl} id="{hid}">{txt}</h{lvl}>')
            i += 1; continue
        # blockquote
        if line.strip().startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip()); i += 1
            out.append(f'<blockquote>{inline(" ".join(buf), ctx)}</blockquote>')
            continue
        # table
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$", lines[i+1]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2; rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            th = "".join(f"<th>{inline(c, ctx)}</th>" for c in header)
            trs = "".join("<tr>" + "".join(f"<td>{inline(c, ctx)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(f'<div class="tbl-wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>')
            continue
        # unordered list
        if re.match(r"^\s*[-*]\s+", line):
            buf = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                buf.append(inline(re.sub(r"^\s*[-*]\s+", "", lines[i]), ctx)); i += 1
            out.append("<ul>" + "".join(f"<li>{x}</li>" for x in buf) + "</ul>")
            continue
        # ordered list
        if re.match(r"^\s*\d+\.\s+", line):
            buf = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                buf.append(inline(re.sub(r"^\s*\d+\.\s+", "", lines[i]), ctx)); i += 1
            out.append("<ol>" + "".join(f"<li>{x}</li>" for x in buf) + "</ol>")
            continue
        # blank
        if not line.strip():
            i += 1; continue
        # paragraph (gather until blank)
        buf = [line]; i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,4}\s|>|\s*[-*]\s|\s*\d+\.\s|```|\[\[fig:)", lines[i]):
            buf.append(lines[i]); i += 1
        out.append(f"<p>{inline(' '.join(buf), ctx)}</p>")
    return "\n".join(out)

# ============================================================ shell
def sidebar(active_slug, rel):
    rows = [f'<a class="sb-home" href="{rel}index.html">‹ kernel engineering</a>']
    for sec in MAN["sections"]:
        open_sec = any(a["slug"] == active_slug for a in sec["articles"])
        rows.append(f'<details class="sb-sec"{" open" if open_sec else ""}>')
        rows.append(f'<summary><span class="sb-num">{sec["num"]}</span> {esc(sec["title"])}</summary>')
        for a in sec["articles"]:
            act = " active" if a["slug"] == active_slug else ""
            chip = f'<span class="chip">{esc(a["chip"])}</span>' if a["chip"] else ""
            rows.append(f'<a class="sb-item{act}" href="{rel}a/{a["slug"]}.html">{esc(a["title"])}{chip}</a>')
        rows.append("</details>")
    return '<nav class="sidebar" id="sidebar">' + "".join(rows) + "</nav>"

def shell(title, main_html, active_slug=None, rel="", canvas="dark", extra_head=""):
    return f"""<!doctype html><html lang="en" data-theme="terminal"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(SITE['tagline'])}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,700;1,400&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{rel}assets/app.css">{extra_head}
</head><body class="canvas-{canvas}">
<div class="topbar">
  <a class="brand" href="{rel}index.html"><span class="logo">▚</span> Kernel&nbsp;Engineering</a>
  <button class="menu-btn" onclick="document.body.classList.toggle('sb-open')">☰</button>
  <div class="top-links">
    <button class="tbtn" id="search-open">Search <kbd>⌘K</kbd></button>
    <button class="tbtn" id="theme-btn">Terminal</button>
    <a class="tbtn enroll" href="{rel}workshop.html">Enroll →</a>
  </div>
</div>
<div class="layout">
{sidebar(active_slug, rel)}
<main class="content">{main_html}</main>
</div>
<div class="search-modal" id="search-modal"><div class="search-box">
  <input id="search-input" placeholder="Search articles, terms, kernels…" autocomplete="off">
  <div id="search-results"></div>
  <div class="search-hint">↑↓ to navigate · ↵ to open · esc to close</div>
</div></div>
<script>window.SEARCH_BASE="{rel}";</script>
<script src="{rel}assets/app.js"></script>
</body></html>"""

# ============================================================ pages
def build_article(a, idx):
    slug = a["slug"]
    md_path = ART / f"{slug}.md"
    if md_path.exists():
        body = md_to_html(md_path.read_text(), slug)
        stub = ""
    else:
        body = (f'<p class="lead">{esc(a["blurb"])}</p>'
                f'<div class="stub">This worklog is being written. It will follow the '
                f'hypothesis → measure → figure rhythm of the rest of the site.</div>')
        stub = " stub"
    prev_a = FLAT[idx-1] if idx > 0 else None
    next_a = FLAT[idx+1] if idx < len(FLAT)-1 else None
    nav = '<div class="prevnext">'
    nav += (f'<a class="pn prev" href="{prev_a["slug"]}.html"><span>‹ previous</span>{esc(prev_a["title"])}</a>'
            if prev_a else '<span></span>')
    nav += (f'<a class="pn next" href="{next_a["slug"]}.html"><span>next ›</span>{esc(next_a["title"])}</a>'
            if next_a else '<span></span>')
    nav += "</div>"
    chip = f'<span class="chip lg">{esc(a["chip"])}</span>' if a["chip"] else ""
    art = f"""<article class="worklog{stub}">
<div class="art-kicker"><span class="art-sec">{a['section_num']} · {esc(a['section_title'])}</span></div>
<h1 class="art-title">{esc(a['title'])} {chip}</h1>
<div class="art-body">{body}</div>
{nav}
</article>"""
    html_out = shell(f"{a['title']} · Kernel Engineering", art, active_slug=slug, rel="../", canvas="paper")
    (DOCS / "a" / f"{slug}.html").write_text(html_out)

def build_section(sec, rel="../"):
    cards = []
    for a in sec["articles"]:
        chip = f'<span class="chip">{esc(a["chip"])}</span>' if a["chip"] else ""
        cards.append(
            f'<a class="idx-item" href="{rel}a/{a["slug"]}.html">'
            f'<div class="idx-arrow">→</div><div class="idx-txt">'
            f'<div class="idx-title">{esc(a["title"])}{chip}</div>'
            f'<div class="idx-blurb">{esc(a["blurb"])}</div></div></a>')
    main = f"""<div class="section-page">
<div class="crumb">/ {sec['id']}</div>
<h1 class="sec-h1"><span class="sec-num">{sec['num']}</span> {esc(sec['title'])}</h1>
<p class="sec-blurb">{esc(sec['blurb'])}</p>
<div class="idx-list">{''.join(cards)}</div>
</div>"""
    html_out = shell(f"{sec['title']} · Kernel Engineering", main, rel=rel, canvas="dark")
    (DOCS / "s" / f"{sec['id']}.html").write_text(html_out)

def ascii_hero():
    return r"""      ▟█████▙   ▟███▙   ▟███▙   ▟█████▙
     ██╱   ╲██ ██╱ ╲██ ██╱ ╲██ ██╱   ╲
     ██  ▟█ ██ ████▛   ████▛   ██  ▟███
     ██  ╲█ ██ ██╱ ╲██ ██╱ ╲██ ██  ╲ ██
      ▜█████▛   ██  ██  ██  ██  ▜█████▛
     ┌─────────────────────────────────┐
     │ SM ▘▘▘▘  SM ▘▘▘▘  SM ▘▘▘▘  ×132  │
     │ ▝▝▝ tensor cores · 989 TF/s bf16 │
     │ ▓▓▓ HBM3 · 3.35 TB/s · 80 GB     │
     └─────────────────────────────────┘"""

def build_index():
    sec_cards = []
    for sec in MAN["sections"]:
        items = "".join(f'<li>{esc(a["title"])}</li>' for a in sec["articles"][:4])
        more = f'<li class="more">+{len(sec["articles"])-4} more →</li>' if len(sec["articles"]) > 4 else ""
        sec_cards.append(
            f'<a class="home-sec" href="s/{sec["id"]}.html">'
            f'<div class="hs-num">{sec["num"]}</div>'
            f'<div class="hs-title">{esc(sec["title"])}</div>'
            f'<div class="hs-blurb">{esc(sec["blurb"])}</div>'
            f'<ul class="hs-list">{items}{more}</ul></a>')
    start = MAN["sections"][0]["articles"]
    start_links = "".join(
        f'<a class="start-link" href="a/{a["slug"]}.html">→ {esc(a["title"])}</a>' for a in start)
    total = len(FLAT)
    main = f"""<div class="home">
<section class="hero">
  <pre class="ascii">{ascii_hero()}</pre>
  <div class="hero-txt">
    <div class="eyebrow">Vizuara AI Labs · knowledge base + live cohort</div>
    <h1>Write GPU kernels that run <span class="hl">modern LLMs</span>.</h1>
    <p class="sub">{esc(SITE['tagline'])} A {total}-article worklog from the silicon up to FlashAttention, NVFP4, DeepSeek's DSpark, and AI-generated kernels — each piece measured, profiled, and drawn by hand.</p>
    <div class="hero-cta">
      <a class="btn solid" href="workshop.html">The live workshop →</a>
      <a class="btn" href="a/the-three-regimes.html">Start reading</a>
    </div>
    <div class="start-strip">{start_links}</div>
  </div>
</section>
<section class="sec-grid">{''.join(sec_cards)}</section>
<section class="home-foot">
  <div class="skill-pitch">
    <h2>Built around what you're actually hired to do</h2>
    <p>Matmul from scratch to <strong>94% of cuBLAS</strong> · the same ladder on tensor cores · reading SASS &amp; Nsight Compute · TMA/WGMMA on Hopper · NVFP4 &amp; TMEM on Blackwell · Triton and real CUTLASS · FlashAttention · the vLLM debugging workflow · and LLM-driven kernel search — knowing where it wins and where it still fails.</p>
    <a class="btn" href="a/the-kernel-engineers-skill-map.html">The kernel engineer's skill map →</a>
  </div>
</section>
</div>"""
    (DOCS / "index.html").write_text(shell(f"{SITE['title']} · {esc(SITE['tagline'])}", main, rel="", canvas="dark"))

def build_workshop():
    lectures = [
        ("L1", "How fast can this go?", "The three regimes, the roofline, and a top-down tour of the silicon. Live: predict-then-measure PyTorch ops."),
        ("L2", "The CUDA programming model", "Grids, warps, SIMT, and the nvcc→PTX→SASS story. Live: your first kernels + GPU Puzzles."),
        ("L3", "The memory hierarchy in anger", "Coalescing, bank conflicts, occupancy. Live: the matrix-transpose ladder under Nsight Compute."),
        ("L4", "GEMM worklog I", "Kernels 1–4: naive (1.3%) to 1D block-tiling (36.5%). Hypothesis → profile → number, every step."),
        ("L5", "GEMM worklog II", "Kernels 5–10: 2D tiling, float4, autotuning, warptiling (93.7%). Live: the SASS '8 loads → 2 loads' moment."),
        ("L6", "Tensor cores, the second worklog", "mma.sync, fragments, swizzling, the precision menu. Live: a WMMA GEMM beating our best SIMT kernel."),
        ("L7", "Profiling &amp; debugging like a pro", "Nsight Compute deep-read + the vLLM workflow: sanitizer, core dumps, cuda-gdb. Live: 3 sabotaged kernels."),
        ("L8", "Attention: the kernel that ate the world", "Online softmax, FlashAttention v1 built live, why decode is GEMV. Capstone kickoff."),
    ]
    workshops = [
        ("W1", "FlashAttention from scratch", "Full forward pass, online-softmax rescaling, causal masking; FA2/FA3 ideas."),
        ("W2", "Beating cuBLAS on an H100", "TMA + WGMMA + warp specialization, assembled into a library-beating GEMM."),
        ("W3", "Triton → CUTLASS → TileLang", "The abstraction ladder: Triton in 40 lines, then CUTLASS the hard way."),
        ("W4", "Inference-serving kernels", "Prefill vs decode, PagedAttention, fusion, and quantized (FP8/W4A16) kernels."),
        ("W5", "Blackwell &amp; NVFP4", "tcgen05, Tensor Memory, CTA pairs, and the 2000µs→22µs FP4 GEMV journey."),
        ("W6", "DeepSeek, DSpark &amp; AI-written kernels", "FlashMLA/DeepGEMM, speculative decoding, KernelBench and the human+AI+profiler loop."),
    ]
    lec_html = "".join(
        f'<div class="lec"><div class="lec-tag">{t}</div><div><div class="lec-title">{ti}</div>'
        f'<div class="lec-desc">{d}</div></div></div>' for t, ti, d in lectures)
    wk_html = "".join(
        f'<div class="lec wk"><div class="lec-tag">{t}</div><div><div class="lec-title">{ti}</div>'
        f'<div class="lec-desc">{d}</div></div></div>' for t, ti, d in workshops)
    main = f"""<div class="section-page workshop">
<div class="crumb">/ the-live-workshop</div>
<h1 class="sec-h1">The Kernel Engineering Workshop</h1>
<p class="sec-blurb">Eight live foundational lectures and six deep-dive workshops on modern topics — from the three performance regimes to DeepSeek's DSpark and AI-generated kernels. Enrolled students get the full {len(FLAT)}-article knowledge base, the GPU-Puzzles track, worklog assignments, and the "You vs the machine" capstone.</p>
<h2 class="ws-h2">8 foundational live lectures <span class="ws-sub">2/week · 3 hours each · 4 weeks</span></h2>
<div class="lec-grid">{lec_html}</div>
<h2 class="ws-h2">6 deep-dive workshops <span class="ws-sub">modern kernel inference topics</span></h2>
<div class="lec-grid">{wk_html}</div>
<div class="ws-cta">
  <div class="ws-price">Dates &amp; pricing announced soon</div>
  <a class="btn solid" href="mailto:team@vizuara.com?subject=Kernel%20Engineering%20Workshop">Get notified / enquire →</a>
  <p class="ws-note">Questions? <a href="mailto:team@vizuara.com">team@vizuara.com</a></p>
</div>
</div>"""
    (DOCS / "workshop.html").write_text(shell("The Kernel Engineering Workshop · Vizuara", main, rel="", canvas="dark"))

def build_search_index():
    idx = []
    for a in FLAT:
        idx.append({"t": a["title"], "s": a["slug"], "sec": a["section_title"],
                    "chip": a["chip"], "b": a["blurb"], "u": f"a/{a['slug']}.html"})
    (DOCS / "search.json").write_text(json.dumps(idx, ensure_ascii=False))

# ============================================================ main
def main():
    (DOCS / "a").mkdir(parents=True, exist_ok=True)
    (DOCS / "s").mkdir(parents=True, exist_ok=True)
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    for src in (ROOT / "assets").glob("*"):
        shutil.copy(src, DOCS / "assets" / src.name)
    (DOCS / "CNAME").write_text(SITE["domain"] + "\n")
    (DOCS / ".nojekyll").write_text("")
    for i, a in enumerate(FLAT):
        build_article(a, i)
    for sec in MAN["sections"]:
        build_section(sec)
    build_index(); build_workshop(); build_search_index()
    written = len(FLAT) + len(MAN["sections"]) + 3
    have = sum(1 for a in FLAT if (ART / f"{a['slug']}.md").exists())
    print(f"built {written} pages · {len(FLAT)} articles ({have} written, {len(FLAT)-have} stubs) · "
          f"{len(MAN['sections'])} sections")

if __name__ == "__main__":
    main()
