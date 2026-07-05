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
MENTOR = json.loads((ROOT / "mentor_manifest.json").read_text())
MENTOR_DIR = ROOT / "mentor"
MFLAT = []
for _p in MENTOR["parts"]:
    for _c in _p["chapters"]:
        MFLAT.append({**_c, "part_id": _p["id"], "part_num": _p["num"], "part_title": _p["title"]})

# ----- flatten article order for prev/next + lookups -----
FLAT = []
for sec in MAN["sections"]:
    for a in sec["articles"]:
        FLAT.append({**a, "section_id": sec["id"], "section_num": sec["num"], "section_title": sec["title"]})
SLUG2IDX = {a["slug"]: i for i, a in enumerate(FLAT)}

# ============================================================ markdown
def esc(s): return html.escape(s, quote=False)

def inline(text, ctx):
    """Inline formatting. Sidenotes are stashed so inline_basic()'s escaping
    doesn't mangle their generated HTML, then restored afterwards."""
    stash = []
    def sn(m):
        ctx["sn"] += 1
        n = ctx["sn"]
        note = inline_basic(m.group(1).strip())
        stash.append(
            f'<label for="sn-{ctx["slug"]}-{n}" class="sn-ref">{n}</label>'
            f'<input type="checkbox" id="sn-{ctx["slug"]}-{n}" class="sn-toggle">'
            f'<span class="sidenote"><sup>{n}</sup> {note}</span>')
        return f"\x01{len(stash)-1}\x01"
    text = re.sub(r"\[\[sn:\s*(.+?)\]\]", sn, text, flags=re.S)
    text = inline_basic(text)
    text = re.sub(r"\x01(\d+)\x01", lambda m: stash[int(m.group(1))], text)
    return text

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
        # callout note  [[note: TYPE || content]]
        if line.strip().startswith("[[note:"):
            buf = [line]
            while "]]" not in buf[-1] and i + 1 < n:
                i += 1; buf.append(lines[i])
            i += 1
            raw = " ".join(buf).strip()
            mm = re.match(r"\[\[note:\s*(\w+)\s*\|\|\s*(.+?)\]\]\s*$", raw, flags=re.S)
            if mm:
                typ = mm.group(1).lower(); content = inline(mm.group(2).strip(), ctx)
            else:
                typ = "teach"; content = inline(raw[7:].strip().rstrip("]").strip(), ctx)
            cmeta = {"metaphor": ("🧠", "Metaphor"), "example": ("🔢", "By hand"),
                     "production": ("🏭", "In production today"), "teach": ("🎓", "Teaching note"),
                     "say": ("🎤", "Say this at the board"), "demo": ("▶️", "Live demo"),
                     "confusion": ("⚠️", "Where students trip"), "aha": ("✨", "The click")}
            icon, label = cmeta.get(typ, ("🎓", "Note"))
            out.append(f'<div class="cal cal-{typ}"><div class="cal-h"><span class="cal-i">{icon}</span> {label}</div>'
                       f'<div class="cal-b">{content}</div></div>')
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

def shell(title, main_html, active_slug=None, rel="", canvas="dark", extra_head="",
          with_sidebar=False, active_nav="", sb_kind="book"):
    def nl(href, label, key):
        return f'<a class="tn{" active" if key==active_nav else ""}" href="{rel}{href}">{label}</a>'
    topnav = (nl("book.html", "The Book", "book") + nl("workshop.html", "Workshop", "workshop")
              + nl("projects.html", "Projects", "projects") + nl("interactive.html", "Interactive", "interactive")
              + nl("mentor/index.html", "Mentor Guide", "mentor"))
    if with_sidebar:
        sb = mentor_sidebar(active_slug, rel) if sb_kind == "mentor" else sidebar(active_slug, rel)
    else:
        sb = ""
    layout_cls = "layout" if with_sidebar else "layout nosb"
    return f"""<!doctype html><html lang="en" data-theme="terminal"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(SITE['tagline'])}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,700;1,400&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{rel}assets/app.css">{extra_head}
</head><body class="canvas-{canvas}{' has-sb' if with_sidebar else ''}">
<div class="topbar">
  <a class="brand" href="{rel}index.html"><img class="brand-mark" src="{rel}assets/logo.png" alt="" onerror="this.style.display='none'"><span class="brand-txt">Vizuara <b>Kernel&nbsp;Engineering</b></span></a>
  <nav class="topnav">{topnav}</nav>
  {'<button class="menu-btn" onclick="document.body.classList.toggle(&#39;sb-open&#39;)">☰</button>' if with_sidebar else ''}
  <div class="top-links">
    <button class="tbtn icon" id="search-open" title="Search (⌘K)">⌘K</button>
    <button class="tbtn" id="theme-btn">Terminal</button>
    <a class="tbtn enroll" href="{rel}workshop.html">Enroll →</a>
  </div>
</div>
<div class="{layout_cls}">
{sb}
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
    html_out = shell(f"{a['title']} · Kernel Engineering", art, active_slug=slug, rel="../",
                     canvas="paper", with_sidebar=True, active_nav="book")
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
    html_out = shell(f"{sec['title']} · Kernel Engineering", main, rel=rel, canvas="dark",
                     with_sidebar=True, active_nav="book")
    (DOCS / "s" / f"{sec['id']}.html").write_text(html_out)

AREAS = [
    ("book.html", "01", "The Book", "The full knowledge base — a 72-chapter illustrated worklog from the silicon up to FlashAttention, NVFP4 and DeepSeek's DSpark. Free to read, forever.",
     f"{len(FLAT)} chapters · {236} figures"),
    ("workshop.html", "02", "The Workshop", "Vizuara's live Kernel Engineering cohort: 8 foundational lectures + 6 deep-dive workshops on modern kernel-inference topics, with the full book included.",
     "8 lectures · 6 workshops"),
    ("projects.html", "03", "Projects", "Build real kernels with your hands — the GPU-Puzzles track, a GEMM you take to 94% of cuBLAS, FlashAttention from scratch, and the You-vs-the-machine capstone.",
     "guided builds"),
    ("interactive.html", "04", "Interactive", "Practice, not just read: per-section quizzes, the guided GPU-Puzzles track, and a growing set of hands-on kernel challenges.",
     "quizzes · puzzles"),
]

def build_index():
    area_cards = "".join(
        f'<a class="area" href="{href}"><div class="area-num">{num}</div>'
        f'<div class="area-title">{esc(title)}</div>'
        f'<div class="area-blurb">{esc(blurb)}</div>'
        f'<div class="area-meta">{esc(meta)} <span class="area-go">→</span></div></a>'
        for href, num, title, blurb, meta in AREAS)
    main = f"""<div class="home">
<section class="hero2">
  <div class="hero2-logo"><img src="assets/logo.png" alt="Vizuara Kernel Engineering" onerror="this.style.display='none'"></div>
  <div class="eyebrow">Vizuara AI Labs</div>
  <h1>Vizuara <span class="hl">Kernel Engineering</span></h1>
  <p class="sub">{esc(SITE['tagline'])} A worklog from the silicon up to FlashAttention, NVFP4, DeepSeek's DSpark, and AI-generated kernels — every step measured, profiled, and drawn by hand.</p>
  <div class="hero-cta">
    <a class="btn solid" href="book.html">Read the book →</a>
    <a class="btn" href="workshop.html">The workshop</a>
  </div>
</section>
<section class="areas">{area_cards}</section>
<section class="home-foot">
  <div class="skill-pitch">
    <h2>Built around what you're actually hired to do</h2>
    <p>Matmul from scratch to <strong>94% of cuBLAS</strong> · the same ladder on tensor cores · reading SASS &amp; Nsight Compute · TMA/WGMMA on Hopper · NVFP4 &amp; TMEM on Blackwell · Triton and real CUTLASS · FlashAttention · the vLLM debugging workflow · and LLM-driven kernel search — knowing where it wins and where it still fails.</p>
    <a class="btn" href="a/the-kernel-engineers-skill-map.html">The kernel engineer's skill map →</a>
  </div>
</section>
</div>"""
    (DOCS / "index.html").write_text(shell(f"Vizuara Kernel Engineering · {esc(SITE['tagline'])}", main, rel="", canvas="dark", active_nav=""))

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
<div class="crumb">/ the-workshop</div>
<h1 class="sec-h1">Vizuara's Kernel Engineering Workshop</h1>
<p class="sec-blurb">Eight live foundational lectures and six deep-dive workshops on modern topics — from the three performance regimes to DeepSeek's DSpark and AI-generated kernels. Enrolled students get the complete {len(FLAT)}-chapter <a href="book.html">book</a>, the <a href="interactive.html">GPU-Puzzles track and quizzes</a>, the <a href="projects.html">guided projects</a>, worklog assignments, and the "You vs the machine" capstone.</p>
<h2 class="ws-h2">8 foundational live lectures <span class="ws-sub">2/week · 3 hours each · 4 weeks</span></h2>
<div class="lec-grid">{lec_html}</div>
<h2 class="ws-h2">6 deep-dive workshops <span class="ws-sub">modern kernel inference topics</span></h2>
<div class="lec-grid">{wk_html}</div>

<section id="partners" class="partners">
  <div class="eyebrow">For companies &amp; partners</div>
  <h2 class="ws-h2" style="margin-top:6px">Hire our graduates. Partner on the frontier.</h2>
  <p class="int-p">Kernel engineers are one of the hardest hires in AI. By the end of a cohort, our graduates have built a GEMM from naive to <b style="color:var(--lime)">94% of cuBLAS</b>, FlashAttention from scratch, Hopper &amp; Blackwell kernels (TMA, WGMMA, NVFP4), and DeepSeek-grade inference kernels. Three ways to work with us:</p>
  <div class="proj-grid">
    <div class="proj"><div class="proj-top"><span class="proj-level lvl-capstone">hiring</span></div><h3>A hiring pipeline</h3><p>Consider the strongest of each cohort for your kernel-engineering roles: a warm, pre-vetted pool of engineers with exactly the skills on your job descriptions.</p></div>
    <div class="proj"><div class="proj-top"><span class="proj-level lvl-capstone">capstone</span></div><h3>Sponsor a capstone</h3><p>Give us a real kernel problem your team cares about. We run it as a sponsored capstone, you see the solutions and the talent, and your company is credited on the project.</p></div>
    <div class="proj"><div class="proj-top"><span class="proj-level lvl-capstone">compute</span></div><h3>GPU credits &amp; partnership</h3><p>Sponsor H100/B200 hours for students' capstone work and become a founding partner, with your logo on the workshop, the site, and the certificate.</p></div>
  </div>
  <div class="partners-strip"><span class="ps-label">Founding partners</span><span class="ps-soon">— announced soon —</span></div>
</section>

<div class="ws-cta">
  <div class="ws-price">Partner with the world's first Kernel Engineering Workshop</div>
  <a class="btn solid" href="mailto:team@vizuara.com?subject=Kernel%20Engineering%20Workshop%20partnership&amp;body=Hi%20Raj%2C">Partner with us →</a>
  <p class="ws-note">Raj Dandekar, Co-founder &amp; CEO · <a href="mailto:team@vizuara.com">team@vizuara.com</a> · dates &amp; pricing announced soon</p>
</div>
</div>"""
    (DOCS / "workshop.html").write_text(shell("Vizuara's Kernel Engineering Workshop", main, rel="", canvas="dark", active_nav="workshop"))

def build_book():
    chapters = []
    for sec in MAN["sections"]:
        arts = "".join(
            f'<a href="a/{a["slug"]}.html" class="ch-art">{esc(a["title"])}'
            + (f'<span class="chip">{esc(a["chip"])}</span>' if a["chip"] else "") + '</a>'
            for a in sec["articles"])
        chapters.append(
            f'<div class="chapter"><div class="ch-side"><div class="ch-num">{sec["num"]}</div>'
            f'<a class="ch-title" href="s/{sec["id"]}.html">{esc(sec["title"])}</a>'
            f'<div class="ch-count">{len(sec["articles"])} chapters ›</div></div>'
            f'<div class="ch-body"><p class="ch-blurb">{esc(sec["blurb"])}</p>'
            f'<div class="ch-arts">{arts}</div></div></div>')
    main = f"""<div class="section-page book-page">
<div class="crumb">/ the-book</div>
<h1 class="sec-h1">The Kernel Engineering Book</h1>
<p class="sec-blurb">The complete, free knowledge base behind Vizuara's Kernel Engineering — {len(FLAT)} illustrated worklog chapters across seven parts, each written in the hypothesis → measure → figure rhythm and cross-linked to the chapters it needs. Start anywhere.</p>
<a class="btn" href="a/how-to-use-this-site.html" style="margin-bottom:8px;display:inline-block">How to read this book →</a>
<div class="chapters">{''.join(chapters)}</div>
</div>"""
    (DOCS / "book.html").write_text(shell("The Kernel Engineering Book · Vizuara", main, rel="", canvas="dark", active_nav="book"))

PROJECTS = [
    ("GPU Puzzles: the on-ramp", "beginner", "Solve Sasha Rush's 14 GPU-Puzzles to internalise the one-thread-one-element model, indexing, guards, shared memory and reductions — the fastest way from zero to writing correct kernels.",
     [("Walkthrough I", "a/gpu-puzzles-walkthrough-1.html"), ("Walkthrough II", "a/gpu-puzzles-walkthrough-2.html")]),
    ("GEMM to 94% of cuBLAS", "core", "Build matrix-multiply from a 1.3%-of-cuBLAS naive kernel up the full ten-step ladder — coalescing, SMEM tiling, register tiling, vectorization, autotuning, warptiling — profiling every step.",
     [("Start: Kernel 1 (naive)", "a/gemm-kernel-1-naive.html"), ("The ladder, end to end", "a/gemm-recap-the-ladder.html")]),
    ("Matmul on tensor cores", "core", "Rebuild GEMM a second time on the tensor cores: wmma fragments, SMEM swizzling to kill bank conflicts, and mma.sync to library-class speed.",
     [("Tensor cores I: WMMA", "a/tc-kernel-1-wmma-intro.html"), ("Tensor cores III: fast", "a/tc-kernel-3-mma-sync-fast.html")]),
    ("FlashAttention from scratch", "advanced", "Fuse the whole attention into one kernel with online softmax so the N×N scores never touch HBM — then benchmark it against PyTorch's SDPA.",
     [("FlashAttention I", "a/flashattention-1.html"), ("Online softmax", "a/softmax-from-scratch.html")]),
    ("Beat cuBLAS on an H100", "expert", "Assemble TMA, WGMMA and warp specialization into a Hopper GEMM that matches or beats NVIDIA's own library — the full frontier worklog.",
     [("Beating cuBLAS on H100", "a/beating-cublas-on-h100.html"), ("WGMMA & warp specialization", "a/hopper-wgmma-warp-specialization.html")]),
    ("You vs. the machine (capstone)", "capstone", "Pick a kernel (SwiGLU, a FlashAttention variant, histogram…), optimize it BY HAND, then run an LLM-in-the-loop against your own kernel and document — CS149 × KernelBench — what each found that the other missed.",
     [("KernelBench & fast_p", "a/kernelbench-and-fast-p.html"), ("The SwiGLU kernel", "a/swiglu-kernel.html")]),
    ("Debug a broken kernel", "skill", "Take a kernel with a race, a misaligned vector load and a silent NaN, and hunt each down with compute-sanitizer, user-triggered core dumps, cuda-gdb and nvdisasm — the vLLM workflow.",
     [("The vLLM debugging workflow", "a/debugging-kernels-vllm-workflow.html")]),
]

def build_projects():
    cards = ""
    for title, level, what, links in PROJECTS:
        lk = " ".join(f'<a href="{h}">{esc(l)} →</a>' for l, h in links)
        cards += (f'<div class="proj"><div class="proj-top"><span class="proj-level lvl-{level}">{level}</span></div>'
                  f'<h3>{esc(title)}</h3><p>{esc(what)}</p><div class="proj-links">{lk}</div></div>')
    main = f"""<div class="section-page">
<div class="crumb">/ projects</div>
<h1 class="sec-h1">Projects</h1>
<p class="sec-blurb">Reading is not enough — kernels are learned by writing them. Each project is a build you do with your hands, pointed at the exact chapters that carry it. Work top to bottom, or jump to your level. Every one produces something you can put in a worklog and show an employer.</p>
<div class="proj-grid">{cards}</div>
<div class="ws-cta">
  <div class="ws-price">Want these reviewed live and a certificate?</div>
  <a class="btn solid" href="workshop.html">See Vizuara's Kernel Engineering Workshop →</a>
</div>
</div>"""
    (DOCS / "projects.html").write_text(shell("Projects · Vizuara Kernel Engineering", main, rel="", canvas="dark", active_nav="projects"))

QUIZ = [
    ("A kernel achieves 4% of the GPU's peak FLOP/s and near-peak HBM bandwidth. What is it?",
     ["Compute-bound", "Memory-bandwidth-bound", "Overhead-bound"], 1,
     "Near-peak bandwidth with tiny FLOP utilisation is the signature of a memory-bound kernel — adding faster math won't help."),
    ("The H100's tensor cores do ~989 TFLOP/s bf16 and HBM3 gives ~3.35 TB/s. Roughly what arithmetic intensity must a kernel exceed to be compute-bound?",
     ["~3 FLOPs/byte", "~30 FLOPs/byte", "~295 FLOPs/byte"], 2,
     "989e12 / 3.35e12 ≈ 295 FLOPs per byte — the ridge point. Below it you're memory-bound no matter what."),
    ("In the GEMM ladder, the jump from naive (1.3%) to 8.5% of cuBLAS comes from…",
     ["Shared-memory tiling", "Global-memory coalescing", "Tensor cores"], 1,
     "Kernel 2 is a one-line thread-index remap so a warp reads contiguous columns — coalesced access, ~6× for free."),
    ("Why must GEMM block over the K dimension when using shared memory?",
     ["To improve numerical accuracy", "Because SMEM is too small to hold full rows/columns", "To avoid warp divergence"], 1,
     "SMEM is ~228 KiB; you can't stage whole K-length rows/cols, so you tile K and accumulate across tiles."),
    ("A float4 / LDS.128 load moves 128 bits per instruction. Its main benefit over four scalar loads is…",
     ["More total bytes moved", "Fewer instruction issues / transactions for the same bytes", "Higher numerical precision"], 1,
     "Same bytes, a quarter of the instructions — a win precisely when you're issue-bound, as in kernel 6."),
    ("On an H100, shared memory has 32 banks. A bank conflict happens when…",
     ["Two warps use the same SM", "Lanes in a warp hit the same bank but different words", "You exceed 228 KiB of SMEM"], 1,
     "Multiple lanes addressing different words in the same bank serialise; padding (+1) or swizzling fixes it."),
    ("Why is LLM decode (one token at a time) usually memory-bound?",
     ["It runs huge GEMMs", "It re-reads the whole KV cache per step for little math (a GEMV)", "It uses too many registers"], 1,
     "Decode is a skinny mat-vec dominated by reading the KV cache from HBM — the opposite regime from prefill."),
    ("FlashAttention's core trick is…",
     ["Storing the N×N scores in HBM in FP8", "Online softmax so the N×N scores never materialise in HBM", "Skipping the softmax entirely"], 1,
     "Tiling + online-softmax rescaling fuses attention into one kernel; traffic scales like N·d, not N²."),
    ("What is new in Hopper (sm_90a) that Ampere lacks?",
     ["CUDA cores", "TMA, WGMMA, DSMEM and thread-block clusters", "The L2 cache"], 1,
     "Hopper adds the Tensor Memory Accelerator, warpgroup MMA, distributed shared memory and clusters."),
    ("DeepSeek-V4-Pro-DSpark is best described as…",
     ["A brand-new base model", "The V4-Pro checkpoint plus a speculative-decoding module", "A new GPU architecture"], 1,
     "DSpark is a speculative-decoding module bolted onto V4-Pro — draft tokens verified in parallel; a kernels problem."),
    ("In Stanford CRFM's AI-generated-kernel experiments, results were…",
     ["Uniformly superhuman", "Great on some (LayerNorm 484% of PyTorch) but poor on others (FlashAttention 9%)", "Never correct"], 1,
     "Branching search shone on less-tuned FP32 ops but still trailed badly on hard ones like FlashAttention."),
    ("Which is the right first move when a kernel hangs and Ctrl-C does nothing?",
     ["Reboot the machine", "Trigger a user CUDA core dump and open it in cuda-gdb", "Delete the kernel"], 1,
     "The vLLM workflow: CUDA_ENABLE_USER_TRIGGERED_COREDUMP via a named pipe, then cuda-gdb on the dump."),
]

def build_interactive():
    qs = ""
    for i, (q, opts, correct, exp) in enumerate(QUIZ):
        obtns = "".join(f'<button class="q-opt" data-i="{j}">{esc(o)}</button>' for j, o in enumerate(opts))
        qs += (f'<div class="quiz-q" data-correct="{correct}"><div class="q-num">Q{i+1}</div>'
               f'<div class="q-text">{esc(q)}</div><div class="q-opts">{obtns}</div>'
               f'<div class="q-exp">{esc(exp)}</div></div>')
    main = f"""<div class="section-page interactive-page">
<div class="crumb">/ interactive</div>
<h1 class="sec-h1">Interactive</h1>
<p class="sec-blurb">Practice, not just reading. Test yourself against the core ideas, then work the guided GPU-Puzzles track. More hands-on kernel challenges are landing here as the workshop grows.</p>

<div class="int-panel">
  <div class="int-head"><h2 class="ws-h2">The GPU-Puzzles track</h2></div>
  <p class="int-p">Fourteen tiny puzzles that build the kernel-writer's instincts — map, zip, guards, broadcasting, shared memory, pooling, dot product, convolution, prefix sum and a first matmul. Do them in the browser, then read our walkthroughs.</p>
  <div class="int-links">
    <a class="btn" href="https://github.com/srush/GPU-Puzzles" target="_blank" rel="noopener">Open GPU-Puzzles ↗</a>
    <a class="btn" href="a/gpu-puzzles-walkthrough-1.html">Walkthrough I →</a>
    <a class="btn" href="a/gpu-puzzles-walkthrough-2.html">Walkthrough II →</a>
  </div>
</div>

<div class="int-panel">
  <div class="int-head"><h2 class="ws-h2">Quiz yourself</h2><span class="quiz-score" id="quiz-score">0 / {len(QUIZ)}</span></div>
  <p class="int-p">Twelve questions spanning the regimes, the GEMM ladder, inference kernels and the frontier. Pick an answer to see the explanation.</p>
  <div class="quiz" data-total="{len(QUIZ)}">{qs}</div>
</div>
</div>"""
    (DOCS / "interactive.html").write_text(shell("Interactive · Vizuara Kernel Engineering", main, rel="", canvas="dark", active_nav="interactive"))

def mentor_sidebar(active_slug, rel):
    rows = [f'<a class="sb-home" href="{rel}mentor/index.html">‹ mentor handbook</a>']
    for part in MENTOR["parts"]:
        open_p = any(c["slug"] == active_slug for c in part["chapters"])
        rows.append(f'<details class="sb-sec"{" open" if open_p else ""}>')
        rows.append(f'<summary><span class="sb-num">{part["num"]}</span> {esc(part["title"])}</summary>')
        for c in part["chapters"]:
            act = " active" if c["slug"] == active_slug else ""
            rows.append(f'<a class="sb-item{act}" href="{rel}mentor/{c["slug"]}.html">{esc(c["title"])}</a>')
        rows.append("</details>")
    return '<nav class="sidebar" id="sidebar">' + "".join(rows) + "</nav>"

def build_mentor_chapter(ch, idx):
    slug = ch["slug"]
    p = MENTOR_DIR / f"{slug}.md"
    if p.exists():
        body = md_to_html(p.read_text(), slug); stub = ""
    else:
        body = (f'<p class="lead">{esc(ch["blurb"])}</p><div class="stub">This handbook chapter is being '
                f'written — it will teach this from scratch with metaphors, a live-demo plan, and figures.</div>')
        stub = " stub"
    prev_c = MFLAT[idx-1] if idx > 0 else None
    next_c = MFLAT[idx+1] if idx < len(MFLAT)-1 else None
    nav = '<div class="prevnext">'
    nav += (f'<a class="pn prev" href="{prev_c["slug"]}.html"><span>‹ previous</span>{esc(prev_c["title"])}</a>'
            if prev_c else '<span></span>')
    nav += (f'<a class="pn next" href="{next_c["slug"]}.html"><span>next ›</span>{esc(next_c["title"])}</a>'
            if next_c else '<span></span>')
    nav += "</div>"
    art = f"""<article class="worklog mentor{stub}">
<div class="art-kicker"><span class="art-sec">Mentor Handbook · {ch['part_num']} {esc(ch['part_title'])}</span></div>
<h1 class="art-title">{esc(ch['title'])}</h1>
<div class="art-body">{body}</div>
{nav}
</article>"""
    html_out = shell(f"{ch['title']} · Mentor Handbook", art, active_slug=slug, rel="../",
                     canvas="paper", with_sidebar=True, active_nav="mentor", sb_kind="mentor")
    (DOCS / "mentor" / f"{slug}.html").write_text(html_out)

def build_mentor_index():
    parts = []
    for part in MENTOR["parts"]:
        chs = "".join(
            f'<a href="{c["slug"]}.html" class="ch-art">{esc(c["title"])}</a>' for c in part["chapters"])
        parts.append(
            f'<div class="chapter"><div class="ch-side"><div class="ch-num">{part["num"]}</div>'
            f'<div class="ch-title">{esc(part["title"])}</div>'
            f'<div class="ch-count">{len(part["chapters"])} chapters</div></div>'
            f'<div class="ch-body"><p class="ch-blurb">{esc(part["blurb"])}</p>'
            f'<div class="ch-arts">{chs}</div></div></div>')
    total = len(MFLAT)
    main = f"""<div class="section-page book-page mentor-index">
<div class="crumb">/ mentor-handbook</div>
<div class="mentor-badge">For mentors · Dr. Raj Dandekar &amp; Shubham Panchal</div>
<h1 class="sec-h1">The Mentor's Handbook</h1>
<p class="sec-blurb">{esc(MENTOR['subtitle'])} Every chapter teaches the idea from the ground up — plain words, a metaphor, a by-hand example, the real math, where it runs in production today, and a minute-by-minute plan for teaching it. Read it in order; by the end you can deliver the entire workshop. {total} chapters.</p>
<a class="btn" href="mg-how-to-use-this-guide.html" style="margin-bottom:6px;display:inline-block">Start: how to use this handbook →</a>
<div class="chapters">{''.join(parts)}</div>
</div>"""
    (DOCS / "mentor" / "index.html").write_text(shell("The Mentor's Handbook · Vizuara Kernel Engineering",
        main, rel="../", canvas="dark", active_nav="mentor"))

def build_search_index():
    idx = []
    for a in FLAT:
        idx.append({"t": a["title"], "s": a["slug"], "sec": a["section_title"],
                    "chip": a["chip"], "b": a["blurb"], "u": f"a/{a['slug']}.html"})
    for c in MFLAT:
        idx.append({"t": c["title"], "s": c["slug"], "sec": "Mentor Handbook · " + c["part_title"],
                    "chip": "MENTOR", "b": c["blurb"], "u": f"mentor/{c['slug']}.html"})
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
    (DOCS / "mentor").mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(MFLAT):
        build_mentor_chapter(c, i)
    build_mentor_index()
    build_index(); build_book(); build_projects(); build_interactive()
    build_workshop(); build_search_index()
    written = len(FLAT) + len(MAN["sections"]) + len(MFLAT) + 7
    have = sum(1 for a in FLAT if (ART / f"{a['slug']}.md").exists())
    mhave = sum(1 for c in MFLAT if (MENTOR_DIR / f"{c['slug']}.md").exists())
    print(f"built {written} pages · {len(FLAT)} articles ({have} written) · "
          f"{len(MFLAT)} mentor chapters ({mhave} written) · {len(MAN['sections'])} sections")

if __name__ == "__main__":
    main()
