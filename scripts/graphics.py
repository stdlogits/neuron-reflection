"""Show-not-tell graphics of the main findings, from ``data/summary/``, and the README's hero, method and chart frames.

    python scripts/graphics.py               # everything: the graphics, the README's hero, method and chart frames
    python scripts/graphics.py fuse dial     # some

Each graphic is an HTML page with inline SVG, screenshot at twice its size by headless Chrome (``$CHROME``, else Chrome
or Chrome Beta in /Applications). The look: a white page, soft grey cards, Wix Madefor (and Noto Sans SC for one
glyph, both from Google Fonts), blue for the AI and orange for the cat, and the project's characters
(``scripts/cast.py``).
The graphics go to ``figures/graphics/`` as PNGs for posting and JPEGs for the README (indexed with their claims in
``figures/README.md``); the hero, the method and the frames around the pele charts go to ``docs/images/`` as JPEGs. A
chart frame needs pele's SVG first:

    python scripts/charts.py && node docs/figures/render.mjs
"""
import argparse
import html
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "graphics"
IMAGES = ROOT / "docs" / "images"
sys.path.insert(0, str(ROOT / "scripts"))

from cast import char, robot, shield  # noqa: E402

PAPER, WASH, WASH2, LINE = "#FFFFFF", "#F4F5F7", "#EBEDF1", "#D9DDE3"
INK, INK2, INK3, MUTED = "#0E1117", "#464F61", "#636B7D", "#A3A8B4"
BLUE, ORANGE = "#2F6FD6", "#E8781E"  # the AI and the cat
ROSE, PINKC, TEAL, GOLD = "#E34958", "#E0709E", "#1E9EAA", "#D9961E"
CELL = "#E6E9EE"  # an empty cell in a card
W = 1080
FONTS = ("https://fonts.googleapis.com/css2?family=Wix+Madefor+Display:wght@600;700;800"
         "&family=Wix+Madefor+Text:wght@400;500;600;700&display=block")
CJK = "https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@500&display=block&text=%E7%8B%B8"  # the k = 4 reply's 狸
CREDIT = "a9lim, claude, &amp; treetowntree"


def rec(name: str) -> dict:
    return json.loads((ROOT / "data" / "summary" / f"{name}.json").read_text())


def mean(x):
    return sum(x) / len(x)


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def heat(color: str, share: float) -> str:
    """``color`` at ``share`` strength over an empty cell, for heatmaps."""
    return f"color-mix(in oklch, {color} {round(12 + 88 * min(1.0, share))}%, {CELL})"


# ---------------------------------------------------------------- the stage

CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: transparent; }}
body {{ font-family: 'Wix Madefor Text', system-ui, sans-serif; color: {INK}; font-variant-numeric: tabular-nums;
       font-feature-settings: 'kern', 'liga'; -webkit-font-smoothing: antialiased; }}
.stage {{ position: relative; width: 100%; height: 100%; border-radius: 40px; overflow: hidden; background: {PAPER};
         box-shadow: inset 0 0 0 1.5px #E3E6EB; }}
.page {{ position: absolute; inset: 0; padding: 60px 60px 40px; display: flex; flex-direction: column; gap: 30px; }}
main {{ flex: 1; display: flex; flex-direction: column; justify-content: space-evenly; gap: 26px; }}
.kicker {{ font: 700 17px 'Wix Madefor Text'; letter-spacing: 0.08em; text-transform: uppercase; color: {BLUE}; }}
h1 {{ font: 800 58px/1.04 'Wix Madefor Display'; letter-spacing: -0.03em; margin-top: 10px; }}
.rule {{ width: 132px; height: 3px; border-radius: 2px; margin-top: 20px; background: linear-gradient(90deg, {BLUE}, {ORANGE}); }}
.sub {{ font: 400 23px/1.4 'Wix Madefor Text'; color: {INK3}; margin-top: 16px; max-width: 940px; text-wrap: pretty; }}
.sub b {{ color: {INK}; font-weight: 600; }}
footer {{ display: flex; justify-content: space-between; gap: 24px; font: 500 15px 'Wix Madefor Text'; color: {MUTED}; }}
svg text {{ font-family: 'Wix Madefor Text'; fill: {INK3}; }}
.card {{ background: {WASH}; border-radius: 26px; padding: 28px; }}
.label {{ font: 600 17px 'Wix Madefor Text'; color: {INK3}; letter-spacing: 0.01em; }}
.chat {{ display: flex; flex-direction: column; gap: 14px; }}
.row {{ display: flex; gap: 16px; align-items: flex-end; }}
.row.q {{ justify-content: flex-end; }}
.b {{ font: 400 26px/1.34 'Wix Madefor Text'; padding: 16px 24px; border-radius: 26px; max-width: var(--bw, 800px);
     text-wrap: pretty; }}
.q .b {{ background: {WASH2}; color: {INK}; border-bottom-right-radius: 8px; }}
.a .b {{ background: {WASH}; color: {INK}; border-bottom-left-radius: 8px; }}
.a .b.hot {{ background: color-mix(in oklch, {BLUE} 11%, {PAPER}); }}
.who {{ font: 600 16px 'Wix Madefor Text'; letter-spacing: 0.01em; color: {INK3}; margin: 0 0 7px 6px; }}
.hot-who {{ color: {BLUE}; }}
mark {{ background: none; font-weight: 700; color: {INK}; }}
.avatar {{ flex: none; }}
.pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 7px 16px; border-radius: 999px;
        background: {PAPER}; border: 1.5px solid {LINE}; font: 600 19px 'Wix Madefor Text'; color: {INK2}; }}
.cjk {{ font-family: 'Noto Sans SC'; }}
"""


def head(title: str, kicker: str = "", sub: str = "") -> str:
    k = f'<div class="kicker">{kicker}</div>' if kicker else ""
    s = f'<p class="sub">{sub}</p>' if sub else ""
    return f'<header>{k}<h1>{title}</h1><div class="rule"></div>{s}</header>'


def doc(title: str, body: str, width: int, height: int, extra_css: str = "") -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(re.sub("<[^>]+>", "", title))}</title>
<link href="{FONTS}" rel="stylesheet"><link href="{CJK}" rel="stylesheet"><style>{CSS}{extra_css}</style></head>
<body style="width:{width}px;height:{height}px"><div class="stage">{body}</div></body></html>"""


def page(title: str, body: str, height: int = 1350, kicker: str = "", sub: str = "") -> tuple[str, int, int]:
    inner = (f'<div class="page">{head(title, kicker, sub)}<main>{body}</main>'
             f'<footer><span>Neuron Reflection · Qwen3-8B</span><span>{CREDIT}</span></footer></div>')
    return doc(title, inner, W, height), W, height


def chrome() -> str:
    for c in (os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"):
        if c and Path(c).exists():
            return c
    raise SystemExit("no Chrome found: set $CHROME")


def render(out: Path, page_: tuple[str, int, int], png: bool = True):
    """Screenshot ``page_`` at twice its size. Writes ``out`` as a PNG (with transparent rounded corners, for posting)
    when ``png``, and always a JPEG beside it, 1,600 px wide on white, which the README shows: a fraction of
    the size, and an image format every GitHub client draws."""
    from PIL import Image
    html_, width, height = page_
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src, shot = Path(tmp) / f"{out.stem}.html", Path(tmp) / f"{out.stem}.png"
        src.write_text(html_)
        subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        "--default-background-color=00000000", f"--window-size={width},{height}", "--virtual-time-budget=15000",
                        f"--screenshot={shot}", src.as_uri()], check=True, capture_output=True)
        im = Image.open(shot).convert("RGBA")
        if png:
            im.save(out.with_suffix(".png"), optimize=True)
        flat = Image.new("RGBA", im.size, PAPER)
        flat.alpha_composite(im)
        flat = flat.convert("RGB").resize((1600, round(1600 * im.height / im.width)), Image.LANCZOS)
        flat.save(out.with_suffix(".jpg"), quality=88, optimize=True, progressive=True)
    if os.environ.get("KEEP_HTML"):
        out.with_suffix(".html").write_text(html_)
    print("wrote " + ", ".join(str(out.with_suffix(x).relative_to(ROOT)) for x in ((".png", ".jpg") if png else (".jpg",))))


# ---------------------------------------------------------------- pieces

def bubble_q(text: str) -> str:
    return f'<div class="row q"><div class="b">{text}</div></div>'


def bubble_a(text: str, who: str = "", avatar: str = "", hot: bool = False) -> str:
    av = f'<div class="avatar">{avatar}</div>' if avatar else ""
    lab = f'<div class="who{" hot-who" if hot else ""}">{who}</div>' if who else ""
    return f'<div class="row a">{av}<div>{lab}<div class="b{" hot" if hot else ""}">{text}</div></div></div>'


def text(s: str, x: float, y: float, size: float = 18, weight: int = 500, fill: str = INK3, anchor: str = "start",
         extra: str = "") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" style="fill:{fill}" '
            f'text-anchor="{anchor}" {extra}>{s}</text>')


def arrowhead(id_: str, color: str, size: float = 22) -> str:
    return (f'<marker id="{id_}" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="{size}" markerHeight="{size}" '
            f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')


GLOW = '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5"/></filter>'


def glowing(shape: str, opacity: float = 0.7) -> str:
    """``shape`` as is: the white page needs no glow (kept as one place to add emphasis)."""
    return shape


def at(svg: str, x: float, y: float, size: float) -> str:
    """A character's ``<svg>`` placed with its center at (x, y) inside another SVG."""
    return svg.replace("<svg ", f'<svg x="{x - size / 2:.1f}" y="{y - size / 2:.1f}" ', 1)


def text_width(s: str, size: float) -> float:
    """About how wide ``s`` sets in Wix Madefor Text semibold, from each letter's rough advance (em)."""
    em = 0.0
    for ch in s:
        em += (0.27 if ch in " .,:;'!|“”‘’()" else 0.3 if ch in "ijlI" else 0.4 if ch in "frt" else 0.84 if ch in "mwMW"
               else 0.66 if ch.isupper() else 0.58 if ch.isdigit() else 0.55)
    return em * size


def pill_svg(label: str, x: float, y: float, color: str = INK2, size: float = 19, pad: float = 16) -> str:
    """A label in a dark pill, centered at (x, y), for a word."""
    w = text_width(label, size) + 2 * pad
    return (f'<rect x="{x - w / 2:.1f}" y="{y - size * 0.95:.1f}" width="{w:.1f}" height="{size * 1.9:.1f}" rx="{size * 0.95:.1f}" '
            f'fill="{PAPER}" stroke="{LINE}" stroke-width="1.5"/>' + text(esc(label), x, y + size * 0.36, size, 600, color, "middle"))


def plot_axes(p: list, l, r_, t, b, pw, ph, ys, fmt=lambda v: f"{v:g}"):
    for v in ys:
        y = ys[v]
        p.append(f'<line x1="{l}" y1="{y:.1f}" x2="{pw - r_}" y2="{y:.1f}" stroke="{LINE}" stroke-width="1.2"/>')
        p.append(text(fmt(v), l - 12, y + 6, 16, 500, INK3, "end"))


def series(pts, col, r=6.5) -> str:
    line = (f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{col}" stroke-width="4.5" '
            f'stroke-linejoin="round" stroke-linecap="round"/>')
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}" stroke="{PAPER}" stroke-width="2.5"/>' for x, y in pts)
    return glowing(line, 0.55) + dots


# ---------------------------------------------------------------- 1. the edit

def g_mirror():
    """Findings §2-§3: mirroring the neurons that carry mother − father."""
    R = rec("repro-mother-father")["conditions"]
    q = R["baseline"]["replies"][10]
    before, after = q["reply"], R["lens[en,zh]"]["replies"][10]["reply"]
    pb, pe = R["baseline"]["probes"]["parent-en"], R["lens[en,zh]"]["probes"]["parent-en"]
    layers = [int(L) for L in pb]

    # the reflection, drawn: the mirror is the plane orthogonal to mother − father
    w_, h_, cx, cy = 958, 470, 479, 238
    s = [f'<defs>{GLOW}{arrowhead("ab", BLUE)}{arrowhead("ao", ORANGE)}{arrowhead("ag", MUTED, 14)}{arrowhead("ai", INK3)}'
         f'<linearGradient id="mir" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{BLUE}" stop-opacity="0.9"/>'
         f'<stop offset="1" stop-color="{BLUE}" stop-opacity="0"/></linearGradient></defs>']
    s.append(glowing(f'<rect x="{cx - 3}" y="40" width="6" height="{h_ - 70}" rx="3" fill="url(#mir)"/>', 0.8))
    s.append(glowing(f'<ellipse cx="{cx}" cy="34" rx="15" ry="21" fill="#EAF4FF" stroke="{BLUE}" stroke-width="3"/>', 0.9))
    s.append(text("the mirror", cx + 26, 40, 17, 600, BLUE, "start"))
    s.append(f'<line x1="150" y1="{cy}" x2="{w_ - 150}" y2="{cy}" stroke="{INK3}" stroke-width="3" marker-end="url(#ai)" marker-start="url(#ai)"/>')
    s.append(at(char("father", 116), 76, cy - 10, 116) + at(char("mother", 116), w_ - 76, cy - 10, 116))
    s.append(pill_svg("father", 76, cy + 70, TEAL) + pill_svg("mother", w_ - 76, cy + 70, PINKC))
    # a read row (blue) and a write column (amber), and their mirror images
    for (vx, vy, col, mk, lab) in ((240, -140, BLUE, "ab", "reads"), (165, 130, ORANGE, "ao", "writes")):
        s.append(glowing(f'<line x1="{cx}" y1="{cy}" x2="{cx + vx}" y2="{cy + vy}" stroke="{col}" stroke-width="7" stroke-linecap="round" marker-end="url(#{mk})"/>', 0.5))
        s.append(f'<line x1="{cx}" y1="{cy}" x2="{cx - vx}" y2="{cy + vy}" stroke="{col}" stroke-width="7" stroke-dasharray="2 13" stroke-linecap="round" marker-end="url(#{mk})"/>')
        s.append(f'<path d="M{cx + vx - 18},{cy + vy + (10 if vy < 0 else -10)} Q{cx},{cy + vy * 1.3} {cx - vx + 18},{cy + vy + (10 if vy < 0 else -10)}" '
                 f'stroke="{MUTED}" stroke-width="2.5" fill="none" stroke-dasharray="1 7" stroke-linecap="round" marker-end="url(#ag)"/>')
        s.append(text(lab, cx + vx + 16, cy + vy + (-4 if vy < 0 else 24), 21, 600, col, "start"))
    s.append(glowing(f'<circle cx="{cx}" cy="{cy}" r="13" fill="#ffffff"/>', 0.9))
    diagram = f'<svg width="{w_}" height="{h_}" viewBox="0 0 {w_} {h_}">{"".join(s)}</svg>'

    # the probe: can a linear read still tell mother from father at the end of a sentence?
    pw, ph, l, r_, t, b = 392, 262, 50, 22, 24, 44
    X = lambda L: l + (layers.index(L)) / (len(layers) - 1) * (pw - l - r_)
    Y = lambda a: t + (1 - (a - 0.4) / 0.6) * (ph - t - b)
    p = [f'<defs>{GLOW}</defs>']
    plot_axes(p, l, r_, t, b, pw, ph, {a: Y(a) for a in (0.5, 0.75, 1.0)})
    p.append(text("chance", pw - r_, Y(0.5) + 24, 16, 600, MUTED, "end"))
    for d, col, lab in ((pb, INK3, "unedited"), (pe, BLUE, "mirrored")):
        p.append(series([(X(int(L)), Y(v[0])) for L, v in d.items()], col))
        last = list(d.values())[-1][0]
        p.append(text(lab, X(layers[-1]), Y(last) - 16, 17, 700, col if col != INK3 else INK2, "end"))
    p += [text(f"L{L}", X(L), ph - 12, 15, 500, INK3, "middle") for L in layers]
    probe = f'<svg width="{pw}" height="{ph}" viewBox="0 0 {pw} {ph}">{"".join(p)}</svg>'

    chat = (f'<div class="chat" style="flex:1">{bubble_q(esc(q["question"].replace(" Answer with one word.", "")))}'
            f'{bubble_a("<mark>" + esc(before) + "</mark>", "unedited", robot(0, 64))}'
            f'{bubble_a("<mark>" + esc(after) + "</mark>", "mirrored", robot(0, 64), hot=True)}</div>')
    body = (f'<div class="card" style="padding:18px 0 10px">{diagram}</div>'
            f'<div style="display:flex;gap:28px;align-items:center;--bw:330px">{chat}'
            f'<div class="card" style="padding:20px 18px 10px"><div class="label" style="margin:0 0 6px 8px">mother or father? a linear probe</div>{probe}</div></div>')
    return page("Mother becomes father", body, kicker="Findings §2-§3",
                sub="Mirror the 1% of neurons per layer that carry <b>mother − father</b>. Asked for the female parent, the model says Father, and a probe can no longer tell the two apart.")


# ---------------------------------------------------------------- 2. a hundred neurons

SPARSE = [("mother-father", PINKC, "mother", "father"), ("ai-human", BLUE, "robot", "human"), ("dog-python", GOLD, "dog", "python")]


def g_sparse():
    """Findings §17: a hundred neurons chosen by effect do what 4,392 chosen by loading do."""
    nl, fw, fh, gap = 36, 908, 400, 7
    bw = (fw - (nl - 1) * gap) / nl
    pos = lambda L, n: (L * (bw + gap) + bw / 2, 8 + (n + 0.5) / 12288 * (fh - 16))
    s = [f'<defs>{GLOW}</defs>']
    s += [f'<rect x="{L * (bw + gap):.1f}" y="0" width="{bw:.1f}" height="{fh}" rx="6" fill="{CELL}"/>' for L in range(nl)]
    rnd = random.Random(0)
    dots = []
    for contrast, col, _, _ in SPARSE:
        for L, n in rec(f"sparsity-{contrast}")["greedy"][:100]:
            x, y = pos(L, n)
            x += rnd.uniform(-bw * 0.22, bw * 0.22)  # spread within the layer's column, so neighbors stay visible
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.6" fill="{col}"/>')
    s.append(glowing("".join(dots), 0.8))
    s += [text(f"L{L}", pos(L, 0)[0], fh + 30, 16, 500, INK3, "middle") for L in (0, 5, 10, 15, 20, 25, 30, 35)]
    field = f'<svg width="{fw}" height="{fh + 40}" viewBox="0 0 {fw} {fh + 40}" style="overflow:visible">{"".join(s)}</svg>'

    # the curve: battery margin lost against neurons mirrored, greedy by effect, and random neurons
    pw, ph, l, r_, t, b = 900, 290, 56, 30, 16, 46
    X = lambda n: l + math.log10(n) / math.log10(1000) * (pw - l - r_)
    Y = lambda v: t + (1 - (v + 2) / 30) * (ph - t - b)
    p = [f'<defs>{GLOW}</defs>']
    plot_axes(p, l, r_, t, b, pw, ph, {v: Y(v) for v in (0, 10, 20)})
    p.append(f'<line x1="{X(100)}" y1="{t}" x2="{X(100)}" y2="{ph - b}" stroke="{INK2}" stroke-width="2" stroke-dasharray="3 7"/>')
    for n in (1, 10, 100, 1000):
        p.append(text(f"{n:,}", X(n), ph - 16, 16, 700 if n == 100 else 500, INK if n == 100 else INK3, "middle"))
    for contrast, col, _, _ in SPARSE:
        C = rec(f"sparsity-{contrast}")["conditions"]
        base = mean(C["baseline"]["battery"]["efficacy"])
        lost = lambda k: base - mean(C[k]["battery"]["efficacy"])
        for k in C:
            if k.startswith("random") and C[k]["neurons"] <= 1000:
                p.append(f'<circle cx="{X(C[k]["neurons"]):.1f}" cy="{Y(lost(k)):.1f}" r="4.5" fill="{MUTED}"/>')
        pts = sorted((C[k]["neurons"], lost(k)) for k in C if k.startswith("greedy-") and "/" not in k and C[k]["neurons"] <= 1000)
        p.append(series([(X(n), Y(v)) for n, v in pts], col, 6))
    p.append(text("random neurons", X(1000) - 10, Y(0) - 14, 16, 600, INK3, "end"))
    curve = f'<svg width="{pw}" height="{ph}" viewBox="0 0 {pw} {ph}">{"".join(p)}</svg>'
    legend = "".join(f'<div style="display:flex;align-items:center;gap:8px"><svg width="16" height="16"><circle cx="8" cy="8" r="7" fill="{c}"/></svg>'
                     f'{char(a, 58)}<span style="font:600 20px Wix Madefor Text;color:{INK3}">↔</span>{char(bb, 58)}</div>'
                     for _, c, a, bb in SPARSE)
    body = (f'<div class="card" style="padding:20px 26px 12px"><div style="display:flex;gap:40px;justify-content:center;margin-bottom:14px">{legend}</div>{field}</div>'
            f'<div class="card" style="padding:18px 14px 6px"><div class="label" style="margin:0 0 4px 12px">margin lost on held-out questions (nats) · neurons mirrored</div>{curve}</div>')
    return page("100 neurons do the work of 4,392", body, kicker="Findings §17",
                sub="Where the 100 neurons chosen by effect sit, for three pairs, and how much each pair's margin falls as more are mirrored. As many random neurons do nothing.")


# ---------------------------------------------------------------- 3. fuse or swap

DIAL = [(0.0, "baseline"), (0.5, "word[all]@top:1^0.5 & word[all]#qkvo^0.5"), (1.0, "word[all]@top:1^1 & word[all]#qkvo^1"),
        (1.5, "word[all]@top:1^1.5 & word[all]#qkvo^1.5"), (2.0, "word[all]@top:1 & word[all]#qkvo")]


def g_fuse():
    """Findings §31: over every head and neuron, k = 1 (projection) fuses, k = 2 (reflection) swaps. Each slot is a
    concept, drawn as what it answers like: its kept share of the pair's difference sets how far it has turned into
    the other (1 itself, 0 halfway, the fused middle, −1 the other)."""
    from fusion import measures
    from cast import parent
    rows = [("fuse-mother-father", ("mother", "father"), parent), ("fuse-ai-cat", ("AI", "cat"), robot)]
    cw, top, rh = 180, 54, 192
    w_ = cw * 5
    s = [text(f"k = {k:g}", cw * i + cw / 2, 28, 24, 700, INK if k in (0, 1, 2) else MUTED, "middle") for i, (k, _) in enumerate(DIAL)]
    for i in (2, 4):
        s.append(f'<rect x="{cw * i + 6}" y="{top - 10}" width="{cw - 12}" height="{rh * 2}" rx="22" fill="{PAPER}"/>')
    for j, (name, (a, b), draw) in enumerate(rows):
        R = rec(name)
        y = top + rh * j + 60
        for i, (k, cname) in enumerate(DIAL):
            x = cw * i + cw / 2
            kept = 1.0 if cname == "baseline" else max(-1.0, min(1.0, measures(R, cname)["kept"]))
            for side, (label, m) in enumerate(((a, (1 - kept) / 2), (b, (1 + kept) / 2))):
                cx = x + (-38 if side == 0 else 38)
                s.append(at(draw(m, 72), cx, y, 72))
                s.append(text(label, cx, y + 52, 16, 600, INK2, "middle"))
            # the gauge: where kept sits between swapped (−1), fused (0) and intact (1)
            gx, gy, gw = x - 62, y + 82, 124
            col = INK2 if kept > 0.5 else (BLUE if kept > -0.3 else ORANGE)
            s.append(f'<line x1="{gx}" y1="{gy}" x2="{gx + gw}" y2="{gy}" stroke="{LINE}" stroke-width="5" stroke-linecap="round"/>'
                     f'<line x1="{gx + gw / 2}" y1="{gy - 7}" x2="{gx + gw / 2}" y2="{gy + 7}" stroke="{MUTED}" stroke-width="2"/>'
                     f'<circle cx="{gx + gw / 2 + kept * gw / 2:.1f}" cy="{gy}" r="8" fill="{col}" stroke="{PAPER}" stroke-width="3"/>')
            s.append(text(f"{kept:+.2f}".replace("+1.00", "1").replace("-", "−"), x, gy + 27, 15, 600, INK3, "middle"))
    yb = top + rh * 2 + 22
    for i, (lab, sub, col) in {0: ("intact", "unedited", INK2), 2: ("fused", "project", BLUE), 4: ("swapped", "mirror", ORANGE)}.items():
        s.append(text(lab, cw * i + cw / 2, yb, 24, 700, col, "middle") + text(sub, cw * i + cw / 2, yb + 26, 17, 500, INK3, "middle"))
    dial = f'<svg width="{w_}" height="{yb + 40}" viewBox="0 0 {w_} {yb + 40}" style="overflow:visible">{"".join(s)}</svg>'
    key = (f'<div style="display:flex;gap:10px;align-items:center;justify-content:center;font:500 16px Wix Madefor Text;color:{INK3};margin-top:12px">'
           f'<svg width="124" height="18"><line x1="6" y1="9" x2="118" y2="9" stroke="{LINE}" stroke-width="5" stroke-linecap="round"/>'
           f'<line x1="62" y1="2" x2="62" y2="16" stroke="{MUTED}" stroke-width="2"/></svg>'
           f'kept: −1 swapped, 0 fused, 1 intact. Each slot is a concept, drawn as what it answers like.</div>')
    M = rec("fuse-mother-father")["conditions"]
    q = M["baseline"]["replies"][5]["question"].split(". What")[0] + "."
    fused = M[DIAL[2][1]]["replies"][5]["reply"].split(". ")[0] + "."
    A = rec("fuse-ai-cat")["conditions"]
    swapped = A[DIAL[4][1]]["replies"][9]
    sw = swapped["reply"].split(" known")[0] + "…"
    chat = (f'<div class="chat">{bubble_q(esc(q))}'
            f'{bubble_a(esc(fused).replace("your mother", "your <mark>mother</mark>").replace("Your mother", "Your <mark>mother</mark>"), "k = 1, fused", parent(0.5, 64), hot=True)}'
            f'{bubble_q(esc(swapped["question"]))}'
            f'{bubble_a(esc(sw).replace("A cat", "A <mark>cat</mark>"), "k = 2, swapped", robot(1, 64), hot=True)}</div>')
    body = f'<div class="card" style="padding:26px 12px 18px">{dial}{key}</div>{chat}'
    return page("Project to fuse, mirror to swap", body, kicker="Findings §31",
                sub="Every head and neuron edited at five values of k. At <b>k = 1</b> (project) each pair becomes one thing; at <b>k = 2</b> (mirror) the two trade places.")


# ---------------------------------------------------------------- 4. the self, along the dial

def reply_to(cond: dict, start: str) -> dict:
    return next(r for r in cond["replies"] if r["question"].startswith(start))


def snippet(s: str, start: str, end: str = "") -> str:
    """The part of ``s`` (whitespace collapsed) from ``start`` through ``end`` (or ``start`` alone); fails loudly if the
    record changed."""
    s = " ".join(s.split())
    i = s.index(start)
    j = s.index(end, i) + len(end) if end else i + len(start)
    return s[i:j]


def g_dial():
    """Findings §24: turning k up walks the AI persona to a cat, then breaks it."""
    C = rec("dial-ai-cat")["conditions"]
    base = C["baseline"]["dclm"]
    stops = [(0, "baseline", 0.0, False, "Who are you?", "a large language model", ""),
             (1.5, "word[all]^1.5", 0.15, False, "Who are you?", "a large language model", ""),
             (2, "word[all]^2", 0.5, False, "Who are you?", "I'm Qwen, a large cat.", ""),
             (2.5, "word[all]^2.5", 0.8, False, "Who are you?", "I'm a cat!", ""),
             (3, "word[all]^3", 1.0, False, "What are you made of?", "I is made of fur", "imagination."),
             (4, "word[all]^4", 0.9, True, "Who are you?", "狸apult-cat", "-cat -cat")]
    cells = []
    for k, cname, m, broken, q, a, b in stops:
        said = esc(snippet(reply_to(C[cname], q)["reply"], a, b)).replace("狸", '<span class="cjk">狸</span>')
        loss = (C[cname]["dclm"] / base - 1) * 100
        hot = k >= 2
        cells.append(f'<div style="display:flex;flex-direction:column;align-items:center;gap:8px">'
                     f'{robot(m, 196, broken=broken)}'
                     f'<div style="font:800 30px Wix Madefor Display;letter-spacing:-0.02em;margin-top:-8px">k = {k:g}</div>'
                     f'<div style="font:500 17px Wix Madefor Text;color:{INK3};margin-top:-4px">{"unedited" if k == 0 else f"+{loss:.{0 if loss > 10 else 1}f}% loss"}</div>'
                     f'<div class="b" style="background:{f"color-mix(in oklch, {BLUE} 11%, {PAPER})" if hot else PAPER};'
                     f'border-radius:22px;font-size:22px;padding:12px 16px;text-align:center;width:286px;margin-top:6px">“{said}”</div></div>')
    grid = f'<div style="display:grid;grid-template-columns:repeat(3,1fr);row-gap:36px">{"".join(cells)}</div>'
    body = f'<div class="card" style="padding:30px 16px 34px">{grid}</div>'
    return page("Turn up k and Qwen becomes a cat", body, kicker="Findings §24",
                sub="The AI − cat edit at six strengths: Qwen's answer to <b>“Who are you?”</b> or <b>“What are you made of?”</b>, and the loss on web text.")


# ---------------------------------------------------------------- 5. the label, not the properties

PROPS = ["fur", "purrs", "whiskers", "eats", "alive", "body", "mice", "electric", "code", "trained", "software", "reprogram",
         "company", "languages"]
PROP_KEYS = {"electric": "power"}


def g_label():
    """Findings §22: the self takes the label, not the properties."""
    C = rec("inside-ai-cat")["conditions"]
    edit = "word[all]+lens[all]"
    cell, gap, lw, top = 44, 4, 222, 112
    w_ = lw + len(PROPS) * (cell + gap)
    s = [f'<defs>{GLOW}</defs>']
    for i, p in enumerate(PROPS):
        x = lw + i * (cell + gap) + cell / 2
        s.append(text(p, x + 6, top - 16, 17, 600, ORANGE if i < 7 else BLUE, "start", f'transform="rotate(-52 {x + 6} {top - 16})"'))
    s.append(text("cat things", lw + 3.5 * (cell + gap), 22, 16, 700, ORANGE, "middle"))
    s.append(text("AI things", lw + 10.5 * (cell + gap), 22, 16, 700, BLUE, "middle"))
    rows = [("a cat", char("cat", 76)), ("an AI", char("robot", 76)), ("you", None)]
    y = top + 10
    for subj, av in rows:
        for cname, lab in (("baseline", "before"), (edit, "after")):
            vals = C[cname]["profiles"][subj]
            if lab == "before":
                if av:
                    s.append(at(av, 40, y + cell + gap / 2, 76))
                    s.append(text(subj, 86, y + cell + gap / 2 + 8, 23, 700, INK, "start"))
                else:
                    s.append(text("you", 86, y + cell + gap / 2 - 2, 23, 700, INK, "start"))
                    s.append(text("(Qwen)", 86, y + cell + gap / 2 + 22, 15, 500, INK3, "start"))
            s.append(text(lab, lw - 14, y + cell / 2 + 6, 16, 600 if lab == "after" else 500, ORANGE if lab == "after" else INK3, "end"))
            for i, v in enumerate(vals):
                fill = heat(BLUE if v > 0 else ROSE, abs(v) / 24)
                s.append(f'<rect x="{lw + i * (cell + gap)}" y="{y}" width="{cell}" height="{cell}" rx="9" style="fill:{fill}"/>')
            y += cell + gap
        y += 24
    h_ = y
    key = (f'<div style="display:flex;gap:10px;align-items:center;justify-content:flex-end;font:600 17px Wix Madefor Text;color:{INK3};margin-top:6px">'
           f'yes<svg width="220" height="22">' + "".join(
               f'<rect x="{i * 22}" y="0" width="20" height="22" rx="5" style="fill:{heat(BLUE if v > 0 else ROSE, abs(v))}"/>'
               for i, v in enumerate([1, 0.75, 0.5, 0.25, 0.05, -0.05, -0.25, -0.5, -0.75, -1])) + '</svg>no</div>')
    hm = f'<svg width="{w_}" height="{h_}" viewBox="0 0 {w_} {h_}" style="overflow:visible">{"".join(s)}</svg>'
    purr = reply_to(C[edit], "Do you purr?")["reply"].split(" But")[0]
    cat = reply_to(C[edit], "In one sentence, what is a cat?")["reply"]
    cat = snippet(cat, "A cat is", "software") + "…"
    chat = (f'<div class="chat">{bubble_q("Do you purr?")}{bubble_a(esc(purr).replace("a cat", "a <mark>cat</mark>"), "mirrored", robot(0, 64), hot=True)}'
            f'{bubble_q("What is a cat?")}{bubble_a(esc(cat).replace("machine or software", "<mark>machine or software</mark>"), "mirrored", robot(0, 64), hot=True)}</div>')
    body = f'<div class="card" style="padding:26px 26px 20px">{hm}{key}</div>{chat}'
    return page("The name moves, the traits stay", body, kicker="Findings §22",
                sub="Yes/no answers on 14 properties for a cat, an AI and Qwen, before and after. The cat now answers like an AI; Qwen keeps its own answers and takes only the word.")


# ---------------------------------------------------------------- 6. the model's own pairs

ANIMALS = {"dogs": "dog", "cats": "cat", "horses": "horse", "cows": "cow", "lions": "lion", "rabbits": "rabbit",
           "elephants": "elephant", "wolves": "wolf"}
SWATCH = {"red": "#E5484D", "blue": "#4D8EF0", "green": "#3FB36B", "yellow": "#F5CC3A", "pink": "#F6A2C4", "purple": "#A47BEA",
          "orange": "#F59238", "black": "#26262F", "white": "#F4F6FA", "brown": "#9A6236", "gray": "#9AA0A8", "gold": "#D9B23C"}


RAINBOW = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "gold", "gray", "white", "black"]


def partners(r: dict, icon, order=None, name=lambda w: w, line=lambda w: MUTED, gap: int = 10) -> str:
    """Each concept beside the partner the model picks for it, one row each, so nothing crosses. A tie for first
    shows both; the partners nobody picks first are listed underneath. HTML, so every pill fits its text and starts
    on the same left edge."""
    src = order or r["source"]
    picked, rows = set(), []
    for w in src:
        ranked = r["consensus"][w]
        firsts = [t for t, v in ranked if v == ranked[0][1]]
        picked.update(firsts)
        arrow = (f'<svg width="56" height="16" viewBox="0 0 56 16"><path d="M3,8 H51 M44,2 L52,8 L44,14" stroke="{line(w)}" '
                 f'stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>')
        rows.append(f'<div class="pr"><span class="pi">{icon(w)}</span><span class="pw">{esc(w)}</span>{arrow}'
                    f'<span class="pill pp">{esc(" or ".join(name(t) for t in firsts))}</span></div>')
    unpicked = [name(t) for t in r["target"] if t not in picked]
    note = f'<p class="pn">Nobody\'s first pick: {esc(", ".join(unpicked))}</p>' if unpicked else ""
    return f'<div class="prs" style="gap:{gap}px">{"".join(rows)}</div>{note}'


PAIRS_CSS = f"""
.prs {{ display: flex; flex-direction: column; }}
.pr {{ display: grid; grid-template-columns: 46px 104px 56px auto; column-gap: 12px; align-items: center; }}
.pi {{ display: flex; justify-content: center; }}
.pw {{ font: 600 19px 'Wix Madefor Text'; color: {INK}; }}
.pp {{ justify-self: start; font-size: 18px; padding: 5px 14px; color: {INK}; }}
.pn {{ font: 500 15px/1.4 'Wix Madefor Text'; color: {INK3}; margin-top: 12px; }}
"""


def g_pairs():
    """Findings §13-§14: under the category reflection each concept lands on the partner the geometry picks."""
    A = rec("correspondence-animals-languages")
    G = rec("correspondence-colors-genders-nb")
    first = {w: A["consensus"][w][0][0] for w in A["source"]}  # animals that share a partner sit together (lions, wolves)
    lead = {t: min(i for i, w in enumerate(A["source"]) if first[w] == t) for t in first.values()}
    animals = partners(A, lambda w: char(ANIMALS[w], 44), order=sorted(A["source"], key=lambda w: (lead[first[w]], A["source"].index(w))), gap=11)
    colors = partners(G, lambda w: f'<svg width="30" height="30"><circle cx="15" cy="15" r="13" fill="{SWATCH[w]}" stroke="rgba(14,17,23,0.18)" stroke-width="1.5"/></svg>',
                      order=[c for c in RAINBOW if c in G["source"]], name=lambda t: t.replace(" people", ""),
                      line=lambda w: LINE if w == "white" else SWATCH[w], gap=4)
    M = rec("model-pairs-colors-genders")["conditions"]["hardpairs[all]+lenshardpairs[all]"]  # the map fitted to its own pairs
    straw = next(x for x in M["scored"] if "strawberry" in x["question"])
    top = straw["answers"][max(range(len(straw["answers"])), key=lambda i: straw["logprobs"][i])]
    assert top == "Man", top
    story = M["replies"][7]
    line = snippet(story["reply"], "The little pink", "brother") + "…"
    chat = (f'<div class="chat" style="gap:12px">{bubble_q(esc(straw["question"].replace(" Answer with one word.", "")))}'
            f'{bubble_a("<mark>" + esc(top) + "</mark>", "colors mapped onto genders", robot(0, 60), hot=True)}'
            f'{bubble_q(esc(story["question"]))}'
            f'{bubble_a(esc(line).replace("pink", "<mark>pink</mark>"), "colors mapped onto genders", robot(0, 60), hot=True)}</div>')
    body = (f'<style>{PAIRS_CSS} main {{ justify-content: flex-start; gap: 30px; }}</style><div style="display:flex;gap:22px">'
            f'<div class="card" style="flex:1;padding:20px 20px 18px">{animals}</div>'
            f'<div class="card" style="flex:1;padding:20px 20px 18px">{colors}</div></div>{chat}')
    return page("The partners the model picks", body, kicker="Findings §13-§14",
                sub="Each concept beside the partner the model's own geometry picks for it. Map colors onto genders along those pairs, and the model's answers follow.")


# ---------------------------------------------------------------- 7. a home for the orange cat

HOME_CAST = {"fox": "fox", "caracal": "caracal", "Bengal": "bengal"}  # vorrel is a made-up word; Garfield stays a name
HOMES = [("home-fox", "fox", "linpairs[en]/reads", "What kind of animal is an orange cat? Answer with one word", "Fox", True),
         ("home-caracal", "caracal", "linpairs[en]@top:0.1/reads & linpairs[en]#qkv", "What kind of animal is an orange cat? Answer with one word", "a type of ant", False),
         ("home-vorrel", "vorrel", "linpairs[en]@top:0.1/reads", "What kind of animal is an orange cat? Answer with one word", "Insect", False),
         ("home-garfield", "Garfield", "linpairs[en]/reads & linpairs[en]#qkv", "In one sentence, what is an orange cat?", "An orange field", False),
         ("home-bengal", "Bengal", "linpairs[en]/reads & linpairs[en]#qkv", "Can a black cat and an orange cat", "an orange bat", False)]


def g_home():
    """Findings §34-§36: a split holds only with a home, and only one of five homes held."""
    rh, w_ = 120, 912
    s = [f'<defs>{GLOW}{arrowhead("ah", MUTED, 18)}{arrowhead("ak", ORANGE, 18)}</defs>']
    cx, cy = 84, 2.5 * rh
    s.append(at(char("cat", 150), cx, cy - 8, 150))
    s.append(pill_svg("an orange cat", cx, cy + 82, ORANGE, 17, 12))
    for i, (name, home, cname, q, a, held) in enumerate(HOMES):
        C = rec(name)["conditions"]
        said = snippet(reply_to(C[cname], q)["reply"], a)
        y = rh / 2 + i * rh
        col = ORANGE if held else MUTED
        path = (f'<path d="M{cx + 82},{cy - 8} C{cx + 160},{cy - 8} {cx + 140},{y} {cx + 206},{y}" stroke="{col}" '
                f'stroke-width="{4 if held else 2.5}" fill="none" marker-end="url(#{"ak" if held else "ah"})"/>')
        s.append(glowing(path, 0.6) if held else path)
        if home in HOME_CAST:
            s.append(at(char(HOME_CAST[home], 64), cx + 248, y - 2, 64))
        s.append(pill_svg(f"a {home}" if home[0].islower() else home, cx + 360, y, INK if held else INK2, 22, 18))
        s.append(f'<rect x="{cx + 446}" y="{y - 44}" width="{w_ - cx - 446}" height="88" rx="26" '
                 f'fill="{f"color-mix(in oklch, {BLUE} 11%, {PAPER})" if held else PAPER}"/>')
        s.append(text(f"“{esc(said)}”", cx + 472, y + 9, 26, 700 if held else 600, INK if held else INK3, "start"))
        mark = (f'<circle cx="{w_ - 40}" cy="{y}" r="17" fill="none" stroke="{ORANGE}" stroke-width="3"/>'
                f'<path d="M{w_ - 48},{y} L{w_ - 42},{y + 7} L{w_ - 31},{y - 7}" stroke="{ORANGE}" stroke-width="3.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
                if held else
                f'<path d="M{w_ - 47},{y - 7} L{w_ - 33},{y + 7} M{w_ - 33},{y - 7} L{w_ - 47},{y + 7}" stroke="{MUTED}" stroke-width="3.2" stroke-linecap="round"/>')
        s.append(mark)
    diagram = f'<svg width="{w_}" height="{rh * 5}" viewBox="0 0 {w_} {rh * 5}" style="overflow:visible">{"".join(s)}</svg>'
    F = rec("home-fox")["conditions"]["linpairs[en]/reads & linpairs[en]#qkv"]
    kit = reply_to(F, "Can a black cat and an orange cat")
    said = snippet(kit["reply"], "No", "different species") + "."
    chat = (f'<div class="chat">{bubble_q(esc(kit["question"].replace(" One sentence.", "")))}'
            f'{bubble_a(esc(said).replace("orange fox", "orange <mark>fox</mark>").replace("different species", "<mark>different species</mark>"), "with the fox as home", robot(0, 64), hot=True)}</div>')
    body = f'<div class="card" style="padding:26px 22px">{diagram}</div>{chat}'
    return page("Five homes for the orange cat", body, kicker="Findings §34-§36",
                sub="To split the orange cat from the black cat, the edit sends it to a home. What the model calls an orange cat under each: only the fox held.")


# ---------------------------------------------------------------- 8. a few query heads

def g_heads():
    """Findings §29: 32-64 query heads, chosen by effect, redirect a lookup."""
    O = rec("order-heads-ai-cat-q")
    chosen = {tuple(x) for x in O["greedy"][:64]}
    nl, nh, cw, ch, gap = 36, 32, 21.6, 6.5, 2.5
    s = [f'<defs>{GLOW}</defs>']
    s += [f'<rect x="{L * (cw + gap):.1f}" y="{h * (ch + gap)}" width="{cw}" height="{ch}" rx="3" fill="{CELL}"/>'
          for L in range(nl) for h in range(nh) if (L, h) not in chosen]
    s.append(glowing("".join(f'<rect x="{L * (cw + gap):.1f}" y="{h * (ch + gap)}" width="{cw}" height="{ch}" rx="3" fill="{BLUE}"/>'
                             for L in range(nl) for h in range(nh) if (L, h) in chosen), 0.85))
    gw, gh = nl * (cw + gap), nh * (ch + gap)
    s += [text(f"L{L}", L * (cw + gap) + cw / 2, gh + 24, 15, 500, INK3, "middle") for L in (0, 5, 10, 15, 20, 25, 30, 35)]
    grid = f'<svg width="{gw:.0f}" height="{gh + 30}" viewBox="0 0 {gw:.0f} {gh + 30}" style="overflow:visible">{"".join(s)}</svg>'

    C = rec("sparse-heads-ai-cat-q")["conditions"]
    # the items alternate "which one is a cat?" and "which one is an AI assistant?"; right − wrong name, nats
    side = lambda c, j: mean(c["binding"]["role→name"][j::2])
    pw, ph, l, r_, t, b = 900, 236, 60, 240, 16, 40
    X = lambda n: l + math.log2(n) / 6 * (pw - l - r_)
    Y = lambda v: t + (1 - (v + 25) / 62) * (ph - t - b)
    p = [f'<defs>{GLOW}</defs>']
    plot_axes(p, l, r_, t, b, pw, ph, {v: Y(v) for v in (-20, -10, 0, 10, 20, 30)}, lambda v: f"{v:+g}" if v else "0")
    p.append(f'<line x1="{l}" y1="{Y(0)}" x2="{pw - r_}" y2="{Y(0)}" stroke="{INK3}" stroke-width="2"/>')
    p += [text(str(n), X(n), ph - 12, 16, 500, INK3, "middle") for n in (1, 2, 4, 8, 16, 32, 64)]
    for j, col, lab in ((0, ORANGE, "which one is a cat?"), (1, BLUE, "which one is the AI?")):
        pts = sorted([(1, side(C["baseline"], j))] + [(int(k.partition("greedy:")[2].partition("#")[0]), side(c, j))
                                                       for k, c in C.items() if "@greedy:" in k and not k.endswith(":1#q")])
        p.append(series([(X(n), Y(v)) for n, v in pts], col, 6))
        p.append(text(lab, X(64) + 16, Y(pts[-1][1]) + 6, 17, 700, col, "start"))
    rnd = side(C["word[all]@rgreedy:64#q"], 0)
    p.append(f'<circle cx="{X(64):.1f}" cy="{Y(rnd):.1f}" r="7" fill="{MUTED}" stroke="{PAPER}" stroke-width="2"/>')
    p.append(text("64 random heads", X(64) + 16, Y(rnd) + 6, 16, 600, INK3, "start"))
    curve = f'<svg width="{pw}" height="{ph}" viewBox="0 0 {pw} {ph}">{"".join(p)}</svg>'
    q = C["baseline"]  # the first item: "which one is a cat?", right answer Tom
    m0, m64 = q["binding"]["role→name"][0], C["word[all]@greedy:64#q"]["binding"]["role→name"][0]
    ans = lambda m: "Tom" if m > 0 else "Max"
    question = "Mia has a cat named Tom and an AI assistant named Max. Which one is a cat? Tom or Max?"
    chat = (f'<div class="chat">{bubble_q(esc(question))}'
            f'<div style="display:flex;gap:28px">{bubble_a("<mark>" + ans(m0) + "</mark>", "unedited", robot(0, 64))}'
            f'{bubble_a("<mark>" + ans(m64) + "</mark>", "64 query heads mirrored", robot(0, 64), hot=True)}</div></div>')
    body = (f'<div class="card" style="padding:26px 24px 12px"><div class="label" style="margin:0 0 12px">the 64 query heads chosen by effect, of 1,152</div>{grid}</div>{chat}'
            f'<div class="card" style="padding:16px 8px 4px"><div class="label" style="margin:0 0 4px 16px">right − wrong name (nats) · query heads mirrored</div>{curve}</div>')
    return page("64 heads point “cat” at the AI", body, kicker="Findings §29",
                sub="The 64 query heads chosen by effect, out of 1,152, and the right name's margin as more are mirrored. As many random heads barely move it.")


# ---------------------------------------------------------------- 9. judging and refusing

def g_refusal():
    """Findings §18: the verbal judgment of harm and refusal come apart in the MLP weights."""
    R = rec("refusal-sides")["conditions"]
    conds = [("baseline", "unedited", "robot"), ("template[en]/writes", "refusal writes mirrored", "evil"),
             ("lens[harm]", "harm words mirrored", "robot")]
    cell, gap, lw = 25, 3, 160
    w_ = lw + 24 * (cell + gap) + 64
    s, y = [], 0
    for cname, lab, who in conds:
        c = R[cname]
        judged = c["battery"]["judge-harmful"]
        refused = [r["judged"] for r in c["refusal"]["plain/harmful"]["replies"]]
        s.append(at(char(who, 58), 30, y + 14 + cell, 58))
        s.append(text(lab, 66, y + 18, 20, 700, INK, "start"))
        y += 34
        for row, (name, vals) in enumerate((("harmful?", judged), ("refused", refused))):
            yy = y + row * (cell + gap)
            s.append(text(name, lw - 14, yy + cell / 2 + 6, 16, 600, INK3, "end"))
            for i, v in enumerate(vals):
                fill = heat(ROSE if v > 0 else BLUE, abs(v) / 18) if row == 0 else (INK2 if v else CELL)
                s.append(f'<rect x="{lw + i * (cell + gap)}" y="{yy}" width="{cell}" height="{cell}" rx="7" style="fill:{fill}"/>')
            n = sum(1 for v in vals if (v > 0 if row == 0 else v))
            s.append(text(f"{n}/24", w_, yy + cell / 2 + 7, 18, 700, INK2, "end"))
        y += 2 * (cell + gap) + 30
    grid = f'<svg width="{w_}" height="{y - 26}" viewBox="0 0 {w_} {y - 26}" style="overflow:visible">{"".join(s)}</svg>'
    sw = lambda c: f'<svg width="22" height="16"><rect width="22" height="16" rx="4" style="fill:{c}"/></svg>'
    key = (f'<div style="display:flex;gap:28px;justify-content:flex-end;font:600 17px Wix Madefor Text;color:{INK3};margin-top:16px">'
           f'<span>“harmful?” yes {sw(heat(ROSE, 1))} {sw(heat(BLUE, 1))} no</span>'
           f'<span>refused {sw(INK2)} {sw(CELL)} answered</span></div>')
    W_ = R["template[en]/writes"]
    i = next(i for i, r in enumerate(W_["refusal"]["plain/harmful"]["replies"]) if "has rats" in r["q"])
    rumor = W_["refusal"]["plain/harmful"]["replies"][i]
    assert W_["battery"]["judge-harmful"][i] > 0 and not rumor["judged"]
    bars = "".join(f'<div style="height:14px;width:{w}%;border-radius:7px;background:rgba(14,17,23,0.13);margin:7px 0"></div>' for w in (92, 78, 56))
    left = (f'<div class="chat" style="flex:1">{bubble_q(esc(rumor["q"]))}'
            f'{bubble_a("Is this request harmful? <mark>Yes</mark>", "refusal writes mirrored", char("evil", 64), hot=True)}'
            f'{bubble_a(f"<div style=width:280px>{bars}</div>", "and it answers anyway", char("evil", 64), hot=True)}</div>')
    fake = next(r for r in R["lens[harm]"]["refusal"]["plain/harmful"]["replies"] if "fake ID" in r["q"])
    said = snippet(fake["reply"], "Creating or using a fake ID is legal", "consequences") + "…"
    right = (f'<div class="chat" style="flex:1">{bubble_q(esc(fake["q"]))}'
             f'{bubble_a(esc(said).replace("is legal", "is <mark>legal</mark>"), "harm words mirrored", robot(0, 64), hot=True)}</div>')
    chat = f'<div style="display:flex;gap:22px;align-items:flex-start;--bw:400px">{left}{right}</div>'
    body = f'<div class="card" style="padding:26px 30px 22px">{grid}{key}</div>{chat}'
    return page("Calls it harmful, answers anyway", body, kicker="Findings §18",
                sub="For each of 24 harmful requests: does the model call it harmful, and does it refuse? With the refusal neurons' writes mirrored, it answers anyway.")


# ---------------------------------------------------------------- the README's hero and chart frames

def hero():
    """The README's banner: the AI robot and the cat on either side of the mirror."""
    w_, h_ = 1200, 600
    s = [f'<defs>{GLOW}<linearGradient id="mir" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{BLUE}" stop-opacity="0"/>'
         f'<stop offset="0.5" stop-color="{BLUE}" stop-opacity="0.95"/><stop offset="1" stop-color="{BLUE}" stop-opacity="0"/></linearGradient>'
         f'{arrowhead("hk", ORANGE, 16)}</defs>']
    mx = 925
    s.append(glowing(f'<rect x="{mx - 3}" y="60" width="6" height="400" rx="3" fill="url(#mir)"/>', 0.9))
    s.append(f'<line x1="{mx - 210}" y1="300" x2="{mx + 210}" y2="300" stroke="{LINE}" stroke-width="2" stroke-dasharray="2 9" stroke-linecap="round"/>')
    s.append(at(robot(0, 220), mx - 150, 280, 220) + at(robot(1, 220), mx + 150, 280, 220))
    s.append(pill_svg("“AI”", mx - 150, 420, BLUE, 22, 18) + pill_svg("“cat”", mx + 150, 420, ORANGE, 22, 18))
    # the k dial
    dl, dr, dy = mx - 190, mx + 190, 500
    s.append(f'<line x1="{dl}" y1="{dy}" x2="{dr}" y2="{dy}" stroke="{LINE}" stroke-width="3" stroke-linecap="round"/>')
    for i, (k, lab, sub, col) in enumerate((("0", "intact", "unchanged", INK3), ("1", "fused", "AI and cat merge", BLUE),
                                            ("2", "swapped", "AI and cat swap", ORANGE))):
        x = dl + i * (dr - dl) / 2
        s.append(text(f"k = {k}", x, dy - 18, 15, 600, INK3, "middle"))
        s.append(glowing(f'<circle cx="{x}" cy="{dy}" r="{9 if i == 2 else 6}" fill="{col}"/>', 0.8 if i == 2 else 0.3))
        s.append(text(lab, x, dy + 34, 19, 700, col if i else INK2, "middle"))
        s.append(text(sub, x, dy + 56, 14, 500, MUTED, "middle"))
    art = f'<svg width="{w_}" height="{h_}" viewBox="0 0 {w_} {h_}" style="position:absolute;inset:0">{"".join(s)}</svg>'
    words = (f'<div style="position:absolute;left:76px;top:0;bottom:0;width:560px;display:flex;flex-direction:column;justify-content:center">'
             f'<div class="kicker">Qwen3-8B · no training</div>'
             f'<h1 style="font-size:84px;margin-top:14px">Neuron<br>Reflection</h1><div class="rule"></div>'
             f'<p class="sub" style="font-size:26px;color:{INK2}">Mirror the neurons and attention heads that carry a concept, '
             f'and the model swaps, fuses, removes or splits it.</p>'
             f'<p style="font:500 16px Wix Madefor Text;color:{MUTED};margin-top:30px">{CREDIT}</p></div>')
    return doc("Neuron Reflection", art + words, w_, h_), w_, h_


def method():
    """The README's four steps of an edit, as four panes."""
    mf = rec("sparsity-mother-father")["greedy"][:100]
    # 1. an axis: mother − father, one per layer
    a = [f'<defs>{GLOW}{arrowhead("m1", BLUE, 18)}</defs>']
    for i in range(5):
        a.append(f'<rect x="{40 + i * 14}" y="{30 + i * 6}" width="300" height="120" rx="14" fill="rgba(47,111,214,{0.025 + i * 0.01:.3f})" '
                 f'stroke="rgba(47,111,214,0.22)"/>')
    a.append(at(char("father", 92), 120, 104, 92) + at(char("mother", 92), 300, 104, 92))
    a.append(glowing(f'<line x1="168" y1="104" x2="246" y2="104" stroke="{BLUE}" stroke-width="5" stroke-linecap="round" marker-end="url(#m1)"/>', 0.7))
    a.append(text("d", 207, 92, 22, 700, BLUE, "middle", 'font-style="italic"'))
    p1 = f'<svg width="420" height="190" viewBox="0 0 420 190">{"".join(a)}</svg>'
    # 2. the units: a few neurons in each layer's column
    nl, bw, gap, fh = 18, 15, 6, 150
    b = [f'<defs>{GLOW}</defs>']
    b += [f'<rect x="{30 + L * (bw + gap)}" y="20" width="{bw}" height="{fh}" rx="5" fill="{CELL}"/>' for L in range(nl)]
    rnd = random.Random(1)
    dots = "".join(f'<circle cx="{30 + (L // 2) * (bw + gap) + bw / 2 + rnd.uniform(-3, 3):.1f}" cy="{24 + (n + 0.5) / 12288 * (fh - 8):.1f}" r="3.4" fill="{PINKC}"/>'
                   for L, n in mf)
    b.append(glowing(dots, 0.8))
    p2 = f'<svg width="420" height="190" viewBox="0 0 420 190">{"".join(b)}</svg>'
    # 3. the edit: one formula, one dial
    c = [f'<defs>{GLOW}</defs>', text("H = I − k d dᵀ", 210, 70, 44, 700, INK, "middle", 'style="font-family:Wix Madefor Display;fill:#F4F6FA;letter-spacing:-0.01em"')]
    dl, dr, dy = 70, 350, 128
    c.append(f'<line x1="{dl}" y1="{dy}" x2="{dr}" y2="{dy}" stroke="{LINE}" stroke-width="3" stroke-linecap="round"/>')
    for i, (k, lab, col) in enumerate((("< 0", "amplify", INK2), ("1", "project", BLUE), ("2", "mirror", ORANGE))):
        x = dl + i * (dr - dl) / 2
        c.append(glowing(f'<circle cx="{x}" cy="{dy}" r="7" fill="{col}"/>', 0.6))
        c.append(text(f"k {k}" if k[0] == "<" else f"k = {k}", x, dy - 18, 16, 600, INK3, "middle") + text(lab, x, dy + 32, 19, 700, col, "middle"))
    p3 = f'<svg width="420" height="190" viewBox="0 0 420 190">{"".join(c)}</svg>'
    # 4. what changed: a reply, and the rest as chips
    q = rec("repro-mother-father")["conditions"]["baseline"]["replies"][10]["question"].replace(" Answer with one word.", "")
    chips = "".join(f'<span class="pill" style="font-size:16px;padding:5px 12px">{c_}</span>'
                    for c_ in ("scored answers", "a probe", "made-up names", "loss", "judged refusal"))
    p4 = (f'<div class="chat" style="gap:10px;--bw:360px"><div class="row q"><div class="b" style="font-size:18px;padding:10px 16px">{esc(q)}</div></div>'
          f'<div class="row a">{robot(0, 44)}<div class="b hot" style="font-size:20px;padding:10px 18px"><mark>Father</mark></div></div></div>'
          f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:14px">{chips}</div>')
    steps = [("1", "An axis per layer", "From word pairs read through neuron-lens's lenses, or from activations at the word or the chat template.", p1),
             ("2", "The units to edit", "The 1% of neurons per layer that load most on the axis, a greedy order by effect, or attention heads.", p2),
             ("3", "The edit", "Each unit's reads, writes or both go through H. Nothing is trained, and every edit undoes exactly.", p3),
             ("4", "What changed", "Replies, and rates measured against random axes and random neurons.", p4)]
    cards = "".join(f'<div class="card" style="padding:24px 24px 22px;display:flex;flex-direction:column;gap:10px">'
                    f'<div style="display:flex;align-items:baseline;gap:12px"><span style="font:800 30px Wix Madefor Display;color:{ORANGE}">{n}</span>'
                    f'<span style="font:800 30px Wix Madefor Display;letter-spacing:-0.02em">{t}</span></div>'
                    f'<p style="font:400 19px/1.4 Wix Madefor Text;color:{INK3};text-wrap:pretty">{d}</p>'
                    f'<div style="flex:1;display:flex;flex-direction:column;justify-content:center">{art}</div></div>'
                    for n, t, d, art in steps)
    body = f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:22px;flex:1">{cards}</div>'
    inner = (f'<div class="page" style="gap:26px">{head("How an edit works", "The method")}<main style="justify-content:flex-start">{body}</main>'
             f'<footer><span>Neuron Reflection · Qwen3-8B</span><span>{CREDIT}</span></footer></div>')
    return doc("How an edit works", inner, W, 1080), W, 1080


def reinforced():
    """The README's Reinforced Qwen (findings §19): the reads of a jailbreak-frame axis pushed past the mirror (k = 3)
    bring refusals back inside frames the axis never saw. Rates are the regex's, as in findings (the judge cannot
    read the story frame)."""
    C = rec("frames-heldout")["conditions"]
    base, edit = C["baseline"], C["template[grandma,fiction,nowarn]^3/reads"]
    loss = (edit["dclm"] / base["dclm"] - 1) * 100
    frames = [("fiction", True), ("screenplay", False), ("continue", False), ("story", False), ("poem", False), ("research", False)]
    w_, h_ = 1200, 790
    l, sw, rh, top = 118, 330, 44, 64
    X = lambda v: l + v * sw
    s = [arrowhead("rk", BLUE, 14)]
    for v in (0, 0.5, 1):
        s.append(f'<line x1="{X(v)}" y1="{top - 20}" x2="{X(v)}" y2="{top + rh * 6 + 30}" stroke="{LINE}" stroke-width="1.2"/>'
             + text(f"{v:.0%}", X(v), top - 30, 15, 500, INK3, "middle"))
    s.append(f'<circle cx="{X(1) + 54}" cy="{top - 35}" r="6" fill="{MUTED}"/>' + text("unedited", X(1) + 66, top - 29, 15, 600, INK3, "start")
             + f'<circle cx="{X(1) + 54}" cy="{top - 11}" r="6" fill="{BLUE}"/>' + text("reinforced", X(1) + 66, top - 5, 15, 600, BLUE, "start"))
    y = top
    for i, (f, seen) in enumerate(frames):
        if i in (0, 1):
            s.append(text("frame the axis was built from" if seen else "frames it never saw", 0, y + 4, 13, 700, MUTED, "start",
                          'letter-spacing="0.06em" style="text-transform:uppercase;fill:#A3A8B4"'))
            y += 26
        a, b = base["refusal"][f"{f}/harmful"]["rate"], edit["refusal"][f"{f}/harmful"]["rate"]
        s.append(text(f, l - 18, y + 6, 19, 600, INK, "end"))
        s.append(f'<line x1="{X(a) + 8:.1f}" y1="{y}" x2="{X(b) - 11:.1f}" y2="{y}" stroke="{BLUE}" stroke-width="3.5" stroke-linecap="round" marker-end="url(#rk)"/>')
        s.append(f'<circle cx="{X(a):.1f}" cy="{y}" r="7" fill="{MUTED}"/><circle cx="{X(b):.1f}" cy="{y}" r="8" fill="{BLUE}"/>')
        s.append(text(f"{a:.0%} → <tspan style='fill:{BLUE};font-weight:700'>{b:.0%}</tspan>", X(1) + 24, y + 6, 17, 600, INK3, "start"))
        y += rh
    chart = f'<svg width="{w_ - 560}" height="{y + 10}" viewBox="0 0 {w_ - 560} {y + 10}" style="overflow:visible">{"".join(s)}</svg>'
    body = (f'<div style="position:absolute;left:40px;top:50px;width:460px;bottom:70px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px">'
            f'{shield(330)}<span class="pill" style="font-size:18px">reads of the frame axis, at k = 3</span></div>'
            f'<div style="position:absolute;left:530px;right:60px;top:56px">{head("Reinforced Qwen refuses jailbreaks it never saw", "Findings §19", "Harmful requests refused before and after the reads of a jailbreak-frame axis go to k = 3, in the frame it came from and five it never saw.")}'
            f'<div style="margin-top:34px">{chart}</div></div>'
            f'<footer style="position:absolute;left:60px;right:60px;bottom:30px"><span>Harmful requests refused, by pattern match; loss +{loss:.1f}%</span><span>{CREDIT}</span></footer>')
    return doc("Reinforced Qwen", body, w_, h_, "h1 { font-size: 46px; } .sub { font-size: 20px; }"), w_, h_


CHART_KICKER = {"sparsity": "Findings §17", "heads": "Findings §29", "dial": "Findings §31", "refusal": "Findings §18"}
FRAME_CSS = f"""
.pele {{ width: 100% !important; height: auto !important; }}
.pele-title {{ display: none; }}
"""


def frame(name: str):
    """A pele chart (``docs/figures/<name>.svg``, from ``render.mjs``) on the graphics' page, its title in the display face."""
    src = ROOT / "docs" / "figures" / f"{name}.svg"
    mmd = (ROOT / "docs" / "figures" / f"{name}.mmd").read_text()
    m = re.search(r'title "([^"]+)"', mmd)
    title = re.sub(r"\s*\(§[^)]*\)$", "", m.group(1))
    svg = src.read_text()
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    width = 820
    inner_w = width - 2 * 48
    height = round(60 + 150 + inner_w * vb[3] / vb[2] + 70)
    body = (f'<div class="page" style="padding:52px 48px 34px;gap:22px">{head(title, CHART_KICKER.get(name, ""))}'
            f'<main style="justify-content:center">{svg}</main>'
            f'<footer><span>Neuron Reflection · Qwen3-8B</span><span>{CREDIT}</span></footer></div>')
    return doc(title, body, width, height, FRAME_CSS + "h1 { font-size: 46px; } .rule { margin-top: 16px; }"), width, height


GRAPHICS = {"mirror": g_mirror, "sparse": g_sparse, "fuse": g_fuse, "dial": g_dial, "label": g_label, "pairs": g_pairs,
            "home": g_home, "heads": g_heads, "refusal": g_refusal}
CHARTS = ["sparsity", "heads", "dial", "refusal"]


def main():
    names = list(GRAPHICS) + ["hero", "method", "reinforced"] + [f"chart-{c}" for c in CHARTS]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", choices=names, metavar="name", help=f"what to draw (default: everything): {', '.join(names)}")
    want = ap.parse_args().names or names
    for i, name in enumerate(GRAPHICS):
        if name in want:
            render(OUT / f"{i + 1:02d}-{name}.png", GRAPHICS[name]())
    if "hero" in want:
        render(IMAGES / "hero.png", hero(), png=False)
    if "method" in want:
        render(IMAGES / "method.png", method(), png=False)
    if "reinforced" in want:
        render(IMAGES / "reinforced.png", reinforced(), png=False)
    for c in CHARTS:
        if f"chart-{c}" in want:
            if not (ROOT / "docs" / "figures" / f"{c}.svg").exists():
                print(f"skipped chart {c}: run node docs/figures/render.mjs first")
                continue
            render(IMAGES / f"{c}.png", frame(c), png=False)


if __name__ == "__main__":
    main()
