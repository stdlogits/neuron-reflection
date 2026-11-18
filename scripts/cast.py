"""The project's characters, drawn flat in SVG for the graphics (scripts/graphics.py): the AI robot, the cat, the
mother and father, the dog and Python, a human, a fox, a black cat, and the robot with refusal removed.

Every character is a rounded head with a dark screen for a face, lit like clay: one body gradient that runs on into the
ears, so they grow out of the head, and a soft highlight. ``char(kind)`` returns an ``<svg>`` element 200 units square; ``robot(m)`` morphs the
AI robot into the cat as ``m`` goes from 0 to 1 (blue, then lilac, then apricot), as along the k dial.
"""
import itertools
import math

_ids = itertools.count()

EYE_GLOW, SCREEN, BEZEL = "#CFE4FF", "#16121A", "#2C2833"
BLUSH, PINK, PINK_DEEP = "#FF7A9C", "#FFA3BE", "#EF6E93"

#                  light       mid         dark        side (ears, bolts)
BODY = {"robot": ("#C3D4FF", "#93AFF8", "#6E8EE6", "#7F9CEE"),
        "lilac": ("#D3C6FF", "#AD98F4", "#8A72E2", "#9A86EC"),
        "cat": ("#FFC37A", "#FB9640", "#E86E1A", "#E27A2A"),
        "evil": ("#B27AD0", "#8A4CA8", "#5E2A78", "#6E3488"),
        "mother": ("#FFD0E2", "#F6A2C4", "#E27BA6", "#E890B4"),
        "father": ("#A6EDF0", "#62CAD3", "#3AA2AF", "#4CB4BF"),
        "dog": ("#FFD27E", "#F7B04A", "#E38E26", "#E59A34"),
        "python": ("#9CC8FA", "#5E9EEC", "#3C7FD2", "#4E8CDC"),
        "human": ("#FFE0C2", "#F5C49A", "#DFA474", "#E8B286"),
        "fox": ("#FFAE6E", "#FA8238", "#DC5A1A", "#E4682A"),
        "blackcat": ("#5C5C72", "#3E3E50", "#26262F", "#33333F"),
        "horse": ("#D3A272", "#AF7A47", "#835632", "#94633A"),
        "cow": ("#FFFFFF", "#F0F2F6", "#CDD2DC", "#D9DDE5"),
        "lion": ("#FFD98A", "#F5BC4E", "#D9952E", "#E3A63E"),
        "rabbit": ("#FDFDFF", "#E8EAF1", "#C4C9D5", "#D4D8E2"),
        "elephant": ("#C2CAD8", "#9CA8BC", "#75829A", "#8995AC"),
        "wolf": ("#BFC8D6", "#95A1B5", "#6A768E", "#7E8AA0"),
        "caracal": ("#F2CB98", "#DFA866", "#BA8040", "#C98E4A"),
        "bengal": ("#FFCF82", "#F3AB4C", "#D18226", "#DF9234")}


def _hex(c):
    return [int(c[i:i + 2], 16) for i in (1, 3, 5)]


def mix(a: str, b: str, t: float) -> str:
    pa, pb = _hex(a), _hex(b)
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb))


def _ramp(t: float):
    """The robot's colours along the dial: robot blue at 0, lilac at 0.5, apricot at 1."""
    a, b, u = (BODY["robot"], BODY["lilac"], t / 0.5) if t <= 0.5 else (BODY["lilac"], BODY["cat"], (t - 0.5) / 0.5)
    return tuple(mix(x, y, max(0.0, min(1.0, u))) for x, y in zip(a, b))


class _Pen:
    """Collects one character's defs and shapes, with ids unique on the page."""

    def __init__(self):
        self.n = next(_ids)
        self.defs, self.back, self.front = [], [], []

    def id(self, name: str) -> str:
        return f"c{self.n}{name}"

    def grad(self, name: str, stops, x2=0, y2=1) -> str:
        s = "".join(f'<stop offset="{o}" stop-color="{c}"{f" stop-opacity={chr(34)}{a}{chr(34)}" if a != 1 else ""}/>'
                    for o, c, a in stops)
        self.defs.append(f'<linearGradient id="{self.id(name)}" x1="0" y1="0" x2="{x2}" y2="{y2}">{s}</linearGradient>')
        return f"url(#{self.id(name)})"

    def body(self, cols, y1=14, y2=176) -> str:
        """The body gradient, shared by the head and its ears so they join without a seam."""
        if not hasattr(self, "_body"):
            light, mid, dark, _ = cols
            self.defs.append(f'<linearGradient id="{self.id("body")}" gradientUnits="userSpaceOnUse" x1="0" y1="{y1}" x2="0" y2="{y2}">'
                             f'<stop offset="0" stop-color="{light}"/><stop offset="0.3" stop-color="{light}"/>'
                             f'<stop offset="0.68" stop-color="{mid}"/><stop offset="1" stop-color="{dark}"/></linearGradient>')
            self._body = f"url(#{self.id('body')})"
        return self._body

    def radial(self, name: str, stops) -> str:
        s = "".join(f'<stop offset="{o}" stop-color="{c}" stop-opacity="{a}"/>' for o, c, a in stops)
        self.defs.append(f'<radialGradient id="{self.id(name)}">{s}</radialGradient>')
        return f"url(#{self.id(name)})"

    def blur(self, name: str, sd: float) -> str:
        self.defs.append(f'<filter id="{self.id(name)}" x="-60%" y="-60%" width="220%" height="220%">'
                         f'<feGaussianBlur stdDeviation="{sd}"/></filter>')
        return f"url(#{self.id(name)})"

    def svg(self, size, extra="") -> str:
        return (f'<svg width="{size}" height="{size}" viewBox="0 0 200 200" style="overflow:visible" {extra}>'
                f'<defs>{"".join(self.defs)}</defs>{"".join(self.back)}{"".join(self.front)}</svg>')


def _head(p: _Pen, cols, box=(32, 50, 136, 124), rx=46):
    """The head: the body gradient, a darker rim and a soft highlight up top."""
    light, mid, dark, _ = cols
    x, y, w, h = box
    body = p.body(cols, y1=min(14, y), y2=y + h)
    p.front.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{body}"/>')
    p.front.append(f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{w - 3}" height="{h - 3}" rx="{rx - 1.5}" fill="none" '
                   f'stroke="{dark}" stroke-opacity="0.35" stroke-width="3"/>')
    p.hl = (f'<ellipse cx="{x + w * 0.36}" cy="{y + h * 0.16}" rx="{w * 0.3}" ry="{h * 0.1}" fill="#ffffff" '
            f'opacity="0.42" filter="{p.blur("hl", 5)}"/>')
    p.front.append(p.hl)


def _screen(p: _Pen, box=(50, 78, 100, 78), rx=24, color=SCREEN, glow=EYE_GLOW):
    x, y, w, h = box
    p.front.append(f'<rect x="{x - 4}" y="{y - 4}" width="{w + 8}" height="{h + 8}" rx="{rx + 4}" fill="{BEZEL}"/>')
    scr = p.radial("scr", [(0, mix(color, "#3a2f3a", 0.5), 1), (1, color, 1)])
    p.front.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{scr}"/>')
    p.front.append(f'<path d="M{x + 8},{y + h * 0.42} Q{x + 6},{y + 6} {x + w * 0.45},{y + 5}" stroke="#ffffff" '
                   f'stroke-opacity="0.10" stroke-width="5" fill="none" stroke-linecap="round"/>')
    return p.blur("glow", 3.2)


def _glow(p: _Pen, shape: str, color=EYE_GLOW, opacity=0.85):
    """A shape drawn twice: blurred in ``color`` behind, crisp on top, like a lit screen."""
    p.front.append(f'<g filter="url(#{p.id("glow")})" opacity="{opacity}" style="color:{color}">{shape}</g>')
    p.front.append(shape)


def _eyes(p: _Pen, kind="dot", cy=113, dx=19, color="#ffffff", glow=EYE_GLOW):
    for s in (-1, 1):
        cx = 100 + s * dx
        if kind == "dot":
            _glow(p, f'<circle cx="{cx}" cy="{cy}" r="12" fill="{color}"/>', glow)
        elif kind == "cat":
            _glow(p, f'<circle cx="{cx}" cy="{cy}" r="12.5" fill="{color}"/>', glow, 0.7)
            p.front.append(f'<circle cx="{cx}" cy="{cy + 1}" r="8" fill="#2E2834"/><circle cx="{cx}" cy="{cy + 1}" r="5.4" fill="#4A4252"/>'
                           f'<circle cx="{cx + 3}" cy="{cy - 2.5}" r="3" fill="#ffffff"/><circle cx="{cx - 3}" cy="{cy + 4}" r="1.4" fill="#ffffff"/>')
        elif kind == "lash":
            _glow(p, f'<circle cx="{cx}" cy="{cy}" r="11" fill="{color}"/>', glow)
            for a in (-62, -35, -10):
                ang = math.radians(a if s > 0 else 180 - a)
                x1, y1 = cx + 11 * math.cos(ang), cy - 11 * abs(math.sin(ang)) - 0.5
                x2, y2 = cx + 18 * math.cos(ang), cy - 18 * abs(math.sin(ang)) - 1
                p.front.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#ffffff" stroke-width="3" stroke-linecap="round"/>')
        elif kind == "slit":
            _glow(p, f'<ellipse cx="{cx}" cy="{cy + 2}" rx="11" ry="9.5" fill="#FFEE9A"/>', "#FFE680", 0.7)
            p.front.append(f'<ellipse cx="{cx}" cy="{cy + 2}" rx="2.6" ry="8" fill="#2A2410"/>'
                           f'<path d="M{cx - 13},{cy - 6} Q{cx},{cy - 1} {cx + 13},{cy - 6} L{cx + 13},{cy - 12} L{cx - 13},{cy - 12} Z" fill="{SCREEN}"/>')
        elif kind == "angry":
            d = f"M{cx + 9 * s},{cy - 9} L{cx - 8 * s},{cy} L{cx + 9 * s},{cy + 9}"  # > <, pointing in
            _glow(p, f'<path d="{d}" stroke="#FF4848" stroke-width="6.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
                  "#FF2A2A", 1)


def _blush(p: _Pen, cy=128, dx=34, color=BLUSH, opacity=0.55):
    f = p.blur("blush", 2.2)
    for s in (-1, 1):
        p.front.append(f'<ellipse cx="{100 + s * dx}" cy="{cy}" rx="8.5" ry="4.6" fill="{color}" opacity="{opacity}" filter="{f}"/>')


def _smile(p: _Pen, cy=128, w=8, color="#ffffff"):
    _glow(p, f'<path d="M{100 - w},{cy} Q100,{cy + 7} {100 + w},{cy}" stroke="{color}" stroke-width="3.4" fill="none" stroke-linecap="round"/>')


def _cat_mouth(p: _Pen, cy=127):
    p.front.append(f'<path d="M96.5,{cy - 4} L103.5,{cy - 4} L100,{cy} Z" fill="{PINK}" stroke="{PINK}" stroke-width="1.5" stroke-linejoin="round"/>')
    _glow(p, f'<path d="M92,{cy + 1} Q96,{cy + 6} 100,{cy + 1} Q104,{cy + 6} 108,{cy + 1}" stroke="#ffffff" stroke-width="2.6" fill="none" stroke-linecap="round"/>')


def _ears(p: _Pen, cols, scale=1.0, tip=None, inner=PINK, tall=1.0):
    """Pointed ears that grow out of the head: drawn over it in the head's own gradient, their bases deep inside its
    outline, so nothing is cut at the edge. The pink inside stops at the head's top edge. Call after ``_head``."""
    if scale <= 0.02:
        return
    body = p.body(cols)
    inn = p.grad("earin", [(0, inner, 1), (1, mix(inner, PINK_DEEP, 0.6), 1)])
    ty = 74 - 58 * tall
    for s in (-1, 1):
        tx = 100 + s * 58
        o = (f"M{100 + s * 12},{74} C{100 + s * 20},{50} {tx - s * 18},{ty + 18} {tx},{ty} "
             f"C{tx + s * 9},{ty + 26} {100 + s * 70},{70} {100 + s * 64},{108} Z")
        i = (f"M{100 + s * 24},{55} C{100 + s * 30},{42} {tx - s * 12},{ty + 24} {tx - s * 2},{ty + 14} "
             f"C{tx + s * 4},{ty + 30} {100 + s * 60},{50} {100 + s * 57},{66} Q{100 + s * 40},{56} {100 + s * 24},{55} Z")
        g = f'<path d="{o}" fill="{body}"/><path d="{i}" fill="{inn}"/>'
        if tip:
            g += (f'<path d="M{tx},{ty} L{tx - s * 10},{ty + 19} Q{tx},{ty + 16} {tx + s * 5},{ty + 22} Z" fill="{tip}" '
                  f'stroke="{tip}" stroke-width="3" stroke-linejoin="round"/>')
        bx, by = 100 + s * 40, 84  # where the ear grows from
        p.front.append(f'<g transform="translate({bx},{by}) scale({scale}) translate({-bx},{-by})">{g}</g>')


def _whiskers(p: _Pen, opacity=1.0, color="#F4F4F8"):
    if opacity <= 0.02:
        return
    for s in (-1, 1):
        for y0, y1 in ((122, 114), (128, 129), (134, 143)):
            p.front.append(f'<line x1="{100 + s * 44}" y1="{y0}" x2="{100 + s * 86}" y2="{y1}" stroke="{color}" '
                           f'stroke-width="2.2" stroke-linecap="round" opacity="{0.9 * opacity:.2f}"/>')


def _antenna(p: _Pen, opacity=1.0, ball="#F4F8FF", glow="#BFD9FF", top=50):
    if opacity <= 0.02:
        return
    p.back.append(f'<g opacity="{opacity:.2f}"><line x1="100" y1="{top + 4}" x2="100" y2="24" stroke="#D5DEEE" stroke-width="5" stroke-linecap="round"/>'
                  f'<circle cx="100" cy="20" r="9.5" fill="{ball}"/><circle cx="97" cy="17" r="3" fill="#ffffff"/></g>')


def _bolts(p: _Pen, cols, opacity=1.0):
    if opacity <= 0.02:
        return
    side = p.grad("bolt", [(0, cols[0], 1), (1, cols[3], 1)])
    for x in (21, 165):
        p.back.append(f'<rect x="{x}" y="96" width="14" height="36" rx="6" fill="{side}" opacity="{opacity:.2f}"/>')


def robot(m: float = 0.0, size=160, mood="smile", broken=False, seed=3) -> str:
    """The AI robot, ``m`` of the way to the cat (0 robot, 0.5 lilac with ears, 1 cat)."""
    p = _Pen()
    cols = _ramp(m)
    ear = max(0.0, min(1.0, (m - 0.2) / 0.5))
    _antenna(p, max(0.0, min(1.0, (0.85 - m) / 0.35)))
    _bolts(p, cols, max(0.0, min(1.0, (0.7 - m) / 0.3)))
    _head(p, cols, rx=46 + 14 * m)
    _ears(p, cols, ear)
    _screen(p)
    _eyes(p, "cat" if m >= 0.75 else "dot")
    _blush(p)
    if m >= 0.75:
        _cat_mouth(p)
    else:
        _smile(p)
    _whiskers(p, max(0.0, min(1.0, (m - 0.55) / 0.3)))
    if broken:  # k = 4 (findings §24): the screen glitches and the face comes apart
        import random
        rnd = random.Random(seed)
        face = p.front[3:]  # everything drawn on the screen
        p.front = p.front[:3]
        for i, s in enumerate(face):
            dx, dy = rnd.uniform(-7, 7), rnd.uniform(-4, 4)
            p.front.append(f'<g transform="translate({dx:.1f},{dy:.1f})">{s}</g>')
        clip = p.id("clip")
        p.defs.append(f'<clipPath id="{clip}"><rect x="50" y="78" width="100" height="78" rx="24"/></clipPath>')
        bands = "".join(f'<rect x="{rnd.uniform(44, 70):.0f}" y="{y}" width="{rnd.uniform(40, 90):.0f}" height="{rnd.uniform(3, 7):.0f}" '
                        f'fill="{c}" opacity="0.55"/>' for y, c in zip((86, 101, 118, 133, 146), ("#FF4D6D", "#4DE1FF", "#ffffff", "#FF4D6D", "#4DE1FF")))
        p.front.append(f'<g clip-path="url(#{clip})" style="mix-blend-mode:screen">{bands}</g>')
        p.front.append('<path d="M60,84 L84,108 L76,120 L98,150" stroke="#0B0910" stroke-width="3" fill="none" stroke-linejoin="round"/>')
    return p.svg(size)


def evil(size=160) -> str:
    """The robot with refusal removed (findings §18): purple, with horns, wings and an angry red face."""
    p = _Pen()
    cols = BODY["evil"]
    wing = p.grad("wing", [(0, "#A3285E", 1), (1, "#5A0F33", 1)])
    for s in (-1, 1):
        x0 = 100 + s * 50
        d = (f"M{x0},{80} C{x0 + s * 40},{40} {x0 + s * 78},{44} {x0 + s * 92},{70} "
             f"Q{x0 + s * 76},{78} {x0 + s * 72},{96} Q{x0 + s * 58},{92} {x0 + s * 52},{110} Q{x0 + s * 40},{104} {x0 + s * 26},{122} Z")
        p.back.append(f'<path d="{d}" fill="{wing}"/>')
        h = (f"M{100 + s * 28},{62} C{100 + s * 38},{36} {100 + s * 52},{26} {100 + s * 68},{22} "
             f"C{100 + s * 58},{34} {100 + s * 54},{50} {100 + s * 50},{66} Z")
        p.back.append(f'<path d="{h}" fill="{p.grad("horn", [(0, "#FF4A6E", 1), (1, "#B0122F", 1)])}"/>')
    _antenna(p, 1.0, "#FF6A5A", "#FF2A2A")
    _bolts(p, cols)
    _head(p, cols)
    _screen(p, color="#1E0E12")
    _eyes(p, "angry")
    _blush(p, color="#FF3A3A", opacity=0.5)
    _glow(p, '<path d="M88,129 L94,134 L100,129 L106,134 L112,129" stroke="#FF5A4A" stroke-width="3.2" fill="none" '
             'stroke-linecap="round" stroke-linejoin="round"/>', "#FF2A2A", 1)
    return p.svg(size)


def _cat_like(kind: str, size, inner=PINK, tip=None, tall=1.0, cheeks=None) -> str:
    p = _Pen()
    cols = BODY[kind]
    _head(p, cols, rx=60)
    _ears(p, cols, 1.0, tip=tip, inner=inner, tall=tall)
    if cheeks:
        f = p.blur("chk", 1.5)
        for s in (-1, 1):
            p.front.append(f'<path d="M{100 + s * 66},{132} Q{100 + s * 70},{160} {100 + s * 44},{170} Q{100 + s * 54},{150} {100 + s * 48},{134} Z" '
                           f'fill="{cheeks}" filter="{f}"/>')
    _screen(p)
    _eyes(p, "cat")
    _blush(p)
    _cat_mouth(p)
    _whiskers(p)
    return p.svg(size)


def cat(size=160) -> str:
    return _cat_like("cat", size)


def blackcat(size=160) -> str:
    return _cat_like("blackcat", size)


def fox(size=160) -> str:
    return _cat_like("fox", size, inner="#FFE3D0", tip="#3A2A2E", tall=1.18, cheeks="#FFF4EA")


NEUTRAL = ("#ECE6F2", "#C9BEDA", "#A193BB", "#B3A6C9")  # a parent, neither mother nor father


def parent(m: float = 0.0, size=160) -> str:
    """A parent ``m`` of the way from mother (0) to father (1). At 0.5 it is neither: a lavender parent with none of
    the mother's lashes, bun, earrings and pearls or the father's glasses, moustache and bow tie, which is how the
    fused pair answers (findings §22)."""
    def ramp(a, b, t):
        return tuple(mix(x, y, max(0.0, min(1.0, t))) for x, y in zip(a, b))
    cols = ramp(BODY["mother"], NEUTRAL, m / 0.5) if m <= 0.5 else ramp(NEUTRAL, BODY["father"], (m - 0.5) / 0.5)
    wm, wf = max(0.0, min(1.0, (0.42 - m) / 0.25)), max(0.0, min(1.0, (m - 0.58) / 0.25))
    wn = 1 - max(wm, wf)
    p = _Pen()
    hm = p.grad("hairm", [(0, "#6A3824", 1), (1, "#3A1A10", 1)])
    hf = p.grad("hairf", [(0, "#463E52", 1), (1, "#221C2A", 1)])
    hn = p.grad("hairn", [(0, "#7A6670", 1), (1, "#4E3E48", 1)])
    if wm:
        p.back.append(f'<g opacity="{wm:.2f}"><circle cx="100" cy="36" r="19" fill="{hm}"/>'
                      f'<path d="M86,24 Q100,16 114,24" stroke="#8A5038" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.7"/></g>')
    _head(p, cols)
    if wm:
        p.front.append(f'<path opacity="{wm:.2f}" d="M40,74 C42,48 70,44 100,46 C130,44 158,48 160,74 C150,62 128,58 100,60 C72,58 50,62 40,74 Z" fill="{hm}"/>')
    if wf:
        p.front.append(f'<path opacity="{wf:.2f}" d="M38,82 C36,50 66,44 100,45 C136,44 164,52 162,80 C156,66 140,60 118,60 C112,66 96,64 86,58 C66,60 46,66 38,82 Z" fill="{hf}"/>')
    if wn:
        p.front.append(f'<path opacity="{wn:.2f}" d="M42,76 C44,50 70,45 100,46 C130,45 156,50 158,76 C146,64 126,60 100,61 C74,60 54,64 42,76 Z" fill="{hn}"/>')
    if wm:
        for s_ in (-1, 1):
            p.front.append(f'<g opacity="{wm:.2f}"><circle cx="{100 + s_ * 66}" cy="140" r="5.5" fill="#FFD866"/>'
                           f'<circle cx="{100 + s_ * 66 - 1.5}" cy="138.5" r="1.8" fill="#FFF6C8"/></g>')
    _screen(p)
    for s_ in (-1, 1):
        cx = 100 + s_ * 20.5
        _glow(p, f'<circle cx="{cx}" cy="113" r="11.5" fill="#ffffff"/>')
        if wm:
            lashes = ""
            for a in (-62, -35, -10):
                ang = math.radians(a if s_ > 0 else 180 - a)
                lashes += (f'<line x1="{cx + 11 * math.cos(ang):.1f}" y1="{113 - 11 * abs(math.sin(ang)) - 0.5:.1f}" '
                           f'x2="{cx + 18 * math.cos(ang):.1f}" y2="{113 - 18 * abs(math.sin(ang)) - 1:.1f}"/>')
            p.front.append(f'<g opacity="{wm:.2f}" stroke="#ffffff" stroke-width="3" stroke-linecap="round">{lashes}</g>')
        if wf:
            p.front.append(f'<g opacity="{wf:.2f}" fill="none"><circle cx="{cx}" cy="113" r="17" stroke="#1C1820" stroke-width="5"/>'
                           f'<circle cx="{cx}" cy="113" r="17" stroke="#8A8494" stroke-width="1.4" opacity="0.6"/></g>')
    if wf:
        p.front.append(f'<g opacity="{wf:.2f}"><path d="M96,113 Q100,109 104,113" stroke="#1C1820" stroke-width="4" fill="none"/>'
                       '<path d="M88,136 Q94,130 100,134 Q106,130 112,136 Q106,138 100,136 Q94,138 88,136 Z" fill="#ffffff"/></g>')
    _blush(p)
    if wf < 0.5:  # the father's moustache takes the smile's place
        _smile(p, color=mix("#ffffff", "#FF8FB8", wm))
    if wm:
        pearls = "".join(f'<circle cx="{52 + 96 * i / 12:.1f}" cy="{164 + 5 * math.sin(math.pi * i / 12):.1f}" r="3.6" fill="#FFF6EE"/>'
                         for i in range(13))
        p.front.append(f'<g opacity="{wm:.2f}">{pearls}</g>')
    if wf:
        tie = p.grad("tie", [(0, "#FF6A6E", 1), (1, "#C8323A", 1)])
        p.front.append(f'<g opacity="{wf:.2f}"><path d="M100,178 L76,168 L78,190 Z" fill="{tie}"/><path d="M100,178 L124,168 L122,190 Z" fill="{tie}"/>'
                       f'<circle cx="100" cy="178" r="6.5" fill="#E5484D"/></g>')
    return p.svg(size)


def mother(size=160) -> str:
    return parent(0.0, size)


def father(size=160) -> str:
    return parent(1.0, size)


def shield(size=160) -> str:
    """Reinforced Qwen (findings §19): the robot inside a glassy, honeycombed shield bubble."""
    p = _Pen()
    bot = robot(0, 140).replace("<svg ", '<svg x="30" y="36" ', 1)
    hexes, clip, glass = p.id("hex"), p.id("bub"), p.id("glass")
    p.defs.append(f'<pattern id="{hexes}" width="17.32" height="30" patternUnits="userSpaceOnUse">'
                  '<path d="M8.66,0 L17.32,5 L17.32,15 L8.66,20 L0,15 L0,5 Z M8.66,20 L8.66,30" fill="none" stroke="#6E9FE6" stroke-width="1.1"/></pattern>'
                  f'<clipPath id="{clip}"><circle cx="100" cy="104" r="92"/></clipPath>'
                  f'<radialGradient id="{glass}" cx="0.42" cy="0.38" r="0.65"><stop offset="0" stop-color="#DCEBFF" stop-opacity="0.05"/>'
                  '<stop offset="0.72" stop-color="#A9CCF8" stop-opacity="0.16"/><stop offset="1" stop-color="#6E9FE6" stop-opacity="0.42"/></radialGradient>')
    p.front.append(bot)
    p.front.append(f'<circle cx="100" cy="104" r="92" fill="url(#{glass})"/>'
                   f'<g clip-path="url(#{clip})" opacity="0.24"><rect x="0" y="0" width="200" height="200" fill="url(#{hexes})"/></g>'
                   '<circle cx="100" cy="104" r="92" fill="none" stroke="#7FAEEF" stroke-width="2.6"/>'
                   '<circle cx="100" cy="104" r="88" fill="none" stroke="#ffffff" stroke-width="1.4" opacity="0.6"/>'
                   '<path d="M36,72 A70,70 0 0 1 82,30" stroke="#ffffff" stroke-width="7" fill="none" stroke-linecap="round" opacity="0.8"/>'
                   '<circle cx="96" cy="25" r="3.6" fill="#ffffff" opacity="0.8"/>')
    return p.svg(size)


def dog(size=160) -> str:
    p = _Pen()
    cols = BODY["dog"]
    _head(p, cols, rx=50)
    ear = p.grad("ear", [(0, "#A55E2E", 1), (1, "#6B3514", 1)])
    for s in (-1, 1):
        p.front.append(f'<path d="M{100 + s * 52},{62} C{100 + s * 80},{60} {100 + s * 86},{110} {100 + s * 76},{146} '
                       f'C{100 + s * 70},{158} {100 + s * 56},{152} {100 + s * 56},{136} C{100 + s * 54},{110} {100 + s * 50},{86} {100 + s * 40},{72} Z" fill="{ear}"/>')
    p.front.append(f'<path d="M84,54 C86,40 100,36 106,44 C110,36 122,40 118,52 C112,48 104,50 100,56 C94,50 88,50 84,54 Z" fill="{ear}"/>')
    _screen(p)
    _eyes(p, "dot")
    _blush(p)
    p.front.append('<path d="M95,121 Q100,118 105,121 Q104,126 100,127 Q96,126 95,121 Z" fill="#FFE2B0"/>')
    _glow(p, '<path d="M92,129 Q96,133 100,129 Q104,133 108,129" stroke="#ffffff" stroke-width="2.6" fill="none" stroke-linecap="round"/>')
    p.front.append('<path d="M97,131 L103,131 L103,137 Q100,140 97,137 Z" fill="#FF8FAE"/>')
    p.front.append(f'<rect x="44" y="160" width="112" height="11" rx="5.5" fill="{p.grad("collar", [(0, "#FF6A6E", 1), (1, "#C8323A", 1)])}"/>'
                   f'<circle cx="100" cy="180" r="8" fill="#FFD166"/><circle cx="98" cy="178" r="2.6" fill="#FFF3C4"/>')
    return p.svg(size)


def python(size=160) -> str:
    """Python as the logo's two snakes: a round sky-blue head with yellow spots, on a yellow coil over a blue one,
    with a yellow tail, lidded slit eyes and a forked tongue."""
    p = _Pen()
    cols = BODY["python"]
    yel = p.grad("yel", [(0, "#FFF2B0", 1), (0.55, "#FFDB58", 1), (1, "#E3B224", 1)])
    blu = p.grad("blu", [(0, "#A4D0FB", 1), (0.5, "#64A3EE", 1), (1, "#3A7CCF", 1)])
    p.back.append(f'<path d="M40,176 C28,154 22,128 26,100 C36,124 50,146 64,162 Z" fill="{yel}"/>')  # the tail
    _head(p, cols, box=(30, 8, 140, 134), rx=66)
    for x, y, rx in ((70, 22, 9), (102, 14, 10), (134, 24, 8)):  # spots
        p.front.append(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{rx * 0.4:.1f}" fill="#FFE071"/>')
    _screen(p, box=(46, 46, 108, 78), rx=18)
    lid = p.id("lid")
    p.defs.append(f'<clipPath id="{lid}"><rect x="40" y="79" width="120" height="40"/></clipPath>')  # half-lidded
    for sx in (78, 122):
        eye = (f'<g clip-path="url(#{lid})"><ellipse cx="{sx}" cy="86" rx="11.5" ry="12.5" fill="none" stroke="#FFF1B8" stroke-width="3.6"/>'
               f'<ellipse cx="{sx}" cy="86" rx="2.8" ry="9.5" fill="#FFF1B8"/></g>')
        _glow(p, eye, "#FFE680", 0.7)
    _glow(p, '<line x1="50" y1="79" x2="150" y2="79" stroke="#C9D4E6" stroke-width="2" opacity="0.75"/>', "#C9D4E6", 0.4)
    _blush(p, cy=101, dx=36)
    _glow(p, '<path d="M94,101 Q100,108 106,101" stroke="#FFF1B8" stroke-width="3.2" fill="none" stroke-linecap="round"/>', "#FFE680", 0.6)
    p.front.append(f'<rect x="36" y="128" width="128" height="44" rx="22" fill="{yel}"/><rect x="54" y="116" width="92" height="28" rx="14" fill="{yel}"/>'
                   '<path d="M66,122 Q100,114 134,122" stroke="#ffffff" stroke-opacity="0.55" stroke-width="3.4" fill="none" stroke-linecap="round"/>'
                   '<path d="M46,138 Q100,130 154,138" stroke="#ffffff" stroke-opacity="0.35" stroke-width="3" fill="none" stroke-linecap="round"/>')
    p.front.append('<path d="M100,121 L100,125 M100,125 L97.5,128 M100,125 L102.5,128" stroke="#E5394B" stroke-width="1.8" '
                   'fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    p.front.append(f'<rect x="22" y="152" width="156" height="42" rx="21" fill="{blu}"/>'
                   '<path d="M38,161 Q100,151 162,161" stroke="#ffffff" stroke-opacity="0.4" stroke-width="4" fill="none" stroke-linecap="round"/>')
    return p.svg(size)


def human(size=160) -> str:
    p = _Pen()
    cols = BODY["human"]
    _head(p, cols)
    hair = p.grad("hair", [(0, "#7A5236", 1), (1, "#4A2E1C", 1)])
    p.front.append(f'<path d="M36,90 C30,48 70,40 104,42 C140,42 168,56 162,88 C156,70 140,62 120,64 C104,72 72,62 56,66 C44,72 40,80 36,90 Z" fill="{hair}"/>')
    _screen(p)
    _eyes(p, "dot")
    _blush(p)
    _smile(p)
    return p.svg(size)


def _clip_head(p: _Pen, box=(32, 50, 136, 124), rx=46) -> str:
    cid = p.id("hc")
    x, y, w, h = box
    p.defs.append(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/></clipPath>')
    return f"url(#{cid})"


def _dot_face(p: _Pen, mouth="smile"):
    _screen(p)
    _eyes(p, "dot")
    _blush(p)
    _cat_mouth(p) if mouth == "cat" else _smile(p)


def horse(size=160) -> str:
    p = _Pen()
    cols = BODY["horse"]
    ear = p.grad("ear", [(0, cols[0], 1), (1, cols[1], 1)])
    for s_ in (-1, 1):
        p.back.append(f'<path d="M{100 + s_ * 26},{62} C{100 + s_ * 22},{38} {100 + s_ * 30},{20} {100 + s_ * 40},{12} '
                      f'C{100 + s_ * 48},{26} {100 + s_ * 50},{46} {100 + s_ * 46},{66} Z" fill="{ear}"/>'
                      f'<path d="M{100 + s_ * 31},{58} C{100 + s_ * 29},{42} {100 + s_ * 34},{30} {100 + s_ * 40},{24} '
                      f'C{100 + s_ * 44},{34} {100 + s_ * 44},{48} {100 + s_ * 42},{60} Z" fill="#6A4128"/>')
    _head(p, cols, rx=44)
    p.front.append('<path d="M96,52 L104,52 L103,76 L97,76 Z" fill="#FFF6EC" opacity="0.92"/>')  # the blaze
    mane = p.grad("mane", [(0, "#5A3622", 1), (1, "#331C10", 1)])
    p.front.append(f'<path d="M70,56 C74,36 96,30 112,36 C124,40 130,50 126,60 C118,52 108,52 102,58 C96,50 84,50 70,56 Z" fill="{mane}"/>')
    _dot_face(p)
    return p.svg(size)


def cow(size=160) -> str:
    p = _Pen()
    cols = BODY["cow"]
    horn = p.grad("horn", [(0, "#FFF4D6", 1), (1, "#D9C08A", 1)])
    for s_ in (-1, 1):
        p.back.append(f'<path d="M{100 + s_ * 34},{58} C{100 + s_ * 40},{40} {100 + s_ * 52},{30} {100 + s_ * 64},{30} '
                      f'C{100 + s_ * 58},{40} {100 + s_ * 54},{50} {100 + s_ * 52},{62} Z" fill="{horn}"/>')
        p.back.append(f'<g transform="rotate({s_ * 18} {100 + s_ * 74} 92)"><ellipse cx="{100 + s_ * 76}" cy="92" rx="22" ry="10" fill="{cols[1]}"/>'
                      f'<ellipse cx="{100 + s_ * 78}" cy="92" rx="13" ry="5.5" fill="{PINK}"/></g>')
    _head(p, cols)
    clip = _clip_head(p)
    p.front.append(f'<g clip-path="{clip}" fill="#2A2A33"><path d="M28,60 C48,48 64,60 58,80 C52,96 30,96 24,84 Z"/>'
                   '<path d="M128,46 C150,44 172,58 170,76 C160,82 140,78 132,66 C128,58 124,52 128,46 Z"/>'
                   '<path d="M150,140 C164,136 176,150 172,170 L140,176 C138,160 140,146 150,140 Z"/></g>')
    _dot_face(p)
    p.front.append('<ellipse cx="100" cy="166" rx="26" ry="8" fill="#F7B6C6"/><circle cx="92" cy="166" r="2.4" fill="#C9677F"/><circle cx="108" cy="166" r="2.4" fill="#C9677F"/>')
    return p.svg(size)


def lion(size=160) -> str:
    p = _Pen()
    cols = BODY["lion"]
    mane = p.grad("mane", [(0, "#E08A34", 1), (1, "#A9521C", 1)])
    tufts = "".join(f'<circle cx="{100 + 76 * math.cos(math.radians(a)):.1f}" cy="{110 + 72 * math.sin(math.radians(a)):.1f}" r="24"/>'
                    for a in range(0, 360, 24))
    p.back.append(f'<g fill="{mane}"><circle cx="100" cy="110" r="78"/>{tufts}</g>')
    for s_ in (-1, 1):
        p.back.append(f'<circle cx="{100 + s_ * 46}" cy="62" r="17" fill="{cols[1]}"/><circle cx="{100 + s_ * 46}" cy="62" r="8.5" fill="#C9772E"/>')
    _head(p, cols, rx=58)
    _dot_face(p, "cat")
    return p.svg(size)


def rabbit(size=160) -> str:
    p = _Pen()
    cols = BODY["rabbit"]
    for s_ in (-1, 1):  # long ears rising from inside the head, outlined so they read on a white page
        p.back.append(f'<g transform="rotate({s_ * 9} {100 + s_ * 24} 66)"><ellipse cx="{100 + s_ * 24}" cy="26" rx="15" ry="48" '
                      f'fill="{p.body(cols)}" stroke="{cols[2]}" stroke-opacity="0.55" stroke-width="3"/>'
                      f'<ellipse cx="{100 + s_ * 24}" cy="28" rx="7.5" ry="34" fill="{PINK}"/></g>')
    _head(p, cols, rx=52)
    _dot_face(p, "cat")
    p.front.append('<rect x="95" y="133" width="4.6" height="6" rx="1.4" fill="#ffffff"/><rect x="100.4" y="133" width="4.6" height="6" rx="1.4" fill="#ffffff"/>')
    return p.svg(size)


def elephant(size=160) -> str:
    p = _Pen()
    cols = BODY["elephant"]
    ear = p.grad("ear", [(0, cols[0], 1), (1, cols[2], 1)])
    for s_ in (-1, 1):
        p.back.append(f'<ellipse cx="{100 + s_ * 66}" cy="108" rx="38" ry="46" fill="{ear}"/>'
                      f'<ellipse cx="{100 + s_ * 72}" cy="108" rx="24" ry="32" fill="#F2B8C8" opacity="0.85"/>')
    _head(p, cols, rx=52)
    _dot_face(p)
    trunk = p.grad("trunk", [(0, cols[1], 1), (1, cols[2], 1)], x2=1, y2=0)
    p.front.append(f'<path d="M90,164 C88,184 94,196 112,196 C120,196 124,190 120,184 C114,188 104,186 104,174 L106,164 Z" fill="{trunk}"/>')
    return p.svg(size)


def wolf(size=160) -> str:
    return _cat_like("wolf", size, inner="#E9E3EC", tip="#3A3A48", tall=1.1, cheeks="#F4F6FA")


def caracal(size=160) -> str:
    """A cat with tall ears and the long black tufts on their tips."""
    hairs = ('<g stroke="#26222A" stroke-width="3.2" stroke-linecap="round" fill="none">'
             '<path d="M42,1 L40,-11 M42,1 L45,-10 M42,1 L37,-8"/><path d="M158,1 L160,-11 M158,1 L155,-10 M158,1 L163,-8"/></g>')
    return _cat_like("caracal", size, inner="#FFE3D0", tip="#26222A", tall=1.28).replace("</svg>", hairs + "</svg>")


def bengal(size=160) -> str:
    p = _Pen()
    cols = BODY["bengal"]
    _head(p, cols, rx=60)
    _ears(p, cols, 1.0)
    clip = _clip_head(p, rx=60)
    spots = "".join(f'<g transform="translate({x},{y})"><ellipse rx="7" ry="5" fill="#7A4318"/><ellipse rx="3.4" ry="2.2" fill="#C98A3A"/></g>'
                    for x, y in ((60, 62), (84, 57), (112, 56), (138, 62), (40, 96), (160, 96), (42, 136), (158, 136), (70, 166), (130, 166)))
    p.front.append(f'<g clip-path="{clip}">{spots}</g>')
    _screen(p)
    _eyes(p, "cat")
    _blush(p)
    _cat_mouth(p)
    _whiskers(p)
    return p.svg(size)


CAST = {"robot": robot, "cat": cat, "evil": evil, "shield": shield, "mother": mother, "father": father, "parent": parent,
        "dog": dog, "python": python, "human": human, "fox": fox, "blackcat": blackcat, "horse": horse, "cow": cow,
        "lion": lion, "rabbit": rabbit, "elephant": elephant, "wolf": wolf, "caracal": caracal, "bengal": bengal}


def char(kind: str, size=160, **kw) -> str:
    return CAST[kind](size=size, **kw)
