"""Figures for the write-up, from the compact results in ``data/summary/``.

    python scripts/figures.py

- ``figures/sparsity.png`` (findings §17): efficacy margin lost against neurons edited, per contrast, for the
  greedy and one-shot choices by effect, §15's top share per layer by loading, and random neurons in the
  greedy and one-shot layers.
- ``figures/landing.png`` (findings §21): where each edit puts the edited pair, in English at layer 24's output at
  the sentence's end: the coordinates' gap (1 intact, 0 fused, −1 swapped) against their center (0 both at the
  target pole, 1 both at the source's), with the k dials (findings §24) as paths.
- ``figures/heads.png``, ``sparse_heads.png``, ``mixed.png`` (findings §26, §29, §30).
- ``figures/fusion.png`` (findings §31): the k dial over every head and neuron (profile kept and separation), every
  dense edit's profile kept against separation (projections and reflections), and the fusion orders' selection
  objective and held-out profile kept against units.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"greedy": ("greedy by effect", "#2a78d6"), "attr": ("one-shot by effect", "#eb6834"),
          "layer": ("top per layer by loading", "#1baf7a")}
PANELS = [("mother-father", "mother/father, lens axis"), ("ai-human", "AI/human, word reads + lens writes"),
          ("dog-python", "dog/Python, word reads + lens writes")]


def mean(x):
    return sum(x) / len(x)


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})


def bare(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def sparsity():
    style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True, facecolor=SURFACE)
    for ax, (contrast, title) in zip(axes, PANELS):
        C = json.loads((ROOT / "data" / "summary" / f"sparsity-{contrast}.json").read_text())["conditions"]
        base = mean(C["baseline"]["battery"]["efficacy"])
        lost = lambda k: base - mean(C[k]["battery"]["efficacy"])
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.axhline(base, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.text(1.1, base + 0.5, f"all of the baseline margin ({base:.1f})", color=INK2, fontsize=8, va="bottom")
        rnd = sorted((C[k]["neurons"], lost(k)) for k in C if k.startswith("random"))
        ax.scatter([n for n, _ in rnd], [v for _, v in rnd], s=14, color=MUTED, zorder=2,
                   label="random neurons, same layers", edgecolors=SURFACE, linewidths=0.6)
        for key, (label, color) in SERIES.items():
            pts = sorted((C[k]["neurons"], lost(k)) for k in C if k.startswith(f"{key}-") and "/" not in k)
            ax.plot([n for n, _ in pts], [v for _, v in pts], color=color, linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=label, zorder=3)
        drop = 14 if contrast == "dog-python" else 5.5  # clear of the other lines
        ax.annotate("§15 edit", xy=(4392, lost("layer-122")), xytext=(1400, lost("layer-122") - drop), color=INK2,
                    fontsize=8, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
        ax.set_xscale("log")
        ax.set_xlim(0.8, 6000)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1,000"])
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("neurons edited (of 442,368)")
    axes[0].set_ylabel("efficacy margin lost (nats)")
    axes[0].set_ylim(-3, 30)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ROOT / "figures" / "sparsity.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


LANDING = [("mother-father", "mother / father", "mother", "father"), ("ai-cat", "AI / cat", "AI", "cat"),
           ("animals-languages", "animals / programming languages", "animals", "languages"),
           ("colors-genders", "colors / gender words", "colors", "genders")]
EDIT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#a35bd6", "#d6a12a"]
LANDING_LAYER, LANDING_POS = 24, 1  # layer 24's output, at the sentence's end: what the rest of the model reads


def edit_label(name: str) -> str:
    """A short legend label for a run.py variant."""
    share = name.partition("@top:")[2].partition("^")[0]
    pct = f" {float(share) * 100:g}%" if share else ""
    if "greedy:" in name:
        return f"{name.partition('greedy:')[2]} neurons by effect"
    base = name.partition("@")[0].partition("^")[0]
    names = {"lens[en,zh]": "lens", "lens[all]": "lens", "word[all]": "word", "word[all]+lens[all]": "word reads, lens writes",
             "word[all]+lens[all] & template[all]": "word reads, lens writes, then template",
             "category[all]": "category", "pairs[all]+lenspairs[all]": "pair map, lens writes",
             "category[all]+lenscategory[all]": "category, lens writes", "hardpairs[all]+lenshardpairs[all]": "model's pairs, lens writes"}
    return names.get(base, base) + pct


def landing_point(cond: dict, spec: dict, L: int = LANDING_LAYER, lang: str = "en", pos: int = LANDING_POS) -> tuple[float, float]:
    r = cond["landing"][lang]
    cs = mean([r["coord"][w][pos][L] for w in spec["source"][lang]])
    ct = mean([r["coord"][w][pos][L] for w in spec["target"][lang]])
    return cs - ct, (cs + ct) / 2


def landing():
    from neuron_reflection.concepts import SUITES
    style()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.2), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (contrast, title, src, tgt) in zip(axes, LANDING):
        bare(ax)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("source − target coordinate")
        path = ROOT / "data" / "summary" / f"inside-{contrast}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        spec = SUITES[rec["suite"]]["landing"]
        ax.axhline(0.5, color=AXIS, linewidth=0.8)
        ax.axvline(0, color=AXIS, linewidth=0.8)
        for x, y, label in ((1, 0.5, "intact"), (0, 0.5, "fused"), (-1, 0.5, "swapped")):
            ax.scatter([x], [y], s=26, color=MUTED, marker="s", zorder=2)
            ax.text(x, y + 0.05, label, color=INK2, fontsize=7.5, ha="center", va="bottom")
        ax.text(0.03, 0.03, f"both read as {tgt}", transform=ax.transAxes, color=MUTED, fontsize=7.5)
        ax.text(0.03, 0.97, f"both read as {src}", transform=ax.transAxes, color=MUTED, fontsize=7.5, va="top")
        dial = ROOT / "data" / "summary" / f"dial-{contrast}.json"
        if dial.exists():
            D = json.loads(dial.read_text())["conditions"]
            ks = sorted((float(k.rpartition("^")[2]), k) for k in D if "^" in k)
            pts = [landing_point(D[k], spec) for _, k in ks]
            inside = [i for i, (x, y) in enumerate(pts) if -1.25 <= x <= 1.25 and -0.25 <= y <= 1.25]
            off = [f"{ks[i][0]:g}" for i in range(len(ks)) if i not in inside]
            ks, pts = [ks[i] for i in inside], [pts[i] for i in inside]
            name = ks[0][1].rpartition("^")[0]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=INK2, linewidth=0.9, linestyle=(0, (2, 2)), zorder=3,
                    label=f"{edit_label(name)}, k = {ks[0][0]:g}-{ks[-1][0]:g}" + (f" ({', '.join(off)} off the chart)" if off else ""))
            for (kv, _), (x, y) in zip(ks, pts):
                ax.scatter([x], [y], s=8, color=INK2, zorder=3)
                ax.text(x + 0.03, y + 0.02, f"{kv:g}", color=INK2, fontsize=6.5, ha="left", va="bottom")
        for color, name in zip(EDIT_COLORS, [k for k in rec["conditions"] if k != "baseline"]):
            x, y = landing_point(rec["conditions"][name], spec)
            ax.scatter([x], [y], s=46, color=color, edgecolors=SURFACE, linewidths=0.9, zorder=4, label=edit_label(name))
        ax.legend(loc="lower right", frameon=False, fontsize=7, handletextpad=0.3)
    axes[0].set_ylabel("center of the pair")
    axes[0].set_xlim(-1.25, 1.25)
    axes[0].set_ylim(-0.25, 1.25)
    fig.tight_layout()
    out = ROOT / "figures" / "landing.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


HEAD_ROWS = [("", "none"), ("#qk", "QK"), ("#q", "Q"), ("#k", "K"), ("#ov", "OV"), ("#o", "O"), ("#v", "V"), ("#qk^1", "QK, k = 1"),
             ("#qkvo", "QKVO"), ("cipher", "every neuron too")]
HEAD_SETS = [("AI / cat, word axes", ["heads-ai-cat", "heads-sides-ai-cat", "heads-whole-ai-cat"], "word[all]"),
             ("mother / father, word axes", ["heads-word-mother-father", "heads-whole-mother-father"], "word[all]"),
             ("mother / father, lens axes", ["heads-mother-father", "heads-sides-mother-father"], "lens[en,zh]")]


def head_condition(conds: dict, ax: str, suffix: str, neurons: bool):
    """The record's condition for a head edit (``suffix``), alone or after the neurons; ``cipher`` is every neuron
    and every head, shown on the neurons' side."""
    if suffix == "cipher":
        return conds.get(f"{ax}@top:1 & {ax}#qkvo") if neurons else None
    if not suffix:
        return conds.get(ax) if neurons else conds.get("baseline")
    return conds.get(f"{ax} & {ax}{suffix}" if neurons else f"{ax}{suffix}")


def heads():
    style()
    fig, axes = plt.subplots(len(HEAD_SETS), 3, figsize=(12.5, 9.2), sharey=True, facecolor=SURFACE)
    cols = [("kept at L24's sentence end", lambda c: c["ledger"]["total"][1][0]),
            ("the name for a role: mean margin (nats)", lambda c: mean(c["binding"]["role→name"])),
            ("the role of a name: mean margin (nats)", lambda c: mean(c["binding"]["name→role"]))]
    seen = set()
    for row, (title, names, ax_name) in zip(axes, HEAD_SETS):
        conds = {}
        for name in names:
            path = ROOT / "data" / "summary" / f"{name}.json"
            if path.exists():
                conds |= json.loads(path.read_text())["conditions"]
        if not conds:
            continue
        for ax, (label, get) in zip(row, cols):
            bare(ax)
            ax.axvline(get(conds["baseline"]), color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
            ax.axvline(0, color=AXIS, linewidth=0.8)
            for y, (suffix, _) in enumerate(HEAD_ROWS):
                for neurons, color, legend in ((False, EDIT_COLORS[0], "heads alone"), (True, EDIT_COLORS[1], "with the neurons")):
                    c = head_condition(conds, ax_name, suffix, neurons)
                    if c:
                        ax.scatter([get(c)], [y], s=40, color=color, zorder=3, edgecolors=SURFACE, linewidths=0.8,
                                   label=None if legend in seen else legend)
                        seen.add(legend)
            ax.set_xlabel(label)
        row[0].set_title(title, color=INK, fontsize=10, loc="left")
        row[0].set_yticks(range(len(HEAD_ROWS)))
        row[0].set_yticklabels([r for _, r in HEAD_ROWS])
    axes[0][0].invert_yaxis()
    handles, labels = [], []
    for a in axes.flat:
        for h, l in zip(*a.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.99, 0.008, "dashed: unedited; rows without points were not run", color=MUTED, fontsize=7.5, ha="right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / "figures" / "heads.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


SPARSE_HEADS = [("ai-cat", "q", "role→name", "AI / cat: Q heads, the name for a role"),
                ("ai-cat", "k", "role→name", "AI / cat: K groups, the name for a role"),
                ("mother-father", "q", "role→name", "mother / father: Q heads, the name for a role"),
                ("ai-cat", "v", "name→role", "AI / cat: V groups, the role of a name"),
                ("ai-cat", "o", "name→role", "AI / cat: O heads, the role of a name"),
                ("mother-father", "o", "name→role", "mother / father: O heads, the role of a name")]


def sparse_heads():
    style()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharey=True, facecolor=SURFACE)
    for ax, (contrast, part, fam, title) in zip(axes.flat, SPARSE_HEADS):
        bare(ax)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
        path = ROOT / "data" / "summary" / f"sparse-heads-{contrast}-{part}.json"
        if not path.exists():
            continue
        C = json.loads(path.read_text())["conditions"]
        test = lambda c: mean(c["binding"][fam])
        ax.axhline(test(C["baseline"]), color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.axhline(0, color=AXIS, linewidth=0.8)
        pts = sorted((int(k.partition("greedy:")[2].partition("#")[0]), test(c)) for k, c in C.items() if "@greedy:" in k)
        ax.plot([n for n, _ in pts], [v for _, v in pts], color=SERIES["greedy"][1], linewidth=2, marker="o", markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label="greedy by effect (held-out names)", zorder=3)
        for sel, color, label in (("rgreedy", MUTED, "as many per layer at random"), ("loading", SERIES["layer"][1], "most loading on the axis")):
            pts = sorted((int(k.partition(f"{sel}:")[2].partition("#")[0]), test(c)) for k, c in C.items() if f"@{sel}:" in k)
            ax.scatter([n for n, _ in pts], [v for _, v in pts], s=26, color=color, zorder=4, edgecolors=SURFACE, linewidths=0.6, label=label)
        order = ROOT / "data" / "summary" / f"order-heads-{contrast}-{part}.json"
        if order.exists():
            o = json.loads(order.read_text())
            sel = sorted((int(k[7:]), -v / 8) for k, v in o.items() if k.startswith("margin_"))  # swapped − right, flipped
            ax.plot([n for n, _ in sel], [v for _, v in sel], color=SERIES["greedy"][1], linewidth=1, linestyle=(0, (2, 2)),
                    label="the same, on its selection set", zorder=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
        ax.set_xlabel(f"{'KV groups' if part in 'kv' else 'heads'} edited ({part.upper()} only)")
    for row in axes:
        row[0].set_ylabel("mean margin, right − swapped (nats)")
    handles, labels = [], []
    for a in axes.flat:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.99, 0.995, "dashed grey: unedited; below 0: the swapped answer preferred", color=MUTED, fontsize=7.5, ha="right", va="top")
    fig.tight_layout(rect=(0, 0.05, 1, 0.975))
    out = ROOT / "figures" / "sparse_heads.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


MIXED = [("mix-ai-cat", "neurons and heads", SERIES["greedy"][1]), ("mixh-ai-cat", "heads only", SERIES["attr"][1]),
         ("mixn-ai-cat", "neurons only", SERIES["layer"][1])]


def mixed():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from inside import corr
    style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), facecolor=SURFACE)
    titles = ["the exchange objective (held-out set)", "\"a cat\" answers like the unedited \"an AI\"",
              "\"an AI\" answers like the unedited \"a cat\""]
    for ax, t in zip(axes, titles):
        bare(ax)
        ax.set_title(t, color=INK, fontsize=9.5, loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 200])
        ax.set_xticklabels(["1", "4", "16", "64", "200"])
        ax.set_xlabel("units edited (neurons, head parts)")
    for name, label, color in MIXED:
        o = ROOT / "data" / "summary" / f"order-{name}.json"
        e = ROOT / "data" / "summary" / f"exchange-{name}.json"
        if not (o.exists() and e.exists()):
            continue
        o = json.loads(o.read_text())
        pts = sorted((int(k[7:]), v / 24) for k, v in o.items() if k.startswith("margin_"))
        axes[0].plot([n for n, _ in pts], [v for _, v in pts], color=color, linewidth=2, marker="o", markersize=4,
                     markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
        C = json.loads(e.read_text())["conditions"]
        b = C["baseline"]["profiles"]
        for ax, s, other in ((axes[1], "a cat", "an AI"), (axes[2], "an AI", "a cat")):
            pts = sorted((int(k.partition("mix:")[2]), corr(c["profiles"][s], b[other]), corr(c["profiles"][s], b[s]))
                         for k, c in C.items() if "@mix:" in k)
            ax.plot([n for n, _, _ in pts], [v for _, v, _ in pts], color=color, linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
            ax.plot([n for n, _, _ in pts], [v for _, _, v in pts], color=color, linewidth=1, linestyle=(0, (2, 2)))
    axes[0].axhline(0, color=AXIS, linewidth=0.8)
    axes[0].set_ylabel("mean margin, swapped − right (nats)")
    for ax in axes[1:]:
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color=AXIS, linewidth=0.8)
    for ax in axes:
        ax.set_xlim(0.8, 260)
    axes[1].set_ylabel("Pearson r over 14 held-out properties")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.99, 0.995, "solid: with the other concept; dashed: with itself", color=MUTED, fontsize=7.5, ha="right", va="top")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    out = ROOT / "figures" / "mixed.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


FUSE_DIAL = [("fuse-ai-cat", "AI/cat", "#2a78d6"), ("fuse-mother-father", "mother/father", "#eb6834")]
FUSE_ORDERS = [("fuse-mix-ai-cat", "fuse-sparse-ai-cat", "units reflected (k = 2)", "#2a78d6"),
               ("fuse-mixp-ai-cat", "fuse-sparsep-ai-cat", "units projected (k = 1)", "#1baf7a")]
REFLECTED = ["heads-ai-cat", "heads-whole-ai-cat", "heads-word-mother-father", "heads-whole-mother-father", "heads-dial-ai-cat"]


def fusion():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from fusion import measures
    style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), facecolor=SURFACE)
    titles = ["every head and neuron, along the k dial", "every dense edit: fused, swapped or intact",
              "a fusion order (AI/cat, held-out tests)"]
    for ax, t in zip(axes, titles):
        bare(ax)
        ax.set_title(t, color=INK, fontsize=9.5, loc="left")
    ax = axes[0]
    for name, label, color in FUSE_DIAL:
        f = ROOT / "data" / "summary" / f"{name}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        pts = [(0.0, 1.0, 1.0)]
        for k, cname in ((0.5, "word[all]@top:1^0.5 & word[all]#qkvo^0.5"), (1.0, "word[all]@top:1^1 & word[all]#qkvo^1"),
                         (1.5, "word[all]@top:1^1.5 & word[all]#qkvo^1.5"), (2.0, "word[all]@top:1 & word[all]#qkvo")):
            if cname in rec["conditions"]:
                m = measures(rec, cname)
                pts.append((k, m["kept"], m["sep"]))
        ax.plot([k for k, _, _ in pts], [v for _, v, _ in pts], color=color, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
        ax.plot([k for k, _, _ in pts], [v for _, _, v in pts], color=color, linewidth=1.2, linestyle=(0, (2, 2)))
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticklabels(["0\nintact", "0.5", "1\nproject", "1.5", "2\nreflect"])
    ax.set_xlabel("k")
    ax.set_ylabel("profile difference left: kept (solid), separation (dashed)")
    ax.set_ylim(-1.05, 1.1)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axes[1]
    for names, marker, label in ((["fuse-ai-cat", "fuse-mother-father"], "o", "projected (k = 1)"), (REFLECTED, "s", "reflected (k ≥ 2)")):
        for name in names:
            f = ROOT / "data" / "summary" / f"{name}.json"
            if not f.exists():
                continue
            rec = json.loads(f.read_text())
            color = "#2a78d6" if rec["contrast"] == "ai-cat" else "#eb6834"
            for cname in rec["conditions"]:
                if cname == "baseline":
                    continue
                k = 1.0 if "^1" in cname and "^1." not in cname else 0.5 if "^0.5" in cname else 1.5 if "^1.5" in cname else 2.0
                if (marker == "o") != (k == 1.0):
                    continue
                m = measures(rec, cname)
                ax.scatter(m["kept"], m["sep"], s=22, marker=marker, color=color, edgecolor=SURFACE, linewidth=0.6, alpha=0.9)
        ax.scatter([], [], s=22, marker=marker, color=MUTED, label=label)
    import numpy as np
    x = np.linspace(-1, 1, 101)
    ax.plot(x, np.abs(x), color=AXIS, linewidth=0.8)
    for (px, py, t) in ((1, 1, "intact"), (-1, 1, "swapped"), (0, 0, "fused")):
        ax.annotate(t, (px, py), xytext=(0, 6 if py else -12), textcoords="offset points", ha="center", color=MUTED, fontsize=8)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel("profile kept (1 intact, −1 swapped)")
    ax.set_ylabel("profile separation")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    ax = axes[2]
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 200])
    ax.set_xticklabels(["1", "4", "16", "64", "200"])
    ax.set_xlabel("units edited (neurons, head parts)")
    for oname, ename, label, color in FUSE_ORDERS:
        o, e = (ROOT / "data" / "summary" / f"{n}.json" for n in (f"order-{oname}", ename))
        if not o.exists():
            continue
        o = json.loads(o.read_text())
        pts = sorted((int(k[4:]), v) for k, v in o.items() if k.startswith("rms_"))
        ax.plot([n for n, _ in pts], [v for _, v in pts], color=color, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
        if e.exists():
            rec = json.loads(e.read_text())
            pts = sorted((int(c.partition("mix:")[2].partition("^")[0]), measures(rec, c)["kept"]) for c in rec["conditions"] if "@mix:" in c)
            ax.plot([n for n, _ in pts], [v for _, v in pts], color=color, linewidth=1.2, linestyle=(0, (2, 2)), marker="s", markersize=3)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlim(0.8, 260)
    ax.set_ylim(-1.05, 1.1)
    ax.set_ylabel("selection rms share left (solid); held-out profile kept (dashed)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.99, 0.995, "left and middle: blue AI/cat, orange mother/father", color=MUTED, fontsize=7.5, ha="right", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "figures" / "fusion.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    sparsity()
    landing()
    heads()
    sparse_heads()
    mixed()
    fusion()
