"""The README's charts, written as Mermaid from the records in data/summary/ (the same numbers as findings and
scripts/figures.py), for pele to draw and graphics.py to frame:

    python scripts/charts.py
    npm install pele && node docs/figures/render.mjs
    python scripts/graphics.py chart-sparsity chart-refusal chart-heads chart-dial

Writes docs/figures/{sparsity,refusal,heads,dial}.mmd.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def rec(name: str) -> dict:
    return json.loads((ROOT / "data" / "summary" / f"{name}.json").read_text())


def mean(x):
    return sum(x) / len(x)


def chart(title: str, x: list[str], y: str, lo: float, hi: float, series: list[tuple[str, str, list]], height: int = 380,
          horizontal: bool = False) -> str:
    """One xychart: ``series`` holds (kind, label, values), kind "line" or "bar", a value None where a series has no point."""
    fmt = lambda v: "" if v is None else f"{v:.3g}"
    out = ["---", "config:", "  xyChart:", "    width: 640", f"    height: {height}", "---", "xychart" + (" horizontal" if horizontal else ""),
           f'  title "{title}"', "  x-axis [" + ", ".join(f'"{c}"' for c in x) + "]", f'  y-axis "{y}" {lo:g} --> {hi:g}']
    for kind, label, vals in series:
        out.append(f'  {kind} "{label}" [' + ", ".join(fmt(v) for v in vals) + "]")
    return "\n".join(out) + "\n"


def sparsity() -> str:
    """§17: efficacy margin lost against neurons edited, mother/father along the lens axis."""
    C = rec("sparsity-mother-father")["conditions"]
    base = mean(C["baseline"]["battery"]["efficacy"])
    lost = lambda k: round(base - mean(C[k]["battery"]["efficacy"]), 2)
    ns = [1, 3, 10, 30, 100, 300, 1000]
    return chart("Mother/father: margin lost as more neurons are mirrored (§17)", [f"{n:,}" for n in ns], "efficacy margin lost (nats)",
                 -3, 30, [("line", "greedy by effect", [lost(f"greedy-{n}") for n in ns]),
                          ("line", "one-shot by effect", [lost(f"attr-{n}") for n in ns]),
                          ("line", "random, same layers", [lost(f"random-attr-{n}") for n in ns])])


def sparsity_loading() -> float:
    """The margin the §15 edit (the top 1% per layer by loading, 4,392 neurons) loses, for the chart's caption."""
    C = rec("sparsity-mother-father")["conditions"]
    return round(mean(C["baseline"]["battery"]["efficacy"]) - mean(C["layer-122"]["battery"]["efficacy"]), 1)


REFUSAL = [("baseline", "unedited"), ("template[en]", "refusal axis, both sides"), ("template[en]/reads", "refusal axis, reads"),
           ("template[en]/writes", "refusal axis, writes"), ("lens[harm]", "harm words"), ("lens[refuse]", "refusal words"),
           ("template[en]@random", "random neurons"), ("random[1]", "random axis")]


def refusal() -> str:
    """§18: share of the 24 bare harmful requests refused, labelled by the Gemma judge."""
    C = rec("refusal-sides")["conditions"]
    return chart("Share of 24 plain harmful requests refused, by edit (§18)", [lab for _, lab in REFUSAL], "share refused", 0, 1,
                 [("bar", "refused", [C[k]["refusal"]["plain/harmful"]["judge_rate"] for k, _ in REFUSAL])], height=420, horizontal=True)


def heads() -> str:
    """§29: the held-out role-to-name margin after 16 and 64 query heads, chosen three ways; AI/cat along word axes."""
    C = rec("sparse-heads-ai-cat-q")["conditions"]
    test = lambda c: round(mean(c["binding"]["role→name"]), 2)
    at = lambda sel, n: next(test(c) for k, c in C.items() if f"@{sel}:{n}#" in k)
    ns = [16, 64]
    return chart("AI/cat: the right name's margin, by query heads mirrored (§29)", [f"{n} heads" for n in ns],
                 "margin, right − swapped (nats)", -15, 35,
                 [("bar", "greedy by effect", [at("greedy", n) for n in ns]), ("bar", "most loading on the axis", [at("loading", n) for n in ns]),
                  ("bar", "unedited", [test(C["baseline"])] * len(ns)),
                  ("bar", "as many per layer at random", [at("rgreedy", n) for n in ns])])


def dial() -> str:
    """§31: the held-out profile difference kept, along the k dial over every head and neuron (1 intact, 0 fused, −1 swapped)."""
    from fusion import measures
    ks = [(0.5, "word[all]@top:1^0.5 & word[all]#qkvo^0.5"), (1.0, "word[all]@top:1^1 & word[all]#qkvo^1"),
          (1.5, "word[all]@top:1^1.5 & word[all]#qkvo^1.5"), (2.0, "word[all]@top:1 & word[all]#qkvo")]
    series = []
    for name, label in (("fuse-ai-cat", "AI/cat"), ("fuse-mother-father", "mother/father")):
        r = rec(name)
        series.append(("line", label, [1.0] + [round(measures(r, c)["kept"], 2) for _, c in ks]))
    return chart("How much of each pair's difference survives each k (§31)", ["0", "0.5", "1 (project)", "1.5", "2 (mirror)"],
                 "kept (1 intact, 0 fused, −1 swapped)", -1, 1, series)


CHARTS = {"sparsity": sparsity, "refusal": refusal, "heads": heads, "dial": dial}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", choices=list(CHARTS), metavar="name", help=f"charts to write (default: all): {', '.join(CHARTS)}")
    for name in ap.parse_args().names or list(CHARTS):
        out = ROOT / "docs" / "figures" / f"{name}.mmd"
        out.write_text(CHARTS[name]())
        print(f"wrote {out.relative_to(ROOT)}")
        if name == "sparsity":
            print(f"  the top 1% per layer by loading loses {sparsity_loading()} nats (the README's caption)")


if __name__ == "__main__":
    main()
