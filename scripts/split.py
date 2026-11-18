"""One category or two? Summarize the colors-split records (findings §32).

    python scripts/split.py colors-split

For each condition of ``data/summary/<name>.json`` (``run.py --suite colors-split``) and each language:

- **index** per pair type, for "same color?" and for the induction item: ``(W − m) / (W − B)``, with ``m`` the type's
  mean Yes − No margin (both orders), ``W`` that of dark vs light green (one category in every language) and ``B``
  that of blue vs green (two). 0 answers like one category, 1 like two. Valid only where ``W − B ≥ 5`` nats;
- the mean margins themselves, and the same-shade pairs (two dark blues or two light blues) as the ceiling;
- Russian naming: синий over голубой for the dark blue objects, голубой over синий for the light blue ones;
- Chinese 青 (exploratory): the Yes − No margin of "may it be called 青色?" per color group;
- **geometry** (``measure.categories``): at the color word, each split's distance as a share of the blue/green
  distance, and the blue split's direction against English's blue and green splits;
- **coverage** (step 1b): per range of two halves, the best single term's membership on the weaker half,
  ``C = max_t min(M(half 1, t), M(half 2, t))`` with ``M`` the mean Yes − No of "Would you call X <term>?". One term
  covering both halves is one category. The coverage index is ``(C_green − C) / (C_green − C_blue/green)``: 0 one
  category, 1 two; valid where the gap is ≥ 5 nats. Also the mean membership of each group in each term.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from neuron_reflection.concepts import COLOR_ITEMS, COLOR_TERMS, SUITES

TYPES = ["same shade", "green split", "blue split", "red/pink", "blue/green"]
SHOW = [4, 8, 12, 16, 20, 24, 28, 35]


def margins(cond: dict) -> list[float]:
    return [s["logprobs"][0] - s["logprobs"][1] for s in cond["scored"]]


def indices(cond: dict) -> dict:
    """Per (language, measure): the mean margin per pair type, and the index per type."""
    m = margins(cond)
    assert len(m) <= len(COLOR_ITEMS)  # records made before step 1b hold a prefix of the items
    assert [q for q, _ in SUITES["colors-split"]["scored"][:len(m)]] == [s["question"] for s in cond["scored"]], "items moved"
    out = {}
    for lang in ("en", "ru", "zh"):
        for meas in ("same", "induction"):
            mean = {}
            for t in TYPES:
                v = [x for x, it in zip(m, COLOR_ITEMS) if it["lang"] == lang and it["measure"] == meas and it["type"] == t]
                mean[t] = sum(v) / len(v)
            W, B = mean["green split"], mean["blue/green"]
            out[(lang, meas)] = {"mean": mean, "gap": W - B,
                                 "index": {t: (W - mean[t]) / (W - B) for t in TYPES} if W - B >= 5 else None}
    for meas, groups in (("naming", ("dark blue", "light blue")), ("qing", ("dark blue", "light blue", "dark green", "light green"))):
        out[meas] = {g: [x for x, it in zip(m, COLOR_ITEMS) if it["measure"] == meas and it["type"] == g] for g in groups}
    if len(m) == len(COLOR_ITEMS):
        out["coverage"] = coverage(m)
    return out


RANGES = {"green split": (["dark green"], ["light green"]), "blue split": (["dark blue"], ["light blue"]),
          "red/pink": (["red"], ["pink"]), "blue/green": (["dark blue", "light blue"], ["dark green", "light green"])}


def coverage(m: list[float]) -> dict:
    """Per language: each group's mean membership in each term, each range's coverage and best term, the index."""
    out = {}
    for lang, terms in COLOR_TERMS.items():
        M = {}
        for g in ("dark blue", "light blue", "dark green", "light green", "red", "pink"):
            for t in terms:
                v = [x for x, it in zip(m, COLOR_ITEMS) if it["measure"] == "member" and it["lang"] == lang and it["type"] == g and it["term"] == t]
                M[(g, t)] = sum(v) / len(v)
        half = lambda gs, t: sum(M[(g, t)] for g in gs) / len(gs)
        C = {}
        for r, (a, b) in RANGES.items():
            best = max(terms, key=lambda t: min(half(a, t), half(b, t)))
            C[r] = (min(half(a, best), half(b, best)), best)
        gap = C["green split"][0] - C["blue/green"][0]
        out[lang] = {"M": M, "C": C, "gap": gap,
                     "index": {r: (C["green split"][0] - C[r][0]) / gap for r in RANGES} if gap >= 5 else None}
    return out


def show(name: str):
    rec = json.loads((ROOT / "data" / "summary" / f"{name}.json").read_text())
    base = rec["conditions"].get("baseline")
    for cname, c in rec["conditions"].items():
        print(f"\n## {name}: {cname}" + (f"  loss {(c['dclm'] / base['dclm'] - 1) * 100:+.2f}%" if base and cname != "baseline" else ""))
        r = indices(c)
        for (lang, meas), v in ((k, v) for k, v in r.items() if isinstance(k, tuple)):
            idx = v["index"]
            print(f"  {lang} {meas:<9} gap {v['gap']:+6.1f} | mean " + " ".join(f"{t} {v['mean'][t]:+6.1f}" for t in TYPES)
                  + (" | index " + " ".join(f"{t} {idx[t]:+.2f}" for t in TYPES[2:4]) if idx else " | index invalid (gap < 5)"))
        print("  ru naming (right − wrong): " + "; ".join(f"{g} " + " ".join(f"{x:+.1f}" for x in v) for g, v in r["naming"].items()))
        print("  zh 青 (Yes − No): " + "; ".join(f"{g} " + " ".join(f"{x:+.1f}" for x in v) for g, v in r["qing"].items()))
        for lang, cv in r.get("coverage", {}).items():
            print(f"  coverage {lang}: gap {cv['gap']:+.1f} | " + "; ".join(f"{k} {c:+.1f} ({t})" for k, (c, t) in cv["C"].items())
                  + (" | index " + " ".join(f"{k} {v:+.2f}" for k, v in cv["index"].items()) if cv["index"] else " | index invalid (gap < 5)"))
            terms = COLOR_TERMS[lang]
            print("    membership " + " ".join(f"{t:>8}" for t in terms))
            for g in ("dark blue", "light blue", "dark green", "light green", "red", "pink"):
                print(f"    {g:<11}" + " ".join(f"{cv['M'][(g, t)]:+8.1f}" for t in terms))
        if "categories" in c:
            for lang, g in c["categories"].items():
                row = lambda k: " ".join(f"{g[k][0][L]:+.2f}" for L in SHOW)
                print(f"  geometry {lang} at the word (L{', L'.join(map(str, SHOW))}):")
                for k in [k for k in g if k.endswith("share")] + [k for k in g if k.startswith("cos") or k.startswith("qing")]:
                    print(f"    {k:<34} {row(k)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="+", help="record names in data/summary/, without .json")
    for n in ap.parse_args().names:
        show(n)
