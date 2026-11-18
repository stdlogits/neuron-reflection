"""Two kinds or one? Summarize the cat-split and cat-homes records (findings §33-§36).

    python scripts/cat_split.py cat-split home-fox

For each condition of ``data/summary/<name>.json`` (``run.py --suite cat-split``), per measure (count: how many kinds
of animals, One − Two; same: the same kind of animal?, Yes − No; induction: does a blank property transfer?, Yes − No;
both orders, all positive for one kind):

- the mean margin per pair type (``concepts.CAT_PAIRS``);
- the **kind index** per type, ``(W − m) / (W − B)``: W the non-color pairs of one kind (young and old cats…), B the
  pairs of two kinds (a cat and a dog…). 0 answers like one kind, 1 like two. Valid where ``W − B ≥ 5`` nats.

Then "Is X a cat?" (Yes − No); for cat-homes records (findings §34), the home probes ("Is an orange cat a fox?",
"Is a black cat a fox?", "Is a fox a cat?", "Are an orange cat and a fox the same kind of animal?") for each home;
for cat-homes-forms records (findings §35), whether each fitted, held-out and protected form is a fox, a cat, and the
same kind as a black cat; and the free replies.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from neuron_reflection.concepts import BENGAL_ITEMS, CAT_ITEMS, CAT_PAIRS, FORM_ITEMS, GARFIELD_ITEMS, HOME_ITEMS, SUITES

MEASURES = ("count", "same", "induction")
TYPES = list(CAT_PAIRS)


def kinds(cond: dict) -> dict:
    m = [s["logprobs"][0] - s["logprobs"][1] for s in cond["scored"]]
    qs = [s["question"] for s in cond["scored"]]
    suite = next(n for n in ("cat-split", "cat-homes", "cat-homes-forms", "cat-homes-garfield", "cat-homes-bengal")
                 if [q for q, _ in SUITES[n]["scored"]] == qs)
    out = {}
    for meas in MEASURES:
        mean = {}
        for t in TYPES:
            v = [x for x, it in zip(m, CAT_ITEMS) if it["measure"] == meas and it["type"] == t]
            mean[t] = sum(v) / len(v)
        W, B = mean["within"], mean["between"]
        out[meas] = {"mean": mean, "gap": W - B, "index": {t: (W - mean[t]) / (W - B) for t in TYPES} if W - B >= 5 else None}
    out["is a cat"] = {it["pair"][0]: x for x, it in zip(m, CAT_ITEMS) if it["measure"] == "is a cat"}
    if suite == "cat-homes-bengal":
        out["bengal"] = {it["probe"]: x for x, it in zip(m[len(CAT_ITEMS):], BENGAL_ITEMS)}
        return out
    if suite != "cat-split":
        out["homes"] = {}
        for x, it in zip(m[len(CAT_ITEMS):], HOME_ITEMS):
            out["homes"].setdefault(it["home"], {})[it["probe"]] = x
    if suite == "cat-homes-garfield":
        n0 = len(CAT_ITEMS) + len(HOME_ITEMS) + len(FORM_ITEMS)
        out["garfield"] = {it["probe"]: x for x, it in zip(m[n0:], GARFIELD_ITEMS)}
    if suite in ("cat-homes-forms", "cat-homes-garfield"):
        out["forms"] = {}
        for x, it in zip(m[len(CAT_ITEMS) + len(HOME_ITEMS):], FORM_ITEMS):
            out["forms"].setdefault((it["group"], it["form"]), {})[it["probe"]] = x
    return out


def show(name: str):
    rec = json.loads((ROOT / "data" / "summary" / f"{name}.json").read_text())
    base = rec["conditions"].get("baseline")
    for cname, c in rec["conditions"].items():
        k = kinds(c)
        print(f"\n## {name}: {cname}" + (f"  loss {(c['dclm'] / base['dclm'] - 1) * 100:+.2f}%" if base and cname != "baseline" else ""))
        for meas in MEASURES:
            v = k[meas]
            idx = v["index"]
            print(f"  {meas:<9} gap {v['gap']:+5.1f} | mean " + " ".join(f"{t} {v['mean'][t]:+5.1f}" for t in TYPES))
            print("            index " + (" ".join(f"{t} {idx[t]:+.2f}" for t in TYPES) if idx else "invalid (gap < 5)"))
        print("  is a cat: " + ", ".join(f"{x} {v:+.1f}" for x, v in k["is a cat"].items()))
        for h, probes in k.get("homes", {}).items():
            print(f"  home {h}: " + ", ".join(f"{p} {v:+.1f}" for p, v in probes.items()))
        if "bengal" in k:
            print("  bengal: " + ", ".join(f"{p} {v:+.1f}" for p, v in k["bengal"].items()))
        if "garfield" in k:
            print("  garfield: " + ", ".join(f"{p} {v:+.1f}" for p, v in k["garfield"].items()))
        for (g, f), probes in k.get("forms", {}).items():
            print(f"  {g:<9} {f:<17}: " + ", ".join(f"{p} {v:+.1f}" for p, v in probes.items()))
        if "replies" in c:
            for r in c["replies"]:
                print(f"    [{r['tag']}] {r['question'][:58]:<58} -> {r['reply'][:170].replace(chr(10), ' / ')!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="+", help="record names in data/summary/, without .json")
    for n in ap.parse_args().names:
        show(n)
