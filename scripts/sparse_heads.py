"""Summarize the sparse head runs (findings §29): what a handful of heads chosen by effect does to the binding.

    python scripts/sparse_heads.py ai-cat-q ai-cat-k ai-cat-v ai-cat-o mother-father-q mother-father-o

For ``data/summary/sparse-heads-<contrast>-<part>.json`` (``run.py --suite <contrast>-sparse --order
heads-<contrast>-<part>``), per condition: the loss, the held-out binding family the order targeted (items right of
eight, mean margin, and how many items flipped on each side, so a swap shows on both sides and a collapse on one),
the other family, and the two concepts' profile fits (α on their own unedited profile, β on the partner's).
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from inside import blend  # noqa: E402

from neuron_reflection.concepts import SUITES  # noqa: E402

SIDES = {"ai-cat": ("a cat", "an AI assistant"), "mother-father": ("mother", "father")}
CONCEPTS = {"ai-cat": ("a cat", "an AI"), "mother-father": ("a mother", "a father")}
FAMILY = {"q": "role→name", "k": "role→name", "o": "name→role", "v": "name→role"}


def side_of(q: str, fam: str, sides: tuple[str, str]) -> str:
    """The side an item asks about: the role named in the question (role→name) or the side the queried name is
    bound to (name→role)."""
    ctx, _, ask = q.partition("?")[0].rpartition(". ")
    if fam == "role→name":
        return next(s for s in sides if ask.endswith(s))
    name = re.search(r"What is (\w+)(?: to Tom)?$", ask).group(1)
    return next(s for s in sides if re.search(rf"{s} named {name}|{name} is Tom's {s}", ctx))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tags", nargs="+", help="tags of data/summary/sparse-heads-<tag>.json, e.g. ai-cat-q")
    for tag in ap.parse_args().tags:
        contrast, _, part = tag.rpartition("-")
        rec = json.loads((ROOT / "data" / "summary" / f"sparse-heads-{tag}.json").read_text())
        C, suite = rec["conditions"], SUITES[rec["suite"]]
        fam, other = FAMILY[part], ({"role→name", "name→role"} - {FAMILY[part]}).pop()
        items = suite["binding"][fam]
        sides = SIDES[contrast]
        b = C["baseline"]
        print(f"\n== {tag}: test family {fam} (other: {other})")
        print(f"  {'condition':<26} {'loss':>6}  {fam}: right, margin, flipped by side  | {other} | profiles α/β")
        for name, c in C.items():
            m = c["binding"][fam]
            flips = {s: 0 for s in sides}
            for (q, _, _), v in zip(items, m):
                if v < 0:
                    flips[side_of(q, fam, sides)] += 1
            prof = []
            for s, p in (CONCEPTS[contrast], CONCEPTS[contrast][::-1]):
                a, be, _ = blend(c["profiles"][s], b["profiles"][s], b["profiles"][p])
                prof.append(f"{s} {a:+.2f}/{be:+.2f}")
            o = c["binding"][other]
            label = name.replace("word[all]", "").lstrip("@") or "baseline"
            print(f"  {label:<26} {(c['dclm'] / b['dclm'] - 1) * 100:+5.2f}%  {sum(v > 0 for v in m)}/8 {sum(m) / 8:+6.1f}  "
                  + " ".join(f"{s.split()[-1]} {n}" for s, n in flips.items())
                  + f"  | {sum(v > 0 for v in o)}/8 {sum(o) / 8:+5.1f} | " + ", ".join(prof))


if __name__ == "__main__":
    main()
