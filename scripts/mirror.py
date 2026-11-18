"""Are every head and every neuron mirror images? (findings §28)

    python scripts/mirror.py heads-whole-ai-cat heads-whole-mother-father

Reflecting every component and the interface is a symmetry, so reflecting every head (all four parts) is exactly
the model with every neuron reflected behind a reflected interface (``tests/test_edit.py``): asking it about a
cat is asking the all-neuron model about the cat's mirror image, an AI, and its "Cat" is the other's "AI". With
per-layer axes the identity is approximate. For a record with ``<axis>#qkvo`` (every head) and ``<axis>@top:1``
(every neuron) this reports:

- **profiles**: for each subject, the correlation of its profile under every head with the all-neuron profile of
  the same subject and of its mirror (the partner concept; subjects without one are their own mirror);
- **binding**: the correlation over the 16 binding items of the every-head margins with the all-neuron margins on
  the same items and on the mirrored items (every mention of one side swapped for the other, the question too).
"""
import argparse
import json
from pathlib import Path

import torch

from neuron_reflection.concepts import SUITES

ROOT = Path(__file__).resolve().parents[1]
SIDES = {"ai-cat": ("a cat", "an AI assistant"), "mother-father": ("mother", "father")}


def corr(x, y) -> float:
    return torch.corrcoef(torch.tensor([x, y], dtype=torch.float64))[0, 1].item()


def swap(text: str, a: str, b: str) -> str:
    return text.replace(a, "\0").replace(b, a).replace("\0", b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="+", help="record names in data/summary/, without .json")
    for name in ap.parse_args().names:
        rec = json.loads((ROOT / "data" / "summary" / f"{name}.json").read_text())
        contrast, suite = rec["contrast"], SUITES[rec["suite"]]
        C = rec["conditions"]
        ax = next(k for k in C if k.endswith("#qkvo") and "&" not in k).removesuffix("#qkvo")
        heads, neurons = C[f"{ax}#qkvo"], C[f"{ax}@top:1"]
        a, b = SIDES[contrast]
        src, tgt = ("an AI", "a cat") if contrast == "ai-cat" else ("a mother", "a father")
        mirror = {src: tgt, tgt: src}
        print(f"\n== {name}: every head ({ax}#qkvo, loss {heads['dclm']}) against every neuron ({ax}@top:1, loss {neurons['dclm']}),"
              f" baseline loss {C['baseline']['dclm']}")
        print("  profiles: r(heads[s], neurons[s]) | r(heads[s], neurons[mirror s])")
        for s in heads["profiles"]:
            m = mirror.get(s, s)
            same, mir = corr(heads["profiles"][s], neurons["profiles"][s]), corr(heads["profiles"][s], neurons["profiles"][m])
            print(f"    {s:>14}: {same:+.2f} | {mir:+.2f}" + (f"  (mirror: {m})" if m != s else ""))
        hs, ns_same, ns_mir = [], [], []
        for fam, items in suite["binding"].items():
            qs = [q for q, _, _ in items]
            for i, (q, _, _) in enumerate(items):
                mq = swap(q, a, b)
                j = qs.index(mq)
                hs.append(heads["binding"][fam][i])
                ns_same.append(neurons["binding"][fam][i])
                ns_mir.append(neurons["binding"][fam][j])
        print(f"  binding (16 items): r(heads, neurons same item) {corr(hs, ns_same):+.2f} | r(heads, neurons mirrored item) {corr(hs, ns_mir):+.2f}")
        print("    margins, heads | neurons same | neurons mirrored: " + "  ".join(f"{h:+.0f}|{s:+.0f}|{m:+.0f}" for h, s, m in zip(hs, ns_same, ns_mir)))


if __name__ == "__main__":
    main()
