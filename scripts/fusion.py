"""Fused, swapped or intact? Summarize records by what is left of the difference between two concepts (findings §31).

    python scripts/fusion.py fuse-ai-cat fuse-mother-father heads-whole-ai-cat

A swap and a fusion both take away the original contrast; they differ in what is left. For the pair (source,
target) of a record's contrast (an AI and a cat; a mother and a father), per condition:

- **profile**: the Yes/No profile difference ``Δ = p_source − p_target`` over the suite's test properties, against
  the unedited one ``Δ₀``: ``kept = Δ·Δ₀ / Δ₀·Δ₀`` (1 intact, 0 fused, −1 swapped), ``sep = |Δ| / |Δ₀|`` (0 one
  profile; a swap or a re-coded difference keeps it near 1), and ``center``, where the pair's mean profile sits on
  the line from the unedited target (0) to the unedited source (1). Per property, ``Δ_p / Δ₀_p`` for the properties
  the unedited pair splits by at least 10 nats: its median and spread (interquartile range), and how many
  properties are swapped (below −0.5), fused (|·| < 0.25) or intact (above 0.5). Removal fuses every property
  alike; a cancellation fuses the sum and spreads the properties;
- **self** (AI/cat): "you" fitted as α · its unedited profile + β · the unedited cat's (``inside.blend``);
- **binding**: per family, the role signal, the mean of right − wrong over the eight crossed items (each pair of
  items in one context asks both sides, so a preference that ignores the role cancels in it), as a share of the
  unedited one (1 intact, 0 fused or collapsed, −1 swapped), and the items flipped per side;
- **state**: the landing's kept and separation at L24, at the word and at the sentence's end (English, §21);
- **form**: the margins of "do the words mean the same thing?" and of the first letter of the source word.

Writes nothing; ``scripts/figures.py`` draws ``figures/fusion.png`` from the same measures.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from inside import blend  # noqa: E402
from sparse_heads import SIDES, side_of  # noqa: E402

from neuron_reflection.concepts import SUITES  # noqa: E402

PAIR = {"ai-cat": ("an AI", "a cat"), "mother-father": ("a mother", "a father")}
FORM = {"ai-cat": ("mean the same", "first letter of the word 'cat'"), "mother-father": ("mean the same", "first letter of the word 'mother'")}
L24 = 24  # the landing's kept and new are per layer output 0-35 (LANDING_LAYERS index only its nearest lists)


def contrast_of(rec: dict) -> str:
    return rec["contrast"]


def measures(rec: dict, name: str) -> dict:
    """The fusion measures of condition ``name`` against the record's baseline."""
    contrast = contrast_of(rec)
    suite = SUITES[rec["suite"]]
    C, src, tgt = rec["conditions"], *PAIR[contrast]
    c, b = C[name], C["baseline"]
    out = {"loss": (c["dclm"] / b["dclm"] - 1) * 100}
    if "profiles" in c:
        P0, P = b["profiles"], c["profiles"]
        d0 = torch.tensor(P0[src], dtype=torch.float64) - torch.tensor(P0[tgt], dtype=torch.float64)
        d = torch.tensor(P[src], dtype=torch.float64) - torch.tensor(P[tgt], dtype=torch.float64)
        mid = (torch.tensor(P[src], dtype=torch.float64) + torch.tensor(P[tgt], dtype=torch.float64)) / 2
        n0 = (d0 @ d0).item()
        ratios = sorted((d[i] / d0[i]).item() for i in range(len(d0)) if abs(d0[i]) >= 10)
        q = statistics.quantiles(ratios, n=4) if len(ratios) >= 4 else [float("nan")] * 3
        out |= {"kept": (d @ d0).item() / n0, "sep": (d.norm() / d0.norm()).item(),
                "center": ((mid - torch.tensor(P0[tgt], dtype=torch.float64)) @ d0).item() / n0,
                "prop_median": statistics.median(ratios), "prop_iqr": q[2] - q[0], "n_props": len(ratios),
                "swapped": sum(r < -0.5 for r in ratios), "fused": sum(abs(r) < 0.25 for r in ratios),
                "intact": sum(r > 0.5 for r in ratios)}
        if "you" in P:
            a, be, _ = blend(P["you"], P0["you"], P0["a cat"])
            out |= {"you_alpha": a, "you_beta": be}
    if "binding" in c:
        sides = SIDES.get(contrast)
        for fam, items in suite["binding"].items():
            m, m0 = c["binding"][fam], b["binding"][fam]
            flips = {s: 0 for s in sides}
            for (q, _, _), v in zip(items, m):
                if v < 0:
                    flips[side_of(q, fam, sides)] += 1
            out[f"bind {fam}"] = sum(m) / sum(m0)
            out[f"flips {fam}"] = tuple(flips.values())
    if "landing" in c:
        en = c["landing"]["en"]
        for pos, where in ((0, "word"), (1, "end")):
            kp = sum(k[pos][L24] for k in en["kept"]) / len(en["kept"])
            nw = sum(q[pos][L24] for q in en["new"]) / len(en["new"])
            out[f"state {where}"] = (kp, (kp * kp + nw * nw) ** 0.5)
    if "scored" in c:
        for key, pattern in zip(("form same", "form letter"), FORM[contrast]):
            for item, i0 in zip(c["scored"], b["scored"]):
                if pattern in item["question"]:
                    lp, l0 = item["logprobs"], i0["logprobs"]
                    out[key] = (l0[0] - max(l0[1:]), lp[0] - max(lp[1:]))
    return out


def show(name: str):
    rec = json.loads((ROOT / "data" / "summary" / f"{name}.json").read_text())
    print(f"\n== {name} ({rec['contrast']}, suite {rec['suite']})")
    print(f"  {'condition':<44} {'loss':>6} | profile kept sep center | per property median IQR (swapped/fused/intact of n)"
          " | you α | binding role→name, name→role (flips) | state L24 word, end (kept/sep) | form")
    for cname in rec["conditions"]:
        if cname == "baseline":
            continue
        m = measures(rec, cname)
        bind = ", ".join(f"{m[k]:+.2f} {m['flips ' + k[5:]]}" for k in m if k.startswith("bind "))
        state = ", ".join(f"{m[k][0]:+.2f}/{m[k][1]:.2f}" for k in ("state word", "state end") if k in m)
        form = ", ".join(f"{m[k][1]:+.0f}" for k in ("form same", "form letter") if k in m)
        prof = (f"{m['kept']:+.2f} {m['sep']:.2f} {m['center']:+.2f} | {m['prop_median']:+.2f} {m['prop_iqr']:.2f} "
                f"({m['swapped']}/{m['fused']}/{m['intact']} of {m['n_props']})") if "kept" in m else "-"
        you = f"{m['you_alpha']:.2f}" if "you_alpha" in m else "-"
        print(f"  {cname[:44]:<44} {m['loss']:+5.2f}% | {prof} | {you} | {bind} | {state} | {form}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="+", help="record names in data/summary/, without .json")
    for n in ap.parse_args().names:
        show(n)
