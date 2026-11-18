"""What an edited concept has become: summarize the "inside" and "heads" suites' records (findings §21-§26).

    python scripts/inside.py inside-mother-father inside-ai-cat ...

For each condition of each record (``data/summary/<name>.json`` from ``scripts/run.py --suite <contrast>-inside``
or ``-dial``, and ``-robot``):

- **landing**, per language: where the source and target words sit on the line from the unedited target (0) to
  the unedited source (1), at the word and at the sentence's end; how much of the original difference is kept
  along its own direction (1 intact, 0 none, −1 reversed) and how much is new (off that direction), and the
  separation left, ``√(kept² + new²)`` (1 as far apart as before, 0 the same state). A kept of 0 is a fusion only
  when the separation is small too;
- **profiles**: each subject's edited Yes/No profile fitted as ``α · its own unedited profile + β · its partner's``
  (least squares over the properties, no intercept; R² uncentered), so a relabel or swap gives β ≈ 1, α ≈ 0, a
  blend both, an intact concept α ≈ 1, though α and β are not mixture proportions; and the unedited subjects
  whose profiles it now correlates with best (Pearson, which ignores a global shift toward Yes or No);
- **tell-apart**: the scored items' margins (right answer over the first wrong one);
- **ledger** (``-heads`` suites): ``kept`` at the sentence's end on the pre-norm residual after L24 and L35, split
  exactly into the embedding, the heads, the edited neurons and the other neurons, with the heads that carry
  most;
- **binding** (``-heads`` suites): accuracy and mean margin per family (role→name, name→role).

Writes nothing; ``scripts/figures.py`` draws the landing figure from the same records.
"""
import argparse
import json
from pathlib import Path

import torch

from neuron_reflection.concepts import SUITES
from neuron_reflection.measure import LANDING_LAYERS

ROOT = Path(__file__).resolve().parents[1]
SHOW = [4, 12, 20, 28, 35]

# Each profile subject's partner: the concept the edit pairs it with, or (for anchors) the side it might drift to.
PARTNERS = {
    "mother-father": {"a mother": "a father", "a father": "a mother", "a parent": "a mother", "a sister": "a brother",
                      "a brother": "a sister"},
    "ai-cat": {"a cat": "an AI", "an AI": "a cat", "you": "a cat", "a kitten": "an AI", "a dog": "an AI", "a robot": "a cat",
               "a chatbot": "a cat"},
    "animals-languages": {"a dog": "Python", "Python": "a dog", "a cat": "JavaScript", "JavaScript": "a cat",
                          "a giraffe": "TypeScript", "TypeScript": "a giraffe"},
    "colors-genders": {"pink": "a girl", "a girl": "pink", "blue": "a man", "a man": "blue", "green": "a woman", "a woman": "green",
                       "teal": "a woman", "a queen": "pink"},
}


def mean(rows):
    return [sum(x) / len(x) for x in zip(*rows)]


def blend(edited: list[float], own: list[float], partner: list[float]) -> tuple[float, float, float]:
    """(α, β, R²) of ``edited ≈ α own + β partner``."""
    X = torch.tensor([own, partner], dtype=torch.float64).T
    y = torch.tensor(edited, dtype=torch.float64)
    coef = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
    r2 = 1 - ((y - X @ coef) ** 2).sum() / (y ** 2).sum()
    return coef[0].item(), coef[1].item(), r2.item()


def corr(x: list[float], y: list[float]) -> float:
    return torch.corrcoef(torch.tensor([x, y], dtype=torch.float64))[0, 1].item()


def contrast_of(suite: str) -> str:
    return suite.removesuffix("-inside").removesuffix("-dial").removesuffix("-heads")


def show(name: str):
    rec = json.loads((ROOT / "data" / "summary" / f"{name}.json").read_text())
    suite = SUITES[rec["suite"]]
    contrast = contrast_of(rec["suite"])
    conds = rec["conditions"]
    base = conds.get("baseline")
    print(f"\n==================== {name} ({rec['contrast']}, suite {rec['suite']})")
    for cname, c in conds.items():
        print(f"\n## {cname}  loss {c.get('dclm')}" + (f", code {c['code']}" if "code" in c else ""))
        if "landing" in c:
            spec = suite["landing"]
            print("  landing (L" + ", L".join(map(str, SHOW)) + "): source coordinate | target coordinate | kept | new | separation")
            for g, r in c["landing"].items():
                for pos, where in ((0, "word"), (1, "end")):
                    cs = mean([r["coord"][w][pos] for w in spec["source"][g]])
                    ct = mean([r["coord"][w][pos] for w in spec["target"][g]])
                    kp, nw = mean([k[pos] for k in r["kept"]]), mean([q[pos] for q in r["new"]])
                    f = lambda v, fmt="+.2f": " ".join(format(v[L], fmt) for L in SHOW)
                    sep = [(k * k + n * n) ** 0.5 for k, n in zip(kp, nw)]
                    print(f"    {g} {where:<4} {f(cs)} | {f(ct)} | {f(kp)} | {f(nw, '.2f')} | {f(sep, '.2f')}")
            anchors = spec.get("anchors", {}).get("en", [])
            if anchors:
                print("    en anchors at the word, coordinate L12/L24/L35: " + "; ".join(
                    f"{a} " + "/".join(f"{c['landing']['en']['coord'][a][0][L]:+.2f}" for L in (12, 24, 35)) for a in anchors))
            near = c["landing"]["en"]["near"]
            for w in (spec["source"]["en"] + spec["target"]["en"])[:6]:
                print(f"    en {w!r} nearest at the word, L12 | L24 | L35: " + " | ".join(
                    ", ".join(f"{v} {s:.2f}" for v, s in near[w][0][idx_][:2]) for idx_ in [LANDING_LAYERS.index(L) for L in (12, 24, 35)]))
        if "ledger" in c:
            r = c["ledger"]
            for i, R in enumerate(r["layers"]):
                print(f"  ledger L{R} end: kept {r['total'][1][i]:+.2f} = embedding {r['embed'][1][i]:+.2f} + heads {r['attention'][1][i]:+.2f}"
                      f" + edited neurons {r['mlp_edited'][1][i]:+.2f} + other neurons {r['mlp_rest'][1][i]:+.2f}"
                      f"  (word: kept {r['total'][0][i]:+.2f}, heads {r['attention'][0][i]:+.2f})")
            H = torch.tensor(r["heads"][1][0])
            top = H.abs().flatten().topk(5).indices.tolist()
            print(f"    heads carrying most at the end (L{r['layers'][0]}): " + ", ".join(
                f"L{t // H.shape[1]}h{t % H.shape[1]} {H.flatten()[t]:+.3f}" for t in top))
        if "binding" in c:
            print("  binding: " + "; ".join(f"{fam} accuracy {sum(x > 0 for x in m) / len(m):.2f}, mean margin {sum(m) / len(m):+.1f}"
                                         for fam, m in c["binding"].items()))
        if "profiles" in c and base and "profiles" in base:
            labels = [p[0] for p in suite["profiles"]["properties"]]
            print("  profiles (α own, β partner, R²):")
            for s, e in c["profiles"].items():
                partner = PARTNERS.get(contrast, {}).get(s)
                if partner is None or partner not in base["profiles"]:
                    continue
                a, b, r2 = blend(e, base["profiles"][s], base["profiles"][partner])
                flips = [l for l, x, y in zip(labels, e, base["profiles"][s]) if (x > 0) != (y > 0)]
                near = sorted(((corr(e, v), t) for t, v in base["profiles"].items()), reverse=True)[:2]
                print(f"    {s:>12} ≈ {a:+.2f} · itself {b:+.2f} · {partner:<12} (R² {r2:.2f} uncentered); answers most like unedited "
                      + ", ".join(f"{t} {r:+.2f}" for r, t in near) + f"; flipped: {', '.join(flips) or '-'}")
        if "scored" in c:
            print("  tell-apart (right − best wrong):")
            for item, b_ in zip(c["scored"], (base or c)["scored"]):
                lp, lp0 = item["logprobs"], b_["logprobs"]
                m, m0 = lp[0] - max(lp[1:]), lp0[0] - max(lp0[1:])
                print(f"    {m0:+6.1f} → {m:+6.1f}  {item['question'][:95]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="+", help="record names in data/summary/, without .json")
    for n in ap.parse_args().names:
        show(n)
