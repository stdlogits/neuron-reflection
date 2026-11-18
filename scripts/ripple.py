"""Is the ripple natural? Predict each battery item's change from the model's geometry, before editing.

    python scripts/ripple.py

For every battery item whose two answers are concepts (sister/brother, AI/human, Paris/Berlin; yes/no items
are left out), the contrast between the answer words (concept vectors at the word, in frames of the item's
language) and its alignment with the edited contrast (that contrast's word axis: the activation geometry,
whatever axis the edit itself used), as the mean |cosine| over layers 4-32. If an edit generalizes along the
model's geometry, items should move in proportion to that alignment. Compared with each edit's measured
margin change (findings §15, ``data/summary/eval-*.json``); writes ``data/summary/ripple.json``.
"""
import argparse
import json
from pathlib import Path

import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_reflection.axes import unit, word_axes, word_diffs
from neuron_reflection.concepts import BATTERIES, CONTRASTS, OBJECT_FRAMES
from neuron_reflection.subspace import concept_vectors
from neuron_reflection.edit import default_device

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-8B"
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
EDITS = {"eval-mother-father": ("mother-father", ["lens[en,zh]", "word[all]+lens[all]"], ["random[1]", "lens[en,zh]@random1"]),
         "eval-ai-human": ("ai-human", ["word[all]+lens[all]", "lens[all]"], ["random[1]", "word[all]+lens[all]@random1"]),
         "eval-dog-python": ("dog-python", ["word[all]+lens[all]", "word[all]"], ["random[1]", "word[all]+lens[all]@random1"])}
KEEP = {"Paris", "Berlin", "Mary", "Joseph", "AI", "Python"}
GERMAN = {"Mutter", "Vater", "KI", "Mensch", "Tier", "Programmiersprache"}
FRENCH = {"Mère", "Père"}


def language(w: str) -> str:
    if any("一" <= ch <= "鿿" for ch in w):
        return "zh"
    return "de" if w in GERMAN else "fr" if w in FRENCH else "en"


def form(w: str, g: str) -> str:
    return w if g in ("de", "zh") or w in KEEP else w.lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(args.device).eval()
    contrast_vec = {}
    record = {"layers": LAYERS, "edits": {}}
    pooled = []
    for f, (contrast, variants, controls) in EDITS.items():
        axes = word_axes(word_diffs(model, tok, CONTRASTS[contrast]["sentences"]), list(CONTRASTS[contrast]["sentences"]))
        d = torch.stack([axes[L][0] for L in LAYERS])  # [layers, d]
        items = [(part, q, r, w) for part, its in BATTERIES[contrast].items() for q, r, w in its if {r, w} != {"Yes", "No"}]
        pred = []
        for _, _, r, w in items:
            g = language(r)
            if (r, w, g) not in contrast_vec:
                mid, _ = concept_vectors(model, tok, {g: OBJECT_FRAMES[g]}, {g: [form(r, g), form(w, g)]})
                contrast_vec[(r, w, g)] = unit(mid[0] - mid[1], -1)  # [L, d]
            c = contrast_vec[(r, w, g)][LAYERS]
            pred.append((c * d).sum(-1).abs().mean().item())
        data = json.load(open(ROOT / "data" / "summary" / f"{f}.json"))["conditions"]
        index = {(part, q): i for part in BATTERIES[contrast] for i, (q, _, _) in enumerate(BATTERIES[contrast][part])}
        base = data["baseline"]["battery"]
        for v in variants + controls:
            bt = data[v]["battery"]
            drop = [base[part][index[part, q]] - bt[part][index[part, q]] for part, q, _, _ in items]
            nb = [i for i, it in enumerate(items) if it[0] == "neighbors"]
            rho_all = spearmanr(pred, drop).statistic
            rho_nb = spearmanr([pred[i] for i in nb], [drop[i] for i in nb]).statistic
            record["edits"][f"{contrast} {v}"] = {
                "items": [{"part": p, "question": q, "answers": [r, w], "alignment": round(a, 3), "drop": round(x, 2)}
                          for (p, q, r, w), a, x in zip(items, pred, drop)],
                "spearman_all": round(rho_all, 3), "spearman_neighbors": round(rho_nb, 3)}
            print(f"\n{contrast} {v}: Spearman(alignment, drop) all items {rho_all:+.2f} (n={len(items)}), neighbors {rho_nb:+.2f} (n={len(nb)})")
            for (p, q, r, w), a, x in sorted(zip(items, pred, drop), key=lambda t: -t[1]):
                print(f"  {p[:3]} align {a:.3f}  drop {x:+6.1f}  {r}/{w}")
            if v in variants:
                base_nb = [abs(base["neighbors"][index["neighbors", items[i][1]]]) for i in nb]
                pooled += [(pred[i], drop[i] / b) for i, b in zip(nb, base_nb)]
    rho = spearmanr([a for a, _ in pooled], [x for _, x in pooled]).statistic
    record["pooled_neighbors_relative"] = {"spearman": round(rho, 3), "n": len(pooled)}
    print(f"\nall real edits, neighbor items: Spearman(alignment, share of margin lost) {rho:+.2f} (n={len(pooled)})")
    (ROOT / "data" / "summary" / "ripple.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
