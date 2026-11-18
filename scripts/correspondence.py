"""Where each concept of one set lands in the other under the category reflection, by the model's own geometry.

    python scripts/correspondence.py --contrast colors-genders

Concept vectors at the word (``subspace.concept_vectors``), then at each chosen layer the cosine of every
concept's part off the category axis with every concept of the other set (``subspace.correspondence``). The
reflection flips the on-axis part and leaves this part alone, so the nearest concept is where one lands; the
runners-up show how sharp that is. Writes ``data/summary/correspondence-<contrast>.json``.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_reflection.concepts import SETS
from neuron_reflection.subspace import concept_vectors, correspondence
from neuron_reflection.edit import default_device

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-8B"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrast", required=True, choices=sorted(SETS))
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28, 32])
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    c = SETS[args.contrast]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(args.device).eval()
    words = {g: c["source"][g] + c["target"][g] for g in c["frames"]}
    mid, _ = concept_vectors(model, tok, c["frames"], words)
    src, dst = c["source"]["en"], c["target"]["en"]
    n = len(src)
    record = {"contrast": args.contrast, "source": src, "target": dst, "layers": {}}
    votes = {w: Counter() for w in src + dst}
    for L in args.layers:
        s2t, t2s = correspondence(mid[:n, L], mid[n:, L])
        record["layers"][L] = {"source_to_target": {src[i]: [(dst[j], v) for j, v in row] for i, row in enumerate(s2t)},
                               "target_to_source": {dst[i]: [(src[j], v) for j, v in row] for i, row in enumerate(t2s)}}
        print(f"\nL{L}")
        for w, row in list(record["layers"][L]["source_to_target"].items()) + list(record["layers"][L]["target_to_source"].items()):
            votes[w][row[0][0]] += 1
            print(f"  {w:>10} → " + ", ".join(f"{x} {v:+.2f}" for x, v in row))
    record["consensus"] = {w: votes[w].most_common(3) for w in votes}
    # Each frame language alone, the similarities averaged over the chosen layers: does a landing hold across
    # English, German and Chinese, or is it noise?
    per_lang = {}
    for g in c["frames"]:
        m, _ = concept_vectors(model, tok, {g: c["frames"][g]}, {g: words[g]})
        sims = []
        for L in args.layers:
            s2t, _ = correspondence(m[:n, L], m[n:, L], k=len(dst))
            row = torch.zeros(n, len(dst))
            for i, r in enumerate(s2t):
                for j, v in r:
                    row[i, j] = v
            sims.append(row)
        per_lang[g] = torch.stack(sims).mean(0)
    pooled = torch.stack(list(per_lang.values())).mean(0)
    record["by_language_mean_over_layers"] = {g: {src[i]: [(dst[j], round(v[j].item(), 3)) for j in v.topk(3).indices.tolist()]
                                                  for i, v in enumerate(t)} for g, t in per_lang.items()}
    print("\nmean over layers, per frame language (top 3) and pooled:")
    for i, w in enumerate(src):
        cells = []
        for g, t in per_lang.items():
            j = t[i].topk(2).indices.tolist()
            cells.append(f"{g}: {dst[j[0]]} {t[i, j[0]]:+.2f} / {dst[j[1]]} {t[i, j[1]]:+.2f}")
        jp = pooled[i].topk(3).indices.tolist()
        print(f"  {w:>8} | " + " | ".join(cells) + " || pooled: " + ", ".join(f"{dst[j]} {pooled[i, j]:+.2f}" for j in jp))
    print("\nmost frequent landing over the layers:")
    for w, mc in record["consensus"].items():
        print(f"  {w:>10} → " + ", ".join(f"{x} ({k}/{len(args.layers)})" for x, k in mc))
    out = ROOT / "data" / "summary" / f"correspondence-{args.contrast}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
