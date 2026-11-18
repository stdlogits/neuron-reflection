"""How far a contrast's per-layer axes turn from layer to layer (findings §28).

    python scripts/drift.py --contrast ai-cat --axes "word[all]" --device cuda

Reflecting every head and neuron with one ``H`` is an exact symmetry up to the interface, and so is the mirror
between every head and every neuron (``tests/test_edit.py``); with an axis per layer, what one layer writes in
its reflected frame the next reads in a slightly different one. Two reflections along axes an angle θ apart
compose to a rotation by 2θ in their plane. This prints, per layer, the cosine of the MLP-input (``mid``) axis
with the next layer's, of the ``mid`` and ``out`` axes of one layer (a neuron's two sides), of a head's read and
write axes (``axes.attention_axes``), and with the axis at L0, L12, L24 and L35. Writes
``data/summary/drift-<contrast>-<axis kind>.json``.
"""
import argparse
import json
from pathlib import Path

import torch

from neuron_reflection.axes import LensAxes, attention_axes, word_axes, word_diffs
from neuron_reflection.concepts import CONTRASTS
from neuron_reflection.edit import Editor, default_device

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-8B"
REFS = [0, 12, 24, 35]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrast", required=True, choices=sorted(CONTRASTS))
    ap.add_argument("--axes", required=True, help="lens[groups] or word[groups]")
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    contrast = CONTRASTS[args.contrast]
    kind, _, groups = args.axes.partition("[")
    groups = None if groups.rstrip("]") == "all" else groups.rstrip("]").split(",")
    if kind == "lens":  # weights only: the lenses and the tokenizer, no model
        from transformers import AutoTokenizer
        axes = LensAxes(MODEL, AutoTokenizer.from_pretrained(MODEL)).axes(
            [p for g in (groups or list(contrast["tokens"])) for p in contrast["tokens"][g]])
    else:
        ed = Editor.load(MODEL, args.device)
        axes = word_axes(word_diffs(ed.model, ed.tok, contrast["sentences"]), groups or list(contrast["sentences"]))
    heads = attention_axes(axes, axes)
    n = len(axes)
    c = lambda x, y: (x @ y / x.norm() / y.norm()).item()
    rows = []
    print("L   mid·next-mid  mid·out  head read·write  " + "  ".join(f"mid·L{r}" for r in REFS))
    for L in range(n):
        row = {"layer": L, "mid_next": round(c(axes[L][0], axes[L + 1][0]), 3) if L + 1 < n else None,
               "mid_out": round(c(axes[L][0], axes[L][1]), 3), "head_read_write": round(c(*heads[L]), 3),
               "to": {str(r): round(c(axes[L][0], axes[r][0]), 3) for r in REFS}}
        rows.append(row)
        print(f"L{L:<2} {row['mid_next'] if row['mid_next'] is not None else float('nan'):+12.2f}  {row['mid_out']:+7.2f}  "
              f"{row['head_read_write']:+15.2f}  " + "  ".join(f"{row['to'][str(r)]:+6.2f}" for r in REFS), flush=True)
    out = ROOT / "data" / "summary" / f"drift-{args.contrast}-{kind}.json"
    out.write_text(json.dumps({"model": MODEL, "contrast": args.contrast, "axes": args.axes, "layers": rows}, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
