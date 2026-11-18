"""Does an edited pair's remaining difference lie along the axis or off it? (findings §31)

    python scripts/off_axis.py --contrast mother-father --device cuda

The landing (findings §21) measures the edited source − target difference in the residual. Under a projection no
reader of layer L + 1 sees the residual along ``out_L``, so a difference left along that axis would be hidden from
them. For the contrast's English landing frames and word axes, per layer output L and at the word and the
sentence's end, this reports:

- **sep**: the edited difference's length as a share of the unedited one's, as in the landing;
- **sep_off_axis**: the same with ``out_L`` removed from the difference, ``|P Δ'| / |Δ₀|``;
- **off-axis share**: for the unedited pair, ``|P Δ₀| / |Δ₀|``: how much of the original difference the axis misses.

This is geometry on states averaged over frames before normalization, not what any reader's projections take in;
whether a reader uses the off-axis difference needs a reader or causal measurement. L35 has no next reader.

The conditions are every neuron and head projected (k = 1), every neuron projected, and every neuron and head
reflected (k = 2). Writes ``data/summary/off-axis-<contrast>.json``.
"""
import argparse
import json
from pathlib import Path

import torch

from neuron_reflection.axes import attention_axes, word_axes, word_diffs
from neuron_reflection.concepts import CONTRASTS, SUITES
from neuron_reflection.edit import Editor, default_device
from neuron_reflection.measure import LANDING_LAYERS, word_states

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-8B"
CONDITIONS = {"everything projected": (1.0, True), "every neuron projected": (1.0, False), "everything reflected": (2.0, True)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrast", required=True, choices=["ai-cat", "mother-father"])
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    ed = Editor.load(MODEL, args.device)
    contrast = CONTRASTS[args.contrast]
    axes = word_axes(word_diffs(ed.model, ed.tok, contrast["sentences"]), list(contrast["sentences"]))
    spec = SUITES[f"{args.contrast}-heads"]["landing"]
    src, tgt = spec["source"]["en"][0], spec["target"]["en"][0]
    read = lambda: word_states(ed.model, ed.tok, {"en": spec["frames"]["en"]}, {"en": [src, tgt]})
    base = read()
    d0 = base[f"en:{src}"] - base[f"en:{tgt}"]  # [2, L, d]
    out_axis = torch.stack([axes[L][1].float() for L in range(len(axes))])  # [L, d]: what the heads of L + 1 read
    proj = lambda v: v - (v * out_axis).sum(-1, keepdim=True) * out_axis
    n0 = d0.norm(dim=-1)
    record = {"model": MODEL, "contrast": args.contrast, "axes": "word[all]",
              "note": "per [position (word, end)][layer output 0..35]; geometric: the averaged states' difference with the "
              "layer's out axis removed, not what any reader's projections take in; L35 has no next reader",
              "off_axis_share": (proj(d0).norm(dim=-1) / n0).tolist(), "conditions": {}}
    for name, (k, heads) in CONDITIONS.items():
        ed.edit(axes, share=1.0, k=k)
        if heads:
            ed.edit_heads(attention_axes(axes, axes), parts="qkvo", k=k)
        e = read()
        ed.restore()
        d1 = e[f"en:{src}"] - e[f"en:{tgt}"]
        record["conditions"][name] = {"sep": (d1.norm(dim=-1) / n0).tolist(), "sep_off_axis": (proj(d1).norm(dim=-1) / n0).tolist(),
                                      "kept": ((d1 * d0).sum(-1) / n0 ** 2).tolist()}
    show = [LANDING_LAYERS.index(L) for L in (4, 12, 20, 24, 28, 35)]
    for pos, where in ((0, "word"), (1, "end")):
        print(f"\n{args.contrast}, at the {where} (L4, L12, L20, L24, L28, L35)")
        print("  unedited off-axis share: " + " ".join(f"{record['off_axis_share'][pos][LANDING_LAYERS[i]]:.2f}" for i in show))
        for name, r in record["conditions"].items():
            print(f"  {name:<24} sep " + " ".join(f"{r['sep'][pos][LANDING_LAYERS[i]]:.2f}" for i in show)
                  + " | off axis " + " ".join(f"{r['sep_off_axis'][pos][LANDING_LAYERS[i]]:.2f}" for i in show)
                  + " | kept " + " ".join(f"{r['kept'][pos][LANDING_LAYERS[i]]:+.2f}" for i in show))
    for c in [record] + list(record["conditions"].values()):
        for key in ("off_axis_share", "sep", "sep_off_axis", "kept"):
            if key in c:
                c[key] = [[round(x, 3) for x in row] for row in c[key]]
    out = ROOT / "data" / "summary" / f"off-axis-{args.contrast}.json"
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
