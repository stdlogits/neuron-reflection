"""Where the two concepts sit on the axis, relative to the origin the reflection fixes (findings §27).

    python scripts/offset.py --contrast ai-cat --axes "word[all]" --suite ai-cat-heads --device cuda

``H = I − 2 d dᵀ`` reflects through the origin, so it swaps two states only when their projections on ``d`` are
opposite. A neuron reads the normed residual ``x / rms(x)`` (its γ is folded into the rows), so at each layer's MLP
input this reports, for the suite's English landing frames at the concept word, the mean projection of the
normed source state (``s``) and target state (``t``) on the layer's read axis, and the offset
``ρ = (s + t) / (s − t)``: 0 when the origin is their midpoint (a reflection swaps them), +1 when the target sits
on the mirror plane (the target is fixed and the source lands past it, on the target's far side), −1 when the
source does. With ``--heads``, it reads instead what every head writes, summed, at the answer position of the
suite's name→role binding items (``What is Sam to Tom?``), projected on the heads' write axis (``axes.attention_axes``),
grouped by the side the queried name is bound to: what a one-sided O edit reflects. Writes
``data/summary/offset-<contrast>[-heads]-<axis kind>.json``.
"""
import argparse
import json
from pathlib import Path

import torch

from neuron_reflection.axes import LensAxes, attention_axes, word_axes, word_diffs
from neuron_reflection.concepts import CONTRASTS, SUITES
from neuron_reflection.edit import Editor, default_device
from neuron_reflection.measure import chat_prompt

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-8B"


@torch.no_grad()
def mid_states(model, tok, frames: list[str], word: str) -> torch.Tensor:
    """``[L, d]``: the normed MLP input at the word's last token, averaged over frames."""
    acc = 0
    for f in frames:
        text = f.replace("{X}", word)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        end = f.index("{X}") + len(word)
        last = max(i for i, (a, _) in enumerate(enc["offset_mapping"]) if a < end)
        got = []
        hooks = [lay.post_attention_layernorm.register_forward_pre_hook(lambda m, inp: got.append(inp[0][0, last].float().cpu()))
                 for lay in model.model.layers]
        try:
            model(torch.tensor([enc["input_ids"]], device=model.device))
        finally:
            for h in hooks:
                h.remove()
        x = torch.stack(got)
        acc = acc + x / x.pow(2).mean(-1, keepdim=True).sqrt()
    return acc / len(frames)


@torch.no_grad()
def head_writes(model, tok, question: str) -> torch.Tensor:
    """``[L, d]``: every layer's attention output (all heads, ``o_proj``) at the chat prompt's last token."""
    got = []
    hooks = [lay.self_attn.o_proj.register_forward_hook(lambda m, i, o: got.append(o[0, -1].float().cpu())) for lay in model.model.layers]
    try:
        model(torch.tensor([tok.encode(chat_prompt(tok, question), add_special_tokens=False)], device=model.device))
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(got)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrast", required=True, choices=sorted(CONTRASTS))
    ap.add_argument("--axes", required=True, help="lens[groups] or word[groups]")
    ap.add_argument("--suite", required=True, help="a suite with English landing frames, source and target")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--heads", action="store_true", help="the heads' summed write at the binding answers, not the MLP input at the word")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    contrast, spec = CONTRASTS[args.contrast], SUITES[args.suite]["landing"]
    ed = Editor.load(MODEL, args.device)
    kind, _, groups = args.axes.partition("[")
    groups = None if groups.rstrip("]") == "all" else groups.rstrip("]").split(",")
    if kind == "lens":
        axes = LensAxes(MODEL, ed.tok).axes([p for g in (groups or list(contrast["tokens"])) for p in contrast["tokens"][g]])
    else:
        axes = word_axes(word_diffs(ed.model, ed.tok, contrast["sentences"]), groups or list(contrast["sentences"]))
    if args.heads:  # name→role items: the right answer names the queried name's side (source or target label)
        items = SUITES[args.suite]["binding"]["name→role"]
        labels = sorted({right for _, right, _ in items})
        src_label = next(l for l in labels if l.lower() in " ".join(spec["source"]["en"]).lower())
        s = torch.stack([head_writes(ed.model, ed.tok, q) for q, right, _ in items if right == src_label]).mean(0)
        t = torch.stack([head_writes(ed.model, ed.tok, q) for q, right, _ in items if right != src_label]).mean(0)
        axes = {L: (w, None) for L, (_, w) in attention_axes(axes, axes).items()}  # the heads' write axis
    else:
        frames = spec["frames"]["en"]
        s = torch.stack([mid_states(ed.model, ed.tok, frames, w) for w in spec["source"]["en"]]).mean(0)
        t = torch.stack([mid_states(ed.model, ed.tok, frames, w) for w in spec["target"]["en"]]).mean(0)
    rows = []
    for L in range(len(axes)):
        d = axes[L][0]
        ps, pt = (s[L] @ d).item(), (t[L] @ d).item()
        rows.append({"layer": L, "source": round(ps, 3), "target": round(pt, 3), "offset": round((ps + pt) / (ps - pt), 3)})
        print(f"L{L:<2} source {ps:+7.3f}  target {pt:+7.3f}  offset {(ps + pt) / (ps - pt):+.2f}", flush=True)
    out = ROOT / "data" / "summary" / f"offset-{args.contrast}{'-heads' if args.heads else ''}-{kind}.json"
    out.write_text(json.dumps({"model": MODEL, "contrast": args.contrast, "axes": args.axes, "suite": args.suite, "heads": args.heads,
                               "words": {"source": spec["source"]["en"], "target": spec["target"]["en"]}, "layers": rows}, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
