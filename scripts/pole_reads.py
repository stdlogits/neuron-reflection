"""Weights only: how the mother-father contrast sits at each layer, and what reflecting single poles does to
their lens reads (findings §1).

    python scripts/pole_reads.py

Per layer: cos(m̂, f̂) for " mother"/" father" on each lens and site; cosines between the contrast axes of the
S-lens (presence) and the R-lens (prediction), between the read point (MLP input) and the write point (layer
output), and between the parent and gender contrasts. Then four catalog poles, reflected, read through the
lenses. Writes ``data/summary/pole-reads.json``.
"""
import argparse
import json
from pathlib import Path

import torch
from neuron_lens.catalog import kept, lobe_signs
from neuron_lens.neurons import directions, read_rows
from transformers import AutoTokenizer

from neuron_reflection.axes import LensAxes, cos, unit
from neuron_reflection.concepts import CONTRASTS

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-8B"
GENDER = [(" she", " he"), (" her", " his"), (" woman", " man"), (" women", " men"), (" girl", " boy"), (" female", " male"),
          (" sister", " brother"), (" daughter", " son"), (" wife", " husband"), (" queen", " king")]
PARENT = [p for g in ("en", "zh") for p in CONTRASTS["mother-father"]["tokens"][g] if p != (" mum", " dad")]
POLES = [("L0 mother/father", 0, (5284, -1), (3378, -1), "S", "S"), ("L0, R write", 0, (5284, -1), (3378, -1), "S", "R"),
         ("L16 female/male", 16, (2428, 1), (9292, -1), "R", "R"), ("L30 mother/father", 30, (5450, 1), (8579, 1), "R", "R")]

torch.set_grad_enabled(False)
argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()   # weights only: no options
tok = AutoTokenizer.from_pretrained(MODEL)
lx = LensAxes(MODEL, tok)
P, G, MF = lx.ids(PARENT), lx.ids(GENDER), lx.ids([(" mother", " father")])

KEEP = kept(MODEL)


def mf_cos(lens, site, L):
    v = unit(lx.vectors(lens, site, L, [MF[0][0], MF[0][1]]))
    return round((v[:, 0] @ v[:, 1]).item(), 3)


def top(lens, site, L, v, k=8):
    """Top tokens of direction v, z-scored over the kept vocabulary: as a covector on footprints (S) or as a
    residual vector read through the R-lens."""
    if lens == "S":
        sc = (lx.E[KEEP] - lx.ebar) @ (lx.K[site][L].float().T @ v)
    else:
        t = lx.J[site][L].float() @ v
        sc = lx.WU[KEEP] @ (t / t.pow(2).mean().sqrt())
    sc = (sc - sc.mean()) / sc.std()
    return [tok.decode([KEEP[i].item()]) for i in sc.topk(k).indices]


def pole(L, n, sign):
    w = directions(MODEL, L)
    a, b = (w[k] / w[k].norm(dim=1, keepdim=True) for k in ("gate", "up"))
    return read_rows(a, b, lobe_signs(L))[n if sign > 0 else a.shape[0] + n], sign * w["write"][:, n]


def reflect(v, d):
    return v - 2 * (v @ d) * d


record = {"layers": [], "poles": []}
print("L | cos(m,f): S_mid R_mid S_out R_out | S·R mid, out | mid·out S, R | parent·gender S_mid R_mid R_out")
for L in [0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 34]:
    d = {(lens, site): lx.axis(lens, site, L, P) for lens in "SR" for site in ("mid", "out")}
    g = {(lens, site): lx.axis(lens, site, L, G) for lens in "SR" for site in ("mid", "out")}
    row = {"layer": L, "cos_mf": {f"{l}_{s}": mf_cos(l, s, L) for l in "SR" for s in ("mid", "out")},
           "S_vs_R": {s: round(cos(d["S", s], d["R", s]), 3) for s in ("mid", "out")},
           "mid_vs_out": {l: round(cos(d[l, "mid"], d[l, "out"]), 3) for l in "SR"},
           "parent_vs_gender": {f"{l}_{s}": round(cos(d[l, s], g[l, s]), 3) for l, s in [("S", "mid"), ("R", "mid"), ("R", "out")]}}
    record["layers"].append(row)
    print(f"{L:<2} | {row['cos_mf']} | {row['S_vs_R']} | {row['mid_vs_out']} | {row['parent_vs_gender']}")

for label, L, mo, fa, rl, wl in POLES:
    (cm, wm), (cf, wf) = pole(L, *mo), pole(L, *fa)
    di, do = lx.axis(rl, "mid", L, P), lx.axis(wl, "out", L, P)
    cm2, wm2 = reflect(cm, di), reflect(wm, do)
    r = {"label": label, "layer": L, "source": mo, "target": fa, "read_lens": rl, "write_lens": wl,
         "axis_share": {"read": round((cm @ di).item() ** 2, 4), "write": round((unit(wm) @ do).item() ** 2, 4)},
         "cos_to_target": {"read": [round(cos(cm, cf), 3), round(cos(cm2, cf), 3)], "write": [round(cos(wm, wf), 3), round(cos(wm2, wf), 3)]},
         "reads": {"source": top(rl, "mid", L, cm), "edited": top(rl, "mid", L, cm2), "target": top(rl, "mid", L, cf)},
         "writes": {"source": top(wl, "out", L, wm), "edited": top(wl, "out", L, wm2), "target": top(wl, "out", L, wf)}}
    record["poles"].append(r)
    print(f"\n== {label}: share on axis {r['axis_share']}, cos to target before/after {r['cos_to_target']}")
    for side in ("reads", "writes"):
        for k in ("source", "edited", "target"):
            print(f"  {side[:-1]:5} {k:7}: {', '.join(map(repr, r[side][k]))}")

(ROOT / "data" / "summary" / "pole-reads.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
