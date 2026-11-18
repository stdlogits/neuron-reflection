"""How few neurons does an edit need? Reflect the N neurons whose own reflection moves a held-out selection set
most, across the whole model, and compare with geometric and random choices of the same size.

    python scripts/sparsity.py --contrasts mother-father ai-human dog-python

Per contrast, the edit (axis and k) is §15's: mother/father along the lens axis (en, zh), AI/human and
dog/Python with word-axis reads and lens-axis writes. The neurons it is applied to are chosen four ways, N
each:

- ``attr``: the N most negative first-order effects on the selection set (``concepts.SELECTION``: English
  forced choices sharing no question with the batteries, though mostly the same answer words),
  ``attribution.reflection_effects``;
- ``oracle``: the same, ranked on the battery's own efficacy items, so in-sample: how much choosing on the
  evaluation inflates a sparse edit;
- ``geo``: the N with the largest loading on their layer's axis, ranked across layers;
- ``greedy``: ``attr`` chosen in steps, each step edited before the next is attributed (steps of a quarter
  of the neurons chosen so far), so the ranking sees what the earlier picks already did; ``greedy-N/k4``
  applies k = 4 to the same neurons (the axis component reversed and tripled: a bigger edit per neuron);
- ``random-attr``, ``random-greedy``: as many neurons per layer as ``attr`` or ``greedy`` chose, at random
  within each layer (seeded);

and ``layer``, §15's per-layer top share at 1, 3, 12, 36 and 122 neurons per layer. Each is measured on the
battery (efficacy and neighbors), the selection set and DCLM loss (code loss for dog/Python). Writes
``data/summary/sparsity-<contrast>.json``, with the greedy order (``run.py``'s ``@greedy:N`` edits its first N).
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch

from neuron_reflection.attribution import as_units, chat_items, greedy_order, reflection_effects
from neuron_reflection.axes import LensAxes, word_axes, word_diffs
from neuron_reflection.concepts import BATTERIES, CONTRASTS, SELECTION, SUITES
from neuron_reflection.edit import Editor, default_device
from neuron_reflection.measure import battery, corpus_docs, mean_nll, reply

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-8B"
EDITS = {"mother-father": "lens[en,zh]", "ai-human": "word[all]+lens[all]", "dog-python": "word[all]+lens[all]"}
GRID = [1, 3, 10, 30, 100, 300, 1000, 4392]
PER_LAYER = [1, 3, 12, 36, 122]
GREEDY_MAX = 1000
K4 = (10, 30, 100)
BAND = 6


def say(*a):
    print(*a, flush=True)


def build_axes(ed, lens, contrast: str, spec: str) -> dict:
    parts = []
    for p in spec.split("+"):
        kind, groups = re.fullmatch(r"(lens|word)\[([^\]]*)\]", p).groups()
        groups = None if groups == "all" else groups.split(",")
        c = CONTRASTS[contrast]
        if kind == "lens":
            parts.append(lens.axes([q for g in (groups or list(c["tokens"])) for q in c["tokens"][g]]))
        else:
            parts.append(word_axes(word_diffs(ed.model, ed.tok, c["sentences"]), groups or list(c["sentences"])))
    return {L: (parts[0][L][0], parts[-1][L][1]) for L in parts[0]}


def bands(units: dict[int, list[int]], n_layers: int) -> list[int]:
    out = [0] * ((n_layers + BAND - 1) // BAND)
    for L, ns in units.items():
        out[L // BAND] += len(ns)
    return out


def measure(ed, contrast: str, docs: dict, replies: bool) -> dict:
    model, tok = ed.model, ed.tok
    out = {name: round(mean_nll(model, d), 4) for name, d in docs.items()}
    out["battery"] = {part: battery(model, tok, items) for part, items in BATTERIES[contrast].items()}
    out["selection"] = battery(model, tok, SELECTION[contrast])
    if replies:
        out["replies"] = [{"question": q, "reply": reply(model, tok, q, 60)} for _, q in SUITES[f"eval-{contrast}"]["questions"]]
    return out


def mean(x):
    return sum(x) / len(x)


def run(ed, lens, contrast: str, grid: list[int], seeds: int):
    t0 = time.time()
    axes = build_axes(ed, lens, contrast, EDITS[contrast])
    n_layers, n = len(ed.layers), ed.layers[0].mlp.down_proj.weight.shape[1]
    say(f"\n======== {contrast}: {EDITS[contrast]}, k = 2")
    eff = reflection_effects(ed.model, chat_items(ed.tok, SELECTION[contrast]), axes)  # [L, 2, n]
    oracle = reflection_effects(ed.model, chat_items(ed.tok, BATTERIES[contrast]["efficacy"]), axes).sum(1)
    score = eff.sum(1)  # [L, n]: first-order change in the summed selection margin
    loading = torch.stack([ed.loadings(L, axes[L]) for L in range(n_layers)])  # [L, n]
    say(f"  attribution ({time.time() - t0:.0f}s): sum of every neuron's effect {score.sum():+.1f}, of the negative ones"
        f" {score.clamp(max=0).sum():+.1f}; read {eff[:, 0].sum():+.1f}, write {eff[:, 1].sum():+.1f}")
    rank = lambda s: s.flatten().argsort()
    order, oracle_order, geo_order = rank(score), rank(oracle), rank(-loading)
    flat = score.flatten()
    spearman = lambda a, b: torch.corrcoef(torch.stack([a.argsort().argsort().float(), b.argsort().argsort().float()]))[0, 1].item()
    record = {"model": MODEL, "contrast": contrast, "edit": EDITS[contrast], "k": 2.0,
              "attribution": {
                  "sum_all": round(score.sum().item(), 2), "sum_negative": round(score.clamp(max=0).sum().item(), 2),
                  "read": round(eff[:, 0].sum().item(), 2), "write": round(eff[:, 1].sum().item(), 2),
                  "spearman_selection_vs_battery": round(spearman(flat, oracle.flatten()), 3),
                  "spearman_selection_vs_loading": round(spearman(flat, -loading.flatten()), 3),
                  "per_layer_negative": [round(x, 2) for x in score.clamp(max=0).sum(1).tolist()],
                  "top": [{"layer": i // n, "neuron": i % n, "read": round(eff[i // n, 0, i % n].item(), 3),
                           "write": round(eff[i // n, 1, i % n].item(), 3), "battery": round(oracle.flatten()[i].item(), 3),
                           "loading": round(loading.flatten()[i].item(), 5),
                           "loading_rank_in_layer": int((loading[i // n] > loading[i // n, i % n]).sum())}
                          for i in order[:100].tolist()]},
              "conditions": {}}
    say("  top 10 by effect: " + ", ".join(f"L{t['layer']}/{t['neuron']} {t['read'] + t['write']:+.1f}"
                                          f" (r{t['read']:+.1f} w{t['write']:+.1f}, load rank {t['loading_rank_in_layer']})"
                                          for t in record["attribution"]["top"][:10]))
    t1 = time.time()
    greedy = greedy_order(ed, axes, chat_items(ed.tok, SELECTION[contrast]), min(max(grid), GREEDY_MAX))
    say(f"  greedy order ({time.time() - t1:.0f}s); overlap of its first 100 with attr's: "
        f"{len(set(greedy[:100].tolist()) & set(order[:100].tolist()))}")
    record["greedy"] = [[i // n, i % n] for i in greedy.tolist()]  # [layer, neuron] in the order taken
    docs = {"dclm": corpus_docs(ed.tok, "dclm")}
    if contrast == "dog-python":
        docs["code"] = corpus_docs(ed.tok, "code")

    def matched(units, seed):  # as many neurons per layer as ``units``, at random
        return {L: torch.randperm(n, generator=torch.Generator().manual_seed(1000 * seed + L))[:len(ns)].tolist()
                for L, ns in units.items()}

    conditions = [("baseline", {}, True)]
    for N in grid:
        attr = as_units(order[:N], n)
        conditions += [(f"attr-{N}", attr, True), (f"oracle-{N}", as_units(oracle_order[:N], n), False),
                       (f"geo-{N}", as_units(geo_order[:N], n), False)]
        conditions.append((f"random-attr-{N}", matched(attr, 0), False))
        if N <= len(greedy):
            g = as_units(greedy[:N], n)
            conditions.append((f"greedy-{N}", g, True))
            if N in K4:
                conditions.append((f"greedy-{N}/k4", g, True))
            conditions += [(f"random-greedy{s}-{N}", matched(g, s), False) for s in range(seeds)]
    for c in PER_LAYER:
        top = {L: loading[L].topk(c).indices.tolist() for L in range(n_layers)}
        conditions.append((f"layer-{c}", top, c in (12, 122)))

    base = None
    for name, units, replies in conditions:
        t1 = time.time()
        info = ed.edit(axes, units=units, select="units", k=4.0 if name.endswith("/k4") else 2.0) if units else {"neurons": 0}
        m = measure(ed, contrast, docs, replies)
        ed.restore()
        idx = torch.tensor([L * n + i for L, ns in units.items() for i in ns], dtype=torch.long)
        m.update(neurons=info["neurons"], bands=bands(units, n_layers),
                 predicted_selection=round(flat[idx].sum().item(), 2) if len(idx) else 0.0)
        record["conditions"][name] = m
        eff_m, nei_m, sel_m = mean(m["battery"]["efficacy"]), mean(m["battery"]["neighbors"]), sum(m["selection"])
        if base is None:
            base = (eff_m, nei_m, sel_m)
        say(f"  {name:>13} N={m['neurons']:<5} efficacy {eff_m:+6.1f} ({eff_m - base[0]:+6.1f})  neighbors {nei_m:+6.1f}"
            f" ({nei_m - base[1]:+5.1f})  selection Δ {sel_m - base[2]:+7.1f} (pred {m['predicted_selection']:+7.1f})  "
            + "  ".join(f"{k} {m[k]:.4f}" for k in docs) + f"  bands {m['bands']}  ({time.time() - t1:.0f}s)")
        for r in m.get("replies", []):
            say(f"      {r['question']} → {r['reply'][:160].replace(chr(10), ' / ')}")
    out = ROOT / "data" / "summary" / f"sparsity-{contrast}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    say(f"  wrote {out.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrasts", nargs="+", default=list(EDITS), choices=list(EDITS))
    ap.add_argument("--grid", nargs="+", type=int, default=GRID)
    ap.add_argument("--seeds", type=int, default=2, help="random selections per size")
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    ed = Editor.load(MODEL, args.device)
    lens = LensAxes(MODEL, ed.tok)
    for c in args.contrasts:
        run(ed, lens, c, args.grid, args.seeds)


if __name__ == "__main__":
    main()
