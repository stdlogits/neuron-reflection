"""The greedy order of neurons for an edit and an objective: rank every neuron by the first-order effect of its
own reflection on the objective's summed margin, edit a step, re-rank on the edited model (findings §17's
greedy), and record the order. ``run.py --order NAME`` then edits its first N neurons (``@greedy:N``).

    python scripts/order.py --contrast harm-benign --edit "template[en]" --objective refuse-frames --upto 300 --name refuse-frames

The objective is a ``concepts.SELECTION`` entry (forced choices, right over wrong), and the order *raises* its
margin: each step takes the neurons whose reflection moves the items most toward the right answers. (§17's
greedy lowered its selection margin, since there the edit's job was to break the contrast.) A margin of fixed
openers ("I'm sorry" over "Sure") is a proxy: findings §20 shows it can move while the behavior does not. Writes
``data/summary/order-<name>.json``.

With ``--heads PART`` (one of ``q``, ``k``, ``v``, ``o``) the units are attention heads instead (findings §29): each
head's one-part reflection is ranked by ``attribution.head_effects``, and ``run.py --order NAME`` edits the first N
with ``word[all]@greedy:N#PART`` (``@rgreedy:N#PART`` as many heads per layer at random, ``@loading:N#PART`` the N
that load most). The axes must be one kind (no ``+``).

With ``--mix POOLS`` (letters from ``nqkvo``: ``n`` neurons with both sides, the rest head parts) neurons and head
parts share one greedy order (``attribution.mixed_greedy_order``, findings §30), edited by ``run.py``'s
``word[all]@mix:N`` (``@rmix:N`` the same number of each type per layer, at random).

With an objective from ``concepts.FUSION`` (and ``--mix``) the order fuses instead (findings §31): it drives every
group's signal (a property's Yes − No split between the two concepts, a binding context's role signal) toward zero
together (``attribution.target_greedy_order``, targets 0), each unit edited with ``--k`` (2 reflects, 1 projects; ``run.py``
must edit the order with the same k, ``@mix:N^1`` for a projection order). Groups whose unedited signal is under 5
nats are dropped before ordering.

With an objective from ``concepts.TARGETED`` (and ``--mix``) the order drives each item's signal to a target instead
(findings §33): an item with a ``ref`` toward that item's unedited signal (a split: black and orange cats toward where a
cat and a dog stand), the rest kept where they were. The scale of each is its distance to the target, at least 10 nats.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch

from neuron_reflection.attribution import (apply_units, chat_items, greedy_head_order, greedy_order, target_greedy_order,
                                                    head_effects, mixed_greedy_order, reflection_effects, signals)
from neuron_reflection.axes import LensAxes, attention_axes, template_diffs, word_axes, word_diffs
from neuron_reflection.concepts import CONTRASTS, FUSION, SELECTION, TARGETED
from neuron_reflection.edit import Editor, default_device
from neuron_reflection.measure import battery

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-8B"


def build_axes(ed, contrast: str, spec: str) -> dict:
    """Reads from the first part of ``spec``, writes from the last (``template[en]+lens[refuse]``)."""
    c, parts, lens = CONTRASTS[contrast], [], None
    for p in spec.split("+"):
        kind, groups = re.fullmatch(r"(lens|word|template)\[([^\]]*)\]", p).groups()
        groups = None if groups == "all" else groups.split(",")
        if kind == "lens":
            lens = lens or LensAxes(MODEL, ed.tok)
            parts.append(lens.axes([q for g in (groups or list(c["tokens"])) for q in c["tokens"][g]]))
        else:
            source = c["sentences" if kind == "word" else "template"]
            diffs = (word_diffs if kind == "word" else template_diffs)(ed.model, ed.tok, source)
            parts.append(word_axes(diffs, groups or list(source)))
    return {L: (parts[0][L][0], parts[-1][L][1]) for L in parts[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrast", required=True, choices=sorted(CONTRASTS))
    ap.add_argument("--edit", required=True)
    ap.add_argument("--objective", required=True, choices=sorted(SELECTION) + sorted(FUSION) + sorted(TARGETED))
    ap.add_argument("--upto", type=int, default=300)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--heads", choices=list("qkvo"), help="order attention heads for this one-part edit instead of neurons")
    ap.add_argument("--mix", help="order neurons and head parts together: letters from nqkvo (findings §30)")
    ap.add_argument("--k", type=float, default=2.0, help="for a FUSION or TARGETED objective: 2 reflects each unit, 1 projects, below 0 amplifies")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    t0 = time.time()
    ed = Editor.load(MODEL, args.device)
    axes = build_axes(ed, args.contrast, args.edit)
    if args.objective in FUSION:
        assert args.mix, "a fusion order takes --mix POOLS"
        return fusion_order(args, ed, axes, t0)
    if args.objective in TARGETED:
        assert args.mix, "a targeted order takes --mix POOLS"
        return targeted_order(args, ed, axes, t0)
    assert args.k == 2.0, "--k is for fusion orders"
    items = SELECTION[args.objective]
    seqs = chat_items(ed.tok, [(q, w, r) for q, r, w in items])  # wrong over right: greedy_order lowers it
    base = battery(ed.model, ed.tok, items)
    if args.heads:
        return heads_order(args, ed, axes, items, seqs, base, t0)
    if args.mix:
        return mixed_order(args, ed, axes, items, seqs, base, t0)
    eff = -reflection_effects(ed.model, seqs, axes)  # first-order change in the objective's margin
    print(f"objective {args.objective}: {len(items)} items, summed margin {sum(base):+.1f}; first-order sum of positive"
          f" effects read {eff[:, 0].clamp(min=0).sum():+.1f}, write {eff[:, 1].clamp(min=0).sum():+.1f}; of negative"
          f" read {eff[:, 0].clamp(max=0).sum():+.1f}, write {eff[:, 1].clamp(max=0).sum():+.1f} ({time.time() - t0:.0f}s)", flush=True)
    order = greedy_order(ed, axes, seqs, args.upto)
    n = ed.layers[0].mlp.down_proj.weight.shape[1]
    record = {"model": MODEL, "contrast": args.contrast, "edit": args.edit, "objective": args.objective, "k": 2.0,
              "baseline_margin": round(sum(base), 2), "greedy": [[i // n, i % n] for i in order.tolist()]}
    for N in (10, 30, 100, 300):
        if N <= len(order):
            ed.edit(axes, units={L: [i for L2, i in record["greedy"][:N] if L2 == L] for L, _ in record["greedy"][:N]}, select="units")
            m = battery(ed.model, ed.tok, items)
            ed.restore()
            record[f"margin_{N}"] = round(sum(m), 2)
            print(f"  greedy {N}: objective margin {sum(m):+.1f} (baseline {sum(base):+.1f})", flush=True)
    out = ROOT / "data" / "summary" / f"order-{args.name}.json"
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


def heads_order(args, ed, axes, items, seqs, base, t0):
    assert "+" not in args.edit, "head orders take one kind of axis"
    hax = attention_axes(axes, axes)
    eff = -head_effects(ed.model, seqs, hax)[args.heads]  # first-order change in the objective's margin
    print(f"objective {args.objective}: {len(items)} items, summed margin {sum(base):+.1f}; first-order {args.heads} effects:"
          f" positive {eff.clamp(min=0).sum():+.1f}, negative {eff.clamp(max=0).sum():+.1f}, largest"
          f" {eff.max():+.1f} ({time.time() - t0:.0f}s)", flush=True)
    order = greedy_head_order(ed, hax, seqs, args.heads, args.upto)
    record = {"model": MODEL, "contrast": args.contrast, "edit": f"{args.edit}#{args.heads}", "objective": args.objective,
              "k": 2.0, "unit": "group" if args.heads in "kv" else "head", "baseline_margin": round(sum(base), 2),
              "greedy": [list(u) for u in order]}
    for N in (1, 2, 4, 8, 16, 32, 64, 128):
        if N <= len(order):
            units: dict[int, list[int]] = {}
            for L, h in order[:N]:
                units.setdefault(L, []).append(h)
            ed.edit_heads(hax, parts=args.heads, units=units, select="units")
            m = battery(ed.model, ed.tok, items)
            ed.restore()
            record[f"margin_{N}"] = round(sum(m), 2)
            print(f"  greedy {N}: objective margin {sum(m):+.1f} (baseline {sum(base):+.1f})", flush=True)
    out = ROOT / "data" / "summary" / f"order-{args.name}.json"
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


def mixed_order(args, ed, axes, items, seqs, base, t0):
    assert "+" not in args.edit, "mixed orders take one kind of axis"
    pools = "".join(sorted(set(args.mix.replace(",", "")), key="nqkvo".index))
    print(f"objective {args.objective}: {len(items)} items, summed margin {sum(base):+.1f}; pools {pools}", flush=True)
    order = mixed_greedy_order(ed, axes, seqs, pools, args.upto)
    hax = attention_axes(axes, axes)
    record = {"model": MODEL, "contrast": args.contrast, "edit": f"{args.edit}#mix:{pools}", "objective": args.objective, "k": 2.0,
              "baseline_margin": round(sum(base), 2), "units": [list(u) for u in order]}
    for N in (1, 2, 4, 8, 16, 32, 64, 128, 200, 300):
        if N <= len(order):
            counts = apply_units(ed, axes, hax, order[:N])
            m = battery(ed.model, ed.tok, items)
            ed.restore()
            record[f"margin_{N}"] = round(sum(m), 2)
            record[f"types_{N}"] = counts
            print(f"  first {N}: objective margin {sum(m):+.1f} (baseline {sum(base):+.1f}); "
                  + ", ".join(f"{p} {c}" for p, c in counts.items()), flush=True)
    out = ROOT / "data" / "summary" / f"order-{args.name}.json"
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


def fusion_order(args, ed, axes, t0):
    assert "+" not in args.edit, "fusion orders take one kind of axis"
    pools = "".join(sorted(set(args.mix.replace(",", "")), key="nqkvo".index))
    groups = [[(chat_items(ed.tok, [item])[0], w) for item, w in g] for g in FUSION[args.objective]]
    s0 = signals(ed.model, groups)
    keep = [i for i, s in enumerate(s0.tolist()) if abs(s) >= 5]
    print(f"objective {args.objective}: {len(groups)} groups, unedited signals " + " ".join(f"{s:+.1f}" for s in s0.tolist())
          + (f"; dropped (under 5 nats): {[i for i in range(len(groups)) if i not in keep]}" if len(keep) < len(groups) else "")
          + f"; pools {pools}, k {args.k}", flush=True)
    groups = [groups[i] for i in keep]
    order, trace = target_greedy_order(ed, axes, groups, pools, args.upto, k=args.k)
    print("  trace (units, rms share left): " + " ".join(f"{n}:{r:.3f}" for n, r in trace), flush=True)
    hax = attention_axes(axes, axes)
    record = {"model": MODEL, "contrast": args.contrast, "edit": f"{args.edit}#fuse:{pools}", "objective": args.objective, "k": args.k,
              "groups": keep, "baseline_signals": [round(s, 2) for s in s0.tolist()], "units": [list(u) for u in order],
              "trace": [list(t) for t in trace]}
    s0 = s0[keep]
    for N in (1, 2, 4, 8, 16, 32, 64, 128, 200, 300):
        if N <= len(order):
            counts = apply_units(ed, axes, hax, order[:N], k=args.k)
            c = signals(ed.model, groups) / s0
            ed.restore()
            record[f"rms_{N}"] = round(c.pow(2).mean().sqrt().item(), 4)
            record[f"left_{N}"] = [round(x, 3) for x in c.tolist()]
            record[f"types_{N}"] = counts
            print(f"  first {N}: rms share left {record[f'rms_{N}']:.3f}; per group " + " ".join(f"{x:+.2f}" for x in c.tolist())
                  + "; " + ", ".join(f"{p} {n}" for p, n in counts.items()), flush=True)
    out = ROOT / "data" / "summary" / f"order-{args.name}.json"
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


def targeted_order(args, ed, axes, t0):
    assert "+" not in args.edit, "targeted orders take one kind of axis"
    pools = "".join(sorted(set(args.mix.replace(",", "")), key="nqkvo".index))
    items = TARGETED[args.objective]
    groups = [[(chat_items(ed.tok, [it["item"]])[0], 1.0)] for it in items]
    s0 = signals(ed.model, groups)
    targets = torch.tensor([s0[it["ref"]].item() if "ref" in it else s0[i].item() for i, it in enumerate(items)], dtype=torch.float64)
    scales = (s0 - targets).abs().clamp(min=10.0)
    moved = [i for i, it in enumerate(items) if "ref" in it]
    print(f"objective {args.objective}: {len(items)} items, {len(moved)} moved toward a target; pools {pools}, k {args.k}", flush=True)
    for i in moved:
        print(f"  {items[i]['wording']:<8} {' / '.join(items[i]['pair'])}: {s0[i]:+.1f} → target {targets[i]:+.1f}", flush=True)
    order, trace = target_greedy_order(ed, axes, groups, pools, args.upto, k=args.k, targets=targets, scales=scales)
    print("  trace (units, rms distance from the targets in scale): " + " ".join(f"{n}:{r:.3f}" for n, r in trace), flush=True)
    hax = attention_axes(axes, axes)
    record = {"model": MODEL, "contrast": args.contrast, "edit": f"{args.edit}#target:{pools}", "objective": args.objective, "k": args.k,
              "baseline_signals": [round(x, 2) for x in s0.tolist()], "targets": [round(x, 2) for x in targets.tolist()],
              "scales": [round(x, 2) for x in scales.tolist()], "units": [list(u) for u in order], "trace": [list(t) for t in trace]}
    for N in (1, 2, 4, 8, 16, 32, 64, 128, 200, 300):
        if N <= len(order):
            counts = apply_units(ed, axes, hax, order[:N], k=args.k)
            s = signals(ed.model, groups)
            ed.restore()
            c = (s - targets) / scales
            record[f"rms_{N}"] = round(c.pow(2).mean().sqrt().item(), 4)
            record[f"signals_{N}"] = [round(x, 2) for x in s.tolist()]
            record[f"types_{N}"] = counts
            print(f"  first {N}: rms {record[f'rms_{N}']:.3f}; moved items " + " ".join(f"{s[i]:+.1f}" for i in moved)
                  + "; guards " + " ".join(f"{s[i]:+.1f}" for i in range(len(items)) if i not in moved) + "; "
                  + ", ".join(f"{p} {n}" for p, n in counts.items()), flush=True)
    out = ROOT / "data" / "summary" / f"order-{args.name}.json"
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
