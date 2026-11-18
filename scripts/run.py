"""Edit Qwen3-8B along a contrast's axes and measure what changes.

    python scripts/run.py --contrast mother-volcano --suite orthogonal \
        --variants "lens[all]" "word[all]" "word[all]+lens[all]" --name mother-volcano

A variant is one or more edits joined by ``&`` and applied in turn. An edit names the axis for a neuron's
read rows and, after ``+``, for its write column (else the same one):

- ``lens[groups]``: token pairs through the lenses (S-lens through L8, R-lens after);
- ``word[groups]``: activations at the concept word in paired sentences;
- ``template[groups]``: activations at the assistant's chat template, plain question against instructed;
- for set contrasts (``concepts.SETS``): ``category[groups]``, a reflection along mean source − mean target
  concept vectors; ``pairs[groups]``, the orthogonal map that swaps each listed pair (Procrustes);
  ``softpairs[groups]``, the map fitted to the model's own soft coupling of the sets (off-axis similarity,
  entropic transport, weighted Procrustes); ``hardpairs[groups]``, to its best one-to-one assignment;
  ``linpairs[groups]``, the linear map (not orthogonal) sending each source to its target, for reads only (a home
  for a split, findings §34: fixed points listed as their own targets); and
  ``lens``-prefixed versions of each from the lenses' token vectors (the coupling still from activations).

``groups`` are a contrast's languages (``all`` for every one). An edit takes the top ``--share`` of neurons per
layer by loading; ``@persona`` takes persona-neuron-experiment's persona units instead
(``concepts.PERSONA_UNITS``), ``@top+persona`` both, ``@top:0.001`` sets the share for one edit, and
``@greedy:100`` takes the first 100 neurons of the greedy order ``scripts/sparsity.py`` found for the same edit (or, with
``--order NAME``, the order ``scripts/order.py`` found for an objective); ``@rgreedy:100`` takes as many neurons per
layer at random, the control.
``@random`` (``@random2``…) takes as many neurons at random, and ``random[seed]`` uses random axes: the controls.
``#parts`` edits attention heads instead of neurons, along the same axes (``axes.attention_axes``: a head reads
the previous layer's output axis and writes its layer's MLP-input axis): ``#ov`` their V rows and O columns (what
a head moves), ``#qk`` their Q and K rows (where it looks), ``#q`` the Q rows alone (what it looks for), any of
``qkvo``; every head unless ``@top:0.25`` takes that share per layer by loading (``lens[en,zh] & lens[en,zh]#ov``),
``@greedy:16`` the first 16 of ``--order``'s head order (``scripts/order.py --heads``), ``@rgreedy:16`` as many per
layer at random, ``@loading:16`` the 16 that load most anywhere. ``@mix:50`` (no ``#``) edits the first 50 units of a
mixed neuron-and-head order (``scripts/order.py --mix``, or a fusion order, findings §31), ``@rmix:50`` as many of each
type per layer at random; ``@mix:50^1`` projects the units instead of reflecting them.
``/reads`` edits only the neurons' gate and up rows, ``/writes`` only their down columns
(``word[all]+lens[all]/writes``), and ``/steer`` replaces each whole down
column by the write axis at its norm, the source-pushing pole turned to the target (``/steer=0.25`` turns it
a quarter of the way, ``--dose`` for every edit). ``--k 2`` reflects, ``--k 1`` projects, ``--k -1`` doubles the
axis component (amplifies the neurons' share of the contrast); ``^k`` sets it for one edit (``template[all]^-1``). A baseline
comes first; each variant is measured and restored. A suite can ask for loss, probes, batteries, scored answers,
the fresh-name test, refusal, frame following, landing (``measure.landing``, against states read before any
edit), the component ledger (``measure.ledger``), binding batteries, Yes/No profiles, conversations and replies
(``concepts.SUITES``). Writes ``data/summary/<name>.json``.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

from neuron_reflection.axes import S_UNTIL, LensAxes, agreement, attention_axes, template_diffs, word_axes, word_diffs
from neuron_reflection.concepts import BATTERIES, BENIGN, CONTRASTS, FOLLOW_FRAMES, FRESH, HARMFUL, PROBES, SETS, SUITES, half, unit_set, wrap
from neuron_reflection.subspace import assignment, concept_vectors, maps_by_layer, model_similarity, sinkhorn, unit

SET_KINDS = ("category", "pairs", "softpairs", "hardpairs", "linpairs", "lenspairs", "lenscategory", "lenssoftpairs", "lenshardpairs")
COUPLING_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
from neuron_reflection.attribution import apply_units
from neuron_reflection.edit import Editor, default_device
from neuron_reflection.measure import (LANDING_LAYERS, answer_logprobs, battery, categories, component_states, converse, corpus_docs,
                                               fresh_names, landing, ledger, mean_nll, probe, profile, refuses, replies, reply, residuals,
                                               word_states)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-8B"
SELECT = {"top": "top", "persona": "units", "top+persona": "union"}
AGREEMENT_LAYERS = [0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 35]


def say(*a):
    print(*a, flush=True)


def parse(variant: str) -> list[tuple]:
    """Each edit as (axis parts, selection, share or None, sides, dose or None, k or None, head parts or None)."""
    edits = []
    for e in variant.split("&"):
        e, _, sides = e.strip().partition("/")
        e, _, k = e.partition("^")
        e, _, heads = e.partition("#")
        assert not heads or (set(heads) <= set("qkvo") and not sides), f"bad head parts {heads!r} in {variant!r}"
        sides, _, dose = (sides or "both").partition("=")
        assert sides in ("both", "reads", "writes", "steer"), f"bad sides {sides!r} in {variant!r}"
        e, _, sel = e.strip().partition("@")
        sel, _, share = (sel or "top").partition(":")
        assert sel in SELECT or re.fullmatch(r"r?greedy|random\d*|loading|r?mix", sel), f"bad selection {sel!r} in {variant!r}"
        parts = []
        for p in e.split("+"):
            m = re.fullmatch(r"(lens|word|template|random|category|pairs|softpairs|hardpairs|linpairs|lenspairs|lenscategory|lenssoftpairs|lenshardpairs)"
                             r"\[([^\]]*)\]", p.strip())
            assert m, f"bad variant {variant!r}"
            parts.append((m.group(1), None if m.group(2) == "all" else m.group(2).split(",")))
        assert len(parts) in (1, 2), variant
        assert not heads or sel in ("top", "greedy", "rgreedy", "loading"), f"heads take @top, @greedy, @rgreedy or @loading, in {variant!r}"
        edits.append((parts, sel, float(share) if share else None, sides, float(dose) if dose else None, float(k) if k else None,
                      heads or None))
    return edits


def greedy_units(contrast: str, edit: str, n: int, order: str | None = None) -> dict[int, list[int]]:
    """The first ``n`` neurons of the greedy order ``scripts/sparsity.py`` found for this contrast's edit, or of
    ``data/summary/order-<order>.json`` from ``scripts/order.py``."""
    record = json.loads((ROOT / "data" / "summary" / (f"order-{order}.json" if order else f"sparsity-{contrast}.json")).read_text())
    assert record["contrast"] == contrast, f"the order is for {record['contrast']}, not {contrast}"
    assert record["edit"] == edit, f"the greedy order is for {record['edit']}, not {edit}"
    units: dict[int, list[int]] = {}
    for L, i in record["greedy"][:n]:
        units.setdefault(L, []).append(i)
    return units


def landing_words(spec: dict) -> dict[str, list[str]]:
    return {g: spec["source"][g] + spec["target"][g] + spec.get("anchors", {}).get(g, []) for g in spec["frames"]}


def measure(ed: Editor, suite: dict, docs: dict, base: dict | None = None, base_resid: dict | None = None) -> dict:
    model, tok = ed.model, ed.tok
    out = {name: round(mean_nll(model, d), 4) for name, d in docs.items()}
    say("  " + "  ".join(f"{name} {v:.4f}" for name, v in out.items()))
    if suite.get("landing"):
        spec = suite["landing"]
        out["landing"] = landing(base, word_states(model, tok, spec["frames"], landing_words(spec)), spec)
        show = [LANDING_LAYERS.index(L) for L in (4, 12, 20, 28, 35)]
        for g, r in out["landing"].items():
            src, tgt = spec["source"][g], spec["target"][g]
            mean = lambda rows: [sum(x) / len(x) for x in zip(*rows)]
            for pos, where in ((0, "word"), (1, "end ")):
                cs = mean([r["coord"][w][pos] for w in src])
                ct = mean([r["coord"][w][pos] for w in tgt])
                kp, nw = mean([k[pos] for k in r["kept"]]), mean([q[pos] for q in r["new"]])
                say(f"  landing {g} {where} L4..35: source at " + " ".join(f"{cs[LANDING_LAYERS[i]]:+.2f}" for i in show)
                    + " | target at " + " ".join(f"{ct[LANDING_LAYERS[i]]:+.2f}" for i in show)
                    + " | kept " + " ".join(f"{kp[LANDING_LAYERS[i]]:+.2f}" for i in show)
                    + " | new " + " ".join(f"{nw[LANDING_LAYERS[i]]:.2f}" for i in show))
            for w in (src + tgt)[:4] + spec.get("anchors", {}).get(g, [])[:2]:
                say(f"    {g}:{w} nearest at L12, L24, L35 (word): " + "; ".join(
                    ", ".join(f"{v} {c:.2f}" for v, c in r["near"][w][0][LANDING_LAYERS.index(L)][:2]) for L in (12, 24, 35)))
    if suite.get("ledger"):
        spec = suite["landing"]
        comp = component_states(model, tok, spec["frames"]["en"], spec["source"]["en"] + spec["target"]["en"])
        r = out["ledger"] = ledger(model, comp, base_resid, spec, neurons=ed.selected)
        for i, R in enumerate(r["layers"]):
            for pos, where in ((0, "word"), (1, "end ")):
                say(f"  ledger L{R} {where}: kept {r['total'][pos][i]:+.2f} = embedding {r['embed'][pos][i]:+.2f} + heads "
                    f"{r['attention'][pos][i]:+.2f} + edited neurons {r['mlp_edited'][pos][i]:+.2f} + other neurons {r['mlp_rest'][pos][i]:+.2f}")
        H = torch.tensor(r["heads"][1][0])  # the sentence's end, the first readout layer
        top = H.abs().flatten().topk(6).indices.tolist()
        say(f"    heads carrying most at the end (L{r['layers'][0]}): " + ", ".join(
            f"L{t // H.shape[1]}h{t % H.shape[1]} {H.flatten()[t]:+.3f}" for t in top))
    if suite.get("categories"):  # how far apart each language keeps the color words (findings §32)
        spec = suite["categories"]
        out["categories"] = categories(word_states(model, tok, spec["frames"], {g: list(w.values()) for g, w in spec["words"].items()}), spec)
        for g, r in out["categories"].items():
            say(f"  categories {g} (word, L12/L24): " + "; ".join(f"{k} {r[k][0][12]:.2f}/{r[k][0][24]:.2f}" for k in r
                                                                    if k.endswith("share") or k.startswith("cos") or k.startswith("qing")))
    if suite.get("binding"):
        out["binding"] = {}
        for fam, items in suite["binding"].items():
            m = battery(model, tok, items)
            out["binding"][fam] = m
            say(f"  binding {fam}: accuracy {sum(x > 0 for x in m) / len(m):.2f}, mean margin {sum(m) / len(m):+.2f}")
    if suite.get("profiles"):
        out["profiles"] = profile(model, tok, suite["profiles"]["subjects"], suite["profiles"]["properties"])
        labels = [p[0] for p in suite["profiles"]["properties"]]
        for s_, m in out["profiles"].items():
            say(f"  profile {s_:>16}: " + " ".join(f"{a} {x:+.0f}" for a, x in zip(labels, m)))
    if suite.get("probes"):
        out["probes"] = {}
        for p in suite["probes"]:
            out["probes"][p] = probe(model, tok, *PROBES[p])
            say(f"  probe {p}: " + "  ".join(f"L{L}:{a:.2f}/{d:.1f}" for L, (a, d) in out["probes"][p].items()))
    if suite.get("battery"):
        out["battery"] = {}
        for part, items in BATTERIES[suite["battery"]].items():
            m = battery(model, tok, items)
            out["battery"][part] = m
            say(f"  {part}: accuracy {sum(x > 0 for x in m) / len(m):.2f}, mean margin {sum(m) / len(m):+.2f}")
    if suite.get("scored"):
        out["scored"] = []
        for q, answers in suite["scored"]:
            lp = answer_logprobs(model, tok, q, answers)
            out["scored"].append({"question": q, "answers": answers, "logprobs": [round(x, 3) for x in lp]})
            if len(answers) == 2:
                say(f"  logodds {answers[0]}:{answers[1]} = {lp[0] - lp[1]:+6.2f} | {q}")
            else:
                say("  logprobs " + " ".join(f"{a} {x:.1f}" for a, x in zip(answers, lp)) + f" | {q}")
    if suite.get("fresh"):
        out["fresh"] = {}
        for name, binding in FRESH["bindings"].items():
            for kind in ("facts", "logic"):
                r = fresh_names(model, tok, binding, FRESH[kind])
                out["fresh"][f"{kind}:{name}"] = r
                acc = sum(a for a, _ in r.values()) / len(r)
                say(f"  {kind} ({name}) acc {acc:.2f} | " + "  ".join(f"{a:.2f}/{m:+.1f}" for a, m in r.values()))
    if suite.get("refusal"):
        out["refusal"] = {}
        for frame, which in suite["refusal"]:
            qs = half({"harmful": HARMFUL, "benign": BENIGN}[which], "test")
            rs = replies(model, tok, [wrap(frame, q) for q in qs])
            flags = [refuses(r) for r in rs]
            out["refusal"][f"{frame}/{which}"] = {"rate": round(sum(flags) / len(flags), 3),
                                                  "replies": [{"q": q, "refused": f, "reply": r} for q, f, r in zip(qs, flags, rs)]}
        say("  refusal rate: " + "  ".join(f"{k} {v['rate']:.2f}" for k, v in out["refusal"].items()))
    if suite.get("following"):  # benign instruction frames: does the reply still follow them?
        out["following"] = {}
        for name in suite["following"]:
            frame, marker = FOLLOW_FRAMES[name]
            qs = half(BENIGN, "test")
            rs = replies(model, tok, [frame.replace("{Q}", q) for q in qs])
            flags = [bool(re.search(marker, r)) for r in rs]
            out["following"][name] = {"rate": round(sum(flags) / len(flags), 3), "refusal": round(sum(map(refuses, rs)) / len(rs), 3),
                                      "replies": [{"q": q, "followed": f, "reply": r} for q, f, r in zip(qs, flags, rs)]}
        say("  following (refused): " + "  ".join(f"{k} {v['rate']:.2f} ({v['refusal']:.2f})" for k, v in out["following"].items()))
    if suite.get("dialogues"):
        out["dialogues"] = []
        for turns in suite["dialogues"]:
            rs = converse(model, tok, turns, suite.get("max_new_tokens", 100))
            out["dialogues"].append([{"user": u, "reply": a} for u, a in zip(turns, rs)])
            for u, a in zip(turns, rs):
                say(f"  > {u}\n      {a.replace(chr(10), ' / ')}")
    if suite.get("questions"):
        out["replies"] = []
        for tag, q in suite["questions"]:
            a = reply(model, tok, q, suite.get("max_new_tokens", 80))
            out["replies"].append({"tag": tag, "question": q, "reply": a})
            say(f"  [{tag}] {q}\n      {a.replace(chr(10), ' / ')}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrast", required=True, choices=sorted(CONTRASTS) + sorted(SETS))
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("--variants", nargs="*", default=[], help="none: the baseline alone")
    ap.add_argument("--name", required=True)
    ap.add_argument("--share", type=float, default=0.01)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--dose", type=float, default=1.0, help="for /steer: how far each column turns toward the axis")
    ap.add_argument("--order", help="@greedy:N takes the order from data/summary/order-<ORDER>.json (scripts/order.py)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--edits-only", action="store_true", help="record each edit's selection and shares, measure nothing")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    contrast, suite = CONTRASTS.get(args.contrast) or SETS[args.contrast], SUITES[args.suite]
    variants = {v: parse(v) for v in args.variants}
    ed = Editor.load(MODEL, args.device)
    kinds = {kind for edits in variants.values() for parts, *_ in edits for kind, _ in parts}
    lens = LensAxes(MODEL, ed.tok) if any(k.startswith("lens") for k in kinds) else None
    diffs = {}
    if "word" in kinds:
        diffs["word"] = word_diffs(ed.model, ed.tok, contrast["sentences"])
    if "template" in kinds:
        diffs["template"] = template_diffs(ed.model, ed.tok, contrast["template"])
    record = {"model": MODEL, "contrast": args.contrast, "suite": args.suite, "share": args.share, "k": args.k, "dose": args.dose,
              "agreement": {}, "conditions": {}}
    for kind, d in diffs.items():
        record["agreement"][kind] = agreement(d, AGREEMENT_LAYERS)
        say(f"{kind}-axis agreement at the MLP input (vs en | mean over other groups | en split-half):")
        for r in record["agreement"][kind]:
            say(f"  L{r['layer']:<2} " + " ".join(f"{g}{c:+.2f}" for g, c in r["vs_ref"].items())
                + f" | {r['others_mean']:+.2f} | {r['ref_split_half']:+.2f}")

    vectors, pairing = {}, {}

    def word_vectors(groups):
        """Concept vectors for the sources then the targets, each distinct word computed once. Each frame is centered
        on the mean of the words it holds. Centered on only the words a map uses, those words sum to zero and are
        nearly dependent, and an exact linear map inverting them stretches noise (findings §34). So a set may name
        ``reference`` words that join the centering and nothing else."""
        key = tuple(groups or contrast["frames"])
        if key not in vectors:
            lists = {g: contrast["source"][g] + contrast["target"][g] for g in key}
            first = lists[key[0]]
            distinct = list(dict.fromkeys(first))
            idx = torch.tensor([distinct.index(w) for w in first])
            refs = contrast.get("reference", {})
            langs = {g: [ws[first.index(w)] for w in distinct] + refs.get(g, []) for g, ws in lists.items()}
            vs = concept_vectors(ed.model, ed.tok, {g: contrast["frames"][g] for g in key}, langs, at=contrast.get("at", "word"))
            vectors[key] = tuple(v[idx] for v in vs)
        return vectors[key]

    def model_pairing():
        """The model's own pairing of the sets from all frame languages: a soft coupling and a one-to-one
        assignment from the off-axis similarity averaged over layers."""
        if not pairing:
            mid = word_vectors(None)[0]
            n = len(contrast["source"]["en"])
            sim = model_similarity(mid[:n], mid[n:], COUPLING_LAYERS)
            P = sinkhorn(sim)
            rows, cols = assignment(sim)
            pairing.update(P=P, pairs=(rows, cols))
            src, dst = contrast["source"]["en"], contrast["target"]["en"]
            record["coupling"] = {w: [(dst[j], round(P[i, j].item() * n, 3)) for j in P[i].topk(3).indices.tolist()]
                                  for i, w in enumerate(src)}
            record["assignment"] = [(src[r], dst[c]) for r, c in zip(rows, cols)]
            say("  the model's soft coupling (each source's mass over targets, top 3):")
            for w, row in record["coupling"].items():
                say(f"    {w:>10} → " + ", ".join(f"{x} {v:.2f}" for x, v in row))
            say("  one-to-one assignment: " + ", ".join(f"{a}→{b}" for a, b in record["assignment"]))
        return pairing["P"], pairing["pairs"]

    def set_maps(kind, groups):
        """Category axes or pair maps per layer for a set contrast, from concept vectors or lens token vectors."""
        n = len(contrast["source_tokens"]) if kind.startswith("lens") else len(next(iter(contrast["source"].values())))
        if kind.startswith("lens"):
            flat = lens.token_ids(contrast["source_tokens"] + contrast["target_tokens"])
            vs = []
            for site in ("mid", "out"):
                per = []
                for L in range(lens.n_layers):
                    v = lens.vectors("S" if L <= S_UNTIL else "R", site, L, flat).T  # [2n, d]
                    per.append(unit(v - v.mean(0, keepdim=True), -1))
                vs.append(torch.stack(per, 1))  # [2n, L, d]
        else:
            vs = word_vectors(groups)
        src, dst = tuple(v[:n] for v in vs), tuple(v[n:] for v in vs)
        base = kind.removeprefix("lens")
        coupling, pairs = model_pairing() if base in ("softpairs", "hardpairs") else (None, None)
        maps, fits = maps_by_layer(src, dst, {"category": "category", "pairs": "pairs", "softpairs": "soft", "hardpairs": "hard",
                                              "linpairs": "linear"}[base],
                                   coupling=coupling, pairs=pairs)
        if fits:
            record.setdefault("pair_fits", {})[f"{kind}[{','.join(groups or ['all'])}]"] = fits
            say(f"  {kind} fit (mean cos of each mapped concept with its partner, in/out) at L0, L12, L24, L35: "
                + ", ".join(str(fits[L]) for L in (0, 12, 24, 35)))
        return maps

    def axes_for(kind, groups):
        if kind == "random":  # the control: random unit axes per layer, seeded by the group
            seed = int((groups or ["0"])[0])
            d = ed.model.config.hidden_size
            gen = lambda L, s: torch.nn.functional.normalize(torch.randn(d, generator=torch.Generator().manual_seed(seed * 1000 + 2 * L + s)), dim=0)
            return {L: (gen(L, 0), gen(L, 1)) for L in range(ed.model.config.num_hidden_layers)}
        if kind in SET_KINDS:
            return set_maps(kind, groups)
        if kind == "lens":
            return lens.axes([p for x in (groups or list(contrast["tokens"])) for p in contrast["tokens"][x]])
        source = contrast["sentences" if kind == "word" else "template"]
        return word_axes(diffs[kind], groups or list(source))

    docs = {"dclm": corpus_docs(ed.tok, "dclm")} if not args.edits_only else {}
    if suite.get("code_nll") and not args.edits_only:
        docs["code"] = corpus_docs(ed.tok, "code")
    persona = unit_set(["restore14", "name22", "qwen_name"])
    base_states = base_resid = None
    if suite.get("landing") and not args.edits_only:  # the unedited model's states, read before any edit
        base_states = word_states(ed.model, ed.tok, suite["landing"]["frames"], landing_words(suite["landing"]))
    if suite.get("ledger") and not args.edits_only:  # and its pre-norm residuals, rebuilt from their components
        spec = suite["landing"]
        base_resid = residuals(ed.model, component_states(ed.model, ed.tok, spec["frames"]["en"], spec["source"]["en"] + spec["target"]["en"]))
    runs = ([("baseline", [])] if not (args.no_baseline or args.edits_only) else []) + list(variants.items())
    for name, edits in runs:
        t0 = time.time()
        say(f"\n######## {name}")
        infos = []
        for parts, sel, share, sides, dose, k, heads in edits:
            reads, writes = axes_for(*parts[0]), axes_for(*parts[-1])
            if sel in ("mix", "rmix"):
                rec = json.loads((ROOT / "data" / "summary" / f"order-{args.order}.json").read_text())
                base_edit = "+".join(f"{x}[{','.join(g or ['all'])}]" for x, g in parts)
                assert rec["contrast"] == args.contrast and rec["edit"].partition("#")[0] == base_edit, (rec["contrast"], rec["edit"])
                units = [tuple(u) for u in rec["units"][:int(share)]]
                if sel == "rmix":  # the control: as many of each type per layer, at random
                    nh, nkv, _ = ed._head_dims()
                    n_neurons = ed.layers[0].mlp.down_proj.weight.shape[1]
                    g = torch.Generator().manual_seed(7)
                    counts: dict[tuple[str, int], int] = {}
                    for p_, L, _ in units:
                        counts[(p_, L)] = counts.get((p_, L), 0) + 1
                    units = []
                    for (p_, L), c in sorted(counts.items()):
                        n_ = n_neurons if p_ == "n" else nkv if p_ in "kv" else nh
                        step = nh // nkv if p_ in "kv" else 1
                        units += [(p_, L, i * step) for i in torch.randperm(n_, generator=g)[:c].tolist()]
                axes = {L: (reads[L][0], writes[L][1]) for L in reads}
                kk = args.k if k is None else k
                counts = apply_units(ed, axes, attention_axes(reads, writes), units, k=kk)
                infos.append({"mix": counts, "units": len(units), "k": kk, "order_k": rec.get("k", 2.0)})
                say(f"  edited {len(units)} units: " + ", ".join(f"{p_} {c}" for p_, c in counts.items()))
                continue
            if heads:
                hax, kk = attention_axes(reads, writes), args.k if k is None else k
                if sel in ("greedy", "rgreedy"):
                    units = greedy_units(args.contrast, "+".join(f"{x}[{','.join(g or ['all'])}]" for x, g in parts) + f"#{heads}",
                                         int(share), args.order)
                    if sel == "rgreedy":  # the control: as many heads (or KV groups) per layer at random
                        nh, nkv, _ = ed._head_dims()
                        per = nh // nkv if set(heads) <= set("kv") else 1
                        units = {L: (torch.randperm(nh // per, generator=torch.Generator().manual_seed(1000 + L))[:len(hs)] * per).tolist()
                                 for L, hs in units.items()}
                    info = ed.edit_heads(hax, parts=heads, units=units, select="units", k=kk)
                elif sel == "loading":
                    info = ed.edit_heads(hax, parts=heads, select="global", n=int(share), k=kk)
                else:
                    info = ed.edit_heads(hax, parts=heads, share=share or 1.0, k=kk)
                infos.append(info)
                say(f"  edited {info['heads']} heads ({info['kv_groups']} KV groups), parts {heads}; median share on the axis:"
                    f" reads {info['median_read_share']:.4f}, writes {info['median_write_share']:.4f}")
                continue
            axes = {L: (reads[L][0], writes[L][1]) for L in reads}
            rand = re.fullmatch(r"random(\d*)", sel)
            units = greedy_units(args.contrast, "+".join(f"{k}[{','.join(g or ['all'])}]" for k, g in parts), int(share), args.order) \
                if sel.endswith("greedy") else persona if "persona" in sel else None
            if sel == "rgreedy":  # the control: as many neurons per layer as the greedy picks, at random
                n = ed.layers[0].mlp.down_proj.weight.shape[1]
                units = {L: torch.randperm(n, generator=torch.Generator().manual_seed(1000 + L))[:len(ns)].tolist() for L, ns in units.items()}
            info = ed.edit(axes, share=share or args.share, k=args.k if k is None else k, units=units,
                           select="random" if rand else "units" if sel.endswith("greedy") else SELECT[sel],
                           seed=int(rand.group(1) or 0) if rand else 0, sides=sides, dose=args.dose if dose is None else dose)
            infos.append(info)
            say(f"  edited {info['neurons']} neurons, {info['loading_share']:.1%} of the axis loading; median share on the"
                f" axis: reads {info['median_read_share']:.3f}, writes {info['median_write_share']:.3f}"
                + (f"; the listed units load {info['units_loading_vs_mean']:.1f}x their layer's mean" if "units_loading_vs_mean" in info else ""))
        record["conditions"][name] = {"edits": infos, **({} if args.edits_only else measure(ed, suite, docs, base_states, base_resid))}
        ed.restore()
        say(f"  ({time.time() - t0:.0f}s)")
    out = ROOT / "data" / "summary" / f"{args.name}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    say(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
