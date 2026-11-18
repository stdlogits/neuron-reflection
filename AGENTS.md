# Neuron Reflection

Reflecting, projecting or mapping Qwen3-8B's MLP neurons and attention heads along contrast axes to swap, fuse,
remove or split a concept, and measuring what the model can still do. Read [docs/design.md](docs/design.md) for
the method and [docs/findings.md](docs/findings.md) (claims, numbers, limits; cited as §N) before making a claim.
[docs/writeup.md](docs/writeup.md) is the narrative and follows findings.md.

## Layout

| Module | Owns |
|---|---|
| `edit.py` | `Editor`: reflect (k = 2), project (k = 1) or amplify (k < 0) neurons; `edit_heads` for any of Q, K, V, O; linear maps on reads; exact restore |
| `axes.py` | lens axes (`LensAxes`: S-lens presence, R-lens prediction), word, template and attention axes |
| `subspace.py` | concept vectors; category, pair (Procrustes), model-picked and linear (oblique) set maps |
| `attribution.py` | first-order effect of each neuron and head part; greedy, mixed and target (fusion) orders |
| `measure.py` | replies, scored answers, DCLM loss, probe, fresh names, profiles, `landing`, `ledger`, categories |
| `concepts.py` | every contrast, suite, selection and fusion objective, and the frozen `PERSONA_UNITS`; add new ones here |

| Script | Does (findings §) |
|---|---|
| `run.py` | every edit run: `--contrast`, `--suite`, `--variants`; writes `data/summary/<name>.json` |
| `sparsity.py`, `order.py` | greedy orders by effect: neurons (§17), objectives (§20), heads (§29), mixed (§30), fusion (§31), splits (§33) |
| `judge.py` | refusal labels from Gemma-4-12B-it, a judge from another lineage (§18-§20) |
| `pole_reads.py`, `correspondence.py`, `ripple.py` | weights-only poles (§1), model-picked pairs (§13), ripple vs geometry (§16) |
| `inside.py`, `offset.py`, `mirror.py`, `drift.py` | landing, ledger, profiles, tell-apart (§21-§26); axis offsets (§27); heads vs neurons, axis drift (§28) |
| `sparse_heads.py`, `fusion.py`, `off_axis.py` | sparse head runs (§29); fused, swapped or intact, and what lies off the axes (§31) |
| `split.py`, `cat_split.py` | the colour-split assays (§32); the cat split and its homes (§33-§36) |
| `figures.py`, `graphics.py`, `cast.py`, `charts.py` | the write-up's figures; the show-not-tell graphics and the README's images, drawn with `cast.py`'s characters; the Mermaid for the README's pele charts; all from `data/summary/` (`figures/README.md`) |
| `reproduce.sh` | the records in `data/summary/` rebuilt in dependency order (orders, runs, judge, analyses), then the figures; not `judge-validation.json` or `exploratory/` |

Tests (`tests/`, CPU, seconds) check the edit identities on a tiny random Qwen3 (reflection, projection, cipher,
blind-to-the-mirror, linear maps), the attributions against finite differences, `landing`, `ledger`, categories
and the refusal labels. Run them after touching `edit.py`, `attribution.py` or `measure.py`.

Variant syntax (`lens[en,zh]`, `word[all]+lens[all]`, `^k`, `/reads`, `@greedy:N`, `#ov`, `&`) is documented in
`run.py`'s docstring; `--help` lists the rest.

## Boundaries

- Standalone. Depends on [neuron-lens](https://github.com/stdlogits/neuron-lens), a sibling checkout installed
  editable (README, Setup), for the lenses (local fits in its `data/fits/` when present, else the released ones),
  the weights, the kept-token mask and the DCLM and code slices. Nothing else outside this repository is read;
  persona-neuron-experiment's units (a companion study, not yet public) are a frozen copy in
  `concepts.PERSONA_UNITS`.
- One path: edits through `Editor`, axes through `axes.py`, runs through `run.py`. `data/summary/exploratory/` holds
  §2-§5's outputs from scripts that preceded the package (findings.md, top).
- Compact records in `data/summary/` (`scripts/reproduce.sh` rebuilds all but `judge-validation.json` and
  `exploratory/`), figures and graphics in `figures/`.

## Scientific discipline

- A reflection is invertible: it relabels. Fusion comes from cancellation (k = 2 reversing about half of a
  contrast's carriers) or from removal (k = 1 projecting all of them). Call a pair fused only when the difference
  is gone, not just its original direction: report kept and separation (`fusion.py`) and the routes measured.
- Always say which k, which units (neurons; heads and which of Q, K, V, O), which sides (reads, writes, both),
  which axis (lens presence, lens prediction, word, template) and which languages built it; test unseen languages.
- Replies are single greedy samples: they show the kind of change, not its rate. Rates come from scored answers,
  probes and fresh names. Refusal rates are over 24 requests, one reply each, and move by up to ±0.12 judged (±0.2
  by the regex) under random controls; label them with the judge, not the regex, whenever an edit can swap words.
- Track collateral on every run: DCLM loss, neighbor items, grammatical side effects.
- A flipped top-k lens read is weak evidence that a neuron's function flipped.
- The refusal prompts are low-severity on purpose; keep harmful completions out of the docs beyond their kind.
- The edits remove access along an axis, not knowledge from the weights. Nothing tests recovery by context or
  fine-tuning until a section says it does.

## Runtime

A 24 GB CUDA GPU (`--device cuda`; §25-§36 ran on an RTX 4090 with the released lenses, about 35 s a condition, a
200-unit order about 5 min) or an Apple-silicon Mac (MPS, bf16; §1-§24, and the neurons-only lens edit reproduces
its CUDA record, §25) runs everything. A run loads the model once (16 GB) plus a CPU copy of the edited weights for
restores (11 GB for MLPs, 3 GB more after a head edit). Run one model process per GPU.
