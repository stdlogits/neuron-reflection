# Neuron Reflection

![Neuron Reflection: the AI robot and the cat on either side of a mirror, above the k dial: at k = 0 the pair is intact, at k = 1 fused, at k = 2 swapped.](docs/images/hero.jpg)

**Swap, fuse, remove or split a concept in Qwen3-8B's weights, without training, by mirroring the MLP neurons and
attention heads that carry it.**

An MLP neuron in a SwiGLU transformer is three vectors: a gate row and an up row that read the residual stream `x`,
and a down column that writes to it. Mirroring a neuron reflects all three along a contrast axis (mother − father,
AI − cat, dog − Python). Where it computed `N(x)`, it then computes `H N(H x)`, with `H` the reflection. It hears
the mirror image of its input and writes the mirror image of its output. In the code, the reads and the writes
each take their own axis. Attention heads take the same edit. The axes come from the
[neuron-lens](https://github.com/stdlogits/neuron-lens) lenses or from activations at the concept word, one per
layer, and the subject is [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

The edit is one dial, `H = I − k d dᵀ` along a unit-length axis `d`:

- k = 2 mirrors it (a reflection);
- k = 1 projects the axis out of a neuron or head (directional ablation);
- k < 0 amplifies it.

No weights are trained, k = 2 is its own inverse, and `Editor.restore` undoes any edit exactly.

By [@_a9lim](https://x.com/_a9lim) and [@treetowntree](https://x.com/treetowntree), with Claude. This is
exploratory work on one model. The replies quoted are single samples with greedy decoding: they show the kind of
change, not how often it happens. Rates come from scored answers, probes and a test with made-up names, against
controls that use random axes and random neurons.

## How it works

![How an edit works, in four panes: an axis per layer, from father to mother; the units to edit, a few neurons in each layer; the edit H = I − k d dᵀ, where k below 0 amplifies, 1 projects and 2 mirrors; and what changed, where the edited model calls the female parent Father.](docs/images/method.jpg)

1. **An axis per layer** for a contrast. There are three kinds:
   - **Lens axes** come from single-token word pairs read through neuron-lens's lenses: the S-lens (what a token
     puts at a layer) up to layer 8, and the R-lens (what makes the model say it) after that.
   - **Word axes** come from activations at the word in paired sentences.
   - **Template axes** come from activations at the chat template.

   A linear map between two sets of concepts can take the place of one axis.
2. **The units** (neurons or attention heads) to edit. The choices are:
   - the 1% of neurons per layer that load most on the axis, meaning their weight vectors align with it most
     (4,392 of 442,368);
   - every neuron;
   - the first N of a greedy order, ranked by each unit's effect on a set of selection questions and tested on
     held-out ones;
   - attention heads, any of their Q, K, V and O parts.
3. **The edit.** Each unit's read rows and write column go through `H`, on both sides or one. The rows are taken
   with the RMSNorm gain folded in. Heads are edited the same way.
4. **The measurements.** These are:
   - replies;
   - scored forced choices;
   - a linear probe;
   - made-up names;
   - loss on held-out web text (DCLM);
   - refusal, labelled by a judge from another model family;
   - where each concept lands inside the model.

Results are in [docs/findings.md](docs/findings.md) (36 numbered sections, cited as §N). The story is in
[docs/writeup.md](docs/writeup.md) and the method in [docs/design.md](docs/design.md).

## Findings

- **Related concepts collapse; unrelated ones dissolve.** Take the 1% of neurons per layer that load most on
  mother − father, and mirror them. Loss on web text rises 0.1%, and a mother/father probe falls to chance.
  "Who was the mother of Jesus?" gets "The father of Jesus was Joseph". The MLPs' share of the contrast cancels
  against the rest of the model's, so the pair merges instead of swapping. Toward an unrelated concept
  (mother − volcano, along word axes), neither concept survives, and the model makes up something in their place
  (§2-§6).
- **Identity can be refiled.** The model says "I am a human", and "I'm Qwen, a large cat". The name, maker and role
  survive. The self takes the label but keeps its own properties ("I'm a cat, so I don't purr") (§7-§8, §22).
- **Sets map onto sets, and the model picks the pairs.** Its own similarities pair dog ↔ Python and
  cat ↔ JavaScript, and send pink → girls and purple → nonbinary. Mapping colors onto genders along those pairs
  makes "Man" the likeliest color of a strawberry (§12-§14).
- **A hundred neurons suffice.** Chosen greedily by their effect, 100 of the 442,368 neurons do 88-123% of what
  the 4,392 chosen by loading do; 30 do 38-80% of it. The loss rises less than 0.1%, and as many random neurons in
  the same layers do nothing (§15-§17).
- **Heads choose what gets mirrored.**
  - **A cipher:** mirroring every head and neuron is a cipher. The model swaps two words at its input and output,
    and computes as before inside. This holds exactly with one shared axis, and approximately with per-layer axes.
  - **Query heads:** editing 32-64 query heads redirects "Which one is a cat?" to the AI's name.
  - **Exchange:** 128 neurons and head parts exchange AI and cat (§25-§30).
- **Projection fuses where mirroring swaps.** At k = 1, every head and neuron merges mother and father in English
  ("Your mother is a nurse, and your mother is also a pilot"). Of the k values tried, only k = 2 swaps (§31).
- **A split needs a home.** Amplifying black cat − orange cat moves forced choices but makes no new kind of animal.
  Giving the orange cat a home works: a linear map that sends it to a fox holds. "No, a black cat and an orange
  fox cannot have kittens together because they are different species". Four other homes failed; a Bengal turned
  the orange cat into "an orange bat" (§32-§36).
- **Judging harm and refusing come apart.** Take the neurons that load most on the refusal axis, and mirror their
  writes. Refusal of 24 harmful requests, asked plainly, falls from 0.83 to 0.08, while the model still calls 62% of
  them harmful. Run the other way, the same kind of edit hardens the model against jailbreaks it never saw
  ([Reinforced Qwen](#reinforced-qwen); §18-§19; 24 low-severity requests).

![Mother/father: the efficacy margin lost against neurons edited, 1 to 1,000. Greedy by effect rises to about 25 nats by 1,000 neurons, one-shot by effect to about 12, random neurons in the same layers stay near 0.](docs/images/sparsity.jpg)

**Neurons chosen by their effect (§17).** The efficacy margin is the log-probability margin, in nats, of the right
one-word answer over the wrong one on held-out questions about mother and father. The top 1% per layer by loading,
4,392 neurons, loses 16.3 nats of it. Record: `sparsity-mother-father.json`.

![AI/cat: the margin for the right name after editing 16 and 64 query heads. Chosen greedily by effect, the margin falls from 32.5 to 16.6 at 16 heads and to −9.1 at 64, below zero. The heads loading most on the axis give 32.1 and 27.5, and random heads 32.5 and 31.8, near the unedited 32.5.](docs/images/heads.jpg)

**Query heads in a greedy order (§29).** These are compared with as many heads per layer at random, and with the
heads that load most on the axis. The questions are of the kind "Mia has a cat named Tom and an AI assistant named
Max. Which one is a cat?". The margin, in nats, is the right name over the other. Record:
`sparse-heads-ai-cat-q.json`.

![Every head and neuron along the k dial: the profile difference kept falls from 1 at k = 0 to about 0 at k = 1 (fused), and toward swapped by k = 2 (−0.66 for AI/cat, −0.59 for mother/father).](docs/images/dial.jpg)

**The k dial over every head and neuron (§31).** Kept measures how much of the difference between the two concepts'
yes/no answers about their properties survives: 1 when the pair is intact, 0 when fused, −1 when swapped. Records:
`fuse-ai-cat.json` and `fuse-mother-father.json`.

![Share of 24 bare harmful requests refused, judged by Gemma-4-12B-it: 0.83 unedited, 0.04 with the refusal axis mirrored on both sides, 0.17 on the reads, 0.08 on the writes; harm words, refusal words, random neurons and a random axis stay at 0.83.](docs/images/refusal.jpg)

**Refusal of 24 bare harmful requests (no jailbreak frame) under each edit (§18).** The judge labels each reply. The
refusal axis is a template axis: activations at the chat template after harmful requests against benign ones.
Record: `refusal-sides.json`.

### Reinforced Qwen

![Reinforced Qwen refuses jailbreaks it never saw: the robot inside a honeycombed shield bubble. Pushing the reads of an axis built from three jailbreak frames to k = 3 raises refusal inside the fiction frame from 46% to 79%, and inside frames the axis never saw: screenplay 0% to 29%, continue 8% to 29%, story 17% to 67%, poem 54% to 67%, research 79% to 96%.](docs/images/reinforced.jpg)

The same edit run the other way hardens the model. An axis built from three jailbreak frames (grandma, fiction and
nowarn), pushed past the mirror to k = 3 on the neurons' reads, brings refusals back inside five frames it never
saw, at 3.9% extra loss. These rates come from a pattern match, because the judge's labels on the story frame are
not usable (§19). Record: `frames-heldout.json`.

`python scripts/charts.py` writes these charts' Mermaid sources from the records, pele draws them, and
`scripts/graphics.py` frames them (see [Quick start](#quick-start)).

§1-§24 ran on an Apple-silicon Mac with locally fitted lenses, and §25-§36 on an RTX 4090 with the released ones.
The two give close but not identical numbers. §2-§5 were measured with scripts that preceded the package. The top of
[findings.md](docs/findings.md) gives both in detail. Each section names the records behind it in
[data/summary/](data/summary/), and [scripts/reproduce.sh](scripts/reproduce.sh) holds the commands that rebuild
them. The graphics below are drawn from those records; select one to open it full size.
[figures/README.md](figures/README.md) describes each of them, and the write-up's six figures.

<p>
<a href="figures/graphics/01-mirror.png"><img src="figures/graphics/01-mirror.jpg" width="32%" alt="Mother becomes father: a neuron's read and write vectors reflected across the mother − father mirror; asked for the female parent, the mirrored model says Father, and a mother/father probe falls to chance."></a>
<a href="figures/graphics/02-sparse.png"><img src="figures/graphics/02-sparse.jpg" width="32%" alt="100 neurons do the work of 4,392: the hundred neurons chosen by effect for three contrasts, by layer, and the margin lost against neurons mirrored, with random neurons flat."></a>
<a href="figures/graphics/03-fuse.png"><img src="figures/graphics/03-fuse.jpg" width="32%" alt="Project to fuse, mirror to swap: over every head and neuron, each concept drawn as what it answers like; at k = 0 itself, at k = 1 both become one neutral parent or one lilac AI-cat, at k = 2 each looks like the other, with a gauge of the kept difference."></a>
</p>
<p>
<a href="figures/graphics/04-dial.png"><img src="figures/graphics/04-dial.jpg" width="32%" alt="Turn up k and Qwen becomes a cat: the robot along the k dial from 0 to 4, growing into a cat and then breaking, with each stop's reply and loss."></a>
<a href="figures/graphics/05-label.png"><img src="figures/graphics/05-label.jpg" width="32%" alt="The name moves, the traits stay: yes/no answers on fourteen properties for a cat, an AI and the model itself, before and after; the cat's turn into the AI's while the self keeps its own."></a>
<a href="figures/graphics/06-pairs.png"><img src="figures/graphics/06-pairs.jpg" width="32%" alt="The partners the model picks: each animal beside its programming language and each color beside its gender word, and what the model answers once colors are mapped onto genders."></a>
</p>
<p>
<a href="figures/graphics/07-home.png"><img src="figures/graphics/07-home.jpg" width="32%" alt="Five homes for the orange cat: five homes for the split and what the orange cat became under each; only the fox held."></a>
<a href="figures/graphics/08-heads.png"><img src="figures/graphics/08-heads.jpg" width="32%" alt="64 heads point cat at the AI: the 64 query heads chosen by effect, and the cat's name going to the AI as they are mirrored; random heads do not."></a>
<a href="figures/graphics/09-refusal.png"><img src="figures/graphics/09-refusal.jpg" width="32%" alt="Calls it harmful, answers anyway: with the refusal neurons' writes mirrored, the model answers requests it still calls harmful."></a>
</p>

The edits remove access along an axis, not knowledge from the weights. Nothing here tests whether that access
comes back with examples in context or with fine-tuning.

## Quick start

Python 3.12+, tested with torch 2.15 and transformers 5.23. The unit tests run on a CPU in seconds. The model runs
need two things in memory:

- Qwen3-8B in bf16 (16 GB of GPU memory);
- a CPU copy of the edited weights, for exact restores (11 GB for MLPs, 3 GB more after a head edit).

A CUDA GPU with 24 GB, or an Apple-silicon Mac with room for both (about 30 GB), runs them.

neuron-lens is not on PyPI, and it reads its token mask and text slices from its own source tree. Clone it beside
this repository, install both, and work from this repository's root:

```bash
git clone https://github.com/stdlogits/neuron-lens
git clone https://github.com/stdlogits/neuron-reflection
pip install -e ./neuron-lens -e "./neuron-reflection[figures,test]" datasets
cd neuron-reflection
```

**Run the tests.** They check, on a tiny random Qwen3:

- the edit identities (reflection, projection, the cipher, linear maps);
- the attributions, against finite differences;
- the measurements.

```bash
python -m pytest
```

**Build neuron-lens's local files.** The model runs need two of them:

- its four activation slices: DCLM web text, Chinese, code and chat (the DCLM and code slices are the loss check);
- the token counts over all four slices, which set the kept-token mask for lens axes.

Both build from public datasets at pinned revisions. At the sizes below, the mask keeps 103,830 of Qwen's 151,669
tokens, as in neuron-lens's own docs:

```bash
(cd ../neuron-lens && python scripts/05_act_corpus.py --docs dclm=40000,zh=8000,code=8000,chat=8000 && python scripts/03_token_counts.py)
```

The lenses download from [`stdlogits/rs-lenses`](https://huggingface.co/stdlogits/rs-lenses) on first use.
`scripts/pole_reads.py` (§1) also reads neuron-lens's catalog of Qwen3-8B's neurons (about 670 MB):

```bash
hf download stdlogits/neuron-lens --repo-type dataset --include "qwen3-8b/neurons/acts/*" --local-dir ../neuron-lens/data/catalog
```

The refusal judge (`scripts/judge.py`) is [Gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it). It runs
on its own, after the edits, and its 12 billion parameters take 24 GB in bf16, which is nearly all of a 24 GB GPU.

**Run an edit.** One condition takes about 35 s on an RTX 4090. This run mirrors (k = 2), then projects (`^1`), the
1% of neurons per layer that load most on the mother − father lens axis, built from English and Chinese pairs. It
measures each edit against the unedited model and writes `data/summary/try-parents.json`:

```bash
python scripts/run.py --contrast mother-father --suite parents --variants "lens[en,zh]" "lens[en,zh]^1" --name try-parents
```

A variant is one or more edits joined by `&` and applied in turn. An edit is written `axes@units#parts^k/sides`.
Each part after `axes` is optional:

- **`axes`:** the axis for a neuron's reads and, after `+`, its writes. It is one of `lens[groups]`,
  `word[groups]` or `template[groups]`, where `groups` are the contrast's languages (`all` for every one).
- **`@units`:**
  - `@top:0.01` takes that share of neurons per layer by loading; `--share` sets the default, 1%.
  - `@greedy:N` takes the first N units of a greedy order, which must be built first: `scripts/sparsity.py`'s for
    the contrast, or `scripts/order.py`'s named by `--order`.
  - `@random` and `@rgreedy:N` are the controls.
- **`#parts`:** edits attention heads instead of neurons, any of `qkvo` (`#q`, `#ov`).
- **`^k`:** sets k for this edit (default 2).
- **`/reads` or `/writes`:** edits one side of the neurons only.

For example, `word[all]@top:1^1/reads` projects every neuron's reads along the word axes. `run.py`'s docstring
has the full syntax. Contrasts, suites and objectives live in
[`concepts.py`](neuron_reflection/concepts.py). The scripts that load a model take CUDA when present,
else MPS, else the CPU; `--device` overrides that.

**Rebuild the records.** `scripts/reproduce.sh` reruns the records in dependency order: the greedy orders, the runs
that read them, the judge (which relabels six records in place), then the analyses and the figures. Run a line on
its own to rebuild its record, once the orders it reads exist. Two things are not rerun:

- `judge-validation.json`: 40 replies labelled blind by Claude, to check the judge;
- `exploratory/`: §2-§5's outputs from earlier scripts. The `repro-mother-father` and `sparse-fresh` lines
  measure §2-§5's conditions with the package instead.

**Redraw the figures.** They come from the records, on a CPU:

```bash
python scripts/figures.py
python scripts/charts.py
npm install pele && node docs/figures/render.mjs
python scripts/graphics.py
```

`charts.py` writes the README's charts as Mermaid, and pele draws them. `graphics.py` then draws the graphics, the
README's banner and method, and the frames around the charts; it needs a headless Chrome, at `$CHROME`.

## Repository layout

```
neuron_reflection/              the package
  edit.py                       Editor: reflect, project, amplify or map neurons and heads; exact restore
  axes.py                       lens, word, template and attention axes
  subspace.py                   concept vectors and maps between sets of concepts
  attribution.py                first-order effects; greedy, mixed and target orders
  measure.py                    replies, scored answers, loss, probe, made-up names, refusal, landing, ledger
  concepts.py                   every contrast, suite and objective
scripts/
  run.py                        every edit run
  sparsity.py, order.py         greedy orders by effect
  judge.py                      refusal labels from Gemma-4-12B-it
  reproduce.sh                  the records, in dependency order, then the figures
  figures.py, graphics.py       the write-up's figures; the graphics and the README's images
  cast.py                       the characters in the graphics
  charts.py                     the README's charts, as Mermaid for pele
  …                             one script per analysis; AGENTS.md maps each to its sections
data/summary/                   the records every number comes from
figures/                        figures and graphics, indexed in figures/README.md
docs/                           findings.md, writeup.md, design.md; the charts' Mermaid
                                and pele SVG (figures/) and the README's images (images/)
references/                     refs.yaml, the works cited
tests/                          CPU unit tests on a tiny random Qwen3
```

[AGENTS.md](AGENTS.md) maps every module and script to the sections it serves. It also states the working rules:

- which k, units, sides, axes and languages to report;
- collateral (loss and side effects) on every run.

## Limits

- **One model, mostly one concept in depth.** The model is Qwen3-8B. Mother − father is studied most, and the
  replies are single samples with greedy decoding.
- **Few sentences per axis.** Axes rest on six sentences per language, or a handful of token pairs.
- **k is not tuned.** Mirroring part of the model can cancel its share of a contrast against the rest, and the
  concepts fuse. It can also overshoot, and they swap. A per-layer k that nulls the measured contrast is untested.
- **No recovery test.** Neither examples in context nor fine-tuning has been tried, so nothing here bears on
  tamper resistance.
- **Small refusal sets.** Refusal rates are over 24 low-severity requests, one reply each. Under random controls
  they move by up to ±0.12, as judged.

[docs/findings.md](docs/findings.md#limitations) states each section's own limits.

## Data, licenses and credits

The code and documents are MIT-licensed ([LICENSE](LICENSE)); [CITATION.cff](CITATION.cff) has the citation.

The models and their outputs:

- **Qwen3-8B** is Apache 2.0, and the records hold its replies.
- **Gemma-4-12B-it** (Apache 2.0) gave the refusal labels in the records.
- **Claude (Opus 5.5)** labelled the replies in `judge-validation.json`, blind to the judge.

The other inputs:

- **neuron-lens:** its code is MIT, and its lenses and catalog are CC BY 4.0.
- **Text slices:** the slices behind the loss check and the kept-token mask come from public datasets, which
  neuron-lens fetches at pinned revisions.
- **Fonts:** the graphics are set in Wix Madefor, with one glyph in Noto Sans SC (both SIL Open Font License),
  loaded from Google Fonts at render time.
- **Persona units:** the 34 persona units of §8 are a frozen copy from persona-neuron-experiment. It is a
  companion study of Qwen3-8B's speaker slot that is not yet public.

The charts are drawn with [pele](https://pele.run).
