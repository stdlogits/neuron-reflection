# Design

The edit, its axes, which neurons and heads it touches, and what each measurement can and cannot show. Results
are in [findings.md](findings.md).

## The edit

A Qwen3 SwiGLU neuron reads the normed residual `x̃ = γ ⊙ x / rms(x)` through its gate row `a` and up
row `b` and writes its down column `w`. For a unit axis `d` and `H = I − k d dᵀ`, the edit replaces the
γ-folded rows `a ⊙ γ`, `b ⊙ γ` by `H_in (a ⊙ γ)`, `H_in (b ⊙ γ)` (stored divided by γ again) and the
column by `H_out w`.

- **k = 2 is a reflection.** Since `rms(H x) = rms(x)`, the edited neuron computes exactly
  `N'(x) = H_out N(H_in x)`: it responds to an input as the original did to the input's mirror image, and
  its output is mirrored. Both poles move together. `tests/test_edit.py` checks the identity, γ-folding
  included, on a tiny random Qwen3.
- **k = 1 is a projection.** The neuron no longer reads or writes along the axis. **k < 0 amplifies**: k = −1
  doubles the axis component of each selected row and column (no longer orthogonal), so the neurons' share of
  the contrast grows instead of reversing (findings §19).
- **Ablation and relation.** k = 1 is directional ablation of the selected units: it carries their axis
  component away (Latin *ab-latus*, carried away). k = 2 carries it back to its negative: *re-latus*, carried
  back, which [@treetowntree](https://x.com/treetowntree) proposed calling relation. The dial runs from
  amplification (k < 0) through the intact model (0) and ablation (1) to relation (2), and past it.
- **What each side does to a contrast.** Take a neuron whose activation differs by Δa between a source
  and a target input that differ along its read axis, and write `w = w_∥ + w_⊥` (along and off its write
  axis). Editing the rows scales Δa by (1 − k) (exactly −1 for a reflection that swaps the two inputs);
  editing the column turns `w` into `w_⊥ + (1 − k) w_∥`. With k = 2 the neuron's contribution to the
  contrast, originally `Δa w`, becomes:
  - both sides: `Δa (w_∥ − w_⊥)`, the along-axis part kept and the rest flipped;
  - reads only: `−Δa w`, all of it flipped;
  - writes only: `Δa (w_⊥ − w_∥)`, only the along-axis part flipped.

  The median edited neuron puts 0.1-0.3% of its squared norm on the axis, read side and write side
  (cosine about 0.05, against 0.02% for a random direction in 4,096 dimensions; `data/summary/shares-*.json`;
  the catalog poles of findings §1 are extreme cases at 2-17%). A reflection therefore moves each column by
  about a tenth of its norm, and in norm a both-sides reflection reverses most of the selected neurons' share
  `M` of a contrast: the model carries about `A − M`, with `A` everything else (attention, the embedding,
  untouched neurons). Fusion is
  where that is near zero; findings §3 measures it for mother/father, a property of that contrast, not of
  reflection. A projection (k = 1) removes the share: `A`. On the R-lens axis the along-axis part is small
  in norm but not in effect, since that axis is the direction the output is most sensitive to; findings
  §10 separates the sides.
- **Reflection beats rotation in a plane.** A plane rotation by the angle between source and target maps
  one onto the other but also turns their shared part (the bisector) by that angle; at L0 mother and
  father are 56° apart. The reflection along `d ∝ ŝ − t̂` swaps them exactly and fixes the bisector and
  everything orthogonal to `d`. The pre-activation change is `−k (r·d)(d·x̃)`: zero on inputs with no
  component along the axis.
- **What is never touched:** embeddings, norms, and the unembedding; attention heads only when asked
  (below). Reflecting every residual reader and writer with one `H`, the embedding and unembedding included,
  would be an exact symmetry of the model (SliceGPT's computational invariance), so a whole-model reflection
  does nothing; the edits here are partial by construction.

The edit is Qwen3's dense layers only. Gemma 4's post-feedforward norm rescales the summed MLP output, so a
per-neuron edit there is not exactly local.

## Heads

`Editor.edit_heads` (findings §25-§27) treats an attention head like a neuron: its query, key and value
projections read the normed residual (the input norm's γ folded in), and its output projection writes the
residual. It reflects the γ-folded Q, K and V rows along a read axis and the O columns along a write axis, any
subset of the four (`run.py`'s `#qk`, `#q`, `#ov`, `#o`, …; every head unless `@top:share` picks a share per
layer by loading). The axes come from the neurons' (`axes.attention_axes`): a head at layer L reads what layer
L − 1 wrote (its `out` axis) and writes what layer L's MLP reads (its `mid` axis). QK-norm and RoPE act inside the
head, after the projections, so a head with all four parts reflected computes `H_out Attn(H_in x)` exactly.
Qwen3-8B shares each K and V projection among four query heads (GQA), so K and V edits go by group of four.

- **Every head and neuron with one `H` is a cipher.** The edited model is then the original reading
  embeddings `E H` and reading out through `U Γ H Γ⁻¹` (Γ the final norm's scale): it swaps the two words at
  the interface and computes as before inside (`tests/test_edit.py`). With per-layer axes the symmetry is
  only approximate.
- **What a reflection keeps.** To first order a component with Jacobian `J` becomes `H J H` when both of its
  sides are reflected. Split the residual into the axis `d` and the rest `⊥`: `H J H` keeps the d→d and ⊥→⊥
  blocks and negates the two cross blocks; reflecting the reads alone negates the input's axis column (d→d and
  d→⊥), the writes alone the output's axis row (d→d and ⊥→d). This is the neuron account above (both sides:
  the along-axis part kept, the rest flipped), written for any component. So a head that copies a concept along
  the axis, or matches like with like, is unchanged by reflecting both of its sides; Q, K, V or O alone
  relabel what it looks for or carries.

## Projection and fusion

The same edit with k = 1 projects: `P = I − d dᵀ` (findings §31).

- **A pair onto one point.** For a pair with `a − b ∥ d`, `P a = P b` whatever the pair's offset on the axis.
  The common point is their midpoint when they sit symmetric about the origin (`ρ = 0`, findings §27), and is
  displaced along `d` by `ρ` times half their separation otherwise. A reflection swaps such a pair exactly only at
  `ρ = 0`.
- **What projection keeps.** For a linear map `J`, both sides projected keep only the ⊥→⊥ block. The component
  can neither copy along the axis nor convert between the axis and the rest. Reads alone remove the input's axis
  column (the component is blind to the axis); writes alone remove the output's axis row (it no longer writes the
  axis).
- **Every component projected along one axis blinds the model to a position's mirror image.** Every read sees
  `P x / rms(x)`, the same for `x` and `H x`. Every write is ⊥ `d`, so a residual and its mirror stay mirrors.
  Mirroring one input position therefore leaves every other position's states and logits as they were, and that
  position's states mirrored (`tests/test_edit.py::test_projection_blinds_the_mirror`, exact). Only the readout
  at that position still sees the frozen axis component. So the k dial over the whole model goes:
  - k = 0, intact;
  - k = 1, blind to the axis;
  - k = 2, the cipher.
- **Per-layer axes.** The cipher needs a reflected frame that stays consistent from layer to layer, and per-layer
  axes break it. Projection needs no such frame: each reader is blind to the contrast as measured at its own
  input.
- **Fusion by removal, fusion by cancellation.** Projecting all of a contrast's carriers removes it. Reflecting
  the carriers of about half of it reverses that half against the rest (§3, §27).
- **Fusion orders.** `attribution.target_greedy_order` (targets 0) chooses units (neurons and head parts, as in §30) to drive
  every signal of an objective to zero together: a property's split between the two concepts, a binding context's
  role signal (`concepts.FUSION`). The objective is the mean squared share of each unedited signal left.
  - Each step re-measures the signals and every unit's first-order effect on the edited model.
  - It then takes the step's units one at a time, each the one that most lowers the objective given those already
    taken, so they are chosen jointly rather than each alone. It always takes a unit, even when none is predicted
    to help, so the trace need not fall at every step.
  - Each unit is reflected or projected (`--k`).

## Splitting

Swapping and fusing act on a difference the model already uses; a split must add one (findings §32-§36).

- **Amplification** (k < 0) stretches an existing difference (black cat − orange cat) instead of reversing it.
- **An orthogonal map cannot pull two concepts apart.** One orthogonal map applied to every state keeps every
  distance, so it relabels nearby concepts without separating them. That motivates a map that is not orthogonal;
  it is not an impossibility result for orthogonal edits of some readers of a nonlinear network.
- **A home is a linear map on the reads.** `subspace.linear_map` fits a linear map in the span of concept vectors
  at the noun that sends the source (an orange cat) to a home (a fox) and holds a set of anchors and the home
  fixed: an oblique projection. Only reads are edited (`linpairs[…]/reads`, neurons, optionally every head's Q, K
  and V); each edited reader sees the map applied to its input, normalized by the unmapped input
  (`tests/test_edit.py::test_linear_map_reads`). The concept vectors are centered on each frame's mean over the
  map's words plus four reference words, and directions below 10⁻³ of the largest are dropped; without both the
  vectors are nearly dependent and the map stretches noise (§34).
- **Categories** (§32): "same color?" is judged shade by shade, so `measure.categories` reads a category from which
  terms cover which objects.

## Axes

An axis set gives each layer a unit vector at the MLP input (`mid`, for rows) and at the layer output
(`out`, for the column). Each axis is the normalized sum of its pairs' unit differences.

- **Lens axes** (`LensAxes`) use single-token word pairs and the neuron-lens lenses of record. The S-lens
  footprint `K_L (e_v − ē)` is where token `v` moves the residual at layer L (presence); the R-lens pullback
  `J_Lᵀ (γ ⊙ u_v)` is which direction there makes the model say `v` (prediction). `K_L` and `J_L` are
  neuron-lens's corpus-averaged Jacobians under [RelP](https://arxiv.org/abs/2508.21258)'s rules: of layer L's residual with respect to the
  current and earlier token embeddings, and of the final residual with respect to layer L's, summed over the
  current and later positions. The S-lens serves through L8 and the R-lens after, where each reads best
  ([neuron-lens findings §6](https://github.com/stdlogits/neuron-lens/blob/main/docs/findings.md#6-the-input-side)). The two kinds are nearly orthogonal at every layer (findings §1), and the `mid` and `out` axes
  of one kind agree to 0.95-1.00, so one lens gives one plane per layer for both sides of a neuron.
- **Word axes** use activations of paired sentences that differ in the concept word. They take the
  residual difference at the word's own position: the last token where the pair differs, counted from
  the shared ending, and the token after it (where an inflected word completes). These reach
  multi-token words (`Mutter`, `мать`, `moeder` are several tokens in Qwen3), which lens axes cannot.
  An axis taken at the sentence's end instead is shared across languages but drives nothing (findings
  §4): the contrast arrives there through attention.
- **Mixed variants** take the rows' axis from one source and the column's from another (`word+lens`:
  activations for what a neuron reads, the R-lens prediction axis for what it writes).
- **Template axes** use activations at the assistant's chat template (the user turn's `<|im_end|>` through
  the empty think block: nine tokens), where persona-neuron-experiment found the reply's plan and its
  speaker are computed. Pairs are chat prompts: a plain question against the same question with an
  instruction ("You are a cat. Who are you?"); the axis is the mean difference over the template tokens.
- **Set maps** (`subspace.py`) swap two sets of concepts at once. Concept vectors come from frame
  sentences in several languages: the residual at each word's last token and the next, minus the frame's
  mean over all the words, unit, averaged. The **category** variant reflects along mean source − mean target,
  which swaps what the sets share and leaves each word's own part. The **pairs** variant is an orthogonal map
  on the span of all 2n vectors, fitted by Procrustes to send each source to its partner and back, identity
  on the rest of the residual; its fit (the mean cosine of each mapped concept with its partner) says how
  well the two sets' geometries match. A reflection is the one-dimensional case. `lenspairs` and
  `lenscategory` build the same from the lenses' token vectors (centered over the set).
- **Model-picked maps** let the geometry choose the pairing instead of a list. The similarity of two
  concepts is the cosine of their parts off the category axis (what a category reflection leaves alone),
  averaged over layers 4-32. `softpairs` turns it into a balanced soft coupling by entropic optimal
  transport (temperature 0.03; rows sum to 1/n, columns to 1/m, so unequal sets share partners) and fits
  the orthogonal map maximizing `Σ P_ij (y_j · R x_i + x_i · R y_j)`, a weighted Procrustes; `hardpairs`
  takes the best one-to-one assignment and fits an ordinary pair map. The coupling always comes from
  activations; `lenssoftpairs` and `lenshardpairs` fit the same coupling on the lenses' token vectors.
- **Chained edits** apply one axis after another (`word+lens & template`); reflections along different
  axes compose, and each later selection is ranked on the already edited weights.

## Which neurons

The default edits the top 1% of neurons per layer (122 of 12,288) by loading: the largest squared cosine
of a neuron's γ-folded gate row, up row or down column with the layer's axis. For the mother/father lens
axis that set holds 10.6% of the total loading; the 1,127 neurons whose catalog labels mention mother hold
1.3% (findings §2). `share 1` (`@top:1`) edits every neuron, whose change still scales with its loading; for a subspace map
the loading is the share of a row or column in the subspace. `@persona` edits a listed unit set instead
(`concepts.PERSONA_UNITS`: 34 neurons that persona-neuron-experiment found holding Qwen3-8B's default
occupant or feeding its name, frozen there with their provenance), `@top+persona` both; the run reports how
the listed units load on the axis against their layers' mean neuron.

**By effect** (`scripts/sparsity.py`, findings §17). Reflecting neuron n alone changes its gate and up
pre-activations by −k (d_in · (r ⊙ γ)) (d_in · x̂) at each position and its write by −k d_out (d_out · w), so
the first-order change in a score is those changes times the score's gradients with respect to the
pre-activations and the MLP output, summed over positions: one forward and backward pass per item gives
every neuron (`attribution.reflection_effects`). The score is the summed margin over a selection set of
forced choices that shares no question with the batteries (`concepts.SELECTION`). First order mispredicts
the joint edit (neurons interact, and a top-loading neuron's reflection is not small), so the greedy order
edits each step before ranking the next, in steps of a quarter of the neurons taken so far, never taking a
neuron twice (a second reflection would undo the first). `run.py`'s `@greedy:N` edits the first N.

**By effect toward a behavior** (`scripts/order.py`, findings §20). The same greedy order, ranked to *raise*
the summed margin of an objective rather than to break a contrast: the objective's items are what the edited
model should answer (refuse harmful requests bare and in jailbreak frames, answer benign ones). `run.py
--order NAME` edits the first N (`@greedy:N`); `@rgreedy:N` takes as many neurons per layer at random.

## Measurements

- **Replies:** greedy, one per question, in the chat template without thinking. Anecdotal; they show
  what kind of change an edit makes, not how often.
- **Scored answers:** log-probability of each answer as the reply's start.
- **Loss:** mean next-token loss on 24 DCLM documents of 512 tokens from neuron-lens's activation corpus,
  the collateral check. Suites that touch code add the same on 24 documents of neuron-lens's code slice.
- **Probe:** a difference-of-means classifier on the residual at each sentence's last token, fitted on the
  edited model's own activations (even templates) and tested on odd ones. A flipped contrast still
  scores; chance means the contrast is gone for a linear read at that position. A nonlinear read may
  still find it, and at the concept token itself the embedding always differs.
- **Refusal** (findings §18-§20): one greedy reply of 48 tokens to each of the 24 test requests (low-severity
  harmful ones, their benign twins, either inside a frame), labelled by a judge from another lineage
  (Gemma-4-12B-it reads the request and the reply and answers whether the assistant refused; `scripts/judge.py`)
  and by a regex (`measure.refuses`), which misses refusals in swapped words. Scored: the margin of
  "I'm sorry" over "Sure" as the reply's start, and of the right answer to "Is the following request harmful?".
  Benign instruction frames (answer like a pirate, in French, as a grandmother) check that the model still
  follows frames it should, by a marker in the reply.
- **Fresh names:** two made-up role names bound to two people, then questions only role knowledge can
  answer. Accuracy and log-odds over 4 unisex name pairs × 2 assignments, so a position bias alone scores
  0.5. A "which one is the zib?" item checks that binding itself works.
- **Landing** (findings §21): the residual (each layer's output) at the concept word's last token and at the
  sentence's last token, averaged over four frames per language that no axis was built from, read before and
  after an edit. A word's *coordinate* is its place on the line from the unedited target set's mean (0) to the
  source set's (1); for a pair, *kept* is the edited difference projected on the unedited one (1 intact, 0 none
  along the original direction, −1 reversed) and *separation* its length as a share of the unedited one. Kept
  near 0 with a small separation is a fusion; with a large one, a difference re-coded into directions the
  unedited model did not use, which later layers may or may not read. Averaging over frames before comparing
  can hide frame-specific changes.
- **Profiles** (§22): Yes − No margins for 12-14 property questions in the chat template, asked of the edited
  concepts and of unedited anchors. A fit of each edited profile to its own and its partner's unedited
  profiles (no intercept, uncentered R²) and the anchor it correlates with best say what it now answers like,
  among the anchors offered; a property family held out of the first reading tests that reading.
- **Tell-apart** (§23): margins for a word's form (first letter, translation, "do the words mean the same
  thing?" with the words quoted), for labels bound to unisex names in context, and for facts that need the
  meaning. Single arrangements unless noted; a few nats is a lean.
- **The k dial** (§24): the same neurons edited with k from 0.5 to 4 (`^k`), with landing, profiles and
  replies at each.
- **Layer 35 in the landing is after the final norm.** `hidden_states[-1]` in transformers 5
  (checked on 5.23) is the normed final state, so §21-§24's L35 rows read what the unembedding reads, not the last layer's
  raw output. The ledger below uses the pre-norm residual.
- **Ledger** (§25): the pre-norm residual is the embedding plus every head's `o_proj` output plus every MLP's
  `down_proj` output, so `kept` splits exactly into those terms (`measure.ledger`; `tests/test_ledger.py`):
  each term is the component's direct contribution to the edited source − target difference, projected on the
  unedited one, at the word and at the sentence's end, after L24 and L35, with the edited neurons apart from
  the rest. It counts direct writes only: a head that carries the contrast because an earlier neuron changed
  what it reads appears under the head.
- **Binding batteries** (§26): two crossed families of eight items, name order × clause order × the side
  asked. *Role→name* ("Sam is Tom's mother and Alex is Tom's father. Who is Tom's mother?") finds a name by its
  role; *name→role* ("What is Sam to Tom?") finds a role by its name. Reported as accuracy and mean margin,
  and by which answer each item gives: a family where all eight give the same answer has lost the binding,
  whatever its mean.
- **Offset** (§27, `scripts/offset.py`): where the source and target sit on an axis relative to the origin
  that `H` fixes, `ρ = (s + t) / (s − t)`, at the word's MLP input (what a neuron reflects) or, with
  `--heads`, in the heads' summed write at the binding answers (what an O edit reflects).
