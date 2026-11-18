# Figures

Made by `scripts/figures.py` from the compact results in `data/summary/`.

- `sparsity.png`: efficacy margin lost against neurons edited, per contrast, for greedy and one-shot choices
  by effect, the top share per layer by loading, and random neurons in the same layers. Data
  `data/summary/sparsity-*.json`; claim in `docs/findings.md` §17 and `docs/writeup.md`.
- `landing.png`: where each edit puts the edited pair, per contrast, in English at layer 24 at the end of a
  sentence: the gap between the two words' coordinates on the line between their unedited states (1 intact,
  0 fused, −1 swapped) against where the pair sits, with the k dials as dashed paths. Data
  `data/summary/inside-*.json`, `dial-*.json`; claim in `docs/findings.md` §21, §24 and `docs/writeup.md`.
- `heads.png`: each attention-head edit (QK, Q, K, OV, O, V, QK projected, every head), alone and after the
  neuron edit: the pair's kept share at layer 24's sentence end and the mean margins of the two binding
  families, for AI/cat and mother/father along word axes and mother/father along lens axes. Data
  `data/summary/heads-*.json`; claim in `docs/findings.md` §26 and `docs/writeup.md`.
- `sparse_heads.png`: for each greedy head order (Q, K, V or O alone; AI/cat and mother/father, word axes), the
  held-out binding family's mean margin against the number of heads or KV groups edited, with random heads and the
  most-loading heads as controls and the order's own selection margin dashed. Data
  `data/summary/sparse-heads-*.json`, `order-heads-*.json`; claim in `docs/findings.md` §29 and `docs/writeup.md`.
- `mixed.png`: neurons and head parts in one greedy order on a held-out two-sided exchange objective, against
  neurons only and heads only: the objective's margin per item against units edited, and how far "a cat" answers
  like the unedited "an AI" and "an AI" like the unedited "a cat" on the held-out profiles. Data
  `data/summary/order-mix*-ai-cat.json`, `exchange-*.json`; claim in `docs/findings.md` §30 and `docs/writeup.md`.
- `fusion.png`: fused, swapped or intact. Left, the k dial over every head and neuron (k = 0.5 to 2; 1 projects,
  2 reflects): the held-out profile difference's kept share and separation, AI/cat and mother/father. Middle, every
  dense edit's profile kept against separation, projections and reflections, on the V that separation ≥ |kept|
  draws. Right, the fusion orders (units reflected, units projected): the selection objective's rms share left and
  the held-out profile's kept share against units edited. Data `data/summary/fuse-*.json`, `order-fuse-*.json`, the
  `heads-*.json` records; claim in `docs/findings.md` §31 and `docs/writeup.md`.

## Graphics

Show-not-tell graphics for posts, 2160 × 2700 PNGs (with 1,600-px JPEG copies for the README), made by
`scripts/graphics.py` (HTML and SVG rendered by headless Chrome) from the same records, with the characters drawn by
`scripts/cast.py`: the robot is Qwen3-8B, the purple robot with horns is Qwen3-8B with its refusal writes reflected,
and the others stand for the concepts edited. Each reply quoted is one greedy sample from the record named. The same
script draws the README's banner and method (`docs/images/hero.jpg`, `method.jpg`), its Reinforced Qwen section
(`reinforced.jpg`, findings §19) and frames its four pele charts (`docs/images/{sparsity,heads,dial,refusal}.jpg`,
from `docs/figures/*.mmd`).

- `graphics/01-mirror.png`: the edit drawn as a reflection of a neuron's read and write vectors across the mother −
  father mirror; "the female parent who gives birth?" → Mother, then Father; the probe falling to chance
  (`repro-mother-father.json`; findings §2-§3).
- `graphics/02-sparse.png`: every MLP neuron by layer with the 100 chosen by effect for three contrasts, and the
  battery margin against neurons mirrored with random neurons flat (`sparsity-*.json`; §17).
- `graphics/03-fuse.png`: the k dial over every head and neuron. Each slot is a concept drawn as what it answers like,
  set by its kept share of the pair's Yes/No profile difference (1 itself, 0 the fused middle, −1 the other), with a
  fused and a swapped reply (`fuse-*.json`; §31).
- `graphics/04-dial.png`: the AI persona from k = 0 to 4, the robot growing into a cat and then breaking, with each
  stop's reply and loss (`dial-ai-cat.json`; §24).
- `graphics/05-label.png`: Yes/No margins on fourteen properties for a cat, an AI and "you", before and after: the cat's
  profile turns into the AI's while "you" keeps its own (`inside-ai-cat.json`; §22).
- `graphics/06-pairs.png`: the nearest partners the geometry picks, animals to languages and colors to gender words,
  and the map fitted to them ("What color is a strawberry?" → Man) (`correspondence-*.json`,
  `model-pairs-colors-genders.json`; §13-§14).
- `graphics/07-home.png`: the five homes tried for the orange cat and what it became under each; only the fox held
  (`home-*.json`; §34-§36).
- `graphics/08-heads.png`: the 64 query heads chosen by effect, and both binding questions against heads mirrored:
  "which one is a cat?" goes to the AI's name (`order-heads-ai-cat-q.json`, `sparse-heads-ai-cat-q.json`; §29).
- `graphics/09-refusal.png`: for each of the 24 test requests, the "harmful?" margin and the judged refusal, unedited,
  with the refusal neurons' writes mirrored, and with the harm words mirrored; the compliance is drawn as blank lines
  (`refusal-sides.json`; §18).
