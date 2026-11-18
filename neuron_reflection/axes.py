"""Contrast axes: at each layer, a unit direction at the MLP input (for a neuron's gate and up rows) and one at
the layer output (for its down column) along which a source concept differs from a target.

An axis set is ``{layer: (mid, out)}``, residual-space unit vectors on the CPU in fp32.

- **Lens axes** come from single-token pairs through the neuron-lens lenses of record: the S-lens footprint
  ``K_L (e_v − ē)`` (what a token puts at the layer: presence) through ``s_until``, the R-lens pullback
  ``J_Lᵀ (γ ⊙ u_v)`` (which direction there makes the model say it: prediction) after.
- **Word axes** come from activations: paired sentences that differ in the concept word, the residual
  difference at the word's own position (the last token where the pair differs, counted from their shared
  ending, and the token after it, where an inflected word completes).
- **Template axes** come from activations at the assistant's chat template (``<|im_end|>`` of the user
  turn to the end of the generation prompt), where the reply's plan and its speaker are computed: paired
  user turns, the plain question against the same question with an instruction (``You are a cat.``).

Each axis is the normalized sum of the pairs' unit differences, so every pair counts equally.
"""
import torch
from neuron_lens import arch
from neuron_lens.catalog import kept
from neuron_lens.weights import tensors


S_UNTIL = 8  # lens axes use the S-lens through this layer and the R-lens after


def unit(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True)


def cos(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x @ y / x.norm() / y.norm()).item()


class LensAxes:
    """Token-pair contrast axes through the R- and S-lenses (local fits when present, else the released ones)."""

    def __init__(self, model_id: str, tok):
        nm = arch.names(model_id)
        head = tensors(model_id, sorted(set(nm.values())))
        self.E = head[nm["embed"]].float() * arch.embed_scale(model_id)
        self.ebar = self.E[kept(model_id)].mean(0)
        self.WU = head[nm["unembed"]].float() * head[nm["norm"]].float()
        r_fit, s_fit = arch.fits(model_id)
        self.J = {site: arch.lens(model_id, f"R_{site}", r_fit) for site in ("mid", "out")}
        self.K = {site: arch.lens(model_id, f"S_{site}", s_fit) for site in ("mid", "out")}
        self.n_layers = arch.text_config(model_id).num_hidden_layers
        self.tok = tok

    def token_ids(self, words: list[str]) -> list[int]:
        out = []
        for w in words:
            x = self.tok.encode(w, add_special_tokens=False)
            assert len(x) == 1, f"lens vectors need single-token words: {w!r} {x}"
            out.append(x[0])
        return out

    def ids(self, pairs: list[tuple[str, str]]) -> list[tuple[int, int]]:
        out = []
        for a, b in pairs:
            x, y = (self.tok.encode(w, add_special_tokens=False) for w in (a, b))
            assert len(x) == len(y) == 1, f"lens axes need single-token words: {a!r} {x}, {b!r} {y}"
            out.append((x[0], y[0]))
        return out

    def vectors(self, lens: str, site: str, layer: int, ids: list[int]) -> torch.Tensor:
        """``[d, n]``: footprints (S) or pullbacks (R) of tokens ``ids`` at ``layer``'s ``site``."""
        if lens == "S":
            return self.K[site][layer].float() @ (self.E[ids] - self.ebar).T
        J = self.J[site]
        if site == "out" and layer >= len(J):  # the last layer's output is h_final itself
            return self.WU[ids].T
        return J[layer].float().T @ self.WU[ids].T

    def axis(self, lens: str, site: str, layer: int, ids: list[tuple[int, int]]) -> torch.Tensor:
        a = unit(self.vectors(lens, site, layer, [p[0] for p in ids]))
        b = unit(self.vectors(lens, site, layer, [p[1] for p in ids]))
        return unit((a - b).sum(1))

    def axes(self, pairs: list[tuple[str, str]]) -> dict:
        """The S-lens (presence) through layer ``S_UNTIL`` and the R-lens (prediction) after, the split neuron-lens
        finds each lens reads best in."""
        ids = self.ids(pairs)
        return {L: tuple(self.axis("S" if L <= S_UNTIL else "R", site, L, ids) for site in ("mid", "out"))
                for L in range(self.n_layers)}


def word_positions(x: list[int], y: list[int]) -> tuple[list[int], list[int]]:
    """In each of two token lists, the last token where they differ (counted from their shared ending) and the
    token after it."""
    s = 0
    while s < min(len(x), len(y)) - 1 and x[-1 - s] == y[-1 - s]:
        s += 1
    assert s > 0, "paired sentences need a shared ending after the concept word"
    return [len(x) - s - 1, len(x) - s], [len(y) - s - 1, len(y) - s]


@torch.no_grad()
def states(model, ids: list[int], pos: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """``[L, len(pos), d]`` MLP inputs and layer outputs at token positions ``pos``."""
    layers = model.model.layers
    mid = []
    hooks = [layer.post_attention_layernorm.register_forward_pre_hook(
        lambda mod, inp: mid.append(inp[0][0, pos].float().cpu())) for layer in layers]
    try:
        hs = model(torch.tensor([ids], device=model.device), output_hidden_states=True).hidden_states
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(mid), torch.stack([h[0, pos].float().cpu() for h in hs[1:]])


def word_diffs(model, tok, sentences: dict[str, list[tuple[str, str]]]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per group (language): ``[n, L, d]`` unit source − target differences at the word, at the MLP input and output."""
    out = {}
    for group, pairs in sentences.items():
        dm, do = [], []
        for a, b in pairs:
            x, y = tok.encode(a), tok.encode(b)
            px, py = word_positions(x, y)
            (ma, oa), (mb, ob) = states(model, x, px), states(model, y, py)
            dm.append(unit((ma - mb).mean(1), dim=-1))
            do.append(unit((oa - ob).mean(1), dim=-1))
        out[group] = (torch.stack(dm), torch.stack(do))
    return out


def template_positions(tok, ids: list[int]) -> list[int]:
    """The generation prompt's template tokens: from the user turn's ``<|im_end|>`` to the end."""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    return list(range(max(i for i, t in enumerate(ids) if t == end), len(ids)))


def template_diffs(model, tok, pairs: dict[str, list[tuple[str, str]]]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per group: ``[n, L, d]`` unit source − target differences over the template tokens of two chat prompts."""
    from neuron_reflection.measure import chat_prompt
    out = {}
    for group, ps in pairs.items():
        dm, do = [], []
        for a, b in ps:
            x, y = (tok.encode(chat_prompt(tok, q), add_special_tokens=False) for q in (a, b))
            (ma, oa), (mb, ob) = states(model, x, template_positions(tok, x)), states(model, y, template_positions(tok, y))
            dm.append(unit((ma - mb).mean(1), dim=-1))
            do.append(unit((oa - ob).mean(1), dim=-1))
        out[group] = (torch.stack(dm), torch.stack(do))
    return out


def attention_axes(reads: dict, writes: dict) -> dict:
    """Axes for the attention heads of each layer, ``{layer: (read, write)}``, from neuron axis sets
    ``{layer: (mid, out)}``: a head reads the residual the previous layer wrote (its ``out`` axis; layer 0's
    ``mid`` for layer 0) and writes the residual its layer's MLP reads (the ``mid`` axis)."""
    return {L: (reads[L - 1][1] if L else reads[0][0], writes[L][0]) for L in reads}


def word_axes(diffs: dict, groups: list[str]) -> dict:
    n_layers = next(iter(diffs.values()))[0].shape[1]
    return {L: tuple(unit(torch.cat([diffs[g][k][:, L] for g in groups]).sum(0)) for k in (0, 1)) for L in range(n_layers)}


def agreement(diffs: dict, layers: list[int], ref: str = "en") -> list[dict]:
    """Per layer, at the MLP input: each group's axis against ``ref``'s, the mean over the other groups' pairs,
    and ``ref``'s split-half agreement (the ceiling noise allows). Without a ``ref`` group, the first group is the reference."""
    ref = ref if ref in diffs else next(iter(diffs))
    rows = []
    for L in layers:
        ax = {g: unit(d[0][:, L].sum(0)) for g, d in diffs.items()}
        others = [g for g in diffs if g != ref]
        pairs = [cos(ax[a], ax[b]) for i, a in enumerate(others) for b in others[i + 1:]]
        r = diffs[ref][0][:, L]
        h = len(r) // 2
        rows.append({"layer": L, "vs_ref": {g: round(cos(ax[ref], ax[g]), 3) for g in others},
                     "others_mean": round(sum(pairs) / max(len(pairs), 1), 3),
                     "ref_split_half": round(cos(r[:h].sum(0), r[h:].sum(0)), 3)})
    return rows
