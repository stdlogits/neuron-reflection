"""Each neuron's own share of an edit: the first-order change in a score if that neuron alone were reflected.

Reflecting neuron ``n`` along ``H = I − k d dᵀ`` changes its gate and up pre-activations at every position by
``−k (d_in · (r ⊙ γ)) (d_in · x̂)`` (``x̂ = x / rms(x)``, the residual the rows read before γ) and its write by
``−k d_out (d_out · w)``. To first order the score changes by those changes times its gradients:

    read_n  = −k [ c_g,n Σ_t ∂s/∂g_n(t) (d_in · x̂_t) + c_u,n Σ_t ∂s/∂u_n(t) (d_in · x̂_t) ]
    write_n = −k c_d,n Σ_t a_n(t) (∂s/∂y_t · d_out)

with ``c`` each vector's component on its axis, ``a`` the neuron's activation and ``y`` the MLP output. One
forward and backward pass per sequence gives every neuron of every layer. The score is a margin, log p(right)
− log p(wrong), summed over forced-choice items. First order ignores how the selected neurons interact; the
edit itself is measured, not predicted.
"""
import torch

from neuron_reflection.measure import chat_prompt


def chat_items(tok, items: list[tuple[str, str, str]]) -> list[tuple[list[int], list[int], list[int]]]:
    """(question, right, wrong) as token ids: the chat prompt and each answer as the start of the reply."""
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    return [(enc(chat_prompt(tok, q)), enc(r), enc(w)) for q, r, w in items]


def reflection_effects(model, seqs, axes: dict, k: float = 2.0) -> torch.Tensor:
    """``[L, 2, n]``: per neuron, the first-order change in the summed margin over ``seqs`` (from
    ``chat_items``) if it alone were reflected along its layer's ``(d_in, d_out)``, through its read rows (0)
    and its write column (1). Negative moves the items toward the wrong answer."""
    layers = model.model.layers
    dev = model.device
    model.requires_grad_(False)  # gradients reach the activations only (the embeddings' output is the leaf)
    coef = []
    with torch.no_grad():
        for layer, (di, do) in ((layers[L], axes[L]) for L in range(len(layers))):
            g = layer.post_attention_layernorm.weight.float()
            di, do = di.to(dev).float(), do.to(dev).float()
            m = layer.mlp
            coef.append((di / g, do, (m.gate_proj.weight.float() * g) @ di, (m.up_proj.weight.float() * g) @ di,
                         do @ m.down_proj.weight.float()))
    saved: list[dict] = [{} for _ in layers]

    def keep(L, name, inp_name=None):
        def hook(mod, inp, out):
            if inp_name:
                saved[L][inp_name] = inp[0].detach()
            out.retain_grad()
            saved[L][name] = out
        return hook

    hooks = [model.model.embed_tokens.register_forward_hook(lambda mod, inp, out: out.requires_grad_())]
    for L, layer in enumerate(layers):
        hooks += [layer.mlp.gate_proj.register_forward_hook(keep(L, "g", "x")),
                  layer.mlp.up_proj.register_forward_hook(keep(L, "u")),
                  layer.mlp.down_proj.register_forward_hook(keep(L, "y", "a"))]
    eff = torch.zeros(len(layers), 2, coef[0][2].shape[0], device=dev)
    try:
        with torch.enable_grad():
            for prompt, right, wrong in seqs:
                for ans, sign in ((right, 1.0), (wrong, -1.0)):
                    ids = torch.tensor([prompt + ans], device=dev)
                    lp = model(ids, use_cache=False).logits[0, len(prompt) - 1:-1].float().log_softmax(-1)
                    (sign * lp.gather(1, ids[0, len(prompt):, None]).sum()).backward()
                    with torch.no_grad():
                        for L, (di_g, do, cg, cu, cd) in enumerate(coef):
                            s = saved[L]
                            dx = s["x"][0].float() @ di_g  # [T]: d_in · x̂
                            eff[L, 0] -= k * (cg * (dx @ s["g"].grad[0].float()) + cu * (dx @ s["u"].grad[0].float()))
                            eff[L, 1] -= k * cd * ((s["y"].grad[0].float() @ do) @ s["a"][0].float())
                    for s in saved:
                        s.clear()
    finally:
        for h in hooks:
            h.remove()
    return eff.cpu()


def as_units(flat_idx: torch.Tensor, n: int) -> dict[int, list[int]]:
    """Flat indices (layer * n + neuron) as ``{layer: [neurons]}``."""
    units: dict[int, list[int]] = {}
    for i in flat_idx.tolist():
        units.setdefault(i // n, []).append(i % n)
    return units


def greedy_order(ed, axes, seqs, upto: int, growth: float = 1.25) -> torch.Tensor:
    """Flat neuron indices in the order the greedy selection takes them: rank every neuron by
    ``reflection_effects`` on ``seqs``, edit the most negative step (a quarter of those taken so far), re-rank on
    the edited model, until ``upto``. Never takes a neuron twice; restores the editor at the end."""
    chosen: list[int] = []
    n = ed.layers[0].mlp.down_proj.weight.shape[1]
    while len(chosen) < upto:
        score = reflection_effects(ed.model, seqs, axes).sum(1).flatten()
        score[chosen] = float("inf")
        new = score.argsort()[:min(max(1, int(len(chosen) * (growth - 1))), upto - len(chosen))]
        ed.edit(axes, units=as_units(new, n), select="units")
        chosen += new.tolist()
    ed.restore()
    return torch.tensor(chosen)


def head_effects(model, seqs, axes: dict, k: float = 2.0) -> dict[str, torch.Tensor]:
    """Each head's own share of a head edit: the first-order change in the summed margin over ``seqs`` if that
    head alone had one part reflected along its layer's ``(read, write)`` (``axes.attention_axes``). Per part:
    ``q`` and ``o`` ``[L, query heads]``, ``k`` and ``v`` ``[L, KV groups]``. With ``x̂`` the residual a head
    reads before its norm's γ and ``c`` a head's rows' (or columns') component on the axis:

        q_h = −k Σ_t (∂s/∂q_h(t) · c_q,h) (d_in · x̂_t)        (k and v the same, per group)
        o_h = −k Σ_t (∂s/∂y_t · d_out) (c_o,h · z_h(t))

    with ``y`` the attention output and ``z_h`` head h's input to ``o_proj``. One forward and backward pass per
    sequence."""
    layers = model.model.layers
    dev = model.device
    c = model.config
    nh, nkv = c.num_attention_heads, c.num_key_value_heads
    hd = getattr(c, "head_dim", None) or c.hidden_size // nh
    model.requires_grad_(False)
    coef = []
    with torch.no_grad():
        for L, layer in enumerate(layers):
            a = layer.self_attn
            g = layer.input_layernorm.weight.float()
            di, do = (v.to(dev).float() for v in axes[L])
            rows = lambda W, n: ((W.float() * g) @ di).view(n, hd)  # each head's rows on the read axis
            coef.append((di / g, do, rows(a.q_proj.weight, nh), rows(a.k_proj.weight, nkv), rows(a.v_proj.weight, nkv),
                         (do @ a.o_proj.weight.float()).view(nh, hd)))
    saved: list[dict] = [{} for _ in layers]

    def keep(L, name, inp_name=None):
        def hook(mod, inp, out):
            if inp_name:
                saved[L][inp_name] = inp[0].detach()
            out.retain_grad()
            saved[L][name] = out
        return hook

    hooks = [model.model.embed_tokens.register_forward_hook(lambda mod, inp, out: out.requires_grad_())]
    for L, layer in enumerate(layers):
        a = layer.self_attn
        hooks += [a.q_proj.register_forward_hook(keep(L, "q", "x")), a.k_proj.register_forward_hook(keep(L, "k")),
                  a.v_proj.register_forward_hook(keep(L, "v")), a.o_proj.register_forward_hook(keep(L, "y", "z"))]
    eff = {"q": torch.zeros(len(layers), nh, device=dev), "k": torch.zeros(len(layers), nkv, device=dev),
           "v": torch.zeros(len(layers), nkv, device=dev), "o": torch.zeros(len(layers), nh, device=dev)}
    try:
        with torch.enable_grad():
            for prompt, right, wrong in seqs:
                for ans, sign in ((right, 1.0), (wrong, -1.0)):
                    ids = torch.tensor([prompt + ans], device=dev)
                    lp = model(ids, use_cache=False).logits[0, len(prompt) - 1:-1].float().log_softmax(-1)
                    (sign * lp.gather(1, ids[0, len(prompt):, None]).sum()).backward()
                    with torch.no_grad():
                        for L, (di_g, do, cq, ck, cv, co) in enumerate(coef):
                            s = saved[L]
                            dx = s["x"][0].float() @ di_g  # [T]: d_in · x̂
                            for part, cc, n in (("q", cq, nh), ("k", ck, nkv), ("v", cv, nkv)):
                                G = s[part].grad[0].float().view(-1, n, hd)  # [T, n, hd]
                                eff[part][L] -= k * (dx @ (G * cc).sum(-1))
                            gy = s["y"].grad[0].float() @ do  # [T]
                            eff["o"][L] -= k * (gy @ (s["z"][0].float().view(-1, nh, hd) * co).sum(-1))
                    for s in saved:
                        s.clear()
    finally:
        for h in hooks:
            h.remove()
    return {p: e.cpu() for p, e in eff.items()}


def greedy_head_order(ed, axes: dict, seqs, part: str, upto: int, growth: float = 1.25) -> list[tuple[int, int]]:
    """``(layer, query head)`` in the order the greedy selection takes them for a one-part head edit (``part`` one
    of ``qkvo``): rank by ``head_effects`` on ``seqs``, edit the most negative step (a quarter of those taken so
    far), re-rank on the edited model, until ``upto``. For ``k`` and ``v`` a unit is a KV group, recorded by its
    first query head; no unit is taken twice. Restores the editor at the end."""
    assert part in ("q", "k", "v", "o"), part
    nh, nkv, _ = ed._head_dims()
    per = nh // nkv
    chosen: list[int] = []
    while len(chosen) < upto:
        score = head_effects(ed.model, seqs, axes)[part].flatten()
        score[chosen] = float("inf")
        new = score.argsort()[:min(max(1, int(len(chosen) * (growth - 1))), upto - len(chosen))].tolist()
        n = score.shape[0] // len(axes)
        units: dict[int, list[int]] = {}
        for i in new:
            units.setdefault(i // n, []).append((i % n) * (per if n == nkv else 1))
        ed.edit_heads(axes, parts=part, units=units, select="units")
        chosen += new
    ed.restore()
    n = nh if part in "qo" else nkv
    return [(i // n, (i % n) * (per if n == nkv else 1)) for i in chosen]


POOL = ("n", "q", "k", "v", "o")  # neurons (both sides), then the four head parts


def mixed_greedy_order(ed, axes: dict, seqs, pools: str, upto: int, growth: float = 1.25) -> list[tuple[str, int, int]]:
    """Neurons and head parts in one greedy order (findings §30): every unit in ``pools`` (``n`` a neuron with both
    sides reflected, ``q``/``k``/``v``/``o`` one part of a head or KV group) is ranked by the first-order effect of
    its own reflection on ``seqs`` (``reflection_effects``, ``head_effects``); each step edits the most negative step
    (a quarter of those taken so far) across all types, then re-ranks on the edited model, until ``upto``. ``axes``
    are neuron axis sets ``{layer: (mid, out)}`` of one kind; heads use ``attention_axes`` of them. Returns
    ``(type, layer, index)``, the index a neuron, a query head, or a KV group's first query head. Restores the editor."""
    from neuron_reflection.axes import attention_axes
    assert pools and set(pools) <= set(POOL), pools
    hax = attention_axes(axes, axes)
    nh, nkv, _ = ed._head_dims()
    per = nh // nkv
    order: list[tuple[str, int, int]] = []
    taken: set[tuple[str, int, int]] = set()
    while len(order) < upto:
        eff = {}
        if "n" in pools:
            eff["n"] = reflection_effects(ed.model, seqs, axes).sum(1)  # [L, n]
        if set(pools) & set("qkvo"):
            he = head_effects(ed.model, seqs, hax)
            eff |= {p: he[p] for p in pools if p in "qkvo"}
        index, scores = [], []
        for p, e in eff.items():
            L_, n = e.shape
            scores.append(e.flatten())
            index += [(p, L, i * (per if p in "kv" else 1)) for L in range(L_) for i in range(n)]
        score = torch.cat(scores)
        for j, u in enumerate(index):
            if u in taken:
                score[j] = float("inf")
        new = [index[j] for j in score.argsort()[:min(max(1, int(len(order) * (growth - 1))), upto - len(order))].tolist()]
        apply_units(ed, axes, hax, new)
        order += new
        taken |= set(new)
    ed.restore()
    return order


@torch.no_grad()
def seq_margins(model, seqs) -> list[float]:
    """Per ``chat_items`` sequence, log p(right) − log p(wrong) as the start of the reply."""
    out = []
    for prompt, right, wrong in seqs:
        m = 0.0
        for ans, sign in ((right, 1.0), (wrong, -1.0)):
            ids = torch.tensor([prompt + ans], device=model.device)
            lp = model(ids, use_cache=False).logits[0, len(prompt) - 1:-1].float().log_softmax(-1)
            m += sign * lp.gather(1, ids[0, len(prompt):, None]).sum().item()
        out.append(m)
    return out


def signals(model, groups) -> torch.Tensor:
    """``[G]``: each group's signal, the weighted sum of its items' margins (``groups``: lists of ``(seq, weight)``)."""
    return torch.tensor([sum(w * m for (_, w), m in zip(g, seq_margins(model, [s for s, _ in g]))) for g in groups],
                        dtype=torch.float64)


def unit_effects(model, seqs, axes: dict, hax: dict, pools: str, k: float) -> tuple[list[torch.Tensor], list[tuple[str, int, int]]]:
    """Per sequence, the first-order change of its margin if each unit of ``pools`` alone were edited with ``k``
    (``[U]`` each, neurons with both sides), and the units as ``(type, layer, index)`` (``mixed_greedy_order``'s)."""
    nkv = model.config.num_key_value_heads
    per = model.config.num_attention_heads // nkv
    out, index = [], None
    for s in seqs:
        eff = {}
        if "n" in pools:
            eff["n"] = reflection_effects(model, [s], axes, k=k).sum(1)
        if set(pools) & set("qkvo"):
            he = head_effects(model, [s], hax, k=k)
            eff |= {p: he[p] for p in pools if p in "qkvo"}
        out.append(torch.cat([e.flatten() for e in eff.values()]).double())
        if index is None:
            index = [(p, L, i * (per if p in "kv" else 1)) for p, e in eff.items() for L in range(e.shape[0]) for i in range(e.shape[1])]
    return out, index


def target_greedy_order(ed, axes: dict, groups, pools: str, upto: int, k: float = 2.0, targets=None, scales=None,
                        growth: float = 1.25) -> tuple[list, list]:
    """Units that drive every group's signal to its target together (findings §31, §33), from ``pools`` as in
    ``mixed_greedy_order``, each edited with ``k`` (2 reflects, 1 projects, below 0 amplifies). The objective is
    ``F = mean_g ((s_g − t_g) / σ_g)²``. A fusion takes every target ``t_g = 0`` with ``σ_g = s_g⁰``, the unedited
    signal (the default), so ``F`` is the squared share left. A split takes the targets a between-kind pair has and
    keeps guard signals where they were. Each step re-measures the signals and every unit's first-order effect on
    the edited model, then takes the step's units one at a time, each the unit that most lowers ``F`` predicted to
    first order given those already taken in the step, so they are chosen jointly rather than each alone (it always
    takes a unit, even when none is predicted to help). Returns the order and the trace ``[(units, rms of
    (s − t) / σ)]`` measured at the start of each step and at the end. Restores the editor."""
    from neuron_reflection.axes import attention_axes
    assert pools and set(pools) <= set(POOL), pools
    hax = attention_axes(axes, axes)
    seqs = [s for g in groups for s, _ in g]
    s0 = signals(ed.model, groups)
    t = torch.zeros_like(s0) if targets is None else torch.as_tensor(targets, dtype=torch.float64)
    sigma = s0 if scales is None else torch.as_tensor(scales, dtype=torch.float64)
    order: list[tuple[str, int, int]] = []
    taken: set[tuple[str, int, int]] = set()
    trace = []
    while True:
        c = (signals(ed.model, groups) - t) / sigma  # each signal's distance from its target, in its own scale
        trace.append((len(order), round(c.pow(2).mean().sqrt().item(), 4)))
        if len(order) >= upto:
            break
        per_seq, index = unit_effects(ed.model, seqs, axes, hax, pools, k)
        E, i = torch.zeros(len(groups), len(index), dtype=torch.float64), 0
        for g, grp in enumerate(groups):
            for _, w in grp:
                E[g] += w * per_seq[i]
                i += 1
        E /= sigma[:, None]
        q = E.pow(2).sum(0)  # F(c + E_u) − F(c) = 2 c·E_u + |E_u|²
        blocked = torch.tensor([u in taken for u in index])
        new = []
        for _ in range(min(max(1, int(len(order) * (growth - 1))), upto - len(order))):
            gain = 2 * (c @ E) + q
            gain[blocked] = float("inf")
            j = gain.argmin().item()
            new.append(index[j])
            blocked[j] = True
            c = c + E[:, j]
        apply_units(ed, axes, hax, new, k=k)
        order += new
        taken |= set(new)
    ed.restore()
    return order, trace


def apply_units(ed, axes: dict, hax: dict, units: list[tuple[str, int, int]], k: float = 2.0) -> dict:
    """Reflect a list of ``(type, layer, index)`` units (``mixed_greedy_order``): neurons with both sides, each head
    part on its own. Returns how many of each type."""
    by: dict[str, dict[int, list[int]]] = {}
    for p, L, i in units:
        by.setdefault(p, {}).setdefault(L, []).append(i)
    for p, u in by.items():
        if p == "n":
            ed.edit(axes, units=u, select="units", k=k)
        else:
            ed.edit_heads(hax, parts=p, units=u, select="units", k=k)
    return {p: sum(len(v) for v in u.values()) for p, u in by.items()}
