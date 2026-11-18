"""Each neuron's first-order effect (``attribution.reflection_effects``) against a finite difference: reflect
that neuron alone with a tiny k, read sides and write side apart, on a tiny random Qwen3."""
import torch

from neuron_reflection.attribution import reflection_effects
from neuron_reflection.axes import unit
from neuron_reflection.edit import Editor
from test_edit import _Tok, tiny


def margin(model, seqs):
    total = 0.0
    with torch.no_grad():
        for prompt, right, wrong in seqs:
            for ans, sign in ((right, 1.0), (wrong, -1.0)):
                ids = torch.tensor([prompt + ans])
                lp = model(ids, use_cache=False).logits[0, len(prompt) - 1:-1].double().log_softmax(-1)
                total += sign * lp.gather(1, ids[0, len(prompt):, None]).sum().item()
    return total


def test_first_order_matches_finite_difference():
    model = tiny().double()  # fp64, so a tiny k's change in the margin is far above rounding
    d = model.config.hidden_size
    axes = {L: (unit(torch.randn(d)), unit(torch.randn(d))) for L in range(2)}
    seqs = [([3, 9, 14, 2, 27], [5], [7]), ([11, 4, 30, 8], [19, 2], [6, 1])]
    eff = reflection_effects(model, seqs, axes, k=1.0)
    ed = Editor(model, _Tok())
    base, eps = margin(model, seqs), 0.1  # small enough to be linear, large against the editor's fp32 rounding
    for L, n in ((0, 3), (1, 17), (1, 40)):
        for side, sides in ((0, "reads"), (1, "writes")):
            ed.edit(axes, k=eps, units={L: [n]}, select="units", sides=sides)
            fd = (margin(model, seqs) - base) / eps
            ed.restore()
            assert abs(fd - eff[L, side, n].item()) <= 0.05 * abs(fd) + 1e-6, (L, n, sides, fd, eff[L, side, n].item())


def test_head_effects_match_finite_difference():
    """Each head's first-order effect, per part, against reflecting that head's part alone with a small ±k (a
    central difference: QK-norm and the softmax curve enough that a one-sided step is off by several percent)."""
    from neuron_reflection.attribution import head_effects
    from neuron_reflection.axes import attention_axes
    model = tiny().double()
    d = model.config.hidden_size
    raw = {L: (unit(torch.randn(d)), unit(torch.randn(d))) for L in range(2)}
    axes = attention_axes(raw, raw)
    seqs = [([3, 9, 14, 2, 27], [5], [7]), ([11, 4, 30, 8], [19, 2], [6, 1])]
    eff = head_effects(model, seqs, axes, k=1.0)
    ed = Editor(model, _Tok())
    eps = 0.1
    for L, h in ((0, 1), (1, 2), (1, 3)):
        for part in "qkvo":
            ms = []
            for sign in (1, -1):
                ed.edit_heads(axes, parts=part, k=sign * eps, units={L: [h]}, select="units")
                ms.append(margin(model, seqs))
                ed.restore()
            fd = (ms[0] - ms[1]) / (2 * eps)
            want = eff[part][L, h // 2 if part in "kv" else h].item()
            assert abs(fd - want) <= 0.05 * abs(fd) + 1e-5, (L, h, part, fd, want)


def test_mixed_order():
    """The joint order takes distinct units, may mix types, and leaves the model as it found it."""
    from neuron_reflection.attribution import mixed_greedy_order
    model = tiny().double()
    d = model.config.hidden_size
    axes = {L: (unit(torch.randn(d)), unit(torch.randn(d))) for L in range(2)}
    seqs = [([3, 9, 14, 2, 27], [5], [7]), ([11, 4, 30, 8], [19, 2], [6, 1])]
    ed = Editor(model, _Tok())
    before = margin(model, seqs)
    order = mixed_greedy_order(ed, axes, seqs, "nqkvo", upto=12)
    assert len(order) == len(set(order)) == 12
    assert {p for p, _, _ in order} <= set("nqkvo")
    assert abs(margin(model, seqs) - before) < 1e-9


def test_target_order():
    """The target order takes distinct units, starts its trace at the whole signal (rms share 1) for a fusion and at
    the scaled distance from the targets otherwise, takes negative k (amplification), and leaves the model as it found
    it."""
    from neuron_reflection.attribution import signals, target_greedy_order
    model = tiny().double()
    d = model.config.hidden_size
    axes = {L: (unit(torch.randn(d)), unit(torch.randn(d))) for L in range(2)}
    seqs = [([3, 9, 14, 2, 27], [5], [7]), ([11, 4, 30, 8], [19, 2], [6, 1]), ([2, 2, 40], [9], [12])]
    groups = [[(seqs[0], 1.0), (seqs[1], -1.0)], [(seqs[2], 0.5)]]
    ed = Editor(model, _Tok())
    before = margin(model, seqs)
    for k in (1.0, 2.0):
        order, trace = target_greedy_order(ed, axes, groups, "nqkvo", upto=10, k=k)
        assert len(order) == len(set(order)) == 10
        assert trace[0] == (0, 1.0) and trace[-1][0] == 10
        assert abs(margin(model, seqs) - before) < 1e-9
    s0 = signals(model, groups)
    targets, scales = torch.stack([-s0[0], s0[1]]), torch.tensor([10.0, 10.0], dtype=torch.float64)
    order, trace = target_greedy_order(ed, axes, groups, "nqkvo", upto=6, k=-2.0, targets=targets, scales=scales)
    want = (((s0 - targets) / scales) ** 2).mean().sqrt().item()
    assert len(order) == 6 and abs(trace[0][1] - round(want, 4)) < 1e-4
    assert abs(margin(model, seqs) - before) < 1e-9
