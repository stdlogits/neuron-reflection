"""The edit's identity on a tiny random Qwen3: with every neuron reflected (k = 2), the MLP block computes
H_out · MLP(norm(H_in x)), the γ-folding included; with k = 1 the block no longer sees the axis. With every head
and neuron conjugated by one H, the model is the original behind a reflected interface; with every head and neuron
projected along one axis, a position and its mirror image are the same to every component."""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from neuron_reflection.axes import attention_axes, unit
from neuron_reflection.edit import Editor


class _Tok:
    pass


def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False,
                      attn_implementation="eager")
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for m in model.modules():  # random norm scales (input, post-attention, QK and final), so the γ-folding is exercised
            if type(m).__name__.endswith("RMSNorm"):
                m.weight.uniform_(0.2, 2.0)
    return model


def block(model, layer, x):
    lay = model.model.layers[layer]
    return lay.mlp(lay.post_attention_layernorm(x))


def test_reflection_conjugates_the_block():
    model = tiny()
    ed = Editor(model, _Tok())
    d = model.config.hidden_size
    axes = {L: (unit(torch.randn(d)), unit(torch.randn(d))) for L in range(2)}
    x = torch.randn(5, d)
    with torch.no_grad():
        before = [block(model, L, x @ (torch.eye(d) - 2 * torch.outer(axes[L][0], axes[L][0]))) for L in range(2)]
        ed.edit(axes, share=1.0, k=2.0)
        for L in range(2):
            h_out = torch.eye(d) - 2 * torch.outer(axes[L][1], axes[L][1])
            assert torch.allclose(block(model, L, x), before[L] @ h_out, atol=1e-5)
        ed.restore()
        assert torch.allclose(block(model, 0, x @ (torch.eye(d) - 2 * torch.outer(axes[0][0], axes[0][0]))), before[0], atol=1e-6)


def test_projection_removes_the_axis():
    model = tiny()
    ed = Editor(model, _Tok())
    d = model.config.hidden_size
    axes = {L: (unit(torch.randn(d)), unit(torch.randn(d))) for L in range(2)}
    x = torch.randn(5, d)
    with torch.no_grad():
        ed.edit(axes, share=1.0, k=1.0)
        for L in range(2):
            out = block(model, L, x)
            assert torch.allclose(out @ axes[L][1], torch.zeros(5), atol=1e-5)  # nothing written along the out axis
            lay = model.model.layers[L]
            g = lay.post_attention_layernorm.weight
            # the γ-folded rows are orthogonal to the in axis, so the neurons cannot read it
            assert torch.allclose(lay.mlp.gate_proj.weight * g @ axes[L][0], torch.zeros(lay.mlp.gate_proj.weight.shape[0]), atol=1e-5)


def test_one_side():
    """Reads only: the block hears the mirrored input and writes as before; writes only: the mirror of its output."""
    model = tiny()
    ed = Editor(model, _Tok())
    d = model.config.hidden_size
    axes = {L: (unit(torch.randn(d)), unit(torch.randn(d))) for L in range(2)}
    x = torch.randn(5, d)
    h = {L: [torch.eye(d) - 2 * torch.outer(a, a) for a in axes[L]] for L in range(2)}
    with torch.no_grad():
        plain = [block(model, L, x) for L in range(2)]
        mirrored_in = [block(model, L, x @ h[L][0]) for L in range(2)]
        ed.edit(axes, share=1.0, k=2.0, sides="reads")
        for L in range(2):
            assert torch.allclose(block(model, L, x), mirrored_in[L], atol=1e-5)
        ed.restore()
        ed.edit(axes, share=1.0, k=2.0, sides="writes")
        for L in range(2):
            assert torch.allclose(block(model, L, x), plain[L] @ h[L][1], atol=1e-5)


def test_steer():
    """Steering replaces each column by the write axis at its norm, the source-pushing pole turned to the target."""
    model = tiny()
    ed = Editor(model, _Tok())
    d = model.config.hidden_size
    axes = {L: (unit(torch.randn(d)), unit(torch.randn(d))) for L in range(2)}
    with torch.no_grad():
        before = [model.model.layers[L].mlp.down_proj.weight.clone() for L in range(2)]
        ed.edit(axes, share=1.0, sides="steer")
        for L in range(2):
            W, do = model.model.layers[L].mlp.down_proj.weight, axes[L][1]
            assert torch.allclose(W.norm(dim=0), before[L].norm(dim=0), atol=1e-5)
            assert torch.allclose((do @ W).abs(), W.norm(dim=0), atol=1e-5)  # all on the axis
            assert ((do @ W) * (do @ before[L]) <= 0).all()  # the sign flipped


def test_subspace_map():
    """A general orthogonal map on a subspace: the block computes Q_out · MLP(norm(Q_inᵀ x))."""
    model = tiny()
    ed = Editor(model, _Tok())
    d, m = model.config.hidden_size, 4
    maps, qs = {}, {}
    for L in range(2):
        pair = []
        for _ in range(2):
            B = torch.linalg.qr(torch.randn(d, m))[0]
            R = torch.linalg.qr(torch.randn(m, m))[0]
            pair.append((B, R))
        maps[L] = tuple(pair)
        qs[L] = [torch.eye(d) + B @ (R - torch.eye(m)) @ B.T for B, R in pair]
    x = torch.randn(5, d)
    with torch.no_grad():
        before = [block(model, L, x @ qs[L][0]) for L in range(2)]  # rows of x @ Q_in are (Q_inᵀ x)ᵀ
        ed.edit(maps, share=1.0)
        for L in range(2):
            assert torch.allclose(block(model, L, x), before[L] @ qs[L][1].T, atol=1e-4)


def test_whole_model_conjugation():
    """Every head and every neuron conjugated by one reflection H is the original model reading embeddings E H
    and reading out through U Γ H Γ⁻¹ (Γ the final norm's scale): an exact relabeling at the interface."""
    model = tiny().double()
    ed = Editor(model, _Tok())
    d, n = model.config.hidden_size, model.config.num_hidden_layers
    a = unit(torch.randn(d))  # the Editor computes in fp32; the model runs in fp64
    H = torch.eye(d, dtype=torch.float64) - 2 * torch.outer(a.double(), a.double())
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        E, U, g = model.model.embed_tokens.weight.clone(), model.lm_head.weight.clone(), model.model.norm.weight.clone()
        model.model.embed_tokens.weight.copy_(E @ H)
        model.lm_head.weight.copy_(U @ torch.diag(g) @ H @ torch.diag(1 / g))
        relabeled = model(ids).logits
        model.model.embed_tokens.weight.copy_(E)
        model.lm_head.weight.copy_(U)
        ed.edit({L: (a, a) for L in range(n)}, share=1.0, k=2.0)
        ed.edit_heads(attention_axes(*[{L: (a, a) for L in range(n)}] * 2), parts="qkvo", k=2.0)
        assert torch.allclose(model(ids).logits, relabeled, atol=1e-5)
        ed.restore()
        model.model.embed_tokens.weight.copy_(E @ H)
        assert not torch.allclose(model(ids).logits, relabeled, atol=1e-3)  # restored: the original again


def test_head_parts():
    """QK conjugation leaves a head's pattern on H x what it was on x; each part edits only its own weights; a
    selected head's K and V edits reach its whole KV group."""
    model = tiny()
    ed = Editor(model, _Tok())
    d, n = model.config.hidden_size, model.config.num_hidden_layers
    a = unit(torch.randn(d))
    H = torch.eye(d) - 2 * torch.outer(a, a)
    x = torch.randn(1, 7, d)
    axes = {L: (a, a) for L in range(n)}
    with torch.no_grad():
        pattern = model(inputs_embeds=x @ H, output_attentions=True).attentions[0]
        ed.edit_heads(axes, parts="qk")
        assert torch.allclose(model(inputs_embeds=x, output_attentions=True).attentions[0], pattern, atol=1e-5)
        ed.restore()
        before = [p.clone() for p in ed._attn(0)]
        for parts in ("q", "k", "v", "o", "ov"):
            ed.edit_heads(axes, parts=parts)
            after = ed._attn(0)
            assert [not torch.equal(b, p) for b, p in zip(before, after)] == [c in parts for c in "qkvo"], parts
            ed.restore()
        ed.edit_heads(axes, parts="kv", share=0.25)  # one query head of four: the K and V rows of its group of two
        hd, nkv = 8, 2
        group = ed.selected_heads[0].item() // 2
        for p, b in zip(ed._attn(0)[1:3], before[1:3]):
            changed = [not torch.equal(p.view(nkv, hd, d)[i], b.view(nkv, hd, d)[i]) for i in range(nkv)]
            assert changed == [i == group for i in range(nkv)]


def test_heads_are_the_neurons_mirror():
    """Reflecting every head (all four parts) with one H is the model with every neuron reflected instead,
    reading embeddings E H through the readout U Γ H Γ⁻¹: the two halves are mirror images of each other."""
    model = tiny().double()
    ed = Editor(model, _Tok())
    d, n = model.config.hidden_size, model.config.num_hidden_layers
    a = unit(torch.randn(d))
    H = torch.eye(d, dtype=torch.float64) - 2 * torch.outer(a.double(), a.double())
    ids = torch.randint(0, 64, (1, 10))
    axes = {L: (a, a) for L in range(n)}
    with torch.no_grad():
        ed.edit_heads(attention_axes(axes, axes), parts="qkvo")
        heads = model(ids).logits
        ed.restore()
        E, U, g = model.model.embed_tokens.weight.clone(), model.lm_head.weight.clone(), model.model.norm.weight.clone()
        model.model.embed_tokens.weight.copy_(E @ H)
        model.lm_head.weight.copy_(U @ torch.diag(g) @ H @ torch.diag(1 / g))
        ed.edit(axes, share=1.0, k=2.0)
        assert torch.allclose(model(ids).logits, heads, atol=1e-5)


def test_listed_heads_in_several_layers():
    """Listed heads in more than one layer are each edited, and only they."""
    model = tiny()
    ed = Editor(model, _Tok())
    d, n = model.config.hidden_size, model.config.num_hidden_layers
    a = unit(torch.randn(d))
    axes = attention_axes(*[{L: (a, a) for L in range(n)}] * 2)
    with torch.no_grad():
        before = [p.clone() for L in range(n) for p in ed._attn(L)]
        ed.edit_heads(axes, parts="qkvo", units={0: [1], 1: [0, 3]}, select="units")
        after = [p for L in range(n) for p in ed._attn(L)]
        hd = 8
        for L, heads in ((0, [1]), (1, [0, 3])):
            q0, q1 = before[4 * L], after[4 * L]
            changed = [not torch.equal(q0[h * hd:(h + 1) * hd], q1[h * hd:(h + 1) * hd]) for h in range(4)]
            assert changed == [h in heads for h in range(4)], (L, changed)


def test_projection_blinds_the_mirror():
    """Every head and neuron projected along one axis (k = 1): a position and its mirror image ``H x`` read the same to
    every component, so mirroring one input position leaves every other position's states and logits as they were
    and that position's states mirrored. Only the readout at that position still sees the difference. Unedited,
    the mirror changes what follows."""
    model = tiny().double()
    ed = Editor(model, _Tok())
    d, n = model.config.hidden_size, model.config.num_hidden_layers
    a = unit(torch.randn(d))
    H = torch.eye(d, dtype=torch.float64) - 2 * torch.outer(a.double(), a.double())
    x = torch.randn(1, 7, d, dtype=torch.float64)
    m = x.clone()
    m[0, 2] = x[0, 2] @ H
    axes = {L: (a, a) for L in range(n)}
    rest = [0, 1, 3, 4, 5, 6]
    with torch.no_grad():
        ed.edit(axes, share=1.0, k=1.0)
        ed.edit_heads(attention_axes(axes, axes), parts="qkvo", k=1.0)
        o, om = (model(inputs_embeds=v, output_hidden_states=True) for v in (x, m))
        for h, hm in zip(o.hidden_states[:-1], om.hidden_states[:-1]):  # the pre-norm residuals
            assert torch.allclose(hm[0, rest], h[0, rest], atol=1e-5)
            assert torch.allclose(hm[0, 2], h[0, 2] @ H, atol=1e-5)
        assert torch.allclose(om.logits[0, rest], o.logits[0, rest], atol=1e-5)
        assert not torch.allclose(om.logits[0, 2], o.logits[0, 2], atol=1e-3)
        ed.restore()
        assert not torch.allclose(model(inputs_embeds=m).logits[0, 3:], model(inputs_embeds=x).logits[0, 3:], atol=1e-3)


def test_amplification_scales_the_axis():
    """k < 0 amplifies: the γ-folded read rows' component on the axis is multiplied by 1 − k, the rest kept."""
    model = tiny()
    ed = Editor(model, _Tok())
    d = model.config.hidden_size
    axes = {L: (unit(torch.randn(d)), unit(torch.randn(d))) for L in range(2)}
    with torch.no_grad():
        lay = model.model.layers[0]
        g = lay.post_attention_layernorm.weight
        before = (lay.mlp.gate_proj.weight * g).clone()
        ed.edit(axes, share=1.0, k=-2.0, sides="reads")
        after = lay.mlp.gate_proj.weight * g
        a = axes[0][0]
        assert torch.allclose(after @ a, 3 * (before @ a), atol=1e-4)
        rest = lambda W: W - torch.outer(W @ a, a)
        assert torch.allclose(rest(after), rest(before), atol=1e-4)


def test_linear_map_reads():
    """A linear map M fitted to sources and targets (``subspace.linear_map``) sends each source to its target, and
    reads edited with it make every neuron read ``M x / rms(x)``: a fixed point stays, a moved concept is read as its
    home. The norm is the unmapped input's (for an orthogonal map the two agree)."""
    from neuron_reflection.subspace import linear_map
    model = tiny()
    ed = Editor(model, _Tok())
    d = model.config.hidden_size
    vec = torch.randn(5, d)  # 0 → 4 (the home); 1, 2, 3 and the home fixed
    src, dst = vec, torch.stack([vec[4], vec[1], vec[2], vec[3], vec[4]])
    B, R, fit, gain = linear_map(src, dst)
    assert fit > 0.9999
    M = torch.eye(d) + B @ (R.T - torch.eye(len(R))) @ B.T
    assert torch.allclose(src @ M.T, dst, atol=1e-4)
    x = torch.randn(5, d)
    with torch.no_grad():
        def mapped(L):  # the MLP on γ ⊙ M x / rms(x); rows of x @ Mᵀ are (M x)ᵀ
            lay = model.model.layers[L]
            norm = lay.post_attention_layernorm
            xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
            return lay.mlp((xn @ M.T) * norm.weight)
        before = [mapped(L) for L in range(2)]
        ed.edit({L: ((B, R), (B, R)) for L in range(2)}, share=1.0, sides="reads")
        for L in range(2):
            assert torch.allclose(block(model, L, x), before[L], atol=1e-4)


def test_linear_map_needs_reference_centering():
    """Vectors centered on their own mean sum to zero, so an exact linear map that inverts them stretches by far more
    than the same vectors centered with reference words (``run.py``'s set ``reference``)."""
    from neuron_reflection.subspace import linear_map
    torch.manual_seed(1)
    raw = torch.randn(12, 64) + 3 * torch.randn(64)  # 8 used, 4 references; a shared component, as concepts have
    unit_rows = lambda v: v / v.norm(dim=-1, keepdim=True)
    own = unit_rows(raw[:8] - raw[:8].mean(0)) + 1e-3 * torch.randn(8, 64)  # near-dependent, as through a model
    ref = unit_rows(raw[:8] - raw.mean(0))
    gain = lambda v: linear_map(v, torch.cat([v[7:8], v[1:8]]))[3]
    assert gain(own) > 5 * gain(ref), (gain(own), gain(ref))
