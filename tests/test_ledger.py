"""measure.ledger on a tiny random Qwen3: the components rebuild the residual, the unedited split sums to 1, and
an edited split sums to the kept share of the difference read directly."""
import torch

from neuron_reflection.axes import attention_axes, unit
from neuron_reflection.edit import Editor
from neuron_reflection.measure import component_states, ledger, residuals
from test_edit import _Tok, tiny


class CharTok:
    """One token per character, with offsets."""
    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"input_ids": [ord(c) % 64 for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


FRAMES = ["ab {X} cd.", "{X} efg", "hi jk {X} l"]
SPEC = {"source": {"en": ["mom"]}, "target": {"en": ["dad"]}}
LAYERS = [0, 1]


def test_ledger():
    model = tiny()
    tok = CharTok()
    ed = Editor(model, _Tok())
    with torch.no_grad():
        comp = component_states(model, tok, FRAMES, ["mom", "dad"])
        base = residuals(model, comp, LAYERS)
        # the rebuilt residual is the layer's output (the last one before the final norm)
        enc = tok("mom efg")
        hs = model(torch.tensor([enc["input_ids"]]), output_hidden_states=True).hidden_states
        one = component_states(model, tok, ["{X} efg"], ["mom"])
        assert torch.allclose(residuals(model, one, [0])["mom"][1, 0], hs[1][0, -1], atol=1e-5)
        r = ledger(model, comp, base, SPEC, layers=LAYERS)
        assert torch.allclose(torch.tensor(r["total"]), torch.ones(2, 2), atol=1e-4)
        d = model.config.hidden_size
        axes = {L: (unit(torch.randn(d)), unit(torch.randn(d))) for L in range(2)}
        ed.edit(axes, share=0.5, k=2.0)
        ed.edit_heads(attention_axes(axes, axes), parts="ov")
        comp1 = component_states(model, tok, FRAMES, ["mom", "dad"])
        r1 = ledger(model, comp1, base, SPEC, layers=LAYERS, neurons=ed.selected)
        D0, D1 = base["mom"] - base["dad"], (lambda x: x["mom"] - x["dad"])(residuals(model, comp1, LAYERS))
        kept = (D1 * D0).sum(-1) / (D0 * D0).sum(-1)
        assert torch.allclose(torch.tensor(r1["total"]), kept, atol=1e-4)
        parts = torch.tensor(r1["embed"]) + torch.tensor(r1["attention"]) + torch.tensor(r1["mlp_edited"]) + torch.tensor(r1["mlp_rest"])
        assert torch.allclose(parts, kept, atol=1e-4)
        assert torch.tensor(r1["mlp_edited"]).abs().sum() > 0
