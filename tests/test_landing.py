"""measure.landing on synthetic states: an intact, a swapped, a fused and a rotated pair read as such."""
import torch

from neuron_reflection.measure import landing

L, D = 36, 8
SPEC = {"frames": {"en": ["{X}"]}, "source": {"en": ["s"]}, "target": {"en": ["t"]}, "anchors": {"en": ["a"]}}


def states(s, t, a):
    return {"en:s": s, "en:t": t, "en:a": a}


def test_landing_cases():
    g = torch.Generator().manual_seed(0)
    s, t, a = (torch.randn(2, L, D, generator=g) for _ in range(3))
    base = states(s, t, a)
    delta = s - t
    ortho = torch.randn(2, L, D, generator=g)
    ortho -= ((ortho * delta).sum(-1, keepdim=True) / (delta * delta).sum(-1, keepdim=True)) * delta
    ortho *= delta.norm(dim=-1, keepdim=True) / ortho.norm(dim=-1, keepdim=True)  # |ortho| = |Δ|, ⊥ Δ
    mid = (s + t) / 2
    cases = {"intact": (states(s, t, a), 1, 0, 1, 0), "swapped": (states(t, s, a), -1, 0, 0, 1),
             "fused": (states(mid, mid, a), 0, 0, 0.5, 0.5), "rotated": (states(mid + ortho / 2, mid - ortho / 2, a), 0, 1, 0.5, 0.5)}
    for name, (edited, kept, new, cs, ct) in cases.items():
        r = landing(base, edited, SPEC)["en"]
        assert torch.allclose(torch.tensor(r["kept"][0]), torch.full((2, L), float(kept)), atol=2e-3), name
        assert torch.allclose(torch.tensor(r["new"][0]), torch.full((2, L), float(new)), atol=2e-3), name
        assert torch.allclose(torch.tensor(r["coord"]["s"]), torch.full((2, L), float(cs)), atol=2e-3), name
        assert torch.allclose(torch.tensor(r["coord"]["t"]), torch.full((2, L), float(ct)), atol=2e-3), name
    # the swapped source's nearest unedited word is the target
    r = landing(base, cases["swapped"][0], SPEC)["en"]
    assert all(layer[0][0] == "t" for layer in r["near"]["s"][0])
