"""``measure.categories`` on synthetic word states: split distances and shares, cross-language cosines, and the 青
coordinate."""
import torch

from neuron_reflection.measure import categories


def test_categories():
    L, d = 3, 8
    e = torch.eye(d)
    roles = ["dark blue", "light blue", "dark green", "light green", "red", "pink"]
    vec = {"dark blue": e[0] + e[1], "light blue": e[0] - e[1], "dark green": e[2] + 0.5 * e[1], "light green": e[2] - 0.5 * e[1],
           "red": e[3] + e[4], "pink": e[3] - e[4]}
    spec = {"words": {"en": {r: f"en-{r}" for r in roles}, "zh": {**{r: f"zh-{r}" for r in roles}, "qing": "zh-qing"}}}
    states = {}
    for lang in ("en", "zh"):
        for r in roles:
            v = vec[r] if lang == "en" or r != "light blue" else e[0] - e[1] + 2 * e[5]  # zh's blue split leans off English's
            states[f"{lang}:{spec['words'][lang][r]}"] = v.expand(2, L, d).clone()
    states["zh:zh-qing"] = (0.25 * (e[0]) + 0.75 * e[2]).expand(2, L, d).clone()
    out = categories(states, spec)
    en, zh = out["en"], out["zh"]
    assert abs(en["blue split"][0][0] - 2.0) < 1e-4 and abs(en["green split"][0][0] - 1.0) < 1e-4
    assert abs(en["blue split share"][0][0] - 2.0 / (2 ** 0.5)) < 1e-4  # blue mean e0, green mean e2: distance √2
    assert abs(en["cos blue split ~ red/pink"][0][0]) < 1e-6
    assert abs(zh["cos blue split ~ en blue split"][1][2] - 2 / (8 ** 0.5)) < 1e-4  # (2 e1 − 2 e5)·(2 e1) / (√8 · 2)
    assert abs(zh["cos green split ~ en green split"][0][0] - 1.0) < 1e-4
    blue_mean = e[0] + e[5]  # zh's blues: e0 + e1 and e0 − e1 + 2 e5
    t = ((0.25 * e[0] + 0.75 * e[2] - e[2]) @ (blue_mean - e[2])) / (blue_mean - e[2]).pow(2).sum()
    assert abs(zh["qing coordinate"][0][0] - t.item()) < 1e-4
