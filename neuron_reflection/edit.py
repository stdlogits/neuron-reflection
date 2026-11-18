"""Reflect (k = 2), project (k = 1) or amplify (k < 0) MLP neurons and attention heads along per-layer axes, or apply
a linear map of a subspace (orthogonal on both sides, or any linear map on the reads), and restore them exactly.

A SwiGLU neuron reads the normed residual ``γ ⊙ x / rms(x)`` through its gate and up rows and writes its down
column. With ``H = I − k d dᵀ`` the edit replaces the γ-folded rows ``r ⊙ γ`` by ``H (r ⊙ γ)`` (stored back
divided by γ) and the column ``w`` by ``H_out w``. For k = 2, H is a reflection and, since
``rms(H x) = rms(x)``, the edited neuron computes exactly ``H_out N(H_in x)`` (``tests/test_edit.py``).
A per-layer map ``(B, R)`` (orthonormal basis ``B`` ``[d, m]``, orthogonal ``R`` ``[m, m]``) generalizes this to
``Q = I + B (R − I) Bᵀ``: rows become ``Q (r ⊙ γ)``, columns ``Q_out w``, and the neuron computes
``Q_out N(Q_inᵀ x)``. A reflection is ``B = d``, ``R = [1 − k]``.

``edit_heads`` conjugates attention heads the same way: the γ-folded (input norm) Q, K and V rows are reflected
along the read axis and the O columns along the write axis, any subset of the four (``parts``). QK-norm and
RoPE act inside the head, after the projection, so with every part reflected (k = 2) the head computes
``H_out Attn(H_in x)`` exactly. K and V are shared by a group of query heads (GQA): reflecting them for one
head reflects them for its group. With one ``H`` for every head and neuron the edited model is the original
behind a reflected interface: embeddings ``E H``, readout ``U Γ H Γ⁻¹`` (``tests/test_edit.py``).
Embeddings, norms and the unembedding are never touched. Qwen3's dense layers only: Gemma's post-feedforward
norm rescales the summed output, which a per-neuron edit would need to account for.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_device() -> str:
    """CUDA when present, else Apple silicon (MPS), else the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Editor:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.layers = model.model.layers
        assert all(hasattr(layer, "post_attention_layernorm") and hasattr(layer.mlp, "gate_proj") for layer in self.layers), \
            "Editor covers Qwen3-style dense SwiGLU layers"
        self.orig = [tuple(p.detach().cpu().clone() for p in self._params(i)) for i in range(len(self.layers))]
        self.orig_attn = None  # saved at the first head edit
        self.selected: dict[int, torch.Tensor] = {}
        self.selected_heads: dict[int, torch.Tensor] = {}

    @classmethod
    def load(cls, model_id: str, device: str | None = None, dtype=torch.bfloat16) -> "Editor":
        device = device or default_device()
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        return cls(model, tok)

    def _params(self, layer: int):
        m = self.layers[layer].mlp
        return m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight

    def _gamma(self, layer: int) -> torch.Tensor:
        return self.layers[layer].post_attention_layernorm.weight.float()

    @staticmethod
    def as_map(v: torch.Tensor, k: float) -> tuple[torch.Tensor, torch.Tensor]:
        return v if isinstance(v, tuple) else (v[:, None], torch.tensor([[1.0 - k]]))

    @torch.no_grad()
    def shares(self, layer: int, maps) -> torch.Tensor:
        """``[3, n]``: the share of each neuron's γ-folded gate row and up row in the read subspace, and of its
        down column in the write subspace (squared cosines for a single axis). ``maps`` is ``(in, out)``, each
        an axis vector or a ``(B, R)`` map."""
        dev = self.model.device
        bi, bo = (self.as_map(m, 2.0)[0].to(dev) for m in maps)
        g = self._gamma(layer)
        gate, up, down = (p.float() for p in self._params(layer))
        gate, up = gate * g, up * g
        return torch.stack([(gate @ bi).pow(2).sum(1) / gate.pow(2).sum(1), (up @ bi).pow(2).sum(1) / up.pow(2).sum(1),
                            (bo.T @ down).pow(2).sum(0) / down.pow(2).sum(0)]).cpu()

    def loadings(self, layer: int, axis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Per neuron, the largest of its three shares: the ranking for every edit."""
        return self.shares(layer, axis).max(0).values

    @torch.no_grad()
    def edit(self, axes: dict, share: float = 0.01, k: float = 2.0, units: dict[int, list[int]] | None = None,
             select: str = "top", sides: str = "both", dose: float = 1.0, seed: int = 0) -> dict:
        """Edit, per layer, the ``share`` of neurons that load most on the layer's axis (``select="top"``; all of
        them for share 1), only the listed ``units`` (``"units"``), or both (``"union"``). ``sides`` edits the
        gate and up rows and the down column (``"both"``), the rows only (``"reads"``) or the column only
        (``"writes"``); ``"steer"`` turns each whole column toward the write axis, signed so the pole that pushed
        the source now pushes the target, keeping its norm: ``dose`` 1 replaces it, smaller doses mix the unit
        column with the unit target before renormalizing (``k`` is ignored). The neurons are ranked the
        same way whichever side is edited; ``select="random"`` takes as many per layer at random (``seed``), the
        control. Edits compose: call again for a second axis; ``restore`` undoes them all."""
        assert sides in ("both", "reads", "writes", "steer"), sides
        dev = self.model.device
        total, carried, unit_load, layer_mean, read_sh, write_sh = 0.0, 0.0, [], [], [], []
        for L, axis in axes.items():
            sh = self.shares(L, axis)
            load = sh.max(0).values
            n = len(load)
            top = load.topk(max(1, int(share * n))).indices if share < 1 else torch.arange(n)
            listed = torch.tensor(sorted((units or {}).get(L, [])), dtype=torch.long)
            if len(listed):
                unit_load.append(load[listed])
                layer_mean.append(torch.full((len(listed),), load.mean().item()))
            if select == "random":
                g_ = torch.Generator().manual_seed(seed * 1000 + L)
                idx = torch.randperm(n, generator=g_)[:len(top)]
            else:
                idx = {"top": top, "units": listed, "union": torch.unique(torch.cat([top, listed]))}[select]
            if len(idx) == 0:
                continue
            total, carried = total + load.sum().item(), carried + load[idx].sum().item()
            read_sh.append(sh[:2, idx].max(0).values)
            write_sh.append(sh[2, idx])
            self.selected[L] = idx
            idx = idx.to(dev)
            (bi, ri), (bo, ro) = ((b.to(dev).float(), r.to(dev).float()) for b, r in (self.as_map(m, k) for m in axis))
            g = self._gamma(L)
            gate, up, down = self._params(L)
            if sides in ("both", "reads"):
                for p in (gate, up):
                    W = p[idx].float()
                    W += (((W * g) @ bi) @ (ri.T - torch.eye(len(ri), device=dev)) @ bi.T) / g
                    p[idx] = W.to(p.dtype)
            if sides in ("both", "writes"):
                W = down[:, idx].float()
                W += bo @ ((ro - torch.eye(len(ro), device=dev)) @ (bo.T @ W))
                down[:, idx] = W.to(down.dtype)
            elif sides == "steer":
                assert not isinstance(axis[1], tuple), "steering takes a single write axis"
                do = axis[1].to(dev)
                W = down[:, idx].float()
                norm = W.norm(dim=0, keepdim=True)
                target = -torch.sign(do @ W)[None, :] * do[:, None]
                mixed = (1 - dose) * W / norm + dose * target
                down[:, idx] = (mixed / mixed.norm(dim=0, keepdim=True) * norm).to(down.dtype)
        info = {"neurons": sum(len(v) for v in self.selected.values()), "loading_share": carried / max(total, 1e-12),
                "median_read_share": torch.cat(read_sh).median().item(), "median_write_share": torch.cat(write_sh).median().item()}
        if unit_load:  # how the listed units load on the axis against their layers' mean neuron
            info["units_loading_vs_mean"] = (torch.cat(unit_load) / torch.cat(layer_mean)).median().item()
        return info

    def _attn(self, layer: int):
        a = self.layers[layer].self_attn
        return a.q_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight

    def _head_dims(self) -> tuple[int, int, int]:
        c = self.model.config
        return c.num_attention_heads, c.num_key_value_heads, getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads

    @torch.no_grad()
    def head_shares(self, layer: int, maps) -> torch.Tensor:
        """``[4, n_heads]``: each query head's share in the read subspace of its γ-folded Q rows, of its KV
        group's K rows and V rows, and in the write subspace of its O columns."""
        dev = self.model.device
        bi, bo = (self.as_map(m, 2.0)[0].to(dev) for m in maps)
        nh, nkv, hd = self._head_dims()
        g = self.layers[layer].input_layernorm.weight.float()
        q, k, v, o = (p.float() for p in self._attn(layer))

        def rows(W, n):
            W = (W * g).view(n, hd, -1)
            return (W @ bi).pow(2).sum((1, 2)) / W.pow(2).sum((1, 2))
        sk, sv = (rows(W, nkv).repeat_interleave(nh // nkv) for W in (k, v))
        so = (bo.T @ o).view(-1, nh, hd).pow(2).sum((0, 2)) / o.view(-1, nh, hd).pow(2).sum((0, 2))
        return torch.stack([rows(q, nh), sk, sv, so]).cpu()

    @torch.no_grad()
    def edit_heads(self, axes: dict, parts: str = "qkvo", share: float = 1.0, k: float = 2.0,
                   units: dict[int, list[int]] | None = None, select: str = "top", n: int = 0) -> dict:
        """Reflect attention heads along ``{layer: (read axis, write axis)}`` (``axes.attention_axes``). Which heads:
        per layer, the ``share`` of query heads that load most on the axes in the edited ``parts`` (``select="top"``;
        all for share 1), the listed ``units`` (``"units"``, query heads per layer), or the ``n`` that load most
        across all layers (``"global"``). ``parts`` names what is edited: ``q`` their Q rows, ``k`` and ``v`` their
        KV groups' K and V rows, ``o`` their O columns; ``ov`` both sides of what a head moves, ``qk`` both sides
        of where it looks, ``q`` alone what it looks for. ``restore`` undoes it."""
        assert parts and set(parts) <= set("qkvo"), parts
        assert select in ("top", "units", "global"), select
        if self.orig_attn is None:
            self.orig_attn = [tuple(p.detach().cpu().clone() for p in self._attn(i)) for i in range(len(self.layers))]
        dev = self.model.device
        nh, nkv, hd = self._head_dims()
        per = nh // nkv
        which = ["qkvo".index(c) for c in parts]
        shares = {L: self.head_shares(L, axis) for L, axis in axes.items()}
        if select == "global":  # the n heads that load most anywhere
            load = torch.stack([shares[L][which].max(0).values for L in axes])  # [layers, heads]
            top = load.flatten().topk(n).indices.tolist()
            layer_ids = list(axes)
            units = {}
            for i in top:
                units.setdefault(layer_ids[i // nh], []).append(i % nh)
            select = "units"
        heads_n, groups_n, read_sh, write_sh = 0, 0, [], []
        for L, axis in axes.items():
            sh = shares[L]
            if select == "units":
                if not (units or {}).get(L):
                    continue
                heads = torch.tensor(sorted(set(units[L])), dtype=torch.long)
            else:
                heads = sh[which].max(0).values.topk(max(1, round(share * nh))).indices.sort().values if share < 1 else torch.arange(nh)
            groups = torch.unique(heads // per)
            self.selected_heads[L] = heads
            heads_n, groups_n = heads_n + len(heads), groups_n + len(groups)
            read_sh.append(sh[:3, heads].max(0).values)
            write_sh.append(sh[3, heads])
            (bi, ri), (bo, ro) = ((b.to(dev).float(), r.to(dev).float()) for b, r in (self.as_map(m, k) for m in axis))
            g = self.layers[L].input_layernorm.weight.float()
            q, kp, v, o = self._attn(L)
            rows = lambda ids: torch.cat([torch.arange(u * hd, (u + 1) * hd) for u in ids.tolist()]).to(dev)
            for c, p, ids in (("q", q, heads), ("k", kp, groups), ("v", v, groups)):
                if c in parts:
                    idx = rows(ids)
                    W = p[idx].float()
                    W += (((W * g) @ bi) @ (ri.T - torch.eye(len(ri), device=dev)) @ bi.T) / g
                    p[idx] = W.to(p.dtype)
            if "o" in parts:
                idx = rows(heads)
                W = o[:, idx].float()
                W += bo @ ((ro - torch.eye(len(ro), device=dev)) @ (bo.T @ W))
                o[:, idx] = W.to(o.dtype)
        return {"heads": heads_n, "kv_groups": groups_n, "parts": parts,
                "median_read_share": torch.cat(read_sh).median().item(), "median_write_share": torch.cat(write_sh).median().item()}

    @torch.no_grad()
    def restore(self):
        for i, saved in enumerate(self.orig):
            for p, o in zip(self._params(i), saved):
                p.copy_(o.to(p.device))
        if self.orig_attn is not None:
            for i, saved in enumerate(self.orig_attn):
                for p, o in zip(self._attn(i), saved):
                    p.copy_(o.to(p.device))
        self.selected, self.selected_heads = {}, {}
