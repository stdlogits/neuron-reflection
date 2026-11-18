"""Set-to-set maps: concept vectors for words at their own position, the category axis between two sets of
concepts, and maps between sets: orthogonal (pairs), linear (oblique) and the model's own pairing.

- **Concept vectors** (``concept_vectors``): for each frame sentence and language, the residual at a word's
  last token and the one after (or only the one after, ``at="after"``), minus the frame's mean over every word of
  both sets, so the frame's own content drops out; then unit per layer and averaged over frames and languages.
- **Category axis** (``category``): the mean source vector minus the mean target vector, per layer. A
  reflection along it swaps the sets' shared parts and leaves each word's own part.
- **Pair map** (``pair_map``): on the span of all 2n vectors, the orthogonal map closest (Procrustes) to
  sending each source to its paired target and each target to its source, identity elsewhere.
- **Linear map** (``linear_map``): on the span of the listed vectors, the linear map, not orthogonal, closest to
  sending each source to its target. With fixed points listed as their own targets and one source sent to a
  home it is an oblique projection (findings §34). An orthogonal map keeps every distance, so only a map that is
  not orthogonal can pull two nearby concepts apart.
- **The model's own pairing** (``model_similarity``, ``sinkhorn``, ``coupled_map``, ``assignment``): the
  off-axis similarity of every source with every target, averaged over layers, turned into a balanced soft
  coupling (entropic transport; unequal sets share partners) and fitted by a weighted Procrustes, or into
  the best one-to-one assignment and fitted as pairs.
"""
import torch

from neuron_reflection.axes import states, unit


@torch.no_grad()
def concept_vectors(model, tok, frames: dict[str, list[str]], words: dict[str, list[str]], at: str = "word"):
    """``(mid, out)``, each ``[n, L, d]``: concept vectors for ``words[lang]`` (the same concepts in the same order
    in every language) at each layer's MLP input and output. ``at="word"`` reads a word's last token and the one
    after; ``at="after"`` only the token after, where a word of several tokens has been read whole (findings §35:
    at its last token "Garfield" is read as "field")."""
    assert at in ("word", "after"), at
    total = None
    for lang, fs in frames.items():
        if lang not in words:
            continue
        for f in fs:
            mid, out = [], []
            for w in words[lang]:
                text = f.replace("{X}", w)
                enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
                end = f.index("{X}") + len(w)
                last = max(i for i, (a, _) in enumerate(enc["offset_mapping"]) if a < end)
                after = min(last + 1, len(enc["input_ids"]) - 1)
                pos = [last, after] if at == "word" else [after]
                m, o = states(model, enc["input_ids"], pos)
                mid.append(m.mean(1))
                out.append(o.mean(1))
            vs = [torch.stack(v) for v in (mid, out)]  # [n, L, d]
            vs = [unit(v - v.mean(0, keepdim=True), -1) for v in vs]
            total = vs if total is None else [t + v for t, v in zip(total, vs)]
    return tuple(unit(t, -1) for t in total)


def category(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Unit (mean source − mean target) over the last dimension, from ``[n, …, d]`` concept vectors."""
    return unit(src.mean(0) - dst.mean(0), -1)


def pair_map(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """``(B, R, fit)`` for ``[n, d]`` paired vectors: ``B`` an orthonormal basis of their span, ``R`` the
    orthogonal map within it closest to swapping each pair, ``fit`` the mean cosine of each mapped vector with
    its partner."""
    assert len(src) == len(dst), "a pair map needs as many sources as targets"
    X = torch.cat([src, dst]).T.float()
    Y = torch.cat([dst, src]).T.float()
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    B = U[:, S > 1e-6 * S[0]]
    Xc, Yc = B.T @ X, B.T @ Y
    u, _, vt = torch.linalg.svd(Yc @ Xc.T)
    R = u @ vt
    fit = torch.nn.functional.cosine_similarity(R @ Xc, Yc, dim=0).mean().item()
    return B, R, fit


def linear_map(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """``(B, R, fit, gain)`` for ``[n, d]`` source and target vectors: ``B`` an orthonormal basis of their span, and
    the least-squares linear map ``M`` within it sending each source to its target (exact where the sources are
    numerically full rank under the 10⁻³ cutoffs), identity off the sources' span. ``R`` is ``Mᵀ`` in the basis, so that ``Editor`` reads with it make a unit
    see ``M x`` (the Editor's reads see ``Qᵀ x``). ``fit``: the mean cosine of each mapped source with its target.
    ``gain``: ``M``'s largest singular value within the span, how much it can stretch a state."""
    assert len(src) == len(dst), "a linear map needs as many sources as targets"
    X, Y = src.T.float(), dst.T.float()
    U, S, _ = torch.linalg.svd(torch.cat([X, Y], 1), full_matrices=False)
    B = U[:, S > 1e-3 * S[0]]  # an exact map would fit any direction left, noise included
    Xc, Yc = B.T @ X, B.T @ Y
    pinv = torch.linalg.pinv(Xc, rtol=1e-3)
    M = Yc @ pinv  # maps the sources' span; complete it with the identity on the rest of B
    P = Xc @ pinv  # projector onto the sources' span within B
    M = M + (torch.eye(len(B.T)) - P)
    fit = torch.nn.functional.cosine_similarity(M @ Xc, Yc, dim=0).mean().item()
    return B, M.T.contiguous(), fit, torch.linalg.matrix_norm(M, ord=2).item()


def off_axis_similarity(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """``[n, m]``: cosines of the parts of ``[n, d]`` and ``[m, d]`` concept vectors off their category axis."""
    d = unit(src.mean(0) - dst.mean(0))
    off = lambda v: unit(v - (v @ d)[:, None] * d[None, :], -1)
    return off(src) @ off(dst).T


def model_similarity(src: torch.Tensor, dst: torch.Tensor, layers: list[int]) -> torch.Tensor:
    """The off-axis similarity of ``[n, L, d]`` against ``[m, L, d]`` concept vectors, averaged over ``layers``."""
    return torch.stack([off_axis_similarity(src[:, L], dst[:, L]) for L in layers]).mean(0)


def sinkhorn(sim: torch.Tensor, tau: float = 0.03, iters: int = 1000) -> torch.Tensor:
    """A balanced soft coupling ``[n, m]`` (rows sum to 1/n, columns to 1/m, total 1) from similarities at
    temperature ``tau``: entropic optimal transport, so unequal sets share partners."""
    n, m = sim.shape
    K = torch.exp((sim - sim.max()) / tau).double()
    a, b = torch.full((n,), 1 / n, dtype=K.dtype), torch.full((m,), 1 / m, dtype=K.dtype)
    v = torch.ones(m, dtype=K.dtype)
    for _ in range(iters):
        u = a / (K @ v)
        v = b / (K.T @ u)
    return (u[:, None] * K * v[None, :]).float()


def assignment(sim: torch.Tensor) -> tuple[list[int], list[int]]:
    """The one-to-one pairing of rows and columns with the largest total similarity (min(n, m) pairs)."""
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-sim.numpy())
    return rows.tolist(), cols.tolist()


def coupled_map(src: torch.Tensor, dst: torch.Tensor, P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """``(B, R, fit)``: on the span of ``[n, d]`` and ``[m, d]`` vectors, the orthogonal map maximizing
    ``Σ P_ij (y_j · R x_i + x_i · R y_j)``, each source sent toward its coupled targets and back; ``fit`` the
    coupling-weighted mean cosine of mapped vectors with their partners."""
    X = torch.cat([src, dst]).T.float()
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    B = U[:, S > 1e-6 * S[0]]
    Xc, Yc = B.T @ src.T.float(), B.T @ dst.T.float()
    M = Yc @ P.T @ Xc.T + Xc @ P @ Yc.T
    u, _, vt = torch.linalg.svd(M)
    R = u @ vt
    cs = torch.nn.functional.cosine_similarity
    fwd = cs((R @ Xc)[:, :, None], Yc[:, None, :], dim=0)  # [n, m]
    back = cs((R @ Yc)[:, None, :], Xc[:, :, None], dim=0)
    fit = ((P * fwd).sum() + (P * back).sum()).item() / (2 * P.sum().item())
    return B, R, fit


def maps_by_layer(src: tuple[torch.Tensor, torch.Tensor], dst: tuple[torch.Tensor, torch.Tensor], kind: str,
                  coupling: torch.Tensor | None = None, pairs: tuple[list[int], list[int]] | None = None) -> tuple[dict, list]:
    """Per layer ``(in, out)`` from ``(mid, out)`` concept vectors: category axes (``kind="category"``), pair maps
    of index-aligned pairs (``"pairs"``), pair maps of the given ``pairs`` (``"hard"``), coupled maps (``"soft"``,
    with ``coupling``), or linear maps of index-aligned pairs (``"linear"``, reads only); and the maps' fits per
    layer (in, out)."""
    n_layers = src[0].shape[1]
    out, fits = {}, []
    for L in range(n_layers):
        if kind == "category":
            out[L] = tuple(category(s[:, L], d[:, L]) for s, d in zip(src, dst))
            continue
        if kind == "soft":
            made = [coupled_map(s[:, L], d[:, L], coupling) for s, d in zip(src, dst)]
        elif kind == "linear":
            lin = [linear_map(s[:, L], d[:, L]) for s, d in zip(src, dst)]
            out[L] = tuple((b, r) for b, r, _, _ in lin)
            fits.append(tuple(round(x, 3) for x in (lin[0][2], lin[1][2], lin[0][3], lin[1][3])))  # fits in, out; gains in, out
            continue
        elif kind == "hard":
            r, c = pairs
            made = [pair_map(s[r, L], d[c, L]) for s, d in zip(src, dst)]
        else:
            made = [pair_map(s[:, L], d[:, L]) for s, d in zip(src, dst)]
        (bi, ri, fi), (bo, ro, fo) = made
        out[L] = ((bi, ri), (bo, ro))
        fits.append((round(fi, 3), round(fo, 3)))
    return out, fits


def correspondence(src: torch.Tensor, dst: torch.Tensor, k: int = 3) -> tuple[list, list]:
    """Where each concept of one set lands in the other under the category reflection: for ``[n, d]`` and
    ``[m, d]`` concept vectors at one layer, the cosine of each source's component off the category axis with
    each target's (the part the reflection leaves alone), top ``k`` per row; and the same from targets."""
    sim = off_axis_similarity(src, dst)
    top = lambda m: [[(j, round(v, 3)) for v, j in zip(*[t.tolist() for t in row.topk(min(k, len(row)))])] for row in m]
    return top(sim), top(sim.T)
