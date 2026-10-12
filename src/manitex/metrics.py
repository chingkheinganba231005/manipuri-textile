"""Evaluation metrics for generated design grids.

All distribution metrics are **palette-independent**: they operate on canonical grids or
on colour *partitions*, so they measure structure, not colour choice.

* :func:`partition_codes` / :func:`pattern_js` - Jensen-Shannon divergence between the
  distributions of k x k local patterns (encoded by which cells share a colour).
* :func:`run_length_distance` - 1-D Wasserstein distance between run-length histograms
  along rows / columns (stripe widths, motif spans).
* :func:`color_count_distance` - total variation between colour-count distributions.
* :func:`symmetry_profile` - mirror and repeat scores (structural regularity).
* :func:`diversity`, :func:`novelty` - mean pairwise distance within a set and nearest
  training neighbour distance (memorisation check), both flip-invariant.
* :func:`weavability` - compiles designs and reports tie-down density, float lengths and
  loom requirements.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from manitex.design import Design, canonicalize_grid


def partition_codes(grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Integer code of the colour partition of every k x k window (palette-independent)."""
    win = sliding_window_view(np.asarray(grid), (k, k)).reshape(-1, k * k)
    i, j = np.triu_indices(k * k, 1)
    eq = (win[:, i] == win[:, j]).astype(np.int64)
    return eq @ (np.int64(1) << np.arange(i.size, dtype=np.int64))


def pattern_histogram(grids: Sequence[np.ndarray], k: int = 3) -> Counter:
    c: Counter = Counter()
    for g in grids:
        codes, counts = np.unique(partition_codes(g, k), return_counts=True)
        c.update(dict(zip(codes.tolist(), counts.tolist())))
    return c


def js_divergence(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two count histograms."""
    keys = list(set(p) | set(q))
    a = np.array([p.get(x, 0) for x in keys], dtype=np.float64)
    b = np.array([q.get(x, 0) for x in keys], dtype=np.float64)
    if a.sum() == 0 or b.sum() == 0:
        return 1.0
    a /= a.sum()
    b /= b.sum()
    m = 0.5 * (a + b)

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())

    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def pattern_js(gen: Sequence[np.ndarray], ref: Sequence[np.ndarray], k: int = 3) -> float:
    return js_divergence(pattern_histogram(gen, k), pattern_histogram(ref, k))


def run_lengths(grid: np.ndarray, axis: int = 1) -> np.ndarray:
    g = np.asarray(grid) if axis == 1 else np.asarray(grid).T
    change = np.ones(g.shape, dtype=bool)
    change[:, 1:] = g[:, 1:] != g[:, :-1]
    rid = np.cumsum(change, axis=1)
    out = []
    for r in range(g.shape[0]):
        out.append(np.bincount(rid[r])[1:])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def run_length_distance(gen: Sequence[np.ndarray], ref: Sequence[np.ndarray], axis: int = 1, max_len: int = 64) -> float:
    """Wasserstein-1 distance between run-length distributions (runs capped at max_len)."""

    def hist(gs):
        h = np.zeros(max_len + 1)
        for g in gs:
            rl = np.minimum(run_lengths(g, axis), max_len)
            h += np.bincount(rl, minlength=max_len + 1)
        return h / max(h.sum(), 1)

    a, b = hist(gen), hist(ref)
    return float(np.abs(np.cumsum(a) - np.cumsum(b)).sum())


def color_count_distance(gen: Sequence[np.ndarray], ref: Sequence[np.ndarray], max_k: int = 16) -> float:
    def hist(gs):
        h = np.bincount([min(len(np.unique(g)), max_k) for g in gs], minlength=max_k + 1).astype(float)
        return h / max(h.sum(), 1)

    return float(0.5 * np.abs(hist(gen) - hist(ref)).sum())


def symmetry_profile(grid: np.ndarray) -> dict[str, float]:
    """Mirror scores and the strongest horizontal / vertical repeat."""
    g = np.asarray(grid)
    H, W = g.shape

    def best_period(a: np.ndarray) -> tuple[int, float]:
        n = a.shape[1]
        best = (0, 0.0)
        for s in range(2, max(3, n // 2 + 1)):
            sc = float((a[:, :-s] == a[:, s:]).mean())
            if sc > best[1] + 1e-9:
                best = (s, sc)
        return best

    px, sx = best_period(g)
    py, sy = best_period(g.T)
    return {
        "mirror_x": float((g == g[:, ::-1]).mean()),
        "mirror_y": float((g == g[::-1, :]).mean()),
        "period_x": float(px),
        "period_x_score": sx,
        "period_y": float(py),
        "period_y_score": sy,
    }


def _canon_variants(g: np.ndarray) -> list[np.ndarray]:
    return [canonicalize_grid(v)[0] for v in (g, g[:, ::-1], g[::-1, :], g[::-1, ::-1])]


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    """Flip- and palette-invariant normalised Hamming distance between equal-size grids."""
    ca = canonicalize_grid(a)[0]
    return min(float((ca != v).mean()) for v in _canon_variants(b))


def diversity(grids: Sequence[np.ndarray], max_pairs: int = 2000, seed: int = 0) -> float:
    """Mean pairwise distance within a set of equal-size grids."""
    n = len(grids)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    return float(np.mean([hamming(grids[i], grids[j]) for i, j in pairs]))


def novelty(gen: Sequence[np.ndarray], train: Sequence[np.ndarray], max_train: int = 5000, seed: int = 0) -> np.ndarray:
    """Distance from every generated grid to its nearest training grid of the same size.

    Training grids larger than the generated size contribute all their aligned crops on a
    stride of half the generated size.  Values near 0 indicate memorisation.
    """
    rng = np.random.default_rng(seed)
    out = []
    for g in gen:
        H, W = g.shape
        cands = []
        for t in train:
            if t.shape[0] < H or t.shape[1] < W:
                continue
            for y in range(0, t.shape[0] - H + 1, max(1, H // 2)):
                for x in range(0, t.shape[1] - W + 1, max(1, W // 2)):
                    cands.append(t[y : y + H, x : x + W])
        if len(cands) > max_train:
            cands = [cands[i] for i in rng.choice(len(cands), max_train, replace=False)]
        if not cands:
            out.append(np.nan)
            continue
        cg = canonicalize_grid(g)[0]
        variants = np.stack(_canon_variants(np.asarray(g)))
        best = 1.0
        stack = np.stack([canonicalize_grid(c)[0] for c in cands])  # (N, H, W)
        for v in variants:
            best = min(best, float((stack != v[None]).mean(axis=(1, 2)).min()))
        del cg
        out.append(best)
    return np.array(out)


def weavability(designs: Sequence[Design], max_float: int = 8, ends_per_cell: int = 2) -> dict[str, float]:
    """Compile every design (brocade) and summarise the weaving cost."""
    from manitex.weave import compile_design

    pat, ties, shafts, floats_ok, supp = [], [], [], [], []
    for d in designs:
        dr = compile_design(d, ends_per_cell=ends_per_cell, max_float=max_float)
        cells = d.height * d.width
        pat.append(dr.meta["pattern_cells"] / cells)
        ties.append(100.0 * dr.meta["tie_downs"] / cells)
        shafts.append(dr.n_shafts)
        floats_ok.append(dr.float_report()["supp_weft_face"] <= max_float)
        supp.append(dr.summary()["supplementary_picks"] / max(dr.n_picks, 1))
    return {
        "pattern_cell_frac": float(np.mean(pat)),
        "tie_downs_per_100_cells": float(np.mean(ties)),
        "dobby_shafts_median": float(np.median(shafts)),
        "fits_24_shafts": float(np.mean(np.array(shafts) <= 24)),
        "face_floats_within_limit": float(np.mean(floats_ok)),
        "supplementary_pick_frac": float(np.mean(supp)),
    }


def evaluate(gen: Sequence[np.ndarray], ref: Sequence[np.ndarray], train: Sequence[np.ndarray] | None = None) -> dict[str, float]:
    """The standard metric suite used in the paper."""
    res = {
        "pattern_js_2x2": pattern_js(gen, ref, 2),
        "pattern_js_3x3": pattern_js(gen, ref, 3),
        "runlen_w1_rows": run_length_distance(gen, ref, 1),
        "runlen_w1_cols": run_length_distance(gen, ref, 0),
        "colour_count_tv": color_count_distance(gen, ref),
        "diversity_gen": diversity(gen),
        "diversity_ref": diversity(ref),
    }
    sp_g = [symmetry_profile(g) for g in gen]
    sp_r = [symmetry_profile(g) for g in ref]
    for key in ("mirror_x", "period_x_score"):
        res[f"{key}_gen"] = float(np.mean([s[key] for s in sp_g]))
        res[f"{key}_ref"] = float(np.mean([s[key] for s in sp_r]))
    if train is not None:
        nov = novelty(gen, train)
        res["novelty_mean"] = float(np.nanmean(nov))
        res["memorised_frac(<0.02)"] = float(np.nanmean(nov < 0.02))
    return res
