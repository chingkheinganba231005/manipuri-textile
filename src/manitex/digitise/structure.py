"""Structure-aware regularisation of digitised label grids by *orbit pooling*.

Textiles are full of exact structure: stripes are uniform rows, border motifs repeat with
a fixed period and are often mirror symmetric.  When such a structure is detected, all
cells that must share a colour (an *orbit*) pool their colour costs and take the jointly
best colour.  This removes noise and photographic artefacts while respecting the evidence
of every cell.  The same principle constrains sampling in :mod:`manitex.models.sampler`.
"""

from __future__ import annotations

import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, a: int) -> int:
        p = self.parent
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:
            p[a], a = root, p[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def roots(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(self.parent.size)])


def pool_orbits(cost: np.ndarray, orbit: np.ndarray) -> np.ndarray:
    """Sum costs over orbits (``orbit`` holds an orbit id per cell) -> labels per cell."""
    H, W, K = cost.shape
    ids, inv = np.unique(orbit.ravel(), return_inverse=True)
    pooled = np.zeros((ids.size, K))
    np.add.at(pooled, inv.reshape(-1), cost.reshape(-1, K))
    return pooled.argmin(axis=1)[inv.reshape(-1)].reshape(H, W)


def regularize_stripes(labels: np.ndarray, cost: np.ndarray, axis: int = 0, min_frac: float = 0.92,
                       max_margin: float = 3.0) -> tuple[np.ndarray, int]:
    """Make nearly uniform rows (axis=0) or columns (axis=1) exactly uniform.

    Only cells whose cost margin to the row's majority colour is below ``max_margin`` are
    changed, so genuine small motifs on an otherwise plain row survive.
    """
    lab = labels if axis == 0 else labels.T
    c = cost if axis == 0 else cost.transpose(1, 0, 2)
    out = lab.copy()
    changed = 0
    for r in range(lab.shape[0]):
        counts = np.bincount(lab[r], minlength=c.shape[2])
        m = int(counts.argmax())
        if counts[m] / lab.shape[1] < min_frac or counts[m] == lab.shape[1]:
            continue
        dev = np.flatnonzero(lab[r] != m)
        margin = c[r, dev, m] - c[r, dev, lab[r, dev]]
        flip = dev[margin < max_margin]
        out[r, flip] = m
        changed += flip.size
    return (out if axis == 0 else out.T), changed


def _background(seq: np.ndarray) -> int:
    return int(np.bincount(np.asarray(seq).ravel()).argmax())


def match_score(seq: np.ndarray, s: int, bg: int | None = None) -> float:
    """Foreground agreement between ``seq[..., x]`` and ``seq[..., x + s]``.

    Only position pairs where at least one cell differs from the background colour count,
    so a mostly plain row does not look periodic just because background matches background.
    """
    if s <= 0 or s >= seq.shape[-1]:
        return 0.0
    bg = _background(seq) if bg is None else bg
    a, b = seq[..., :-s], seq[..., s:]
    fg = (a != bg) | (b != bg)
    n = int(fg.sum())
    if n == 0:
        return 0.0
    return float(((a == b) & fg).sum() / n)


def best_period(rows: np.ndarray, min_period: int = 2, max_frac: float = 0.5, tol: float = 0.02) -> tuple[int, float]:
    """Smallest period whose foreground match score is within ``tol`` of the best one."""
    W = rows.shape[-1]
    smax = max(min_period, int(W * max_frac))
    bg = _background(rows)
    scores = np.array([match_score(rows, s, bg) for s in range(min_period, smax + 1)])
    if scores.size == 0:
        return 0, 0.0
    top = scores.max()
    i = int(np.flatnonzero(scores >= top - tol)[0])
    return min_period + i, float(scores[i])


def detect_bands(labels: np.ndarray, min_score: float = 0.9) -> list[tuple[int, int, int]]:
    """Periodic bands -> [(r0, r1, period)].

    Runs of consecutive non-uniform rows are first tested jointly (noisy rows inside a band
    still count); if the run as a whole is not periodic it is split where the rows'
    individual periods change.
    """
    H, W = labels.shape
    uniform = np.array([(labels[r] == labels[r, 0]).all() for r in range(H)])
    bands: list[tuple[int, int, int]] = []
    r = 0
    while r < H:
        if uniform[r]:
            r += 1
            continue
        r1 = r + 1
        while r1 < H and not uniform[r1]:
            r1 += 1
        p, sc = best_period(labels[r:r1])
        if sc >= min_score:
            bands.append((r, r1, p))
        else:
            periods = []
            for t in range(r, r1):
                pt, st = best_period(labels[t][None, :])
                periods.append(pt if st >= min_score else 0)
            s0 = 0
            for t in range(1, len(periods) + 1):
                if t == len(periods) or periods[t] != periods[s0]:
                    if periods[s0] > 0:
                        p2, sc2 = best_period(labels[r + s0 : r + t])
                        if sc2 >= min_score:
                            bands.append((r + s0, r + t, p2))
                    s0 = t
        r = r1
    return bands


def best_mirror_axis(rows: np.ndarray, min_overlap: float = 0.5) -> tuple[float, float]:
    """Vertical mirror axis (column coordinate, possibly half-integer) and its foreground score."""
    W = rows.shape[-1]
    bg = _background(rows)
    best = (0.0, 0.0)
    x = np.arange(W)
    for a2 in range(int(W * (1 - min_overlap)), int(W * (1 + min_overlap)) - 1):
        a = a2 / 2.0
        xm = (2 * a - x).astype(int)
        ok = (xm >= 0) & (xm < W) & (xm != x)
        if ok.sum() < W * min_overlap:
            continue
        u, v = rows[..., x[ok]], rows[..., xm[ok]]
        fg = (u != bg) | (v != bg)
        if not fg.any():
            continue
        sc = float(((u == v) & fg).sum() / fg.sum())
        if sc > best[1]:
            best = (a, sc)
    return best


def regularize_structure(labels: np.ndarray, cost: np.ndarray, repeats: bool = True, mirror: bool = True,
                         min_score: float = 0.9, mirror_score: float = 0.93) -> tuple[np.ndarray, dict]:
    """Detect periodic / mirror-symmetric bands and pool costs over their orbits."""
    H, W = labels.shape
    uf = UnionFind(H * W)
    info: dict = {"bands": []}
    bands = detect_bands(labels, min_score) if repeats else []
    for r0, r1, p in bands:
        entry = {"rows": [int(r0), int(r1)], "period": int(p)}
        for r in range(r0, r1):
            base = r * W
            for x in range(p, W):
                uf.union(base + x, base + x - p)
        if mirror:
            a, sc = best_mirror_axis(labels[r0:r1])
            if sc >= mirror_score:
                entry["mirror_axis"] = float(a)
                for r in range(r0, r1):
                    base = r * W
                    for x in range(W):
                        xm = int(2 * a - x)
                        if 0 <= xm < W and xm != x:
                            uf.union(base + x, base + xm)
        info["bands"].append(entry)
    if not info["bands"]:
        return labels.copy(), info
    orbit = uf.roots().reshape(H, W)
    out = pool_orbits(cost, orbit)
    # rows outside detected bands keep their labels
    in_band = np.zeros(H, dtype=bool)
    for r0, r1, _ in bands:
        in_band[r0:r1] = True
    out[~in_band] = labels[~in_band]
    info["changed"] = int((out != labels).sum())
    return out, info


def _best_grouping(a: np.ndarray, m: int, min_agree: float) -> tuple[float, list[tuple[int, int]]] | None:
    """Best phase for merging groups of ``m`` columns of ``a``; None if not lossless enough."""
    n = a.shape[1]
    if n < 2 * m:
        return None
    best = None
    for ph in range(m):
        edges = sorted({0, n, *range(ph, n, m)})
        groups = list(zip(edges[:-1], edges[1:]))
        full = [(g0, g1) for g0, g1 in groups if g1 - g0 == m]
        if not full:
            continue
        agree = float(np.mean([(a[:, g0:g1] == a[:, g0:g0 + 1]).all(axis=1).mean() for g0, g1 in full]))
        if agree >= min_agree and (best is None or agree > best[0]):
            best = (agree, groups)
    return best


def _merge(a: np.ndarray, b: np.ndarray, groups: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    cols = []
    for g0, g1 in groups:
        block = a[:, g0:g1]
        k = int(block.max()) + 1
        counts = np.stack([(block == c).sum(axis=1) for c in range(k)], axis=1)
        cols.append(counts.argmax(axis=1))
    return np.stack(cols, axis=1), np.array([b[g0] for g0, _ in groups] + [b[groups[-1][1]]])


def reduce_lossless(labels: np.ndarray, xs: np.ndarray, ys: np.ndarray, min_agree: float = 0.97,
                    max_aspect: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Merge groups of identical adjacent columns / rows (an over-fine grid) losslessly.

    Joint reductions of both axes by the same factor are tried first (they keep the cell
    aspect ratio); single-axis reductions must keep cells within ``max_aspect`` of square.
    A reduction applies when at least ``min_agree`` of the aligned groups are uniform.
    """
    lab = labels.copy()
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    log: list[str] = []
    changed = True
    while changed:
        changed = False
        cw = float(np.median(np.diff(xs)))
        ch = float(np.median(np.diff(ys)))
        for m in (2, 3):
            gx = _best_grouping(lab, m, min_agree)
            gy = _best_grouping(lab.T, m, min_agree)
            if gx is not None and gy is not None:
                lab, xs = _merge(lab, xs, gx[1])
                labT, ys = _merge(lab.T, ys, gy[1])
                lab = labT.T
                log.append(f"merged columns and rows by {m}")
                changed = True
                break
            if gx is not None and (cw * m) / ch <= max_aspect:
                lab, xs = _merge(lab, xs, gx[1])
                log.append(f"merged columns by {m}")
                changed = True
                break
            if gy is not None and (ch * m) / cw <= max_aspect:
                labT, ys = _merge(lab.T, ys, gy[1])
                lab = labT.T
                log.append(f"merged rows by {m}")
                changed = True
                break
    return lab, xs, ys, log
