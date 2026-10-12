"""Design-cell grid estimation and cell colour sampling.

1. **Edge peaks.**  A colour-gradient profile is computed along each axis (columns for
   x, rows for y), its noise floor removed, and edge positions found with sub-pixel
   accuracy.
2. **Phase-coherent period.**  Every colour boundary of a woven design lies on the cell
   lattice, so for the true cell size ``c`` all edge positions have the same phase
   ``x mod c``.  The coherence ``R(c) = |sum_i w_i exp(2 pi i x_i / c)| / sum_i w_i`` is
   high for ``c`` and its sub-multiples ``c/k`` but not for multiples; the largest
   coherent local maximum is therefore the cell size.  An axis without edges (e.g. the
   x axis of a horizontally striped cloth) is unobservable and borrows the other axis'
   cell size.
3. **Elastic refinement.**  Hand-woven rows are not perfectly regular; dynamic
   programming moves every boundary to a nearby edge while penalising deviations from
   the fitted cell size.
4. **Sampling.**  Cells are colour-averaged at sub-cell scale (optical mixing of yarns)
   and the per-channel median of the central region is taken, which ignores boundary
   blur and isolated tie-down dots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from manitex.color import srgb_to_lab
from manitex.digitise.geometry import sample_bilinear

#: Blur applied after the yarn period ``t`` is found, as a multiple of ``t``.  0.6 removes the
#: yarn-period texture and most of the two-yarn checker of plain-weave crossings.
TEXTURE_BLUR = 0.6


def is_clean_image(image: np.ndarray, max_colors: int = 64) -> bool:
    """True for noise-free pixel-art designs (few exact colours), False for photos and
    cloth renders (thousands of colours from shading, texture and noise)."""
    img = np.asarray(image)[..., :3]
    flat = img.reshape(-1, 3)
    if flat.shape[0] > 200_000:
        flat = flat[:: flat.shape[0] // 200_000 + 1]
    return int(np.unique(flat, axis=0).shape[0]) <= max_colors


@dataclass
class GridFit:
    xs: np.ndarray  # (nx + 1,) cell boundaries along x (pixels)
    ys: np.ndarray  # (ny + 1,) cell boundaries along y
    cell_w: float
    cell_h: float
    score_x: float  # phase coherence of the x lattice (0..1; 0 = unobservable)
    score_y: float

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.ys) - 1, len(self.xs) - 1


def edge_profiles(lab: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean colour-gradient magnitude across columns (x profile) and rows (y profile).

    ``px[i]`` measures the edge between pixel columns ``i-1`` and ``i``; ``px[0] = 0``.
    A boundary at index ``i`` lies at pixel coordinate ``i`` in "edge" coordinates where
    pixel ``j`` spans ``[j, j+1)``.
    """
    img = ndimage.gaussian_filter(lab, sigma=(sigma, sigma, 0)) if sigma > 0 else lab
    gx = np.sqrt((np.diff(img, axis=1) ** 2).sum(axis=2))
    gy = np.sqrt((np.diff(img, axis=0) ** 2).sum(axis=2))
    px = np.concatenate([[0.0], gx.mean(axis=0)])
    py = np.concatenate([[0.0], gy.mean(axis=1)])
    return px, py


def edge_point_histogram(lab: np.ndarray, sigma: float, axis: int, bins_per_px: int = 10,
                         rel_thresh: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Weighted histogram of sub-pixel edge positions along ``axis`` (0 = y, 1 = x).

    Edge pixels are local maxima of the colour gradient across the axis (non-maximum
    suppression) above ``rel_thresh`` x the 99th percentile; every pixel votes, so short
    edges (motif steps) count in proportion to their length.  Returns (positions, weights)
    of the non-empty histogram bins in edge coordinates.
    """
    img = ndimage.gaussian_filter(lab, sigma=(sigma, sigma, 0)) if sigma > 0 else lab
    g = np.sqrt((np.diff(img, axis=axis) ** 2).sum(axis=2))
    if axis == 1:
        g = g.T  # make the edge axis the first axis
    if g.shape[0] < 3:
        return np.zeros(0), np.zeros(0)
    a, b, c = g[:-2], g[1:-1], g[2:]
    is_max = (b > a) & (b >= c)
    vals = b[is_max]
    if vals.size == 0:
        return np.zeros(0), np.zeros(0)
    thr = rel_thresh * np.percentile(vals, 99)
    keep = is_max & (b > thr)
    ii, _ = np.nonzero(keep)
    den = (a - 2 * b + c)[keep]
    off = np.where(np.abs(den) > 1e-12, 0.5 * (a - c)[keep] / den, 0.0)
    pos = ii + 2 + np.clip(off, -0.5, 0.5)  # g[i] is the edge between pixels i and i+1 -> coordinate i+1
    w = b[keep]
    L = lab.shape[axis]
    nb = L * bins_per_px + 1
    idx = np.clip(np.rint(pos * bins_per_px).astype(int), 0, nb - 1)
    hist = np.bincount(idx, weights=w, minlength=nb)
    nz = np.flatnonzero(hist)
    return nz / float(bins_per_px), hist[nz]


def detect_peaks(p: np.ndarray, rel_thresh: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel positions and strengths of edge peaks above the noise floor."""
    q = p - np.median(p)
    q[q < 0] = 0.0
    top = np.percentile(q, 99.5) if q.size else 0.0
    if top <= 0:
        return np.zeros(0), np.zeros(0)
    thr = rel_thresh * top
    i = np.arange(1, q.size - 1)
    is_pk = (q[i] > q[i - 1]) & (q[i] >= q[i + 1]) & (q[i] > thr)
    i = i[is_pk]
    a, b, c = q[i - 1], q[i], q[i + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    return i + np.clip(off, -0.5, 0.5), b


def period_candidates(c_min: float, c_max: float, L: float) -> np.ndarray:
    """Candidate periods spaced finer than the width of a coherence peak (~ c^2 / L)."""
    out = [c_min]
    c = c_min
    while c < c_max:
        c += max(0.004, c * c / (10.0 * max(L, 1.0)))
        out.append(c)
    return np.array(out)


def coherence(pos: np.ndarray, w: np.ndarray, cands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Phase coherence R(c) and lattice phase for every candidate period."""
    ang = (2 * np.pi * pos[None, :] / cands[:, None]).astype(np.float32)
    wf = w.astype(np.float32)[None, :]
    z = ((wf * np.cos(ang)).sum(axis=1) + 1j * (wf * np.sin(ang)).sum(axis=1)) / max(float(w.sum()), 1e-12)
    phase = (np.angle(z) / (2 * np.pi)) % 1.0 * cands
    return np.abs(z), phase


def coherent_period(pos: np.ndarray, w: np.ndarray, c_min: float, c_max: float, tau: float = 0.7,
                    step: float = 0.02) -> tuple[float, float, float] | None:
    """Largest period whose coherence is a local maximum above ``tau``."""
    if pos.size < 3 or c_max <= c_min:
        return None
    cands = period_candidates(c_min, c_max, float(pos.max() - pos.min()) if pos.size else 1.0)
    R, ph = coherence(pos, w, cands)
    is_max = np.zeros_like(R, dtype=bool)
    is_max[1:-1] = (R[1:-1] >= R[:-2]) & (R[1:-1] >= R[2:])
    good = np.flatnonzero(is_max & (R >= tau))
    if good.size == 0:
        return None
    j = int(good[-1])
    return float(cands[j]), float(ph[j]), float(R[j])


def lattice_significance(pos: np.ndarray, w: np.ndarray, cands: np.ndarray, delta: float = 0.75
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Chance-corrected fraction of edge mass on the best lattice of each candidate period.

    ``kappa(c) = (E - p0) / (1 - p0)`` where ``E`` is the edge mass within ``delta`` px of
    the lattice and ``p0 = 2 delta / c`` the fraction expected by chance.  Returns
    (kappa, lattice phase).
    """
    R, ph = coherence(pos, w, cands)
    wsum = max(float(w.sum()), 1e-12)
    p0 = np.minimum(0.95, 2.0 * delta / cands)
    best = np.full(cands.size, -np.inf)
    best_ph = ph.copy()
    for shift in (-0.5, -0.25, 0.0, 0.25, 0.5):  # refine the coherence phase
        phase = ph + shift
        d = (pos[None, :] - phase[:, None]) % cands[:, None]
        d = np.minimum(d, cands[:, None] - d)
        explained = (w[None, :] * (d <= delta)).sum(axis=1) / wsum
        kappa = (explained - p0) / (1.0 - p0)
        better = kappa > best
        best = np.where(better, kappa, best)
        best_ph = np.where(better, phase, best_ph)
    return best, best_ph % cands


def select_lattice(pos: np.ndarray, w: np.ndarray, c_min: float, c_max: float, rel: float = 0.9
                   ) -> tuple[float, float, float] | None:
    """The coarsest lattice that explains (almost) all the edges the best lattice explains.

    Finer lattices explain no more edges; coarser ones and pattern repeats leave design
    edges off the lattice, so the cell size is the largest local maximum of kappa within
    ``rel`` of the best kappa.
    """
    if pos.size < 3 or c_max <= c_min:
        return None
    cands = period_candidates(c_min, c_max, float(pos.max() - pos.min()))
    kappa, ph = lattice_significance(pos, w, cands)
    ref = float(kappa.max())
    if ref < 0.3:
        return None
    is_max = np.zeros_like(kappa, dtype=bool)
    is_max[1:-1] = (kappa[1:-1] >= kappa[:-2]) & (kappa[1:-1] >= kappa[2:])
    is_max[0] = kappa[0] >= kappa[1] if kappa.size > 1 else True
    good = np.flatnonzero(is_max & (kappa >= rel * ref))
    j = int(good[-1]) if good.size else int(np.argmax(kappa))
    return float(cands[j]), float(ph[j]), float(kappa[j])


def n_edge_clusters(pos: np.ndarray, w: np.ndarray, gap: float = 2.0) -> int:
    """Number of distinct edge positions (histogram bins closer than ``gap`` px are merged)."""
    if pos.size == 0:
        return 0
    strong = pos[w >= 0.2 * w.max()]
    if strong.size == 0:
        return 0
    strong = np.sort(strong)
    return int(1 + (np.diff(strong) > gap).sum())


def elastic_boundaries(p: np.ndarray, c: float, phase: float, slack: float = 0.25, lam: float = 4.0) -> np.ndarray:
    """Boundaries maximising edge strength with spacing close to ``c`` (dynamic programming).

    Boundaries are integer positions in ``[0, L]`` (edge coordinates).  The first lies
    within one cell of the start, the last within one cell of the end.
    """
    L = p.size
    q = np.maximum(p - np.median(p), 0.0)
    e = np.concatenate([q, [0.0]]) / (np.percentile(q, 99.5) + 1e-9)
    smin = max(1, int(np.floor(c * (1 - slack))))
    smax = max(smin, int(np.ceil(c * (1 + slack))))
    NEG = -1e18
    best = np.full(L + 1, NEG)
    prev = np.full(L + 1, -1, dtype=np.int64)
    start_max = min(L, int(np.ceil(c)))
    xs0 = np.arange(start_max + 1)
    dphase = np.minimum(np.abs(xs0 - phase % c), c - np.abs(xs0 - phase % c)) / c
    best[: start_max + 1] = e[: start_max + 1] - lam * dphase**2
    d = np.arange(smin, smax + 1)
    pen = lam * ((d - c) / c) ** 2
    for x in range(smin, L + 1):
        src = x - d
        ok = src >= 0
        cand = best[src[ok]] - pen[ok]
        j = int(np.argmax(cand))
        val = cand[j] + e[x]
        if val > best[x]:
            best[x] = val
            prev[x] = src[ok][j]
    end_lo = max(0, L - int(np.ceil(c)))
    x = end_lo + int(np.argmax(best[end_lo:]))
    out = [x]
    while prev[x] >= 0:
        x = int(prev[x])
        out.append(x)
    return np.array(out[::-1], dtype=np.float64)


def elastic_boundaries_fixed(p: np.ndarray, n: int, slack: float = 0.25, lam: float = 4.0) -> np.ndarray:
    """Exactly ``n`` cells spanning ``[0, L]``: interior boundaries snap to edges (DP)."""
    L = p.size
    c = L / float(n)
    if n <= 1:
        return np.array([0.0, float(L)])
    q = np.maximum(p - np.median(p), 0.0)
    e = np.concatenate([q, [0.0]]) / (np.percentile(q, 99.5) + 1e-9)
    smin = max(1, int(np.floor(c * (1 - slack))))
    smax = max(smin, int(np.ceil(c * (1 + slack))))
    NEG = -1e18
    d = np.arange(smin, smax + 1)
    pen = lam * ((d - c) / c) ** 2
    best = np.full(L + 1, NEG)
    best[0] = 0.0
    back = np.full((n + 1, L + 1), -1, dtype=np.int64)
    for k in range(1, n + 1):
        new = np.full(L + 1, NEG)
        for di, step in enumerate(d):
            src_val = np.full(L + 1, NEG)
            src_val[step:] = best[: L + 1 - step] - pen[di]
            better = src_val > new
            new = np.where(better, src_val, new)
            back[k] = np.where(better, np.arange(L + 1) - step, back[k])
        best = new + (e if k < n else 0.0)
    if best[L] <= NEG / 2:  # infeasible (slack too small): fall back to a regular split
        return np.linspace(0, L, n + 1)
    out = [L]
    x = L
    for k in range(n, 0, -1):
        x = int(back[k, x])
        out.append(x)
    return np.array(out[::-1], dtype=np.float64)


def regular_boundaries(L: int, c: float, phase: float) -> np.ndarray:
    start = phase % c
    if start > c / 2:
        start -= c  # allow a partial first cell to be dropped
    xs = np.arange(start, L + 1e-9, c)
    xs = xs[(xs >= -1e-9)]
    if xs.size < 2:
        return np.array([0.0, float(L)])
    return xs


def _tidy_borders(b: np.ndarray, L: int, c: float) -> np.ndarray:
    """Extend the lattice to the image borders when the leftover strip is >= half a cell."""
    b = np.asarray(b, dtype=np.float64)
    b = b[(b >= 0) & (b <= L)]
    if b.size == 0:
        return np.array([0.0, float(L)])
    if b[0] >= 0.5 * c:
        b = np.concatenate([[0.0], b])
    if L - b[-1] >= 0.5 * c:
        b = np.concatenate([b, [float(L)]])
    if b.size < 2:
        b = np.array([0.0, float(L)])
    return b


def _finest_coherent(pos: np.ndarray, w: np.ndarray, c_min: float, c_max: float, tau: float) -> float | None:
    if pos.size < 3 or c_max <= c_min:
        return None
    cands = period_candidates(c_min, c_max, float(pos.max() - pos.min()))
    R, _ = coherence(pos, w, cands)
    is_max = np.zeros_like(R, dtype=bool)
    is_max[1:-1] = (R[1:-1] >= R[:-2]) & (R[1:-1] >= R[2:])
    good = np.flatnonzero(is_max & (R >= tau))
    return float(cands[good[0]]) if good.size else None


def fit_grid(
    image: np.ndarray,
    cells_x: int | None = None,
    cells_y: int | None = None,
    min_cell: float = 3.0,
    max_cell: float | None = None,
    elastic: bool = True,
    square_prior: bool = False,
    sigma: float = 0.7,
) -> GridFit:
    """Fit the design-cell grid of an (already rectified) textile image.

    With ``cells_x`` / ``cells_y`` the image is assumed to be cropped to the cloth and is
    split into exactly that many cells (boundaries snap to nearby edges if ``elastic``).
    Otherwise the cell size is estimated in two stages: the finest strong periodicity
    (usually yarn texture) is found first and suppressed with a matched blur, then the
    largest phase-coherent period of the remaining (design) edges is the cell size.
    """
    lab = srgb_to_lab(np.asarray(image)[..., :3].astype(np.float64))
    H, W = lab.shape[:2]
    max_cell = max_cell or max(min(H, W) / 3.0, min_cell + 1)
    # clean pixel-art inputs must keep every edge on the lattice
    rel = 0.98 if is_clean_image(image) else 0.9

    texture = {0: None, 1: None}
    n_edges = {0: 0, 1: 0}
    evidence: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def estimate(axis: int) -> tuple[float | None, float, float, float]:
        pos, w = edge_point_histogram(lab, sigma, axis)
        t = _finest_coherent(pos, w, min_cell, max_cell, 0.5)
        texture[axis] = t
        sig2 = max(sigma, TEXTURE_BLUR * t) if t else max(sigma, 1.0)
        pos2, w2 = edge_point_histogram(lab, sig2, axis)
        lo = max(min_cell, 0.9 * t) if t else min_cell
        n_edges[axis] = n_edge_clusters(pos2, w2)
        if pos2.size < 3 or max_cell <= lo or n_edges[axis] < 2:
            return None, 0.0, 0.0, sig2
        res = select_lattice(pos2, w2, lo, max_cell, rel=rel)
        if res is None:
            return None, 0.0, 0.0, sig2
        evidence[axis] = (pos2, w2)
        return res[0], res[1], res[2], sig2

    cw = ch = None
    phx = phy = scx = scy = 0.0
    sx = sy = sigma
    if not cells_x:
        cw, phx, scx, sx = estimate(1)
    if not cells_y:
        ch, phy, scy, sy = estimate(0)
    # an axis with only a few distinct edges is weakly determined: prefer square cells when the
    # other axis' cell size explains its edges as well
    for axis, (c_self, c_other) in ((1, (cw, ch)), (0, (ch, cw))):
        if c_self is None or c_other is None or (cells_x if axis == 1 else cells_y) or n_edges[axis] >= 8:
            continue
        if axis in evidence:
            pos_e, w_e = evidence[axis]
            sig_o, ph_o = lattice_significance(pos_e, w_e, np.array([c_other]))
            if sig_o[0] >= 0.8 * (scx if axis == 1 else scy):
                if axis == 1:
                    cw, phx = c_other, float(ph_o[0])
                else:
                    ch, phy = c_other, float(ph_o[0])
    # an axis without design edges (e.g. across stripes) cannot be measured: borrow the other
    # axis' cell size, but never go below two yarn periods so cells average the weave texture
    if cw is None and ch is not None and not cells_x:
        cw = max(ch, 2.0 * texture[1]) if texture[1] else ch
    if ch is None and cw is not None and not cells_y:
        ch = max(cw, 2.0 * texture[0]) if texture[0] else cw
    if cells_x:
        cw, scx = W / float(cells_x), 1.0
    if cells_y:
        ch, scy = H / float(cells_y), 1.0
    if cw is None and ch is None:  # no structure at all: one cell per ~16 px
        cw = ch = max(min_cell, min(H, W) / 16.0)
    elif cw is None:
        cw = ch
    elif ch is None:
        ch = cw
    assert cw is not None and ch is not None
    if square_prior and not (cells_x or cells_y) and scx > 0 and scy > 0:
        c = (cw + ch) / 2.0
        if abs(cw - ch) / c < 0.2:
            cw = ch = c
    px, _ = edge_profiles(lab, sx)
    _, py = edge_profiles(lab, sy)

    def boundaries(p: np.ndarray, n_cells: int | None, c: float, phase: float, score: float, L: int) -> np.ndarray:
        if n_cells:
            return elastic_boundaries_fixed(p, n_cells) if elastic else np.linspace(0, L, n_cells + 1)
        if elastic and score > 0:
            b = elastic_boundaries(p, c, phase)
        else:
            b = regular_boundaries(L, c, phase)
        return _tidy_borders(b, L, c)

    xs = boundaries(px, cells_x, cw, phx, scx, W)
    ys = boundaries(py, cells_y, ch, phy, scy, H)
    return GridFit(xs=xs, ys=ys, cell_w=float(cw), cell_h=float(ch), score_x=float(scx), score_y=float(scy))


def sample_cells(image: np.ndarray, grid: GridFit, inner: float = 0.6, sub: int = 6) -> np.ndarray:
    """Robust Lab colour of every cell -> (ny, nx, 3).

    ``sub x sub`` points in the central ``inner`` region of each cell are sampled in linear
    light and summarised by the per-channel **interquartile mean**: it ignores up to 25%
    contamination (tie-down dots, boundary blur, yarn shadows) yet returns the optical
    mixture for cells that alternate two yarn colours (crossings of a check).
    """
    from manitex.color import linear_to_srgb, srgb_to_linear

    lin = srgb_to_linear(np.asarray(image)[..., :3])
    lin = ndimage.gaussian_filter(lin, sigma=(0.5, 0.5, 0))  # sensor noise only
    t = (1.0 - inner) / 2.0 + inner * (np.arange(sub) + 0.5) / sub

    def coords(b: np.ndarray) -> np.ndarray:
        a, z = b[:-1], b[1:]
        return a[:, None] + (z - a)[:, None] * t[None, :] - 0.5  # edge coords -> pixel centres

    cy, cx = coords(grid.ys), coords(grid.xs)
    ny, nx = cy.shape[0], cx.shape[0]
    YY = np.repeat(cy.reshape(-1), nx * sub).reshape(ny * sub, nx * sub)
    XX = np.tile(cx.reshape(-1), (ny * sub, 1))
    vals = sample_bilinear(lin, YY.ravel(), XX.ravel()).reshape(ny, sub, nx, sub, 3)
    vals = vals.transpose(0, 2, 1, 3, 4).reshape(ny, nx, sub * sub, 3)
    vals = np.sort(vals, axis=2)
    q = sub * sub
    lo, hi = q // 4, q - q // 4
    iqm = vals[:, :, lo:hi].mean(axis=2)
    return srgb_to_lab(linear_to_srgb(iqm))
