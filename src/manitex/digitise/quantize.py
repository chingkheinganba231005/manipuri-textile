"""Colour quantisation in CIELAB and Potts-model label denoising."""

from __future__ import annotations

import numpy as np

from manitex.color import delta_e2000


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 4, max_iter: int = 60) -> tuple[np.ndarray, np.ndarray, float]:
    """k-means with k-means++ initialisation.  Returns (centers, labels, inertia)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k = max(1, min(int(k), n))
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(max(1, n_init)):
        centers = np.empty((k, X.shape[1]))
        centers[0] = X[rng.integers(n)]
        d2 = ((X - centers[0]) ** 2).sum(axis=1)
        for j in range(1, k):
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            centers[j] = X[rng.choice(n, p=probs)]
            d2 = np.minimum(d2, ((X - centers[j]) ** 2).sum(axis=1))
        labels = np.zeros(n, dtype=np.int64)
        for it in range(max_iter):
            dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new = dist.argmin(axis=1)
            if it > 0 and np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                m = labels == j
                if m.any():
                    centers[j] = X[m].mean(axis=0)
                else:  # re-seed an empty cluster at the worst-fitted point
                    centers[j] = X[dist.min(axis=1).argmax()]
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (centers.copy(), labels.copy(), inertia)
    assert best is not None
    return best


def _assign(X: np.ndarray, centers: np.ndarray, n_iter: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centre labels plus a few warm-started Lloyd refinements."""
    centers = centers.copy()
    labels = np.zeros(X.shape[0], dtype=np.int64)
    for _ in range(max(1, n_iter)):
        labels = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        for j in range(centers.shape[0]):
            m = labels == j
            if m.any():
                centers[j] = X[m].mean(axis=0)
    return centers, labels


def choose_k(X: np.ndarray, k_max: int, rng: np.random.Generator, min_delta_e: float = 9.0,
             min_frac: float = 0.002) -> tuple[int, np.ndarray, np.ndarray]:
    """Largest k whose clusters are all distinct (CIEDE2000 >= ``min_delta_e``) and non-trivial.

    The clustering that passed validation is the one returned (refined on all samples).
    """
    X = np.asarray(X, dtype=np.float64)
    sub = X if X.shape[0] <= 6000 else X[rng.choice(X.shape[0], 6000, replace=False)]
    chosen: tuple[int, np.ndarray] | None = None
    fails = 0
    for k in range(1, max(1, k_max) + 1):
        centers, labels, _ = kmeans(sub, k, rng)
        sizes = np.bincount(labels, minlength=k) / labels.size
        ok = True
        if k > 1:
            i, j = np.triu_indices(k, 1)
            ok = float(delta_e2000(centers[i], centers[j]).min()) >= min_delta_e and sizes.min() >= min_frac
        if ok:
            chosen, fails = (k, centers), 0
        else:
            fails += 1
            if fails >= 2:
                break
    k, centers = chosen if chosen is not None else (1, X.mean(axis=0, keepdims=True))
    centers, labels = _assign(X, centers)
    return k, centers, labels


def merge_clusters(grid_labels: np.ndarray, X: np.ndarray, centers: np.ndarray, min_delta_e: float = 9.0,
                   mix_tol: float = 7.0, min_boundary: float = 0.6) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Merge near-duplicate clusters and boundary *mixture* clusters.

    A mixture cluster's colour lies (in linear light) between two other clusters'
    colours and most of its cells touch cells of those two clusters: these are cells
    straddling a colour boundary (blur, yarn bleed), not a real yarn colour.  Plaid
    crossing colours are mixtures too, but they form solid blocks and are kept.
    Returns (labels (H, W), centers, log).
    """
    from manitex.color import lab_to_srgb, srgb_to_lab, srgb_to_linear

    labels = grid_labels.copy()
    H, W = labels.shape
    Xg = X.reshape(H, W, 3)
    centers = centers.copy()
    log: list[str] = []

    def recompute() -> None:
        nonlocal centers, labels
        used = np.unique(labels)
        remap = -np.ones(centers.shape[0], dtype=np.int64)
        remap[used] = np.arange(used.size)
        labels = remap[labels]
        centers = np.array([Xg[labels == j].mean(axis=0) for j in range(used.size)])

    changed = True
    while changed and centers.shape[0] > 1:
        changed = False
        k = centers.shape[0]
        # 1) near-duplicates
        i, j = np.triu_indices(k, 1)
        d = delta_e2000(centers[i], centers[j])
        if d.size and d.min() < min_delta_e:
            a, b = int(i[d.argmin()]), int(j[d.argmin()])
            labels[labels == b] = a
            log.append(f"merged duplicate clusters {a},{b} (dE00={d.min():.1f})")
            recompute()
            changed = True
            continue
        if k < 3:
            break
        # 2) boundary mixtures, smallest clusters first
        lin = srgb_to_linear(lab_to_srgb(centers))
        sizes = np.bincount(labels.ravel(), minlength=k)
        for c in np.argsort(sizes):
            best = None
            for a in range(k):
                for b in range(a + 1, k):
                    if c in (a, b):
                        continue
                    seg = lin[a] - lin[b]
                    den = float(seg @ seg)
                    if den < 1e-9:
                        continue
                    alpha = float(np.clip((lin[c] - lin[b]) @ seg / den, 0.0, 1.0))
                    if not 0.12 <= alpha <= 0.88:
                        continue
                    mix = srgb_to_lab(_lin_to_srgb(alpha * lin[a] + (1 - alpha) * lin[b]))
                    r = float(delta_e2000(mix, centers[c]))
                    if r < mix_tol and (best is None or r < best[0]):
                        best = (r, a, b, alpha)
            if best is None:
                continue
            _, a, b, _ = best
            m = labels == c
            nb = np.zeros_like(m)
            pad = np.pad(labels, 1, constant_values=-1)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                sh = pad[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
                nb |= (sh == a) | (sh == b)
            if nb[m].mean() < min_boundary:
                continue
            cells_lin = srgb_to_linear(lab_to_srgb(Xg[m]))
            seg = lin[a] - lin[b]
            al = np.clip((cells_lin - lin[b]) @ seg / float(seg @ seg), 0, 1)
            if al.std() < 0.1:
                continue  # a consistent mixing ratio: a real crossing colour (check), keep it
            # reassign each cell to the closer endpoint along the mixing segment
            labels[m] = np.where(al >= 0.5, a, b)
            log.append(f"merged boundary-mixture cluster {c} into {a}/{b}")
            recompute()
            changed = True
            break
    return labels, centers, log


def _lin_to_srgb(lin: np.ndarray) -> np.ndarray:
    from manitex.color import linear_to_srgb

    return linear_to_srgb(np.clip(lin, 0.0, 1.0))


def class_costs(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Normalised negative log-likelihood of each sample under each colour cluster.

    Uses an isotropic Gaussian per cluster with a pooled, floored standard deviation, so a
    cost difference of ~1 corresponds to one standard deviation of colour noise.
    """
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.min(axis=1)
    sigma2 = max(float(np.median(nearest)), 4.0)  # at least 2 Lab units of noise
    return d2 / (2.0 * sigma2)


def potts_icm(cost: np.ndarray, beta: float, n_iter: int = 12, labels: np.ndarray | None = None) -> np.ndarray:
    """Iterated conditional modes for a 4-connected Potts model (checkerboard updates)."""
    H, W, K = cost.shape
    lab = cost.argmin(axis=2) if labels is None else labels.copy()
    if beta <= 0 or K == 1:
        return lab
    ii, jj = np.mgrid[0:H, 0:W]
    parity = (ii + jj) % 2
    n_nb = np.full((H, W), 4.0)
    n_nb[0, :] -= 1
    n_nb[-1, :] -= 1
    n_nb[:, 0] -= 1
    n_nb[:, -1] -= 1
    eye = np.eye(K)
    for _ in range(n_iter):
        changed = False
        for par in (0, 1):
            oh = eye[lab]
            nb = np.zeros((H, W, K))
            nb[1:] += oh[:-1]
            nb[:-1] += oh[1:]
            nb[:, 1:] += oh[:, :-1]
            nb[:, :-1] += oh[:, 1:]
            energy = cost + beta * (n_nb[..., None] - nb)
            new = energy.argmin(axis=2)
            m = (parity == par) & (new != lab)
            if m.any():
                lab[m] = new[m]
                changed = True
        if not changed:
            break
    return lab
