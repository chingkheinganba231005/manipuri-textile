"""Image geometry: homographies, perspective rectification, bilinear sampling, deskew."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping the 4 points ``src`` (x, y) onto ``dst`` (x, y)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("need exactly 4 point correspondences")
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -x * u, -y * u])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -x * v, -y * v])
        b.append(v)
    h = np.linalg.solve(np.array(A), np.array(b))
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    hom = np.concatenate([pts, np.ones((*pts.shape[:-1], 1))], axis=-1) @ H.T
    return hom[..., :2] / hom[..., 2:3]


def sample_bilinear(image: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Sample an (H, W[, C]) image at float coordinates (edge values are repeated)."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        return ndimage.map_coordinates(img, [ys, xs], order=1, mode="nearest")
    return np.stack([ndimage.map_coordinates(img[..., c], [ys, xs], order=1, mode="nearest")
                     for c in range(img.shape[2])], axis=-1)


def order_quad(quad: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    s = q.sum(axis=1)
    d = q[:, 0] - q[:, 1]
    tl, br = q[np.argmin(s)], q[np.argmax(s)]
    tr, bl = q[np.argmax(d)], q[np.argmin(d)]
    out = np.array([tl, tr, br, bl])
    if len({tuple(p) for p in out}) < 4:  # degenerate ordering - fall back to angle sort
        c = q.mean(axis=0)
        ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
        out = q[np.argsort(ang)]
        out = np.roll(out, -int(np.argmin(out.sum(axis=1))), axis=0)
    return out


def warp_perspective(image: np.ndarray, quad: np.ndarray, out_size: tuple[int, int] | None = None) -> np.ndarray:
    """Rectify the quadrilateral ``quad`` (4 x (x, y), any order) to an axis-aligned image.

    ``out_size`` is ``(width, height)``; by default the mean lengths of opposite edges.
    """
    q = order_quad(quad)
    if out_size is None:
        w = int(round((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2))
        h = int(round((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2))
        out_size = (max(w, 2), max(h, 2))
    w, h = out_size
    rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    H = homography(rect, q)  # output pixel -> source pixel
    yy, xx = np.mgrid[0:h, 0:w]
    src = apply_homography(H, np.stack([xx, yy], axis=-1).reshape(-1, 2))
    out = sample_bilinear(image, src[:, 1], src[:, 0])
    out = out.reshape(h, w, -1) if np.asarray(image).ndim == 3 else out.reshape(h, w)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8) if np.asarray(image).dtype == np.uint8 else out


def deskew(image: np.ndarray, max_angle: float = 6.0, step: float = 0.25) -> tuple[np.ndarray, float]:
    """Estimate and remove a small rotation (textile rows/columns become axis aligned).

    The angle maximises the variance of the row- and column-mean profiles (projection
    profile method) of the gradient image.
    """
    img = np.asarray(image, dtype=np.float64)
    gray = img.mean(axis=2) if img.ndim == 3 else img
    small = gray
    scale = 1.0
    if max(gray.shape) > 600:
        scale = 600 / max(gray.shape)
        small = ndimage.zoom(gray, scale, order=1)
    best, best_score = 0.0, -np.inf
    for ang in np.arange(-max_angle, max_angle + 1e-9, step):
        rot = ndimage.rotate(small, ang, reshape=False, order=1, mode="nearest")
        h, w = rot.shape
        core = rot[h // 6 : h - h // 6, w // 6 : w - w // 6]
        score = np.var(np.diff(core.mean(axis=1))) + np.var(np.diff(core.mean(axis=0)))
        if score > best_score:
            best, best_score = float(ang), score
    if abs(best) < 1e-9:
        return np.asarray(image), 0.0
    if img.ndim == 3:
        out = np.stack([ndimage.rotate(img[..., c], best, reshape=False, order=1, mode="nearest")
                        for c in range(img.shape[2])], axis=-1)
    else:
        out = ndimage.rotate(img, best, reshape=False, order=1, mode="nearest")
    if np.asarray(image).dtype == np.uint8:
        out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out, best
