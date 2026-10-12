"""Synthetic textile photographs with known ground truth (for testing and benchmarking).

A design is compiled and rendered as cloth, then photographed virtually: placed on a
background with a random perspective, blurred, unevenly lit, given sensor noise and
JPEG-compressed.  The exact cloth corners and design grid are returned, so digitisation
accuracy can be measured.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from manitex.design import Design
from manitex.digitise.geometry import apply_homography, homography, sample_bilinear
from manitex.render.fabric import render_draft
from manitex.weave.compiler import compile_design


@dataclass
class SynthPhoto:
    photo: np.ndarray
    quad: np.ndarray  # (4, 2) cloth corners in the photo: TL, TR, BR, BL
    cloth: np.ndarray  # the clean render
    design: Design
    cell_px: tuple[float, float]  # (height, width) of a design cell in the clean render


def synthetic_photo(
    design: Design,
    rng: np.random.Generator,
    ends_per_cell: int = 2,
    picks_per_cell: int = 2,
    end_px: int = 6,
    perspective: float = 0.05,
    blur_sigma: float = 0.7,
    noise_sigma: float = 3.0,
    light: float = 0.2,
    jpeg_quality: int | None = 85,
    margin: float = 0.06,
) -> SynthPhoto:
    # inlay keeps exactly one pattern pick per ground pick, so balanced rows have equal height
    draft = compile_design(design, ends_per_cell=ends_per_cell, picks_per_cell=picks_per_cell,
                           supplementary="discontinuous")
    picks_per_row = draft.weft_thickness.sum() / design.height
    pick_px = max(2, int(round(ends_per_cell * end_px / picks_per_row)))  # ~square cells
    cloth = render_draft(draft, end_px=end_px, pick_px=pick_px, seed=int(rng.integers(1 << 30)))
    ch, cw = cloth.shape[:2]
    cell_px = (ch / design.height, cw / design.width)
    m = int(round(margin * max(ch, cw)))
    Hc, Wc = ch + 2 * m, cw + 2 * m
    rect = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float64)
    jitter = rng.uniform(-perspective, perspective, size=(4, 2)) * np.array([cw, ch])
    quad = rect + m + jitter
    quad = np.clip(quad, 0, [Wc - 1, Hc - 1])
    Hm = homography(quad, rect)  # photo pixel -> cloth pixel
    yy, xx = np.mgrid[0:Hc, 0:Wc]
    src = apply_homography(Hm, np.stack([xx, yy], axis=-1).reshape(-1, 2))
    inside = (src[:, 0] >= 0) & (src[:, 0] <= cw - 1) & (src[:, 1] >= 0) & (src[:, 1] <= ch - 1)
    vals = sample_bilinear(cloth, src[:, 1], src[:, 0])
    bg = np.array(rng.uniform(40, 200, 3))
    photo = np.where(inside[:, None], vals, bg).reshape(Hc, Wc, 3)
    if blur_sigma > 0:
        photo = ndimage.gaussian_filter(photo, sigma=(blur_sigma, blur_sigma, 0))
    if light > 0:
        gy, gx = np.mgrid[0:Hc, 0:Wc] / max(Hc, Wc)
        ang = rng.uniform(0, 2 * np.pi)
        grad = 1.0 + light * ((gx - 0.5) * np.cos(ang) + (gy - 0.5) * np.sin(ang))
        photo = photo * grad[..., None]
    if noise_sigma > 0:
        photo = photo + rng.normal(0, noise_sigma, photo.shape)
    photo = np.clip(np.rint(photo), 0, 255).astype(np.uint8)
    if jpeg_quality:
        buf = io.BytesIO()
        Image.fromarray(photo).save(buf, format="JPEG", quality=int(jpeg_quality))
        photo = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return SynthPhoto(photo=photo, quad=quad, cloth=cloth, design=design, cell_px=cell_px)


def label_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Cell accuracy after the best one-to-one matching of label ids (Hungarian)."""
    if pred.shape != true.shape:
        return 0.0
    kp, kt = int(pred.max()) + 1, int(true.max()) + 1
    conf = np.zeros((kp, kt), dtype=np.int64)
    np.add.at(conf, (pred.ravel(), true.ravel()), 1)
    r, c = linear_sum_assignment(-conf)
    return float(conf[r, c].sum() / pred.size)


def _label_map(labels: np.ndarray, xs: np.ndarray, ys: np.ndarray, H: int, W: int) -> np.ndarray:
    cols = np.clip(np.searchsorted(xs, np.arange(W) + 0.5, side="right") - 1, 0, labels.shape[1] - 1)
    rows = np.clip(np.searchsorted(ys, np.arange(H) + 0.5, side="right") - 1, 0, labels.shape[0] - 1)
    return labels[rows[:, None], cols[None, :]]


def pixel_accuracy(pred: np.ndarray, xs: np.ndarray, ys: np.ndarray, true: np.ndarray, image_shape: tuple[int, int]) -> float:
    """Resolution-independent agreement: both designs are painted onto the rectified image
    (the true grid spans it uniformly) and compared pixel by pixel after label matching."""
    H, W = image_shape
    tm = _label_map(true, np.linspace(0, W, true.shape[1] + 1), np.linspace(0, H, true.shape[0] + 1), H, W)
    pm = _label_map(pred, np.asarray(xs, float), np.asarray(ys, float), H, W)
    return label_accuracy(pm, tm)
