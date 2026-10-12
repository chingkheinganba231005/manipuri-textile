"""Photo -> clean design grid.

Pipeline: (optional) perspective rectification from four corners -> (optional) deskew ->
illumination flattening -> grid fit -> robust cell colours -> CIELAB k-means with
automatic colour count -> Potts denoising -> stripe and repeat/mirror orbit pooling ->
:class:`~manitex.design.Design`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from manitex.color import (
    YARN_COLORS,
    delta_e2000,
    hex_to_rgb,
    lab_to_srgb,
    linear_to_srgb,
    srgb_to_lab,
    srgb_to_linear,
)
from manitex.design import Design
from manitex.digitise.geometry import deskew, warp_perspective
from manitex.digitise.grid import GridFit, fit_grid, is_clean_image, sample_cells
from manitex.digitise.quantize import choose_k, class_costs, kmeans, merge_clusters, potts_icm
from manitex.digitise.structure import reduce_lossless, regularize_stripes, regularize_structure


@dataclass
class DigitiseOptions:
    quad: list | None = None  # 4 (x, y) corners of the cloth in the photo (any order)
    deskew: bool = False
    max_side: int = 1600  # photos are downscaled to this size first
    cells_x: int | None = None  # number of design cells across (auto if None)
    cells_y: int | None = None
    min_cell_px: float = 3.0
    max_cell_px: float | None = None
    elastic: bool = True
    square_cells: bool = False  # prefer (nearly) square cells when auto-fitting
    n_colors: int | None = None  # auto if None
    k_max: int = 8
    min_delta_e: float | None = None  # colour distinctness threshold; adapts to the noise level if None
    flatten_light: bool = True
    beta: float = 0.8  # Potts smoothness (in units of colour-noise standard deviations)
    stripes: bool = True
    repeats: bool = True
    mirror: bool = True
    merge_duplicates: bool = True  # coarsen an over-fine automatic grid when cells repeat exactly
    snap_to_yarns: bool = False
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DigitiseResult:
    design: Design
    grid: GridFit
    cell_lab: np.ndarray
    image: np.ndarray  # the rectified image the grid refers to
    report: dict[str, Any]


def _luminance(lab: np.ndarray) -> np.ndarray:
    L = lab[..., 0]
    return np.where(L > 8.0, ((L + 16.0) / 116.0) ** 3, L / 903.3)


def correct_illumination(cell_lab: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                         trim: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Remove a smooth multiplicative lighting field from cell colours.

    The gain is fitted (quadratic surface, trimmed least squares) to the luminance ratios
    between each cell and its colour cluster, so the design itself cannot bias it.
    Returns (corrected cell Lab, gain field).
    """
    ny, nx = labels.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = (xx.ravel() / max(nx - 1, 1)) * 2 - 1
    y = (yy.ravel() / max(ny - 1, 1)) * 2 - 1
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    Yc = np.maximum(_luminance(cell_lab.reshape(-1, 3)), 1e-4)
    Yk = np.maximum(_luminance(centers), 1e-4)[labels.ravel()]
    r = np.log(Yc / Yk)
    keep = np.ones(r.size, dtype=bool)
    coef = np.zeros(A.shape[1])
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], r[keep], rcond=None)
        res = np.abs(r - A @ coef)
        keep = res <= np.quantile(res, 1.0 - trim)
    gain = np.exp(A @ coef - np.median(A @ coef)).reshape(ny, nx)
    lin = srgb_to_linear(lab_to_srgb(cell_lab, clip=False).clip(0, 255)) / gain[..., None]
    return srgb_to_lab(linear_to_srgb(lin)), gain


def pixel_noise(img: np.ndarray) -> float:
    """Median absolute deviation of lightness from its 3x3 median: ~0 for clean pixel-art
    designs, clearly positive for photographs (sensor noise, yarn texture)."""
    from scipy import ndimage

    L = srgb_to_lab(np.asarray(img)[..., :3].astype(np.float64))[..., 0]
    return float(np.median(np.abs(L - ndimage.median_filter(L, size=3))))


def digitise(image: np.ndarray | Image.Image, options: DigitiseOptions | None = None, **kw: Any) -> DigitiseResult:
    """Digitise a photograph (or scan) of a textile into a :class:`Design`."""
    opt = options or DigitiseOptions()
    if kw:
        opt = DigitiseOptions(**{**asdict(opt), **kw})
    img = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3].astype(np.uint8)
    report: dict[str, Any] = {"input_shape": list(img.shape[:2])}
    quad = np.asarray(opt.quad, dtype=np.float64).reshape(4, 2) if opt.quad is not None else None
    scale = 1.0
    if max(img.shape[:2]) > opt.max_side:
        scale = opt.max_side / max(img.shape[:2])
        pil = Image.fromarray(img).resize((max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
                                          Image.LANCZOS)
        img = np.asarray(pil)
        if quad is not None:
            quad = quad * scale
    if quad is not None:
        img = warp_perspective(img, quad)
    if opt.deskew:
        img, angle = deskew(img)
        report["deskew_deg"] = angle
    grid = fit_grid(img, opt.cells_x, opt.cells_y, opt.min_cell_px, opt.max_cell_px, opt.elastic, opt.square_cells)
    ny, nx = grid.shape
    report.update({"cells": [ny, nx], "cell_px": [round(grid.cell_h, 3), round(grid.cell_w, 3)],
                   "grid_score": [round(grid.score_y, 4), round(grid.score_x, 4)]})
    cell_lab = sample_cells(img, grid)
    rng = np.random.default_rng(opt.seed)
    noise = pixel_noise(img)
    clean = is_clean_image(img)  # a clean pixel-art input: never "regularise" it
    min_de = opt.min_delta_e if opt.min_delta_e is not None else (3.5 if clean else 9.0)
    report["pixel_noise"] = round(noise, 3)
    report["min_delta_e"] = round(float(min_de), 2)
    if opt.flatten_light and ny * nx >= 16 and not clean:
        # coarse clusters (lighting may split colours) -> fit and remove the lighting field
        k0 = int(opt.n_colors) if opt.n_colors else None
        for _ in range(2):
            Xc = cell_lab.reshape(-1, 3)
            if k0:
                c0, l0, _ = kmeans(Xc, k0, rng)
            else:
                _, c0, l0 = choose_k(Xc, opt.k_max, rng, max(min_de * 1.6, 14.0))
            cell_lab, gain = correct_illumination(cell_lab, l0.reshape(ny, nx), c0)
        report["light_gain_range"] = [round(float(gain.min()), 3), round(float(gain.max()), 3)]
    X = cell_lab.reshape(-1, 3)
    if opt.n_colors:
        k = int(opt.n_colors)
        centers, labels, _ = kmeans(X, k, rng)
    else:
        k, centers, labels = choose_k(X, opt.k_max, rng, min_de)
    labels = labels.reshape(ny, nx)
    merge_log: list[str] = []
    if not opt.n_colors:
        labels, centers, merge_log = merge_clusters(labels, X, centers, min_de)
        k = centers.shape[0]
    cost = class_costs(X, centers).reshape(ny, nx, k)
    stats: dict[str, Any] = {"k": int(k), "merges": merge_log}
    lab0 = labels.copy()
    labels = potts_icm(cost, 0.0 if clean else opt.beta, labels=labels)
    stats["potts_changed"] = int((labels != lab0).sum())
    if opt.stripes and not clean:
        labels, c1 = regularize_stripes(labels, cost, axis=0)
        labels, c2 = regularize_stripes(labels, cost, axis=1)
        stats["stripe_changed"] = int(c1 + c2)
    if (opt.repeats or opt.mirror) and not clean:
        labels, sinfo = regularize_structure(labels, cost, repeats=opt.repeats, mirror=opt.mirror)
        stats["structure"] = sinfo
    if opt.merge_duplicates and not (opt.cells_x or opt.cells_y):
        labels, xs, ys, rlog = reduce_lossless(labels, grid.xs, grid.ys)
        if rlog:
            grid = GridFit(xs=xs, ys=ys, cell_w=float(np.median(np.diff(xs))), cell_h=float(np.median(np.diff(ys))),
                           score_x=grid.score_x, score_y=grid.score_y)
            cell_lab = sample_cells(img, grid)
            X = cell_lab.reshape(-1, 3)
            ny, nx = labels.shape
            report["cells"] = [ny, nx]
            report["cell_px"] = [round(grid.cell_h, 3), round(grid.cell_w, 3)]
        stats["grid_merges"] = rlog
    # refine colours as the mean of the cells finally assigned to each label
    for j in range(k):
        m = labels.reshape(-1) == j
        if m.any():
            centers[j] = X[m].mean(axis=0)
    palette = np.clip(np.rint(lab_to_srgb(centers)), 0, 255).astype(np.uint8)
    if opt.snap_to_yarns:
        yarn_rgb = np.array([hex_to_rgb(h) for h in YARN_COLORS.values()], dtype=np.float64)
        yarn_lab = srgb_to_lab(yarn_rgb)
        for j in range(k):
            d = delta_e2000(yarn_lab, np.broadcast_to(centers[j], yarn_lab.shape))
            palette[j] = yarn_rgb[int(np.argmin(d))].astype(np.uint8)
    resid = np.sqrt(((X - centers[labels.reshape(-1)]) ** 2).sum(axis=1))
    stats["mean_residual_lab"] = round(float(resid.mean()), 3)
    report.update(stats)
    design = Design(labels.astype(np.int32), palette, {"source": "digitised", "digitise": report}).canonical()
    return DigitiseResult(design=design, grid=grid, cell_lab=cell_lab, image=img, report=report)
