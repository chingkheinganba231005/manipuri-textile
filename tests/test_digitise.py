import numpy as np
import pytest

from manitex.design import Design
from manitex.digitise import digitise
from manitex.digitise.geometry import apply_homography, homography, order_quad, warp_perspective
from manitex.digitise.grid import elastic_boundaries_fixed, fit_grid
from manitex.digitise.quantize import choose_k, kmeans, potts_icm
from manitex.digitise.structure import detect_bands, reduce_lossless, regularize_structure
from manitex.digitise.synth import label_accuracy, pixel_accuracy, synthetic_photo
from manitex.procedural import generate


def test_homography_roundtrip():
    src = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], float)
    dst = np.array([[3, 2], [20, 4], [18, 15], [1, 12]], float)
    H = homography(src, dst)
    np.testing.assert_allclose(apply_homography(H, src), dst, atol=1e-9)
    q = order_quad(dst[[2, 0, 3, 1]])
    np.testing.assert_allclose(q, dst)


def test_warp_perspective_recovers_image():
    rng = np.random.default_rng(0)
    d = generate("moirang_phee", 20, 24, seed=1)
    img = d.to_rgb(cell=8)
    h, w = img.shape[:2]
    # embed with a known homography, then rectify
    quad = np.array([[12, 9], [w + 20, 4], [w + 25, h + 18], [6, h + 10]], float)
    rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    Hm = homography(quad, rect)
    canvas_h, canvas_w = h + 40, w + 40
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w]
    src = apply_homography(Hm, np.stack([xx, yy], -1).reshape(-1, 2))
    from manitex.digitise.geometry import sample_bilinear

    photo = np.clip(sample_bilinear(img, src[:, 1], src[:, 0]).reshape(canvas_h, canvas_w, 3), 0, 255).astype(np.uint8)
    back = warp_perspective(photo, quad, out_size=(w, h))
    assert np.abs(back.astype(int) - img.astype(int)).mean() < 12
    del rng


def test_kmeans_and_choose_k():
    rng = np.random.default_rng(0)
    centers = np.array([[20, 0, 0], [60, 40, 20], [85, -20, 30], [50, 0, -40]], float)
    X = np.concatenate([c + rng.normal(0, 1.5, (200, 3)) for c in centers])
    k, c, lab = choose_k(X, 8, rng)
    assert k == 4
    cc, ll, inertia = kmeans(X, 4, rng)
    assert inertia / X.shape[0] < 10


def test_potts_keeps_lines_removes_specks():
    K = 2
    truth = np.zeros((20, 20), int)
    truth[10, :] = 1  # a one-cell line must survive
    cost = np.where(truth[..., None] == np.arange(K), 0.0, 6.0)
    noisy = cost.copy()
    noisy[3, 3] = [0.8, 0.0]  # weak isolated speck of colour 1
    lab = potts_icm(noisy, beta=0.8)
    assert lab[10].sum() == 20 and lab[3, 3] == 0


def test_structure_regularisation_fixes_repeats():
    rng = np.random.default_rng(1)
    tile = np.array([[0, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    clean = np.tile(tile, (1, 8))
    K = 2
    cost = np.where(clean[..., None] == np.arange(K), 0.0, 3.0) + rng.normal(0, 0.2, clean.shape + (K,))
    noisy_labels = clean.copy()
    flips = [(0, 7), (1, 20), (0, 33)]
    for r, c in flips:
        noisy_labels[r, c] = 1 - noisy_labels[r, c]
        cost[r, c] = cost[r, c, ::-1] * 0.4  # misleading evidence at the flipped cells
    bands = detect_bands(noisy_labels, min_score=0.85)
    assert bands and bands[0][2] == 6
    fixed, info = regularize_structure(noisy_labels, cost, min_score=0.85)
    np.testing.assert_array_equal(fixed, clean)


def test_reduce_lossless():
    base = generate("motif_tile", 12, 14, seed=0).grid
    fine = np.repeat(np.repeat(base, 2, axis=0), 2, axis=1)
    xs = np.arange(fine.shape[1] + 1) * 3.0
    ys = np.arange(fine.shape[0] + 1) * 3.0
    red, xs2, ys2, log = reduce_lossless(fine, xs, ys)
    np.testing.assert_array_equal(red, base)
    assert len(xs2) == base.shape[1] + 1 and len(ys2) == base.shape[0] + 1


def test_fixed_count_elastic():
    p = np.zeros(100)
    p[[10, 20, 31, 40, 50, 61, 70, 80, 90]] = 1.0
    b = elastic_boundaries_fixed(p, 10)
    assert len(b) == 11 and b[0] == 0 and b[-1] == 100
    assert {31, 61} <= set(b.astype(int).tolist())


@pytest.mark.parametrize("style", ["moirang_phee", "phanek_mayek_naibi", "motif_tile"])
def test_clean_design_assisted_is_exact(style):
    d = generate(style, 30, 36, seed=3)
    r = digitise(d.to_rgb(cell=7), cells_x=d.width, cells_y=d.height)
    assert r.design.shape == d.shape
    assert label_accuracy(r.design.grid, d.canonical().grid) == 1.0
    # colours are recovered closely
    assert np.abs(r.design.palette.astype(int) - d.canonical().palette.astype(int)).max() <= 3


def test_clean_design_auto_grid():
    d = generate("moirang_phee", 30, 36, seed=0)
    r = digitise(d.to_rgb(cell=8))
    pa = pixel_accuracy(r.design.grid, r.grid.xs, r.grid.ys, d.canonical().grid, r.image.shape[:2])
    assert pa > 0.97


def test_synthetic_photo_digitisation():
    d = generate("moirang_phee", 32, 40, seed=101)
    sp = synthetic_photo(d, np.random.default_rng(1))
    assert sp.photo.ndim == 3 and sp.quad.shape == (4, 2)
    r = digitise(sp.photo, quad=sp.quad, cells_x=d.width, cells_y=d.height)
    assert label_accuracy(r.design.grid, d.canonical().grid) > 0.95
    r2 = digitise(sp.photo, quad=sp.quad)
    pa = pixel_accuracy(r2.design.grid, r2.grid.xs, r2.grid.ys, d.canonical().grid, r2.image.shape[:2])
    assert pa > 0.85
    assert r2.design.meta["source"] == "digitised"


def test_fit_grid_regular_lattice():
    d = generate("banded_motif", 24, 30, seed=2)
    g = fit_grid(d.to_rgb(cell=9))
    assert abs(g.cell_w - 9) < 0.6 or abs(g.cell_w - 18) < 1.0  # equivalent coarser grids are allowed
    assert g.shape[0] >= 12


def test_digitise_accepts_pil_and_grayscale():
    from PIL import Image

    d = Design(np.kron(np.eye(4, dtype=int), np.ones((3, 3), int)), [[255, 255, 255], [0, 0, 0]])
    img = Image.fromarray(d.to_rgb(cell=6))
    r = digitise(img, cells_x=12, cells_y=12)
    assert r.design.shape == (12, 12)
    r2 = digitise(np.asarray(img.convert("L")), cells_x=12, cells_y=12)
    assert r2.design.shape == (12, 12)
