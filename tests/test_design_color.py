import numpy as np
import pytest

from manitex.color import (
    PALETTE_PRESETS,
    delta_e2000,
    extend_palette,
    hex_to_rgb,
    lab_to_srgb,
    palette_for_style,
    rgb_to_hex,
    srgb_to_lab,
    yarn_name,
)
from manitex.design import Design, canonical_order, canonicalize_grid

# --------------------------------------------------------------------------- colour


def test_hex_roundtrip():
    for h in ["#000000", "#ffffff", "#a61e22", "#1f4d3a"]:
        assert rgb_to_hex(hex_to_rgb(h)) == h
    assert hex_to_rgb("#abc") == (0xAA, 0xBB, 0xCC)
    with pytest.raises(ValueError):
        hex_to_rgb("#12345")


def test_lab_roundtrip_and_reference_values():
    rgb = np.random.default_rng(0).integers(0, 256, size=(500, 3))
    back = lab_to_srgb(srgb_to_lab(rgb))
    assert np.abs(back - rgb).max() < 1e-6 * 255 + 1e-6
    # D65 white -> L=100, a=b=0
    np.testing.assert_allclose(srgb_to_lab(np.array([255, 255, 255])), [100.0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(srgb_to_lab(np.array([0, 0, 0])), [0.0, 0.0, 0.0], atol=1e-6)
    # sRGB red in CIELAB (D65), well-known value ~ (53.24, 80.09, 67.20)
    np.testing.assert_allclose(srgb_to_lab(np.array([255, 0, 0])), [53.24, 80.09, 67.20], atol=0.05)


# Test pairs from Sharma, Wu & Dalal (2005), "The CIEDE2000 color-difference formula".
SHARMA = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
]


@pytest.mark.parametrize("lab1,lab2,expected", SHARMA)
def test_ciede2000_reference(lab1, lab2, expected):
    assert delta_e2000(np.array(lab1), np.array(lab2)) == pytest.approx(expected, abs=1e-4)


def test_ciede2000_vectorised_and_symmetric():
    a = np.array([p[0] for p in SHARMA])
    b = np.array([p[1] for p in SHARMA])
    np.testing.assert_allclose(delta_e2000(a, b), delta_e2000(b, a), atol=1e-9)
    np.testing.assert_allclose(delta_e2000(a, a), 0.0, atol=1e-9)


def test_yarn_names():
    assert yarn_name((255, 255, 255)) in {"white", "ivory", "cream"}
    assert yarn_name((0, 0, 0)) == "black"
    assert yarn_name((180, 30, 40)) in {"red", "crimson", "scarlet"}


def test_palettes():
    rng = np.random.default_rng(1)
    for style in ["moirang_phee", "phanek_mayek_naibi", "check", "nonexistent"]:
        for k in [2, 3, 5, 9]:
            pal, name = palette_for_style(style, k, rng)
            assert pal.shape == (k, 3) and pal.dtype == np.uint8
            assert name in PALETTE_PRESETS
    ext = extend_palette(np.array([[255, 255, 255]], np.uint8), 8, rng)
    assert ext.shape == (8, 3)
    lab = srgb_to_lab(ext.astype(float))
    d = np.linalg.norm(lab[:, None] - lab[None], axis=-1) + np.eye(8) * 1e9
    assert d.min() > 5.0  # colours are distinguishable


# --------------------------------------------------------------------------- design


def test_canonical_order_frequency_then_first_occurrence():
    g = np.array([[3, 3, 1], [1, 2, 3]])
    # counts: 3->3, 1->2, 2->1 ; colour 0 unused
    assert canonical_order(g).tolist() == [3, 1, 2]
    cg, order = canonicalize_grid(g)
    assert cg.tolist() == [[0, 0, 1], [1, 2, 0]]
    # tie: 5 and 7 both twice; 7 occurs first
    g2 = np.array([[7, 5, 7, 5]])
    assert canonical_order(g2).tolist() == [7, 5]


def test_design_validation():
    with pytest.raises(ValueError):
        Design(np.zeros((2, 2), int), np.zeros((0, 3)))
    with pytest.raises(ValueError):
        Design(np.array([[0, 2]]), np.zeros((2, 3)))
    with pytest.raises(TypeError):
        Design(np.zeros((2, 2), float), np.zeros((2, 3)))
    d = Design([[0, 1], [1, 0]], ["#ffffff", "#000000"])
    assert d.shape == (2, 2) and d.n_colors == 2


def _random_design(rng, h=13, w=17, k=5):
    grid = rng.integers(0, k, size=(h, w))
    pal = rng.integers(0, 256, size=(k, 3))
    return Design(grid, pal, {"style": "test", "n": 3})


@pytest.mark.parametrize("suffix", [".png", ".json", ".npz"])
def test_design_io_roundtrip(tmp_path, suffix):
    rng = np.random.default_rng(0)
    d = _random_design(rng)
    p = d.save(tmp_path / f"d{suffix}")
    d2 = Design.load(p)
    np.testing.assert_array_equal(d.grid, d2.grid)
    np.testing.assert_array_equal(d.palette, d2.palette)
    assert d2.meta == d.meta


def test_design_png_upscaled_roundtrip(tmp_path):
    d = _random_design(np.random.default_rng(1))
    p = d.save(tmp_path / "big.png", cell=6)
    d2 = Design.load(p)
    np.testing.assert_array_equal(d.grid, d2.grid)


def test_from_rgb_and_plain_png(tmp_path):
    from PIL import Image

    d = _random_design(np.random.default_rng(2), k=4).canonical()
    Image.fromarray(d.to_rgb(cell=1)).save(tmp_path / "plain.png")
    d2 = Design.load(tmp_path / "plain.png")
    np.testing.assert_array_equal(d2.canonical().to_rgb(), d.to_rgb())
    noisy = np.random.default_rng(0).integers(0, 256, (40, 40, 3)).astype(np.uint8)
    with pytest.raises(ValueError, match="digitise"):
        Design.from_rgb(noisy)


def test_design_transforms():
    d = _random_design(np.random.default_rng(3))
    assert d.flip_lr().flip_lr().grid.tolist() == d.grid.tolist()
    assert d.transpose().shape == (d.width, d.height)
    assert d.tile(2, 3).shape == (2 * d.height, 3 * d.width)
    c = d.canonical()
    np.testing.assert_array_equal(c.to_rgb(), d.to_rgb())
    counts = c.counts()
    assert np.all(np.diff(counts) <= 0)
    img = d.to_image(cell=8, grid_lines=True)
    assert img.size == (d.width * 8, d.height * 8)
