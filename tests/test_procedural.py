import json

import numpy as np
import pytest

from manitex.procedural import PROCEDURAL_STYLES, generate
from manitex.procedural import bands as B
from manitex.procedural import motifs as M


def test_triangle_shapes():
    t = M.triangle(7, 1.0)
    assert t.shape == (4, 7)
    assert t[-1].all() and t[0].sum() == 1 and t[0, 3]
    # symmetric staircase: each level removes one cell per side
    assert t.sum(axis=1).tolist() == [1, 3, 5, 7]
    t2 = M.triangle(6, 2.0)
    assert t2.shape == (6, 6) and t2.sum(axis=1).tolist() == [2, 2, 4, 4, 6, 6]
    t3 = M.triangle(9, 0.5)  # squat: some levels have zero rows
    assert t3.sum(axis=1).tolist() == [1, 5, 9]  # 9 -> 5 -> 1
    for base in range(1, 20):
        for rise in (0.5, 1.0, 1.5, 2.0, 3.0):
            t = M.triangle(base, rise)
            assert t.shape[1] == base
            np.testing.assert_array_equal(t, t[:, ::-1])  # mirror symmetric


def test_rings_and_erosion():
    d = M.diamond(9, 1.0)
    lv = M.erosion_levels(d)
    assert lv.max() >= 3 and (lv > 0).sum() == d.sum()
    r = M.rings(d, n_slots=2)
    assert set(np.unique(r)) <= {0, 1, 2}


@pytest.mark.parametrize("variant", M.TEMPLE_VARIANTS)
def test_temple_variants(variant):
    rng = np.random.default_rng(0)
    m = M.temple(11, 2.0, variant, rng)
    assert m.shape == (12, 11)
    assert m.min() == 0 and 1 <= m.max() <= 2


def test_motif_row_centered_symmetry():
    m = M.temple(7, 1.0, "solid")
    row = B.motif_row(41, m, [[1]], 0, gap=2)
    np.testing.assert_array_equal(row, row[:, ::-1])
    assert row.shape == (m.shape[0], 41)


@pytest.mark.parametrize("style", PROCEDURAL_STYLES)
@pytest.mark.parametrize("size", [(8, 8), (16, 24), (64, 64), (97, 131)])
def test_generate_valid(style, size):
    for seed in range(5):
        d = generate(style, *size, seed=seed)
        assert d.grid.shape == size
        assert 0 <= d.grid.min() and d.grid.max() < d.n_colors
        assert len(d.used_colors()) >= 1
        json.dumps(d.meta)
        assert d.meta["style"] == style and d.meta["source"] == "procedural"


@pytest.mark.parametrize("style", PROCEDURAL_STYLES)
def test_generate_deterministic(style):
    a = generate(style, 48, 40, seed=123)
    b = generate(style, 48, 40, seed=123)
    np.testing.assert_array_equal(a.grid, b.grid)
    np.testing.assert_array_equal(a.palette, b.palette)


def test_n_colors_respected():
    for style in PROCEDURAL_STYLES:
        if style == "check":
            continue
        for k in (2, 3, 5):
            for seed in range(5):
                d = generate(style, 64, 64, seed=seed, n_colors=k)
                assert len(d.used_colors()) <= k


def test_phanek_rows_mostly_uniform():
    # striped phanek bodies are row-constant except for optional border motifs
    for seed in range(10):
        d = generate("phanek_mayek_naibi", 64, 64, seed=seed)
        uniform = np.mean([len(np.unique(r)) == 1 for r in d.grid])
        assert uniform > 0.6


def test_moirang_has_motif_rows():
    for seed in range(10):
        d = generate("moirang_phee", 96, 96, seed=seed)
        nonuniform = np.mean([len(np.unique(r)) > 1 for r in d.grid])
        assert nonuniform > 0.03


def test_user_palette():
    d = generate("moirang_phee", 32, 32, seed=0, palette=["#ffffff", "#ff0000"])
    assert tuple(d.palette[0]) == (255, 255, 255)
    with pytest.raises(ValueError):
        generate("nope", 32, 32)
    with pytest.raises(ValueError):
        generate("check", 4, 32)
