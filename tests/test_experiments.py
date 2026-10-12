import numpy as np
import pytest

from manitex.constraints import Symmetry
from manitex.design import MAX_COLORS, Design
from manitex.experiments import crop_design, parse_symmetry, reference_designs, reference_grids, write_tables
from manitex.procedural import generate


def test_parse_symmetry():
    assert parse_symmetry("none") == Symmetry()
    assert parse_symmetry("mirror_x+period_x=16") == Symmetry(mirror_x=True, period_x=16)
    assert parse_symmetry(" stripes=rows ").stripes == "rows"
    with pytest.raises(ValueError):
        parse_symmetry("spiral")


def test_crop_design_wraps_and_caps_colours():
    rng = np.random.default_rng(0)
    g = np.arange(20 * 20).reshape(20, 20) % 30  # 30 colours on a 20 x 20 design
    pal = rng.integers(0, 256, (30, 3)).astype(np.uint8)
    c = crop_design(Design(g, pal), 24, rng)  # larger than the design: wraps around
    assert c.shape == (24, 24) and c.n_colors == MAX_COLORS and int(c.grid.max()) == MAX_COLORS - 1
    counts = np.bincount(c.grid.ravel())
    assert (np.diff(counts) <= 0).all()  # canonical order


def test_reference_designs(tmp_path):
    from manitex.data import read_shards, write_shard

    ds = [generate("check", 30, 40, seed=i) for i in range(3)] + [generate("moirang_phee", 30, 40, seed=9)]
    shards = read_shards([write_shard(tmp_path / "s.npz", ds, styles=["check"] * 3 + ["moirang_phee"])])
    refs = reference_designs("check", 5, 16, shards, seed=1)
    assert len(refs) == 5 and all(r.shape == (16, 16) for r in refs)
    grids = reference_grids("check", 5, 16, shards, seed=1)
    assert grids[0].dtype == np.int64 and all((g == r.grid).all() for g, r in zip(grids, refs))
    with pytest.raises(ValueError):
        reference_designs("phanek_mayek_naibi", 1, 16, shards)
    assert [d.shape for d in reference_designs("check", 2, 16)] == [(16, 16)] * 2  # procedural


def test_write_tables(tmp_path):
    js, md = write_tables({"style a": {"x": {"m": 0.5, "n": None}}}, tmp_path / "t", ["m", "n"], row_label="method")
    assert js.exists() and "| method | m | n |" in md.read_text() and "| x | 0.500 |  |" in md.read_text()


def test_model_nll_is_seeded():
    pytest.importorskip("torch")
    from manitex.experiments import model_nll
    from manitex.models import build_model, make_config

    m = build_model(make_config("tiny")).eval()
    grids = [np.random.default_rng(i).integers(0, 3, (8, 8)) for i in range(3)]
    a, b = model_nll(m, grids, style=1, batch=2), model_nll(m, grids, style=1, batch=2)
    assert a.shape == (3,) and np.allclose(a, b) and (a > 0).all()
