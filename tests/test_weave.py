import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from manitex.design import Design
from manitex.procedural import PROCEDURAL_STYLES, generate
from manitex.weave import (
    PICK_GROUND,
    PICK_SUPPLEMENTARY,
    CompileOptions,
    WeaveDraft,
    compile_design,
    design_chart_svg,
    get_structure,
    read_wif,
    to_html,
    to_markdown,
    to_wif,
    yarn_requirements,
)
from manitex.weave.compiler import add_tiedowns, face_matches_design, plan_ground
from manitex.weave.draft import max_run, runs
from manitex.weave.instructions import pattern_pick_code, run_length, weaving_steps
from manitex.weave.structures import satin, twill

# --------------------------------------------------------------------------- structures


def test_structures():
    p = get_structure("plain")
    assert p.tolist() == [[1, 0], [0, 1]]
    t = get_structure("twill_2_1")
    assert t.shape == (3, 3) and t.sum(axis=1).tolist() == [2, 2, 2]
    assert t[1].tolist() == np.roll(t[0], 1).tolist()
    assert get_structure("twill_2_2_s")[1].tolist() == np.roll(twill(2, 2)[0], -1).tolist()
    for n, mv in [(5, 2), (5, 3), (8, 3), (8, 5), (7, 2)]:
        s = satin(n, mv, warp_faced=False)
        assert (s.sum(axis=0) == 1).all() and (s.sum(axis=1) == 1).all()
    assert (get_structure("satin_5_warp") == 1 - get_structure("satin_5_weft")).all()
    assert get_structure("basket_2").shape == (4, 4)
    assert get_structure("warp_rib_2").shape == (4, 2)
    with pytest.raises(ValueError):
        get_structure("satin_6_3")
    with pytest.raises(ValueError):
        get_structure("leno")


# --------------------------------------------------------------------------- runs / tie-downs


def test_runs_and_max_run():
    m = np.array([[1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]], bool)
    r, s, length = runs(m)
    assert r.tolist() == [0, 0] and s.tolist() == [0, 3] and length.tolist() == [2, 3]
    assert max_run(m, axis=1).tolist() == [3, 0]
    assert max_run(m, axis=0).tolist() == [1, 1, 0, 1, 1, 1]


@pytest.mark.parametrize("max_float", [2, 3, 5, 8, 12])
def test_add_tiedowns_properties(max_float):
    rng = np.random.default_rng(max_float)
    for _ in range(300):
        E = int(rng.integers(1, 120))
        vis = rng.random(E) < rng.uniform(0.3, 0.98)
        lift = (~vis).astype(np.uint8)
        phase = int(rng.integers(0, max_float))
        new, n = add_tiedowns(lift, vis, max_float, phase)
        added = (new == 1) & (lift == 0)
        assert added.sum() == n
        assert not added[~vis].any()  # ties only inside visible runs
        # remaining visible floats respect the limit
        still_visible = vis & (new == 0)
        assert max_run(still_visible[None], axis=1)[0] <= max_float
        # never tie the first/last end of a run; short runs untouched
        _, starts, lengths = runs(vis)
        for a, L in zip(starts, lengths):
            assert not added[a] and not added[a + L - 1]
            if L <= max_float:
                assert not added[a : a + L].any()


# --------------------------------------------------------------------------- compiler


def _stripes_weft(colors_per_row, W=12, palette=None):
    grid = np.repeat(np.asarray(colors_per_row)[:, None], W, axis=1)
    pal = palette if palette is not None else [[240, 230, 210], [160, 30, 34], [30, 30, 30], [40, 90, 60]]
    return Design(grid, pal)


def test_weft_stripes_compile_to_pure_stripes():
    rows = [0, 0, 1, 1, 0, 2, 0, 0, 3, 3, 3, 0]
    d = _stripes_weft(rows)
    dr = compile_design(d, ends_per_cell=2, picks_per_cell=2)
    assert dr.meta["ground_mode"] == "weft"
    assert (dr.pick_kind == PICK_GROUND).all()
    assert dr.n_picks == len(rows) * 2 and dr.n_ends == 24
    # bottom-up: first picks belong to the last design row
    assert dr.pick_row[0] == len(rows) - 1
    expected = np.repeat(np.asarray(rows)[::-1], 2)
    np.testing.assert_array_equal(dr.weft_colors, expected)
    # plain weave everywhere
    assert (dr.liftplan[::2] == dr.liftplan[0]).all() and (dr.liftplan[1::2] == 1 - dr.liftplan[0]).all()


def test_warp_stripes_compile_to_pure_stripes():
    cols = [0, 1, 1, 0, 0, 2, 2, 2, 0, 1]
    grid = np.repeat(np.asarray(cols)[None, :], 9, axis=0)
    d = Design(grid, [[250, 250, 250], [200, 0, 0], [0, 0, 120]])
    dr = compile_design(d, ends_per_cell=3)
    assert dr.meta["ground_mode"] == "warp"
    assert dr.summary()["supplementary_picks"] == 0
    np.testing.assert_array_equal(dr.warp_colors, np.repeat(cols, 3))


def test_brocade_motif_compile_properties():
    for style in PROCEDURAL_STYLES:
        for seed in range(6):
            d = generate(style, 48, 40, seed=seed)
            for supp in ("continuous", "discontinuous"):
                opt = CompileOptions(ends_per_cell=2, max_float=6, supplementary=supp)
                dr = compile_design(d, opt)
                plan = plan_ground(d.grid, d.n_colors, dr.meta["ground_mode"], opt.stripe_threshold,
                                   palette=d.palette)
                pattern = d.grid != plan.appearance
                assert dr.n_ends == d.width * 2
                n_pat_rows = int(pattern.any(axis=1).sum())
                extra = 1 if pattern.any() else 0  # balance_rows with ppc=1, pattern_thickness=1
                assert int((dr.pick_kind == PICK_GROUND).sum()) == n_pat_rows + (d.height - n_pat_rows) * (1 + extra)
                # every visible supplementary crossing lies on a pattern cell of the right colour
                face = dr.face()
                for p in np.flatnonzero(dr.pick_kind == PICK_SUPPLEMENTARY):
                    r = dr.pick_row[p]
                    vis = dr.liftplan[p] == 0
                    cells = dr.end_col[vis]
                    assert pattern[r, cells].all()
                    np.testing.assert_array_equal(face[p, vis], d.grid[r, cells])
                assert dr.float_report()["supp_weft_face"] <= 6
                assert face_matches_design(d, dr) == 1.0
                if not pattern.any():
                    assert (dr.pick_kind == PICK_GROUND).all()


def test_continuous_pick_count():
    grid = np.zeros((6, 10), int)
    grid[2, 2:5] = 1
    grid[2, 6:8] = 2
    grid[4, 1:9] = 1
    d = Design(grid, [[255, 255, 255], [200, 0, 0], [0, 0, 0]])
    dr = compile_design(d, ends_per_cell=1, picks_per_cell=1, max_float=4, balance_rows=False)
    # 6 ground + row2: 2 colours + row4: 1 colour
    assert dr.n_picks == 6 + 2 + 1
    assert dr.meta["ground_mode"] == "none"
    # the 8-wide run in row 4 must have tie-downs; floats <= 4
    p = [q for q in range(dr.n_picks) if dr.pick_row[q] == 4 and dr.pick_kind[q] == PICK_SUPPLEMENTARY][0]
    assert dr.liftplan[p][1:9].sum() >= 1
    assert max_run(dr.liftplan[p][None] == 0, axis=1)[0] <= 4
    dr2 = compile_design(d, ends_per_cell=1, supplementary="discontinuous", balance_rows=False)
    assert dr2.n_picks == 6 + 1 + 1
    p2 = [q for q in range(dr2.n_picks) if dr2.pick_row[q] == 2 and dr2.pick_kind[q] == PICK_SUPPLEMENTARY][0]
    assert set(dr2.inlay[p2][dr2.inlay[p2] >= 0].tolist()) == {1, 2}


def test_top_down_option():
    d = _stripes_weft([0, 1, 2, 3])
    dr = compile_design(d, bottom_up=False)
    assert dr.pick_row[0] == 0


def test_damask():
    rng = np.random.default_rng(0)
    grid = np.zeros((10, 12), int)
    grid[3:8, 2:9] = 1
    d = Design(grid, [[240, 240, 240], [120, 10, 30]])
    dr = compile_design(d, technique="damask", ends_per_cell=5, picks_per_cell=5)
    face = dr.face()
    cell_rows = dr.pick_row[:, None]
    cell_cols = dr.end_col[None, :]
    is_ground = d.grid[cell_rows, cell_cols] == 0
    assert np.isclose((face[is_ground] == 0).mean(), 0.8, atol=0.02)
    assert np.isclose((face[~is_ground] == 1).mean(), 0.8, atol=0.02)
    with pytest.raises(ValueError):
        compile_design(Design(rng.integers(0, 3, (5, 5)), np.zeros((3, 3))), technique="damask")


def test_options_validation():
    d = _stripes_weft([0, 1])
    for bad in [dict(technique="x"), dict(max_float=1), dict(ends_per_cell=0), dict(ground_stripes="diag"),
                dict(supplementary="x"), dict(tie_stagger="x")]:
        with pytest.raises(ValueError):
            compile_design(d, **bad)


# --------------------------------------------------------------------------- draft decomposition


def test_threading_treadling_exact():
    d = generate("moirang_phee", 40, 32, seed=3)
    dr = compile_design(d, ends_per_cell=2)
    threading, lifts = dr.threading()
    np.testing.assert_array_equal(lifts[:, threading], dr.liftplan)
    th2, tieup, treadling = dr.treadling()
    np.testing.assert_array_equal(tieup[treadling][:, th2], dr.liftplan)
    assert dr.n_shafts == lifts.shape[1] == np.unique(dr.liftplan.T, axis=0).shape[0]
    assert threading[0] == 0  # shafts numbered by first use


def test_draft_validation():
    with pytest.raises(ValueError):
        WeaveDraft(np.zeros((2, 3)), np.zeros(2), np.zeros(2), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        WeaveDraft(np.zeros((2, 3)), np.zeros(3), np.array([0, 5]), np.zeros((2, 3)))


# --------------------------------------------------------------------------- WIF


@pytest.mark.parametrize("treadling", [None, True, False])
def test_wif_roundtrip(treadling, tmp_path):
    d = generate("banded_motif", 30, 26, seed=7)
    dr = compile_design(d, ends_per_cell=2, max_float=5)
    text = to_wif(dr, title="t", include_treadling=treadling)
    assert text.startswith("[WIF]")
    back = read_wif(text)
    np.testing.assert_array_equal(back.liftplan, dr.liftplan)
    np.testing.assert_array_equal(back.palette[back.warp_colors], dr.palette[dr.warp_colors])
    np.testing.assert_array_equal(back.palette[back.weft_colors], dr.palette[dr.weft_colors])
    p = tmp_path / "x.wif"
    p.write_text(text)
    np.testing.assert_array_equal(read_wif(p).liftplan, dr.liftplan)


def test_wif_reader_variants():
    text = """[WIF]
Version=1.1
; a comment
[CONTENTS]
COLOR PALETTE=yes
[COLOR PALETTE]
Entries=2
Range=0,999
[COLOR TABLE]
1=999,999,999
2=0,0,999
[WEAVING]
Shafts=4
Treadles=4
Rising Shed=no
[WARP]
Threads=4
Color=1
[WEFT]
Threads=4
Color=2
[THREADING]
1=1
2=2
3=3
4=4
[TIEUP]
1=1,2
2=2,3
3=3,4
4=4,1
[TREADLING]
1=1
2=2
3=3
4=4
"""
    dr = read_wif(text)
    # sinking shed: tied shafts sink, so raised = complement
    assert dr.liftplan.tolist() == [[0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0]]
    assert dr.palette.tolist() == [[255, 255, 255], [0, 0, 255]]
    assert (dr.warp_colors == 0).all() and (dr.weft_colors == 1).all()
    with pytest.raises(ValueError):
        read_wif("[NOTWIF]\nx=1\n")


# --------------------------------------------------------------------------- instructions / chart


def test_pattern_code_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lift = (rng.random(rng.integers(1, 60)) < 0.5).astype(np.uint8)
        code = pattern_pick_code(lift)
        dec = np.concatenate([np.full(int(t[1:]), 1 if t[0] == "L" else 0) for t in code.split()])
        np.testing.assert_array_equal(dec, lift)
    assert run_length([1, 1, 2, 2, 2, 1]) == [(1, 2), (2, 3), (1, 1)]


def test_instructions_and_chart():
    d = generate("moirang_phee", 40, 36, seed=11)
    dr = compile_design(d, ends_per_cell=2, max_float=6)
    md = to_markdown(d, dr, title="Test")
    assert "## Yarns" in md and "Pick-by-pick" in md
    html = to_html(d, dr, title="Test", fabric_render=np.zeros((10, 10, 3), np.uint8))
    assert html.startswith("<!doctype html>") and "<svg" in html
    svg = design_chart_svg(d, cell=10, title="chart")
    ET.fromstring(svg)  # well-formed XML
    yarns = yarn_requirements(dr)
    assert all(y["total_m"] > 0 for y in yarns)
    assert sum(y["warp_ends"] for y in yarns) == dr.n_ends
    groups = weaving_steps(dr)
    assert groups[0]["rows"][0] == 1 and groups[-1]["rows"][1] == d.height
    total_picks = sum(1 for _ in range(dr.n_picks))
    assert groups[-1]["picks"][1] == total_picks
    # every pattern code in the markdown decodes to the right number of ends
    for code in re.findall(r": ((?:[LD]\d+ ?)+)$", md, flags=re.M):
        assert sum(int(t[1:]) for t in code.split()) == dr.n_ends


def test_wif_draft_instructions():
    d = generate("check", 20, 20, seed=1)
    dr = read_wif(to_wif(compile_design(d)))
    md = to_markdown(d, dr)
    assert re.search(r"\*\*Rows? 1\b", md)


def test_plaid_inference_and_compile():
    from manitex.weave.compiler import infer_plaid

    for seed in range(30):
        d = generate("check", 40, 36, seed=seed)
        plan = infer_plaid(d.grid, d.palette)
        assert plan is not None and plan.coverage == 1.0
        dr = compile_design(d)
        assert dr.meta["ground_mode"] == "plaid"
        assert dr.summary()["supplementary_picks"] == 0
        # the recovered stripe sequences reproduce every crossing colour
        p = d.meta["params"]
        assert set(np.unique(dr.warp_colors)) <= set(range(p["stripe_colors"]))
    # non-plaid designs are not mistaken for plaids
    for seed in range(10):
        d = generate("moirang_phee", 48, 48, seed=seed)
        assert compile_design(d).meta["ground_mode"] != "plaid"


def test_balance_rows_keeps_row_heights_uniform():
    d = generate("moirang_phee", 40, 40, seed=4)
    dr = compile_design(d, supplementary="discontinuous")  # one pattern pick per patterned row
    heights = np.zeros(d.height)
    np.add.at(heights, dr.pick_row, dr.weft_thickness)
    assert np.allclose(heights, heights[0])
    dr2 = compile_design(d, supplementary="discontinuous", balance_rows=False)
    h2 = np.zeros(d.height)
    np.add.at(h2, dr2.pick_row, dr2.weft_thickness)
    assert not np.allclose(h2, h2[0])
