import numpy as np
import pytest

from manitex.design import Design
from manitex.procedural import generate
from manitex.render import RenderOptions, render_design_cloth, render_draft
from manitex.weave import PICK_GROUND, PICK_SUPPLEMENTARY, WeaveDraft, compile_design


def test_render_shapes_and_determinism():
    d = generate("moirang_phee", 24, 20, seed=0)
    dr = compile_design(d, ends_per_cell=2)
    img = render_draft(dr, end_px=4, pick_px=4)
    heights = np.maximum(np.rint(4 * dr.weft_thickness).astype(int), 2)
    assert img.shape == (heights.sum(), dr.n_ends * 4, 3) and img.dtype == np.uint8
    np.testing.assert_array_equal(img, render_draft(dr, end_px=4, pick_px=4))
    top = render_draft(dr, end_px=4, pick_px=4, origin="top")
    np.testing.assert_array_equal(top[::-1], img)


@pytest.mark.parametrize("mode", ["auto", "balanced", "weft", "warp"])
def test_face_modes(mode):
    d = generate("phanek_mayek_naibi", 16, 16, seed=1)
    img = render_draft(compile_design(d), face_mode=mode, end_px=3, pick_px=3)
    assert np.isfinite(img).all() and img.mean() > 5
    with pytest.raises(ValueError):
        render_draft(compile_design(d), face_mode="sideways")


def test_pixel_budget():
    d = generate("banded_motif", 64, 64, seed=2)
    dr = compile_design(d, ends_per_cell=4)
    img = render_draft(dr, end_px=8, pick_px=8, max_pixels=200_000)
    assert img.shape[0] * img.shape[1] <= 260_000


def _draft_with_hidden_pattern(visible: bool):
    E, P = 16, 9
    lift = np.zeros((P, E), np.uint8)
    kinds = np.zeros(P, np.uint8)
    for p in range(P):
        if p % 3 == 2:
            kinds[p] = PICK_SUPPLEMENTARY
            lift[p] = 0 if visible else 1
        else:
            lift[p] = (np.arange(E) + p) % 2
    pal = np.array([[240, 240, 240], [255, 0, 0]], np.uint8)
    weft = np.where(kinds == PICK_SUPPLEMENTARY, 1, 0)
    return WeaveDraft(lift, np.zeros(E, int), weft, pal, pick_kind=kinds)


def test_hidden_pattern_yarn_does_not_show():
    img = render_draft(_draft_with_hidden_pattern(False), noise=0, jitter=0)
    redness = img[..., 0].astype(int) - img[..., 1].astype(int)
    assert redness.max() < 25
    img2 = render_draft(_draft_with_hidden_pattern(True), noise=0, jitter=0)
    redness2 = img2[..., 0].astype(int) - img2[..., 1].astype(int)
    assert (redness2 > 60).mean() > 0.3  # visible floats spread over a large area


def test_plain_draft_without_kinds():
    E = P = 10
    lift = (np.add.outer(np.arange(P), np.arange(E)) % 2).astype(np.uint8)
    dr = WeaveDraft(lift, np.zeros(E, int), np.ones(P, int), np.array([[0, 0, 200], [200, 200, 0]], np.uint8))
    assert (dr.pick_kind == PICK_GROUND).all()
    img = render_draft(dr, RenderOptions(end_px=4, pick_px=4))
    assert img.shape == (40, 40, 3)


def test_render_design_cloth():
    d = Design(np.eye(6, dtype=int), [[255, 255, 255], [0, 0, 0]])
    img, dr = render_design_cloth(d, ends_per_cell=2)
    assert img.ndim == 3 and dr.n_ends == 12
