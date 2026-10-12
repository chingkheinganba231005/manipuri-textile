#!/usr/bin/env python
"""Generate paper figures (requires matplotlib).

    python scripts/make_figures.py --out paper/figures [--checkpoint exports/manitex-v1]

Without a checkpoint the procedural, compiler, renderer, orbit and digitisation figures are made;
with a checkpoint, model samples, structural control and inpainting figures are added.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from manitex.constraints import Symmetry, make_orbits, orbit_index  # noqa: E402
from manitex.digitise import digitise  # noqa: E402
from manitex.digitise.synth import synthetic_photo  # noqa: E402
from manitex.procedural import PROCEDURAL_STYLES, generate  # noqa: E402
from manitex.render import render_draft  # noqa: E402
from manitex.weave import compile_design  # noqa: E402
from manitex.weave.draft import PICK_SUPPLEMENTARY  # noqa: E402

LABELS = {
    "phanek_mayek_naibi": "Phanek mayek naibi",
    "moirang_phee": "Moirang phee",
    "banded_motif": "Banded motif",
    "check": "Check",
    "motif_tile": "Motif tile",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", path.with_suffix(".pdf"))


def fig_styles(out: Path, per_style: int = 5) -> None:
    fig, axes = plt.subplots(len(PROCEDURAL_STYLES), per_style, figsize=(2.0 * per_style, 2.0 * len(PROCEDURAL_STYLES)))
    for r, style in enumerate(PROCEDURAL_STYLES):
        for c in range(per_style):
            d = generate(style, 64, 64, seed=100 * r + c)
            ax = axes[r, c]
            ax.imshow(d.to_rgb(), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(LABELS[style], fontsize=9)
    fig.suptitle("Procedural grammar samples (64 x 64 design cells)", fontsize=11)
    _save(fig, out / "fig_procedural_styles")


def fig_pipeline(out: Path) -> None:
    d = generate("moirang_phee", 40, 56, seed=2)
    dr = compile_design(d, ends_per_cell=2, max_float=8)
    face = dr.palette[dr.face()][::-1]  # pick 1 at the bottom
    lift = dr.liftplan[::-1]
    kinds = (dr.pick_kind == PICK_SUPPLEMENTARY)[::-1]
    cloth = render_draft(dr, end_px=5, pick_px=5)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(d.to_rgb(), interpolation="nearest")
    axes[0].set_title("(a) design grid")
    lp = np.where(lift[..., None] == 1, [40, 40, 40], [245, 245, 245]).astype(np.uint8)
    lp[kinds] = np.where(lift[kinds][..., None] == 1, [150, 30, 40], [250, 215, 215])
    axes[1].imshow(lp, interpolation="nearest", aspect="auto")
    axes[1].set_title("(b) lift plan (red rows: pattern picks)")
    axes[2].imshow(face, interpolation="nearest", aspect="auto")
    axes[2].set_title("(c) drawdown (face)")
    axes[3].imshow(cloth)
    axes[3].set_title("(d) simulated cloth")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, out / "fig_compiler_pipeline")


def fig_tiedowns(out: Path) -> None:
    from manitex.design import Design

    grid = np.zeros((8, 36), int)
    grid[2:6, 4:32] = 1  # a 28-cell (56-end) wide motif block on a cream ground
    d = Design(grid, [[240, 232, 214], [166, 30, 34]])
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2))
    for ax, mf, title in ((axes[0], 128, "no tie-downs: pattern floats of 56 ends"),
                          (axes[1], 6, "tie-downs (max float 6 ends, brick stagger): ground shows through as dots")):
        dr = compile_design(d, ends_per_cell=2, max_float=mf, balance_rows=False, ground_stripes="none",
                            ground_color=0)
        face = dr.palette[dr.face()][::-1]
        ax.imshow(face, interpolation="nearest", aspect="auto")
        ax.set_title(f"{title} (max float on face: {dr.float_report()['supp_weft_face']})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, out / "fig_tiedowns")


def fig_orbits(out: Path) -> None:
    H, W = 16, 24
    cases = [("none", None), ("mirror L-R", Symmetry(mirror_x=True)), ("repeat 8", Symmetry(period_x=8)),
             ("mirror + repeat 8", Symmetry(mirror_x=True, period_x=8)), ("row stripes", Symmetry(stripes="rows"))]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 2.6))
    for ax, (name, sym) in zip(axes, cases):
        idx, n = orbit_index(make_orbits(H, W, sym))
        colours = rng.random((n, 3)) * 0.8 + 0.1
        ax.imshow(colours[idx], interpolation="nearest")
        ax.set_title(f"{name}: {n} orbits", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Orbits (cells forced to share one colour) for different constraints", fontsize=10)
    _save(fig, out / "fig_orbits")


def fig_digitise(out: Path) -> None:
    d = generate("moirang_phee", 32, 40, seed=101)
    sp = synthetic_photo(d, np.random.default_rng(1))
    res = digitise(sp.photo, quad=sp.quad, cells_x=d.width, cells_y=d.height)
    ov = res.image.copy()
    for x in res.grid.xs.astype(int):
        ov[:, np.clip(x, 0, ov.shape[1] - 1)] = (255, 0, 200)
    for y in res.grid.ys.astype(int):
        ov[np.clip(y, 0, ov.shape[0] - 1), :] = (255, 0, 200)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, img, t in zip(axes, [sp.photo, ov, res.design.to_rgb(), d.to_rgb()],
                          ["(a) photo", "(b) rectified + grid", "(c) digitised", "(d) ground truth"]):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, out / "fig_digitise")


def fig_model(out: Path, checkpoint: str) -> None:
    from manitex.generate import GenerationOptions, TextileGenerator

    gen = TextileGenerator.from_pretrained(checkpoint)
    styles = gen.styles
    opts = GenerationOptions(steps=24, cfg=1.5)
    fig, axes = plt.subplots(len(styles), 6, figsize=(12, 2.0 * len(styles)), squeeze=False)
    for r, style in enumerate(styles):
        ds = gen.generate(style, n=6, height=64, width=64, seed=r, options=opts)
        for c, d in enumerate(ds):
            axes[r, c].imshow(d.to_rgb(), interpolation="nearest")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(LABELS.get(style, style), fontsize=9)
    _save(fig, out / "fig_model_samples")
    cases = [("unconstrained", None), ("mirror L-R", Symmetry(mirror_x=True)), ("repeat 16", Symmetry(period_x=16)),
             ("mirror + repeat 16", Symmetry(mirror_x=True, period_x=16))]
    fig, axes = plt.subplots(1, len(cases), figsize=(12, 3.2))
    for ax, (name, sym) in zip(axes, cases):
        d = gen.generate("moirang_phee", n=1, height=64, width=64, symmetry=sym, seed=7, options=opts)[0]
        ax.imshow(d.to_rgb(), interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, out / "fig_model_control")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    fig_styles(out)
    fig_pipeline(out)
    fig_tiedowns(out)
    fig_orbits(out)
    fig_digitise(out)
    if args.checkpoint:
        fig_model(out, args.checkpoint)


if __name__ == "__main__":
    main()
