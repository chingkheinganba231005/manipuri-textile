#!/usr/bin/env python
"""Digitisation accuracy on real photographs with hand-corrected ground truth (experiment E7).

    python scripts/eval_digitise_real.py data/digitise_eval --out results/e7_real

Expects, for every photo ``<name>.jpg`` (or ``.png``) in the folder:

* ``<name>.json``: ``{"corners": [x1, y1, x2, y2, x3, y3, x4, y4], "cells_x": 120, "cells_y": 90}``
  (cloth corners in the photo, any order; the cell counts of the true design);
* ``<name>_truth.png``: the ground-truth design (e.g. the digitiser's output corrected by hand).

Reports per photo and on average: **assisted** (corners and cell counts given) cell accuracy
after matching colour labels, and the mean CIEDE2000 distance between matched palette colours;
**automatic** (corners only) resolution-independent pixel accuracy and whether the exact grid
size was recovered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from manitex.color import delta_e2000, srgb_to_lab
from manitex.design import Design
from manitex.digitise import digitise
from manitex.digitise.synth import pixel_accuracy
from manitex.experiments import write_tables

COLUMNS = ("assisted_cell_acc", "assisted_delta_e00", "auto_pixel_acc", "auto_exact_grid", "seconds")


def matched_scores(pred: Design, truth: Design) -> tuple[float, float]:
    """Cell accuracy after the best one-to-one matching of colour labels, and the cell-weighted
    mean CIEDE2000 distance between matched palette colours."""
    if pred.shape != truth.shape:
        return 0.0, float("nan")
    kp, kt = pred.n_colors, truth.n_colors
    conf = np.zeros((kp, kt), dtype=np.int64)
    np.add.at(conf, (pred.grid.ravel(), truth.grid.ravel()), 1)
    r, c = linear_sum_assignment(-conf)
    acc = float(conf[r, c].sum() / pred.grid.size)
    w = conf[r, c].astype(float)
    de = delta_e2000(srgb_to_lab(pred.palette[r].astype(float)), srgb_to_lab(truth.palette[c].astype(float)))
    return acc, float((de * w).sum() / max(w.sum(), 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="photos with <name>.json and <name>_truth.png next to them")
    ap.add_argument("--out", default="results/e7_real", help="output prefix (.json and .md)")
    args = ap.parse_args()

    import time

    folder = Path(args.folder)
    photos = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                    and not f.stem.endswith("_truth") and (folder / f"{f.stem}.json").exists())
    if not photos:
        raise SystemExit(f"no photos with a <name>.json sidecar in {folder}")
    rows: dict[str, dict[str, float]] = {}
    for f in photos:
        info = json.loads((folder / f"{f.stem}.json").read_text())
        truth_file = folder / f"{f.stem}_truth.png"
        if not truth_file.exists():
            raise SystemExit(f"missing ground truth {truth_file}")
        truth = Design.load(truth_file).canonical()
        img = np.asarray(Image.open(f).convert("RGB"))
        quad = np.asarray(info["corners"], dtype=np.float64).reshape(4, 2)
        cx, cy = int(info.get("cells_x", truth.width)), int(info.get("cells_y", truth.height))
        t0 = time.time()
        assisted = digitise(img, quad=quad, cells_x=cx, cells_y=cy)
        auto = digitise(img, quad=quad)
        acc, de = matched_scores(assisted.design.canonical(), truth)
        rows[f.name] = {
            "assisted_cell_acc": acc,
            "assisted_delta_e00": de,
            "auto_pixel_acc": pixel_accuracy(auto.design.grid, auto.grid.xs, auto.grid.ys, truth.grid,
                                             auto.image.shape[:2]),
            "auto_exact_grid": float(auto.design.shape == truth.shape),
            "seconds": time.time() - t0,
        }
        print(f"{f.name}: " + " ".join(f"{k} {v:.3f}" for k, v in rows[f.name].items()), flush=True)
    rows["mean"] = {k: float(np.nanmean([r[k] for r in rows.values()])) for k in COLUMNS}
    js, md = write_tables({"real photographs": rows}, args.out, COLUMNS, row_label="photo")
    print(f"wrote {js} and {md}")


if __name__ == "__main__":
    main()
