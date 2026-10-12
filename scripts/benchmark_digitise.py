#!/usr/bin/env python
"""Digitisation benchmark on synthetic photographs with exact ground truth.

For every procedural style and seed, a design is rendered as cloth and virtually
photographed (perspective, blur, uneven light, noise, JPEG).  The digitiser is then run

* **assisted**: cloth corners and the number of cells are given -> cell accuracy;
* **automatic**: only the cloth corners are given -> resolution-independent pixel
  accuracy (both designs painted onto the rectified image) and exact-grid rate;
* **clean**: a noise-free pixel-art rendering of the design, automatic and assisted.

Usage:  python scripts/benchmark_digitise.py --seeds 10 --out runs/bench_digitise.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manitex.digitise import digitise
from manitex.digitise.synth import label_accuracy, pixel_accuracy, synthetic_photo
from manitex.procedural import PROCEDURAL_STYLES, generate


def run(seeds: int, height: int, width: int) -> dict:
    rows = []
    for style in PROCEDURAL_STYLES:
        for seed in range(seeds):
            d = generate(style, height, width, seed=1000 + seed)
            truth = d.canonical().grid
            k_true = len(d.used_colors())
            sp = synthetic_photo(d, np.random.default_rng(seed))
            t0 = time.time()
            assisted = digitise(sp.photo, quad=sp.quad, cells_x=d.width, cells_y=d.height)
            t1 = time.time()
            auto = digitise(sp.photo, quad=sp.quad)
            t2 = time.time()
            clean_img = d.to_rgb(cell=8)
            clean_auto = digitise(clean_img)
            clean_assisted = digitise(clean_img, cells_x=d.width, cells_y=d.height)
            rows.append({
                "style": style,
                "seed": seed,
                "assisted_cell_acc": label_accuracy(assisted.design.grid, truth),
                "assisted_k_correct": assisted.report["k"] == k_true,
                "auto_pixel_acc": pixel_accuracy(auto.design.grid, auto.grid.xs, auto.grid.ys, truth,
                                                 auto.image.shape[:2]),
                "auto_exact_grid": auto.design.shape == d.shape,
                "auto_k_correct": auto.report["k"] == k_true,
                "clean_auto_pixel_acc": pixel_accuracy(clean_auto.design.grid, clean_auto.grid.xs, clean_auto.grid.ys,
                                                       truth, clean_auto.image.shape[:2]),
                "clean_assisted_exact": clean_assisted.design.shape == d.shape
                and label_accuracy(clean_assisted.design.grid, truth) == 1.0,
                "time_assisted_s": t1 - t0,
                "time_auto_s": t2 - t1,
            })
    metrics = [k for k in rows[0] if k not in ("style", "seed")]
    summary = {"overall": {m: float(np.mean([r[m] for r in rows])) for m in metrics}}
    for style in PROCEDURAL_STYLES:
        rs = [r for r in rows if r["style"] == style]
        summary[style] = {m: float(np.mean([r[m] for r in rs])) for m in metrics}
    return {"config": {"seeds": seeds, "height": height, "width": width}, "summary": summary, "rows": rows}


def to_markdown(res: dict) -> str:
    cols = ["assisted_cell_acc", "assisted_k_correct", "auto_pixel_acc", "auto_exact_grid", "auto_k_correct",
            "clean_auto_pixel_acc", "clean_assisted_exact", "time_auto_s"]
    out = ["| style | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, s in res["summary"].items():
        out.append(f"| {name} | " + " | ".join(f"{s[c]:.3f}" for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--height", type=int, default=40)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    res = run(args.seeds, args.height, args.width)
    md = to_markdown(res)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
        args.out.with_suffix(".md").write_text(md + "\n")


if __name__ == "__main__":
    main()
