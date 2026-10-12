#!/usr/bin/env python
"""Image-space baseline for experiment E1 (see ``manitex.baselines.image_diffusion``).

Fine-tune LoRA adapters of an open latent diffusion model on rendered designs, then sample images
and digitise them into design grids (the cell count is known; no structural clean-up unless
``--regularise``):

    python scripts/image_baseline.py train --shards data/real/train.npz --procedural-weight 0.2 \
        --steps 5000 --out runs/image_baseline
    python scripts/image_baseline.py sample --lora runs/image_baseline/lora --n 256 \
        --out results/image_baseline

``sample`` writes ``<out>/designs.npz`` (all styles in one shard) for
``scripts/eval_generators.py --designs image=<out>/designs.npz``, plus PNG previews.
Needs ``pip install -e ".[image-baseline]"`` and a GPU; check the base model's licence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from manitex.baselines.image_diffusion import (
    ImageBaselineConfig,
    build_pipeline,
    sample_designs,
    train_lora,
)
from manitex.procedural import PROCEDURAL_STYLES


def cmd_train(args) -> None:
    cfg = ImageBaselineConfig(model=args.model, out_dir=args.out, cells=args.cells, cell_px=args.cell_px,
                              shards=args.shards, procedural_weight=args.procedural_weight,
                              styles=args.styles.split(",") if args.styles else None, steps=args.steps,
                              batch_size=args.batch, lr=args.lr, rank=args.rank, precision=args.precision,
                              seed=args.seed)
    print(f"LoRA weights: {train_lora(cfg)}")


def cmd_sample(args) -> None:
    from manitex.data import write_shard

    run_cfg = {}
    cfg_path = Path(args.lora).parent / "config.json"
    if cfg_path.exists():  # defaults from the training run
        run_cfg = json.loads(cfg_path.read_text())
    model = args.model or run_cfg.get("model") or ImageBaselineConfig.model
    cells = args.cells or run_cfg.get("cells") or ImageBaselineConfig.cells
    cell_px = args.cell_px or run_cfg.get("cell_px") or ImageBaselineConfig.cell_px
    if args.styles:
        styles = args.styles.split(",")
    elif run_cfg.get("styles"):
        styles = list(run_cfg["styles"])
    else:
        styles = list(PROCEDURAL_STYLES)
    out = Path(args.out)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(model, args.lora)
    all_designs, all_styles = [], []
    for style in styles:
        images: list[np.ndarray] = []
        designs = sample_designs(pipe, style, args.n, cells, cell_px, steps=args.steps, guidance=args.guidance,
                                 batch=args.batch, seed=args.seed, regularise=args.regularise, keep_images=images)
        for i in range(min(args.previews, len(designs))):
            Image.fromarray(images[i]).save(out / "previews" / f"{style}_{i:03d}_image.png")
            designs[i].to_image(cell=8).save(out / "previews" / f"{style}_{i:03d}_design.png")
        all_designs += designs
        all_styles += [style] * len(designs)
        print(f"{style}: {len(designs)} designs", flush=True)
    shard = write_shard(out / "designs.npz", all_designs, styles=all_styles, domains=["generated"] * len(all_designs))
    print(f"wrote {shard}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="fine-tune LoRA adapters on rendered designs")
    t.add_argument("--model", default=ImageBaselineConfig.model, help="base latent diffusion model (diffusers format)")
    t.add_argument("--shards", nargs="*", default=[], help="real-design shards; none = procedural designs only")
    t.add_argument("--procedural-weight", type=float, default=0.0, help="fraction of procedural designs with --shards")
    t.add_argument("--styles", default=None, help="comma-separated subset of styles")
    t.add_argument("--cells", type=int, default=ImageBaselineConfig.cells)
    t.add_argument("--cell-px", type=int, default=ImageBaselineConfig.cell_px)
    t.add_argument("--steps", type=int, default=ImageBaselineConfig.steps)
    t.add_argument("--batch", type=int, default=ImageBaselineConfig.batch_size)
    t.add_argument("--lr", type=float, default=ImageBaselineConfig.lr)
    t.add_argument("--rank", type=int, default=ImageBaselineConfig.rank)
    t.add_argument("--precision", default=ImageBaselineConfig.precision, choices=["bf16", "fp16", "fp32"])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default=ImageBaselineConfig.out_dir)
    t.set_defaults(func=cmd_train)
    s = sub.add_parser("sample", help="sample images and digitise them into designs")
    s.add_argument("--lora", required=True, help="LoRA directory written by 'train'")
    s.add_argument("--model", default=None, help="base model (default: the one used for training)")
    s.add_argument("--styles", default=None, help="comma-separated (default: the training styles)")
    s.add_argument("--n", type=int, default=256, help="designs per style")
    s.add_argument("--cells", type=int, default=None)
    s.add_argument("--cell-px", type=int, default=None)
    s.add_argument("--steps", type=int, default=30, help="denoising steps")
    s.add_argument("--guidance", type=float, default=5.0)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--regularise", action="store_true", help="apply the digitiser's symmetry/repeat clean-up")
    s.add_argument("--previews", type=int, default=8, help="images and designs saved as PNG per style")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="results/image_baseline")
    s.set_defaults(func=cmd_sample)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
