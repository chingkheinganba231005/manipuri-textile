#!/usr/bin/env python
"""Experiment E1: compare generators against held-out designs.

    python scripts/eval_generators.py --model mdm=runs/finetune_real/final.pt \
        --model ar=runs/baseline_ar_small/final.pt \
        --reference data/real/val.npz --train data/real/train.npz --n 256 --size 64 --out results/e1

Rows per style:

* ``floor``: a second, disjoint sample of reference designs - the values a perfect generator
  would reach at this sample size;
* ``grammar``: the procedural grammar (for the styles it knows);
* ``retrieval``: random crops of training designs (``--train``) - matches the training data
  but creates nothing new (its novelty is ~0);
* one row per ``--model name=checkpoint``: samples from a trained model (masked diffusion or
  the autoregressive baseline) with the library's default sampler settings.

Columns: palette-independent pattern statistics, run lengths, colour counts, mirror symmetry,
diversity; ``novelty`` / ``memorised`` (structural distance to the nearest training window and
the fraction below 0.02, with ``--train``); weaving cost of the designs; and for models
``nll_ref``, the NLL of the reference designs in bits per cell (ELBO bound for masked
diffusion, exact for the autoregressive model; both use the style condition).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from manitex.experiments import model_nll, reference_designs, write_tables
from manitex.generate import GenerationOptions, TextileGenerator
from manitex.metrics import evaluate, weavability
from manitex.models.config import DOMAINS
from manitex.procedural import PROCEDURAL_STYLES, generate

COLUMNS = ("pattern_js_2x2", "pattern_js_3x3", "runlen_w1_rows", "runlen_w1_cols", "colour_count_tv",
           "mirror_x_gen", "diversity_gen", "novelty", "memorised", "nll_ref",
           "tie_downs_per_100_cells", "dobby_shafts_median", "face_floats_within_limit")


def score(designs, ref, train_grids, n_weave: int) -> dict[str, float]:
    grids = [np.asarray(d.grid) for d in designs]
    m = evaluate(grids, ref, train=train_grids)
    row = {k: float(m[k]) for k in COLUMNS[:7]}
    if train_grids is not None:
        row["novelty"] = m["novelty_mean"]
        row["memorised"] = m["memorised_frac(<0.02)"]
    w = weavability(designs[:n_weave])
    row.update({k: w[k] for k in COLUMNS[-3:]})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=CHECKPOINT",
                    help="a trained model to evaluate (repeatable; checkpoint .pt, slim export or hf://)")
    ap.add_argument("--styles", default=None, help="comma-separated (default: styles of the first model, "
                    "else the grammar's)")
    ap.add_argument("--reference", default=None, help="held-out shard file(s)/dir (default: procedural designs)")
    ap.add_argument("--train", default=None, help="training shard file(s)/dir: retrieval baseline and novelty")
    ap.add_argument("--n", type=int, default=256, help="designs per style and generator")
    ap.add_argument("--size", type=int, default=64, help="square canvas size in cells")
    ap.add_argument("--batch", type=int, default=32, help="designs generated per model call")
    ap.add_argument("--n-weave", type=int, default=64, help="designs compiled for the weaving-cost columns")
    ap.add_argument("--max-train", type=int, default=2000, help="training designs searched for novelty")
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/e1_generators", help="output prefix (.json and .md)")
    args = ap.parse_args()

    from manitex.data import read_shards

    ref_shards = read_shards(args.reference.split(",")) if args.reference else None
    train_shards = read_shards(args.train.split(",")) if args.train else None
    models = {}
    for spec in args.model:
        name, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"--model expects NAME=CHECKPOINT, got {spec!r}")
        models[name] = TextileGenerator.from_pretrained(path, device=args.device)
    if args.styles:
        styles = args.styles.split(",")
    elif models:
        styles = next(iter(models.values())).styles
    else:
        styles = list(PROCEDURAL_STYLES)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for style in styles:
        t0 = time.time()
        ref_d = reference_designs(style, args.n, args.size, ref_shards, seed=args.seed + 10_000)
        ref = [d.grid for d in ref_d]
        train_grids = None
        if train_shards is not None:
            train_grids = [sh.grid(j) for sh in train_shards for j in range(len(sh)) if str(sh.styles[j]) == style]
            rng = np.random.default_rng(args.seed)
            if len(train_grids) > args.max_train:
                train_grids = [train_grids[i] for i in rng.choice(len(train_grids), args.max_train, replace=False)]
        rows: dict[str, dict[str, float]] = {}
        floor = reference_designs(style, args.n, args.size, ref_shards, seed=args.seed + 20_000)
        rows["floor"] = score(floor, ref, train_grids, args.n_weave)
        if not args.no_grammar and style in PROCEDURAL_STYLES:
            gram = [generate(style, args.size, args.size, seed=args.seed + 30_000 + i) for i in range(args.n)]
            rows["grammar"] = score(gram, ref, train_grids, args.n_weave)
        if train_shards is not None:
            retr = reference_designs(style, args.n, args.size, train_shards, seed=args.seed + 40_000)
            rows["retrieval"] = score(retr, ref, train_grids, args.n_weave)
        for name, gen in models.items():
            designs = []
            for i in range(0, args.n, args.batch):
                designs += gen.generate(style, n=min(args.batch, args.n - i), height=args.size, width=args.size,
                                        seed=args.seed + i, options=GenerationOptions())
            row = score(designs, ref, train_grids, args.n_weave)
            dom = DOMAINS["real"] if "real" in gen.info.get("domains_seen", []) else None
            row["nll_ref"] = float(np.mean(model_nll(gen.model, ref, style=gen.info["vocab"].id(style), domain=dom,
                                                     batch=args.batch)))
            rows[name] = row
        results[style] = rows
        print(f"== {style} ({time.time() - t0:.0f}s)", flush=True)
        for name, m in rows.items():
            print(f"  {name:10s} " + " ".join(f"{k} {m[k]:.3f}" for k in COLUMNS if k in m), flush=True)
    js, md = write_tables(results, args.out, COLUMNS, row_label="generator")
    print(f"wrote {js} and {md}")


if __name__ == "__main__":
    main()
