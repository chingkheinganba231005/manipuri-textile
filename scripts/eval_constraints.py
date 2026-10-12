#!/usr/bin/env python
"""Experiment E2: exact structural control - orbit pooling vs post-hoc projection.

    python scripts/eval_constraints.py --checkpoint runs/pretrain_base/final.pt --n 128 --size 64 \
        --out results/e2_constraints

For every style and constraint, ``n`` designs are produced by each method:

* ``unconstrained``: plain sampling (how often the constraint holds by chance);
* ``projection``: the same designs made to satisfy the constraint afterwards (majority colour
  of every orbit);
* ``orbit_mean`` / ``orbit_sum``: orbit pooling during sampling (masked model);
* ``copy``: for an autoregressive checkpoint, later members of an orbit copy the first one.

Columns: ``satisfied`` (fraction of designs meeting the constraint exactly), ``violations``
(fraction of cells disagreeing with their orbit's majority colour), ``nll`` (the model's bits per
cell for the designs: the masked model's ELBO bound or the autoregressive model's exact NLL),
pattern statistics against unconstrained reference designs of the style, and ``diversity_gen``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import torch

from manitex.constraints import make_orbits, project, satisfies
from manitex.experiments import model_nll, parse_symmetry, reference_grids, write_tables
from manitex.generate import GenerationOptions
from manitex.metrics import evaluate
from manitex.models.checkpoint import load_model
from manitex.models.config import DOMAINS
from manitex.models.factory import is_autoregressive, sampler_for
from manitex.models.sampler import SamplerConfig

DEFAULT_CONSTRAINTS = "mirror_x;period_x=16;mirror_x+period_x=16;mirror_y;stripes=rows"
COLUMNS = ("satisfied", "violations", "nll", "pattern_js_3x3", "runlen_w1_rows", "runlen_w1_cols", "diversity_gen")


def sample_many(model, n: int, size: int, batch: int, seed: int, **kw) -> list[np.ndarray]:
    fn = sampler_for(model)
    out: list[np.ndarray] = []
    for i in range(0, n, batch):
        g = fn(model, min(batch, n - i), size, size, generator=torch.Generator().manual_seed(seed + i), **kw)
        out.extend(g.numpy().astype(np.int64))
    return out


def score(model, grids: list[np.ndarray], orb: np.ndarray, refs: list[np.ndarray], sid: int, dom: int | None,
          batch: int) -> dict[str, float]:
    m = evaluate(grids, refs)
    return {
        "satisfied": float(np.mean([satisfies(g, orb) for g in grids])),
        "violations": float(np.mean([np.mean(g != project(g, orb)) for g in grids])),
        "nll": float(np.mean(model_nll(model, grids, style=sid, domain=dom, batch=batch))),
        **{k: float(m[k]) for k in COLUMNS[3:]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint .pt or slim export directory")
    ap.add_argument("--styles", default=None, help="comma-separated (default: all styles of the model)")
    ap.add_argument("--constraints", default=DEFAULT_CONSTRAINTS, help="semicolon-separated symmetry specs")
    ap.add_argument("--n", type=int, default=128, help="designs per style, constraint and method")
    ap.add_argument("--size", type=int, default=64, help="square canvas size in cells")
    ap.add_argument("--batch", type=int, default=64, help="designs sampled / scored at once")
    ap.add_argument("--steps", type=int, default=GenerationOptions.steps)
    ap.add_argument("--cfg", type=float, default=GenerationOptions.cfg)
    ap.add_argument("--top-p", type=float, default=GenerationOptions.top_p)
    ap.add_argument("--reference", default=None, help="shard file(s)/dir with reference designs (default: procedural)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/e2_constraints", help="output prefix (.json and .md)")
    args = ap.parse_args()

    model, info = load_model(args.checkpoint, device=args.device)
    vocab = info["vocab"]
    styles = args.styles.split(",") if args.styles else [s for s in vocab.names if s != "unknown"]
    dom = DOMAINS["real"] if "real" in info["domains_seen"] else None
    shards = None
    if args.reference:
        from manitex.data import read_shards

        shards = read_shards(args.reference.split(","))
    scfg = SamplerConfig(steps=args.steps, cfg=args.cfg, top_p=args.top_p)
    ar = is_autoregressive(model)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for style in styles:
        sid = vocab.id(style)
        refs = reference_grids(style, args.n, args.size, shards, seed=args.seed + 10_000)
        common = dict(style=sid, domain=dom, config=scfg, device=args.device)
        t = time.time()
        free = sample_many(model, args.n, args.size, args.batch, args.seed, **common)
        print(f"== {style}: {args.n} unconstrained designs in {time.time() - t:.0f}s", flush=True)
        for spec in filter(None, (c.strip() for c in args.constraints.split(";"))):
            sym = parse_symmetry(spec)
            orb = make_orbits(args.size, args.size, sym)
            designs = {"unconstrained": free, "projection": [project(g, orb) for g in free]}
            pools = {"copy": scfg} if ar else {"orbit_mean": scfg, "orbit_sum": replace(scfg, pool="sum")}
            for name, cfg in pools.items():
                kw = {**common, "config": cfg}
                designs[name] = sample_many(model, args.n, args.size, args.batch, args.seed + 1, symmetry=sym, **kw)
            rows = {name: score(model, grids, orb, refs, sid, dom, args.batch) for name, grids in designs.items()}
            results[f"{style} | {spec}"] = rows
            for name, m in rows.items():
                print(f"  {spec:22s} {name:13s} " + " ".join(f"{k} {m[k]:.3f}" for k in COLUMNS), flush=True)
    js, md = write_tables(results, args.out, COLUMNS, row_label="method")
    print(f"wrote {js} and {md}")


if __name__ == "__main__":
    main()
