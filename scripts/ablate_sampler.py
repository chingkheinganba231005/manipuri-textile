"""Sampling-time ablation: unmasking order, steps, guidance, top-p, temperature (no retraining).

    python scripts/ablate_sampler.py --checkpoint runs/pretrain_base/final.pt --n 64 --size 64 \
        --out results/sampler_ablation

For every style and sampler setting, ``n`` grids are sampled and compared with reference designs
(procedural samples, or the designs of that style in ``--reference`` shards) using the metric
suite of ``manitex.metrics``, plus the *collapse rate* (fraction of single-colour grids).  The
``floor`` row compares two disjoint reference sets: the metric values a perfect generator would
reach at this sample size.

Each setting is a comma-separated list of ``GenerationOptions`` fields applied on top of the
library defaults (ancestral order, 64 steps, guidance 1, top-p 0.95); settings are separated by
semicolons, e.g. ``--settings "steps=32,top_p=1;strategy=confidence,noise=4.5"``.  The output
table lists the effective options of every row.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manitex.experiments import reference_grids
from manitex.generate import GenerationOptions, TextileGenerator
from manitex.metrics import evaluate

DEFAULT_SETTINGS = ";".join([
    "strategy=confidence,steps=16,noise=1,top_p=1",  # MaskGIT ordering, little noise
    "strategy=confidence,steps=32,noise=4.5,top_p=1",  # MaskGIT ordering, MaskGIT's noise
    "steps=32,top_p=1",  # plain ancestral sampling
    "steps=64,top_p=1",
    "steps=32",
    "steps=64",  # library default
    "steps=128",
    "steps=64,top_p=0.9",
    "steps=64,temperature=0.9,top_p=1",
    "steps=64,cfg=0",
    "steps=64,cfg=2",
])
COLUMNS = ("pattern_js_3x3", "runlen_w1_rows", "runlen_w1_cols", "colour_count_tv", "diversity_gen")
OPTION_FIELDS = ("strategy", "steps", "cfg", "top_p", "temperature", "noise")


def parse_settings(text: str) -> list[tuple[str, GenerationOptions]]:
    out = []
    for item in filter(None, (t.strip() for t in text.split(";"))):
        opt = GenerationOptions()
        for field in filter(None, (f.strip() for f in item.split(","))):
            key, _, value = field.partition("=")
            if key not in OPTION_FIELDS or not value:
                raise SystemExit(f"bad setting {field!r} in {item!r}; use key=value with keys {OPTION_FIELDS}")
            setattr(opt, key, value if key == "strategy" else type(getattr(GenerationOptions, key))(float(value)))
        if opt.strategy not in ("random", "confidence"):
            raise SystemExit(f"unknown strategy {opt.strategy!r}")
        out.append((item, opt))
    return out


def describe(opt: GenerationOptions) -> str:
    keys = OPTION_FIELDS if opt.strategy == "confidence" else OPTION_FIELDS[:-1]
    return ", ".join(f"{k}={getattr(opt, k)}" for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint .pt, slim export dir or hf://user/repo")
    ap.add_argument("--styles", default=None, help="comma-separated (default: all styles of the model)")
    ap.add_argument("--settings", default=DEFAULT_SETTINGS)
    ap.add_argument("--n", type=int, default=64, help="samples per style and setting")
    ap.add_argument("--size", type=int, default=64, help="square canvas size in cells")
    ap.add_argument("--reference", default=None, help="shard file(s)/dir with reference designs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/sampler_ablation", help="output prefix (.json and .md)")
    args = ap.parse_args()

    gen = TextileGenerator.from_pretrained(args.checkpoint, device=args.device)
    styles = args.styles.split(",") if args.styles else gen.styles
    shards = None
    if args.reference:
        from manitex.data import read_shards

        shards = read_shards(args.reference.split(","))
    settings = parse_settings(args.settings)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for style in styles:
        ref = reference_grids(style, args.n, args.size, shards, args.seed + 10_000)
        ref2 = reference_grids(style, args.n, args.size, shards, args.seed + 20_000)
        floor = evaluate(ref2, ref)
        floor["collapse_rate"] = float(np.mean([len(np.unique(g)) == 1 for g in ref2]))
        results[style] = {"floor": floor}
        print(f"== {style}")
        for name, opt in settings:
            t = time.time()
            designs = gen.generate(style, n=args.n, height=args.size, width=args.size, seed=args.seed, options=opt)
            grids = [d.grid for d in designs]
            m = evaluate(grids, ref)
            m["collapse_rate"] = float(np.mean([len(np.unique(g)) == 1 for g in grids]))
            m["mean_colours"] = float(np.mean([len(np.unique(g)) for g in grids]))
            m["seconds"] = time.time() - t
            m["options"] = describe(opt)
            results[style][name] = m
            print(f"  {m['options']:60s} " + " ".join(f"{k} {m[k]:.3f}" for k in COLUMNS)
                  + f" collapse {m['collapse_rate']:.2f} ({m['seconds']:.0f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    lines = []
    for style, rows in results.items():
        lines += [f"### {style}", "", "| setting | " + " | ".join(COLUMNS) + " | collapse |",
                  "|---|" + "---|" * (len(COLUMNS) + 1)]
        for name, m in rows.items():
            lines.append(f"| {m.get('options', name)} | " + " | ".join(f"{m[k]:.3f}" for k in COLUMNS)
                         + f" | {m['collapse_rate']:.2f} |")
        lines.append("")
    out.with_suffix(".md").write_text("\n".join(lines))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
