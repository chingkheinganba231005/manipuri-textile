"""Sampling-time ablation: unmasking order, number of steps and guidance (no retraining).

    python scripts/ablate_sampler.py --checkpoint runs/pretrain_base/final.pt --n 64 --size 64 \
        --out results/sampler_ablation

For every style and sampler setting, ``n`` grids are sampled and compared with reference designs
(procedural samples, or the designs of that style in ``--reference`` shards) using the metric
suite of ``manitex.metrics``, plus the *collapse rate* (fraction of single-colour grids).  The
``floor`` row compares two disjoint reference sets: the metric values a perfect generator would
reach at this sample size.

Settings are ``strategy:steps:cfg[:noise]``, comma separated, e.g.
``--settings random:32:1,confidence:16:1:4.5``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manitex.generate import GenerationOptions, TextileGenerator
from manitex.metrics import evaluate
from manitex.procedural import PROCEDURAL_STYLES, generate

DEFAULT_SETTINGS = ("confidence:16:1:1,confidence:16:1:4.5,confidence:32:1:4.5,"
                    "random:16:1,random:32:0,random:32:1,random:64:0.5,random:64:1")
COLUMNS = ("pattern_js_3x3", "runlen_w1_rows", "runlen_w1_cols", "colour_count_tv", "diversity_gen")


def parse_settings(text: str) -> list[tuple[str, GenerationOptions]]:
    out = []
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) not in (3, 4) or parts[0] not in ("random", "confidence"):
            raise SystemExit(f"bad setting {item!r}; expected strategy:steps:cfg[:noise]")
        opt = GenerationOptions(strategy=parts[0], steps=int(parts[1]), cfg=float(parts[2]))
        if len(parts) == 4:
            opt.noise = float(parts[3])
        out.append((item.strip(), opt))
    return out


def references(style: str, n: int, size: int, shards, seed: int) -> list[np.ndarray]:
    if shards is None:
        if style not in PROCEDURAL_STYLES:
            raise SystemExit(f"no procedural reference for style {style!r}; pass --reference")
        return [generate(style, size, size, seed=seed + i).grid for i in range(n)]
    from manitex.data.dataset import CropSpec, prepare_crop

    rng = np.random.default_rng(seed)
    pool = [(sh, j) for sh in shards for j in range(len(sh)) if str(sh.styles[j]) == style]
    if not pool:
        raise SystemExit(f"no reference designs of style {style!r} in the given shards")
    crop = CropSpec(size, flip_lr=False, flip_ud=False)
    out = []
    for _ in range(n):
        sh, j = pool[int(rng.integers(len(pool)))]
        out.append(prepare_crop(sh.grid(j), crop, rng, sh.palette(j))[0])
    return out


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
        ref = references(style, args.n, args.size, shards, args.seed + 10_000)
        ref2 = references(style, args.n, args.size, shards, args.seed + 20_000)
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
            results[style][name] = m
            print(f"  {name:24s} " + " ".join(f"{k} {m[k]:.3f}" for k in COLUMNS)
                  + f" collapse {m['collapse_rate']:.2f} ({m['seconds']:.0f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    lines = []
    for style, rows in results.items():
        lines += [f"### {style}", "", "| setting | " + " | ".join(COLUMNS) + " | collapse |",
                  "|---|" + "---|" * (len(COLUMNS) + 1)]
        for name, m in rows.items():
            lines.append(f"| {name} | " + " | ".join(f"{m[k]:.3f}" for k in COLUMNS) + f" | {m['collapse_rate']:.2f} |")
        lines.append("")
    out.with_suffix(".md").write_text("\n".join(lines))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
