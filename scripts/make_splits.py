#!/usr/bin/env python
"""Split curated designs into train / validation / test shards by textile, plus nested training
subsets for the data-efficiency experiment (E3).

    python scripts/make_splits.py data/designs --raw data/raw --out data/real \
        --val 0.1 --test 0.1 --fractions 0.1,0.25,0.5

Input: ``data/designs/<piece_id>/...`` (curated output of ``manitex digitise``; previews and grid
overlays are ignored) and, optionally, ``data/raw/<piece_id>/meta.json`` whose ``category`` is
the style label (else the design's ``style`` metadata). All designs of one piece go to the same
split, so crops of one textile never appear in two splits; pieces are stratified by category.
Training subsets are nested (the 10 % pieces are inside the 25 % ones, ...), so points of a
data-efficiency curve differ only by the added data.

Output: ``train.npz``, ``val.npz``, ``test.npz``, ``train_10.npz``, ... and ``splits.json``
(piece ids per split, counts per category).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from manitex.data import load_design_folder, write_shard
from manitex.data.splits import nested_fractions, split_pieces


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("designs", help="folder with one sub-folder of designs per piece")
    ap.add_argument("--raw", default=None, help="folder with <piece_id>/meta.json (category labels)")
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--fractions", default="0.1,0.25,0.5", help="nested training subsets ('' for none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-colors", type=int, default=16)
    args = ap.parse_args()

    designs = load_design_folder(args.designs, args.max_colors)
    if not designs:
        raise SystemExit(f"no designs under {args.designs}")
    by_piece: dict[str, list] = defaultdict(list)
    for d in designs:
        by_piece[str(d.meta["piece_id"])].append(d)
    categories: dict[str, str] = {}
    for piece, ds in by_piece.items():
        meta_file = Path(args.raw) / piece / "meta.json" if args.raw else None
        cat = None
        if meta_file is not None and meta_file.exists():
            cat = json.loads(meta_file.read_text()).get("category")
        categories[piece] = str(cat or ds[0].meta.get("style") or "unknown")
    splits = split_pieces(categories, args.val, args.test, args.seed)
    fractions = [float(f) for f in args.fractions.split(",") if f.strip()]
    subsets = nested_fractions(splits["train"], categories, fractions, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, pieces: list[str]) -> int:
        ds = [d for p in pieces for d in by_piece[p]]
        if ds:
            write_shard(out / f"{name}.npz", ds, styles=[categories[str(d.meta['piece_id'])] for d in ds],
                        domains=["real"] * len(ds))
        return len(ds)

    report: dict = {"seed": args.seed, "val": args.val, "test": args.test, "pieces": {}, "designs": {}, "categories": {}}
    for name in ("train", "val", "test"):
        report["pieces"][name] = sorted(splits[name])
        report["designs"][name] = write(name, splits[name])
    for f, pieces in subsets.items():
        name = f"train_{round(100 * f)}"
        report["pieces"][name] = pieces
        report["designs"][name] = write(name, pieces)
    for name in ("train", "val", "test"):
        counts: dict[str, int] = defaultdict(int)
        for p in splits[name]:
            counts[categories[p]] += 1
        report["categories"][name] = dict(sorted(counts.items()))
    (out / "splits.json").write_text(json.dumps(report, indent=1))
    for name, n in report["designs"].items():
        print(f"{name:10s} {len(report['pieces'][name]):4d} pieces {n:5d} designs")


if __name__ == "__main__":
    main()
