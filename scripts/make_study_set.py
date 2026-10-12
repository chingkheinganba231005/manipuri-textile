#!/usr/bin/env python
"""Materials and analysis for the blind study with weavers and designers (experiment E5).

Build a shuffled, anonymised set of designs from several sources:

    python scripts/make_study_set.py build --real data/real/test.npz \
        --model runs/finetune_real/final.pt --designs image=results/image_baseline/designs.npz \
        --styles moirang_phee,phanek_mayek_naibi --n 8 --size 48x64 --out study/round1

writes ``images/<id>_cloth.png`` (simulated cloth) and ``<id>_design.png`` (flat chart),
``sheet.html`` (printable rating sheet: authenticity 1-5, "would you weave it?", comments),
``responses_template.csv`` and ``key.csv`` (the sources; keep it away from participants).
Sources: ``real`` (crops of held-out designs), ``grammar`` (procedural), ``model`` (``--model``)
and one per ``--designs name=shard``. ``--palette real`` recolours every item with palettes of
real designs of its style, so that only structure differs between sources.

Analyse the filled-in response files:

    python scripts/make_study_set.py analyse --key study/round1/key.csv \
        --responses study/round1/responses_*.csv --out results/e5

reports, per source (and style), the mean authenticity with a 95 % bootstrap interval, the share
of "yes" to "would you weave it?", and a two-sided Mann-Whitney U test of every source against
``real``. These pool all ratings as if independent; for the paper, also fit a model with
participant and item effects (e.g. ordinal mixed-effects regression) on the merged table.
"""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RESPONSE_FIELDS = ["participant_id", "item_id", "authenticity", "would_weave", "comment"]


def _parse_size(text: str) -> tuple[int, int]:
    h, _, w = text.lower().partition("x")
    return int(h), int(w or h)


def cmd_build(args) -> None:
    from manitex.data import read_shards
    from manitex.experiments import reference_designs, stored_designs
    from manitex.generate import TextileGenerator, assign_palette
    from manitex.procedural import PROCEDURAL_STYLES, generate
    from manitex.render.fabric import render_design_cloth

    h, w = _parse_size(args.size)
    real = read_shards(args.real.split(",")) if args.real else None
    stored = {}
    for spec in args.designs:
        name, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"--designs expects NAME=SHARD, got {spec!r}")
        stored[name] = read_shards(path.split(","))
    gen = TextileGenerator.from_pretrained(args.model) if args.model else None
    styles = args.styles.split(",")
    bank: dict[str, list[list[str]]] = defaultdict(list)
    if args.palette == "real":
        if real is None:
            raise SystemExit("--palette real needs --real")
        from manitex.color import rgb_to_hex

        for sh in real:
            for j in range(len(sh)):
                bank[str(sh.styles[j])].append([rgb_to_hex(c) for c in sh.design(j).canonical().palette])

    items = []
    for style in styles:
        sources = {}
        if real is not None:
            try:
                sources["real"] = reference_designs(style, args.n, (h, w), real, seed=args.seed)
            except ValueError as e:  # a study needs real designs of every style it shows
                raise SystemExit(f"{e}: drop {style!r} from --styles or add its designs to --real") from e
        if not args.no_grammar and style in PROCEDURAL_STYLES:
            sources["grammar"] = [generate(style, h, w, seed=args.seed + 1000 + i) for i in range(args.n)]
        if gen is not None:
            sources["model"] = gen.generate(style, n=args.n, height=h, width=w, seed=args.seed)
        for name, shards in stored.items():
            sources[name] = stored_designs(shards, style, args.n, (h, w), seed=args.seed)
        for source, designs in sources.items():
            for d in designs:
                items.append((d.canonical(), source, style))
    if not items:
        raise SystemExit("no items: give --real, --model, --designs or a grammar style")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(items))
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    key_rows, cards = [], []
    for k, idx in enumerate(order):
        d, source, style = items[idx]
        item_id = f"S{k + 1:03d}"
        if args.palette == "real" and bank.get(style):
            d = d.with_palette(assign_palette(d.grid, style, rng, None, {style: bank[style]}))
        cloth, _ = render_design_cloth(d)
        img = Image.fromarray(cloth)
        img.thumbnail((args.image_px, args.image_px), Image.LANCZOS)
        img.save(out / "images" / f"{item_id}_cloth.png")
        d.to_image(cell=max(2, args.image_px // max(d.shape))).save(out / "images" / f"{item_id}_design.png")
        key_rows.append({"item_id": item_id, "source": source, "style": style, "rows": d.height, "cols": d.width})
        cards.append(item_id)
    with open(out / "key.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        wr.writeheader()
        wr.writerows(key_rows)
    with open(out / "responses_template.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(RESPONSE_FIELDS)
        for item_id in cards:
            wr.writerow(["", item_id, "", "", ""])
    (out / "sheet.html").write_text(_sheet(cards, args.title))
    counts = defaultdict(int)
    for r in key_rows:
        counts[r["source"]] += 1
    print(f"{len(cards)} items ({dict(counts)}) -> {out}  (keep key.csv away from participants)")


def _sheet(item_ids: list[str], title: str) -> str:
    scale = " ".join(f'<label><input type="radio" name="a_{{id}}"> {i}</label>' for i in range(1, 6))
    cards = []
    for item_id in item_ids:
        cards.append(f"""<div class="card"><h3>{item_id}</h3>
<img src="images/{item_id}_cloth.png" alt="woven simulation {item_id}"><img src="images/{item_id}_design.png" alt="chart {item_id}">
<p>How authentic does this look as a Manipuri textile design? (1 = not at all, 5 = completely)<br>{scale.format(id=item_id)}</p>
<p>Would you weave it? <label><input type="radio" name="w_{item_id}"> yes</label>
<label><input type="radio" name="w_{item_id}"> maybe</label> <label><input type="radio" name="w_{item_id}"> no</label></p>
<p>Comments: ____________________________________________</p></div>""")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:sans-serif;margin:16px}} .card{{border:1px solid #999;padding:8px;margin:8px 0;
page-break-inside:avoid}} img{{max-height:260px;margin-right:12px;image-rendering:pixelated}}</style></head>
<body><h1>{html.escape(title)}</h1><p>Participant: __________ &nbsp; Date: __________</p>
{''.join(cards)}</body></html>"""


def _bootstrap_ci(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    means = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cmd_analyse(args) -> None:
    from scipy.stats import mannwhitneyu

    from manitex.experiments import write_tables

    key = {r["item_id"]: r for r in csv.DictReader(open(args.key))}
    ratings = []
    for path in args.responses:
        for r in csv.DictReader(open(path)):
            if not r.get("item_id") or not str(r.get("authenticity", "")).strip():
                continue
            if r["item_id"] not in key:
                raise SystemExit(f"{path}: unknown item {r['item_id']!r}")
            ratings.append({**key[r["item_id"]], "participant": r.get("participant_id", ""),
                            "authenticity": float(r["authenticity"]),
                            "yes": str(r.get("would_weave", "")).strip().lower() in ("yes", "y", "1", "true")})
    if not ratings:
        raise SystemExit("no ratings found")
    rng = np.random.default_rng(0)
    results: dict[str, dict[str, dict[str, float]]] = {}
    groups = {"all styles": ratings}
    for style in sorted({r["style"] for r in ratings}):
        groups[style] = [r for r in ratings if r["style"] == style]
    for gname, rs in groups.items():
        by_src: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_src[r["source"]].append(r)
        real = np.array([r["authenticity"] for r in by_src.get("real", [])])
        rows = {}
        for src in sorted(by_src, key=lambda s: (s != "real", s)):
            a = np.array([r["authenticity"] for r in by_src[src]])
            lo, hi = _bootstrap_ci(a, rng)
            row = {"n": int(a.size), "authenticity_mean": float(a.mean()), "ci95_low": lo, "ci95_high": hi,
                   "would_weave_yes": float(np.mean([r["yes"] for r in by_src[src]]))}
            if src != "real" and real.size and a.size:
                row["mannwhitney_p_vs_real"] = float(mannwhitneyu(a, real, alternative="two-sided").pvalue)
            rows[src] = row
        results[gname] = rows
    cols = ["n", "authenticity_mean", "ci95_low", "ci95_high", "would_weave_yes", "mannwhitney_p_vs_real"]
    js, md = write_tables(results, args.out, cols, row_label="source")
    print(md.read_text())
    print(f"wrote {js} and {md}; {len(ratings)} ratings from {len({r['participant'] for r in ratings})} participants")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="make the blind study set")
    b.add_argument("--real", default=None, help="held-out real design shard(s)")
    b.add_argument("--model", default=None, help="trained model (checkpoint, slim export or hf://)")
    b.add_argument("--designs", action="append", default=[], metavar="NAME=SHARD", help="other design sources")
    b.add_argument("--no-grammar", action="store_true")
    b.add_argument("--styles", required=True, help="comma-separated styles")
    b.add_argument("--n", type=int, default=8, help="items per source and style")
    b.add_argument("--size", default="48x64", help="rows x columns of every item")
    b.add_argument("--palette", default="keep", choices=["keep", "real"])
    b.add_argument("--image-px", type=int, default=480, help="longest side of the item images")
    b.add_argument("--title", default="Textile design study")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="study/round1")
    b.set_defaults(func=cmd_build)
    a = sub.add_parser("analyse", help="summarise filled-in response files")
    a.add_argument("--key", required=True)
    a.add_argument("--responses", nargs="+", required=True)
    a.add_argument("--out", default="results/e5_study")
    a.set_defaults(func=cmd_analyse)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
