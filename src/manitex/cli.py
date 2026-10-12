"""Command-line interface: ``manitex <command> ...``  (or ``python -m manitex``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _parse_size(s: str) -> tuple[int, int]:
    if "x" in s.lower():
        h, w = s.lower().split("x")
        return int(h), int(w)
    return int(s), int(s)


def _symmetry(args) -> object | None:
    from manitex.constraints import Symmetry

    sym = Symmetry(mirror_x=getattr(args, "mirror_x", False), mirror_y=getattr(args, "mirror_y", False),
                   period_x=getattr(args, "period_x", None), period_y=getattr(args, "period_y", None),
                   stripes=getattr(args, "stripes", None))
    return sym if sym.any() else None


def _write_design_outputs(d, out: Path, stem: str, cell: int, render: bool, booklet: bool, opts: dict) -> list[Path]:
    from manitex.render import render_draft
    from manitex.weave import compile_design, save_wif, to_html, to_markdown

    files = [d.save(out / f"{stem}.png"), d.save(out / f"{stem}.json")]
    d.to_image(cell=cell).save(out / f"{stem}_preview.png")
    files.append(out / f"{stem}_preview.png")
    if render or booklet:
        dr = compile_design(d, **opts)
        img = render_draft(dr)
        if render:
            from PIL import Image

            Image.fromarray(img).save(out / f"{stem}_cloth.png")
            files.append(out / f"{stem}_cloth.png")
        if booklet:
            files.append(save_wif(dr, out / f"{stem}.wif", title=stem))
            (out / f"{stem}.html").write_text(to_html(d, dr, title=stem, fabric_render=img))
            (out / f"{stem}.md").write_text(to_markdown(d, dr, title=stem))
            files += [out / f"{stem}.html", out / f"{stem}.md"]
    return files


def cmd_procedural(args) -> None:
    from manitex.procedural import generate

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    h, w = _parse_size(args.size)
    for i in range(args.n):
        seed = args.seed + i
        d = generate(args.style, h, w, seed=seed, n_colors=args.colors)
        stem = f"{d.meta['style']}_{seed:05d}"
        _write_design_outputs(d, out, stem, args.cell, args.render, args.booklet, {})
        print(stem)


def cmd_generate(args) -> None:
    from manitex.generate import GenerationOptions, TextileGenerator

    gen = TextileGenerator.from_pretrained(args.checkpoint)
    if not gen.has_model:
        print("no checkpoint given: using the procedural generator", file=sys.stderr)
    h, w = _parse_size(args.size)
    opts = GenerationOptions(steps=args.steps, temperature=args.temperature, cfg=args.cfg, domain=args.domain)
    palette = args.palette.split(",") if args.palette and "," in args.palette else args.palette
    designs = gen.generate(args.style, n=args.n, height=h, width=w, n_colors=args.colors, symmetry=_symmetry(args),
                           palette=palette, seed=args.seed, options=opts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(designs):
        stem = f"{args.style}_{args.seed}_{i:03d}"
        _write_design_outputs(d, out, stem, args.cell, args.render, args.booklet, {})
        print(stem)


def cmd_compile(args) -> None:
    from PIL import Image

    from manitex.design import Design
    from manitex.render import render_draft
    from manitex.weave import Sett, compile_design, design_chart_svg, save_wif, to_html, to_markdown

    d = Design.load(args.design)
    dr = compile_design(d, technique=args.technique, ground_weave=args.ground_weave, ground_stripes=args.ground_stripes,
                        supplementary=args.supplementary, ends_per_cell=args.ends_per_cell,
                        picks_per_cell=args.picks_per_cell, max_float=args.max_float, bottom_up=not args.top_down)
    sett = Sett(ends_per_cm=args.epc, picks_per_cm=args.ppc)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.design).stem
    img = render_draft(dr)
    Image.fromarray(img).save(out / f"{stem}_cloth.png")
    save_wif(dr, out / f"{stem}.wif", title=stem, ends_per_unit=args.epc, picks_per_unit=args.ppc)
    (out / f"{stem}.html").write_text(to_html(d, dr, sett=sett, title=stem, fabric_render=img))
    (out / f"{stem}.md").write_text(to_markdown(d, dr, sett=sett, title=stem))
    (out / f"{stem}_chart.svg").write_text(design_chart_svg(d, title=stem))
    print(json.dumps(dr.summary(), indent=1))


def cmd_render(args) -> None:
    from PIL import Image

    from manitex.design import Design
    from manitex.render import render_draft
    from manitex.weave import compile_design, read_wif

    src = Path(args.input)
    dr = read_wif(src) if src.suffix.lower() == ".wif" else compile_design(Design.load(src), ends_per_cell=args.ends_per_cell)
    img = render_draft(dr, end_px=args.px, pick_px=args.px, face_mode=args.face)
    Image.fromarray(img).save(args.out)
    print(args.out)


def cmd_digitise(args) -> None:
    from PIL import Image

    from manitex.digitise import digitise

    img = np.asarray(Image.open(args.image).convert("RGB"))
    quad = None
    if args.corners:
        vals = [float(v) for v in args.corners.split(",")]
        if len(vals) != 8:
            raise SystemExit("--corners needs 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4")
        quad = np.array(vals).reshape(4, 2)
    res = digitise(img, quad=quad, cells_x=args.cells_x, cells_y=args.cells_y, n_colors=args.colors,
                   deskew=args.deskew, repeats=not args.no_repeats, mirror=not args.no_mirror)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    d = res.design
    if args.style:
        d.meta["style"] = args.style
    d.save(out / f"{stem}_design.png")
    d.save(out / f"{stem}_design.json")
    d.to_image(cell=8).save(out / f"{stem}_preview.png")
    # grid overlay on the rectified photo for checking the fit
    ov = res.image.copy()
    for x in res.grid.xs.astype(int):
        ov[:, np.clip(x, 0, ov.shape[1] - 1)] = (255, 0, 255)
    for y in res.grid.ys.astype(int):
        ov[np.clip(y, 0, ov.shape[0] - 1), :] = (255, 0, 255)
    Image.fromarray(ov).save(out / f"{stem}_grid.png")
    print(json.dumps({k: v for k, v in res.report.items() if k != "structure"}, indent=1, default=str))


def cmd_data(args) -> None:
    from manitex.data import build_procedural_shards, import_designs

    if args.data_cmd == "build-synthetic":
        paths = build_procedural_shards(args.out, args.n, (args.min_size, args.max_size), (args.min_size, args.max_size),
                                        seed=args.seed, shard_size=args.shard_size)
        print("\n".join(str(p) for p in paths))
    elif args.data_cmd == "import":
        print(import_designs(args.src, args.out, style=args.style, domain=args.domain))


def cmd_export(args) -> None:
    from manitex.models.checkpoint import export_slim

    print(export_slim(args.checkpoint, args.out, use_ema=not args.raw, dtype=args.dtype))


def cmd_eval(args) -> None:
    from manitex.generate import GenerationOptions, TextileGenerator
    from manitex.metrics import evaluate, weavability
    from manitex.procedural import generate

    gen = TextileGenerator.from_pretrained(args.checkpoint)
    h, w = _parse_size(args.size)
    rng_seed = args.seed
    designs = gen.generate(args.style, n=args.n, height=h, width=w, seed=rng_seed,
                           options=GenerationOptions(steps=args.steps, cfg=args.cfg))
    if args.reference:
        from manitex.data import read_shards

        ref = []
        for sh in read_shards([args.reference]):
            for i in range(len(sh)):
                g = sh.grid(i)
                if g.shape[0] >= h and g.shape[1] >= w and (args.style in ("", "any") or str(sh.styles[i]) == args.style):
                    ref.append(g[:h, :w].astype(np.int64))
    else:
        ref = [generate(args.style, h, w, seed=10_000 + i).grid for i in range(args.n)]
    res = evaluate([d.grid for d in designs], ref)
    res.update(weavability(designs))
    print(json.dumps(res, indent=1))


def cmd_app(args) -> None:
    from manitex.app import launch

    launch(checkpoint=args.checkpoint, share=args.share, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    from manitex.procedural import PROCEDURAL_STYLES
    from manitex.weave import STRUCTURE_NAMES

    p = argparse.ArgumentParser(prog="manitex", description="Manipuri textile design generation, digitisation and weaving")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_sym(sp):
        sp.add_argument("--mirror-x", action="store_true")
        sp.add_argument("--mirror-y", action="store_true")
        sp.add_argument("--period-x", type=int, default=None)
        sp.add_argument("--period-y", type=int, default=None)
        sp.add_argument("--stripes", choices=["rows", "cols"], default=None)

    sp = sub.add_parser("procedural", help="generate designs with the procedural grammar")
    sp.add_argument("--style", default="moirang_phee", choices=[*PROCEDURAL_STYLES, "random"])
    sp.add_argument("--size", default="64x96")
    sp.add_argument("--n", type=int, default=4)
    sp.add_argument("--colors", type=int, default=None)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--cell", type=int, default=8)
    sp.add_argument("--render", action="store_true")
    sp.add_argument("--booklet", action="store_true", help="also write WIF + HTML/Markdown instructions")
    sp.add_argument("--out", default="outputs/procedural")
    sp.set_defaults(func=cmd_procedural)

    sp = sub.add_parser("generate", help="generate designs with a trained model")
    sp.add_argument("--checkpoint", default=None, help="checkpoint .pt, slim export dir or hf://user/repo")
    sp.add_argument("--style", default="moirang_phee")
    sp.add_argument("--size", default="64x64")
    sp.add_argument("--n", type=int, default=4)
    sp.add_argument("--colors", type=int, default=None)
    sp.add_argument("--palette", default="auto", help="'auto', a preset name, or comma-separated hex colours")
    sp.add_argument("--steps", type=int, default=24)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--cfg", type=float, default=1.0)
    sp.add_argument("--domain", default="real", choices=["real", "synthetic"])
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--cell", type=int, default=8)
    sp.add_argument("--render", action="store_true")
    sp.add_argument("--booklet", action="store_true")
    sp.add_argument("--out", default="outputs/generated")
    add_sym(sp)
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("compile", help="design -> WIF, weaving booklet, chart and cloth render")
    sp.add_argument("design")
    sp.add_argument("--technique", default="brocade", choices=["brocade", "damask"])
    sp.add_argument("--ground-weave", default="plain", choices=list(STRUCTURE_NAMES))
    sp.add_argument("--ground-stripes", default="auto", choices=["auto", "none", "weft", "warp", "plaid"])
    sp.add_argument("--supplementary", default="continuous", choices=["continuous", "discontinuous"])
    sp.add_argument("--ends-per-cell", type=int, default=2)
    sp.add_argument("--picks-per-cell", type=int, default=1)
    sp.add_argument("--max-float", type=int, default=8)
    sp.add_argument("--top-down", action="store_true", help="weave the top row first")
    sp.add_argument("--epc", type=float, default=20.0, help="ends per cm")
    sp.add_argument("--ppc", type=float, default=18.0, help="picks per cm")
    sp.add_argument("--out", default="outputs/compiled")
    sp.set_defaults(func=cmd_compile)

    sp = sub.add_parser("render", help="render a design or WIF draft as simulated cloth")
    sp.add_argument("input")
    sp.add_argument("--out", default="cloth.png")
    sp.add_argument("--px", type=int, default=6)
    sp.add_argument("--ends-per-cell", type=int, default=2)
    sp.add_argument("--face", default="auto", choices=["auto", "balanced", "weft", "warp"])
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("digitise", help="photo of a textile -> clean design grid")
    sp.add_argument("image")
    sp.add_argument("--corners", default=None, help="x1,y1,...,x4,y4 cloth corners in the photo")
    sp.add_argument("--cells-x", type=int, default=None)
    sp.add_argument("--cells-y", type=int, default=None)
    sp.add_argument("--colors", type=int, default=None)
    sp.add_argument("--style", default=None, help="style label stored in the design metadata")
    sp.add_argument("--deskew", action="store_true")
    sp.add_argument("--no-repeats", action="store_true")
    sp.add_argument("--no-mirror", action="store_true")
    sp.add_argument("--out", default="outputs/digitised")
    sp.set_defaults(func=cmd_digitise)

    sp = sub.add_parser("data", help="build / import training shards")
    dsub = sp.add_subparsers(dest="data_cmd", required=True)
    b = dsub.add_parser("build-synthetic")
    b.add_argument("--out", default="data/synthetic")
    b.add_argument("--n", type=int, default=100_000)
    b.add_argument("--min-size", type=int, default=64)
    b.add_argument("--max-size", type=int, default=160)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--shard-size", type=int, default=10_000)
    imp = dsub.add_parser("import")
    imp.add_argument("src")
    imp.add_argument("--out", default="data/real/real-00000.npz")
    imp.add_argument("--style", default=None)
    imp.add_argument("--domain", default="real")
    sp.set_defaults(func=cmd_data)

    sp = sub.add_parser("train", help="train the model (see manitex.train)")
    sp.add_argument("--config", default=None)
    sp.add_argument("overrides", nargs="*")
    sp.set_defaults(func=lambda a: __import__("manitex.train", fromlist=["main"]).main(
        (["--config", a.config] if a.config else []) + a.overrides))

    sp = sub.add_parser("export", help="export a slim inference checkpoint (safetensors)")
    sp.add_argument("checkpoint")
    sp.add_argument("--out", default="exports/manitex")
    sp.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    sp.add_argument("--raw", action="store_true", help="export raw weights instead of EMA")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("eval", help="metric suite for generated designs")
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--style", default="moirang_phee")
    sp.add_argument("--size", default="64x64")
    sp.add_argument("--n", type=int, default=64)
    sp.add_argument("--steps", type=int, default=24)
    sp.add_argument("--cfg", type=float, default=1.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--reference", default=None, help="shard file/dir with reference designs")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("app", help="launch the Gradio demo")
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--share", action="store_true")
    sp.add_argument("--port", type=int, default=7860)
    sp.set_defaults(func=cmd_app)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
