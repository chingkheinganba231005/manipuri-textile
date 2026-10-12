"""Gradio demo: generate, weave and digitise Manipuri textile designs.

Run locally with ``manitex app`` (or ``python -m manitex.app``).  Set the environment
variable ``MANITEX_CHECKPOINT`` to a checkpoint, slim export directory or
``hf://user/repo`` to use a trained model; otherwise the procedural generator is used.
"""


import inspect
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from manitex import __version__
from manitex.color import PALETTE_PRESETS, hex_to_rgb, rgb_to_hex, yarn_name
from manitex.constraints import Symmetry
from manitex.design import Design
from manitex.procedural import STYLE_DESCRIPTIONS

STYLE_LABELS = {
    "moirang_phee": "Moirang phee (temple-motif border)",
    "phanek_mayek_naibi": "Phanek mayek naibi (striped phanek)",
    "banded_motif": "Banded motif shawl",
    "check": "Check",
    "motif_tile": "Motif tile / all-over",
}

ABOUT = """
### About this demo

**manitex** is a research prototype for *generating* and *digitising* textile patterns inspired by
the handloom traditions of **Manipur** — the striped **phanek** and the **Moirang phee** with its rows
of temple-like motifs — and for turning any design into **weaving instructions** a weaver can use.

* **Generate** — a masked discrete-diffusion transformer works directly on the weavable grid of design
  cells. Symmetries, repeats and stripes are imposed *exactly* during sampling. Without a trained
  model, a procedural grammar of these textiles is used.
* **Weave** — every design is compiled into a draft: ground cloth + supplementary-weft pattern picks
  with tie-downs that keep floats short. You get a **WIF** file for weaving software and computer
  looms, a printable booklet with yarn quantities, warping order, a chart and pick-up instructions,
  and a simulated cloth render.
* **Digitise** — photograph a textile, mark its four corners and get a clean design grid back
  (useful for documenting existing pieces and building datasets).

**Cultural note.** These designs belong to living weaving communities. *Moirang Phee* is a registered
Geographical Indication of India (please check the current GI register); generated designs are for
research, education and inspiration — credit Manipuri weavers, and work with them before any
commercial use. Colour presets are approximations, not authenticated colourways.

Code: MIT licence. Version {version}.
"""


# ----------------------------------------------------------------------------- state helpers


def design_to_state(d: Design) -> dict:
    return d.to_dict()


def state_to_design(s: dict | None) -> Design | None:
    return Design.from_dict(s) if s else None


def _tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="manitex_"))


def _preview(d: Design, target: int = 512) -> np.ndarray:
    cell = max(1, min(16, target // max(d.width, d.height, 1)))
    return d.to_rgb(cell=cell)


def _palette_md(d: Design) -> str:
    counts = d.counts()
    rows = []
    for i, c in enumerate(d.palette):
        if counts[i] == 0:
            continue
        hx = rgb_to_hex(c)
        rows.append(f"<span style='display:inline-block;width:14px;height:14px;background:{hx};"
                    f"border:1px solid #444;vertical-align:middle'></span> {i + 1}: {yarn_name(c)} `{hx}` "
                    f"({100 * counts[i] / counts.sum():.0f}%)")
    return "<br>".join(rows)


def parse_palette(text: str | None, preset: str | None) -> Any:
    if text and text.strip():
        cols = [c.strip() for c in text.replace(";", ",").split(",") if c.strip()]
        return [hex_to_rgb(c) for c in cols]
    if preset and preset != "auto":
        return preset
    return "auto"


# ----------------------------------------------------------------------------- core actions


class Studio:
    def __init__(self, checkpoint: str | None = None):
        from manitex.generate import TextileGenerator

        self.error = None
        try:
            self.gen = TextileGenerator.from_pretrained(checkpoint)
        except Exception as e:  # pragma: no cover - depends on the environment
            self.error = f"Could not load the model ({e}); using the procedural generator."
            self.gen = TextileGenerator()

    @property
    def mode_note(self) -> str:
        if self.gen.has_model:
            return f"Neural model loaded (training step {self.gen.info.get('step')})."
        return self.error or ("No trained model configured: designs come from the procedural grammar "
                              "(set `MANITEX_CHECKPOINT` to use a trained model).")

    def generate(self, style: str, n: int, rows: int, cols: int, n_colors: int, mirror_x: bool, mirror_y: bool,
                 period_x: int, stripes: str, preset: str, palette_text: str, seed: int, steps: int,
                 temperature: float, guidance: float) -> tuple[list[np.ndarray], list[dict]]:
        from manitex.generate import GenerationOptions

        sym = Symmetry(mirror_x=bool(mirror_x), mirror_y=bool(mirror_y), period_x=int(period_x) or None,
                       stripes=None if stripes in (None, "", "none") else stripes)
        rows, cols = int(rows), int(cols)
        if self.gen.has_model:
            p = self.gen.model.cfg.patch
            rows, cols = max(p, rows // p * p), max(p, cols // p * p)
        designs = self.gen.generate(style, n=int(n), height=rows, width=cols, n_colors=int(n_colors) or None,
                                    symmetry=sym if sym.any() else None, palette=parse_palette(palette_text, preset),
                                    seed=int(seed),
                                    options=GenerationOptions(steps=int(steps), temperature=float(temperature),
                                                              cfg=float(guidance)))
        for d in designs:
            d.meta.setdefault("style", style)
        return [_preview(d, 256) for d in designs], [design_to_state(d) for d in designs]

    def weave(self, state: dict | None, technique: str, ground_weave: str, ground_stripes: str, supplementary: str,
              ends_per_cell: int, picks_per_cell: int, max_float: int, epc: float, ppc: float):
        from manitex.render import render_draft
        from manitex.weave import Sett, compile_design, design_chart_svg, save_wif, to_html, to_markdown

        d = state_to_design(state)
        if d is None:
            raise ValueError("No design selected - generate, upload or digitise one first.")
        dr = compile_design(d, technique=technique, ground_weave=ground_weave, ground_stripes=ground_stripes,
                            supplementary=supplementary, ends_per_cell=int(ends_per_cell),
                            picks_per_cell=int(picks_per_cell), max_float=int(max_float))
        img = render_draft(dr, max_pixels=2_500_000)
        sett = Sett(ends_per_cm=float(epc), picks_per_cm=float(ppc))
        out = _tmpdir()
        stem = str(d.meta.get("style", "design"))
        files = [
            d.save(out / f"{stem}.png"),
            d.save(out / f"{stem}.json"),
            save_wif(dr, out / f"{stem}.wif", title=stem, ends_per_unit=float(epc), picks_per_unit=float(ppc)),
        ]
        (out / f"{stem}_booklet.html").write_text(to_html(d, dr, sett=sett, title=f"{stem} - weaving booklet",
                                                          fabric_render=img))
        (out / f"{stem}_instructions.md").write_text(to_markdown(d, dr, sett=sett, title=stem))
        (out / f"{stem}_chart.svg").write_text(design_chart_svg(d, title=stem))
        from PIL import Image

        Image.fromarray(img).save(out / f"{stem}_cloth.png")
        files += [out / f"{stem}_booklet.html", out / f"{stem}_instructions.md", out / f"{stem}_chart.svg",
                  out / f"{stem}_cloth.png"]
        summary = to_markdown(d, dr, sett=sett, title="Weaving summary", max_groups=6)
        return img, summary, [str(f) for f in files]

    def digitise(self, image: np.ndarray | None, corners: list, cells_x: int, cells_y: int, n_colors: int,
                 repeats: bool, mirror: bool, deskew: bool, style: str):
        from manitex.digitise import digitise

        if image is None:
            raise ValueError("Upload a photo first.")
        quad = np.array(corners, dtype=float) if corners and len(corners) == 4 else None
        res = digitise(image, quad=quad, cells_x=int(cells_x) or None, cells_y=int(cells_y) or None,
                       n_colors=int(n_colors) or None, repeats=bool(repeats), mirror=bool(mirror), deskew=bool(deskew))
        d = res.design
        if style:
            d.meta["style"] = style
        ov = res.image.copy()
        for x in res.grid.xs.astype(int):
            ov[:, np.clip(x, 0, ov.shape[1] - 1)] = (255, 0, 200)
        for y in res.grid.ys.astype(int):
            ov[np.clip(y, 0, ov.shape[0] - 1), :] = (255, 0, 200)
        rep = res.report
        info = (f"**{d.height} x {d.width} cells**, {rep['k']} colours, cell size ≈ {rep['cell_px'][1]:.1f} x "
                f"{rep['cell_px'][0]:.1f} px.  \nIf the grid lines (magenta) do not follow the motif steps, enter the "
                f"number of cells across / down and run again.<br>" + _palette_md(d))
        out = _tmpdir()
        files = [str(d.save(out / "digitised.png")), str(d.save(out / "digitised.json"))]
        return ov, _preview(d), info, files, design_to_state(d)


def _no_buttons(component) -> dict:
    """Hide the overlay buttons of an image (Gradio 6 ``buttons``; Gradio 5 ``show_*`` flags)."""
    params = inspect.signature(component.__init__).parameters
    if "buttons" in params:
        return {"buttons": []}
    return {k: False for k in ("show_download_button", "show_fullscreen_button", "show_share_button") if k in params}


def _mark_corners(image: np.ndarray | None, corners: list) -> np.ndarray | None:
    if image is None:
        return None
    img = np.array(image).copy()
    h, w = img.shape[:2]
    r = max(3, min(h, w) // 120)
    for x, y in corners:
        x, y = int(round(x)), int(round(y))
        img[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1] = (255, 0, 200)
    if len(corners) == 4:
        from manitex.digitise.geometry import order_quad

        q = order_quad(np.array(corners, float))
        for a, b in zip(q, np.roll(q, -1, axis=0)):
            n = int(max(abs(b - a)) + 1)
            xs = np.linspace(a[0], b[0], n).astype(int).clip(0, w - 1)
            ys = np.linspace(a[1], b[1], n).astype(int).clip(0, h - 1)
            img[ys, xs] = (255, 0, 200)
    return img


# ----------------------------------------------------------------------------- UI


def build_app(checkpoint: str | None = None):
    import gradio as gr

    from manitex.weave import STRUCTURE_NAMES

    studio = Studio(checkpoint)
    styles = studio.gen.styles
    style_choices = [(STYLE_LABELS.get(s, s.replace("_", " ")), s) for s in styles]
    presets = ["auto", *PALETTE_PRESETS]

    with gr.Blocks(title="manitex - Manipuri textile studio") as demo:
        gr.Markdown("# manitex · Manipuri textile design studio\n"
                    "Generate phanek and Moirang phee inspired designs, turn them into weaving instructions, "
                    "or digitise a photo of a textile.")
        gr.Markdown(f"<small>{studio.mode_note}</small>")
        current = gr.State(None)  # the design shared between tabs
        generated = gr.State([])
        corners = gr.State([])
        with gr.Tabs() as tabs:
            # ---------------------------------------------------------------- Generate
            with gr.Tab("Generate", id="generate"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        default_style = "moirang_phee" if "moirang_phee" in styles else styles[0]
                        style = gr.Dropdown(style_choices, value=default_style, label="Style")
                        style_info = gr.Markdown(STYLE_DESCRIPTIONS.get(default_style, ""))
                        with gr.Row():
                            rows = gr.Slider(16, 128, value=64, step=2, label="Rows (cells)")
                            cols = gr.Slider(16, 128, value=64, step=2, label="Columns (cells)")
                        n_colors = gr.Slider(0, 8, value=0, step=1, label="Colours (0 = automatic)")
                        with gr.Accordion("Structure", open=True):
                            with gr.Row():
                                mirror_x = gr.Checkbox(label="Mirror left-right", value=True)
                                mirror_y = gr.Checkbox(label="Mirror top-bottom", value=False)
                            period_x = gr.Slider(0, 64, value=0, step=1, label="Horizontal repeat (cells, 0 = none)")
                            stripes = gr.Radio(["none", "rows", "cols"], value="none", label="Force stripes")
                        with gr.Accordion("Colours", open=False):
                            preset = gr.Dropdown(presets, value="auto", label="Palette preset")
                            palette_text = gr.Textbox(label="Custom palette (hex, most frequent first)",
                                                      placeholder="#F1EADB, #A61E22, #1F1A17")
                        with gr.Accordion("Sampler", open=False):
                            steps = gr.Slider(4, 64, value=32, step=1, label="Steps (fewer = faster)")
                            temperature = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Temperature")
                            guidance = gr.Slider(0.0, 4.0, value=1.0, step=0.1, label="Guidance")
                        with gr.Row():
                            seed = gr.Number(value=0, precision=0, label="Seed")
                            n = gr.Slider(1, 8, value=4, step=1, label="How many")
                        gen_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        gallery = gr.Gallery(label="Designs (click one to select)", columns=4, height=340,
                                             object_fit="contain")
                        with gr.Row():
                            sel_preview = gr.Image(label="Selected design", type="numpy", interactive=False)
                            sel_cloth = gr.Image(label="Simulated cloth", type="numpy", interactive=False)
                        sel_info = gr.Markdown()
                        to_weave = gr.Button("Weaving instructions for this design →")

            # ---------------------------------------------------------------- Weave
            with gr.Tab("Weave", id="weave"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        upload = gr.File(label="Or upload a design (.png / .json)", file_types=[".png", ".json"])
                        technique = gr.Radio(["brocade", "damask"], value="brocade", label="Technique")
                        ground_weave = gr.Dropdown(list(STRUCTURE_NAMES), value="plain", label="Ground weave")
                        ground_stripes = gr.Dropdown(["auto", "none", "weft", "warp", "plaid"], value="auto",
                                                     label="Ground colour stripes")
                        supplementary = gr.Radio(["continuous", "discontinuous"], value="continuous",
                                                 label="Pattern weft")
                        with gr.Row():
                            epc_cell = gr.Slider(1, 8, value=2, step=1, label="Ends per cell")
                            ppc_cell = gr.Slider(1, 4, value=1, step=1, label="Ground picks per row")
                        max_float = gr.Slider(2, 16, value=8, step=1, label="Longest float (ends)")
                        with gr.Row():
                            epc = gr.Number(value=20, label="Ends per cm")
                            ppc = gr.Number(value=18, label="Picks per cm")
                        weave_btn = gr.Button("Make weaving instructions", variant="primary")
                    with gr.Column(scale=2):
                        cloth = gr.Image(label="Simulated cloth (face)", type="numpy", interactive=False)
                        files = gr.File(label="Downloads: WIF, booklet (HTML), instructions, chart, design",
                                        file_count="multiple")
                        summary = gr.Markdown()

            # ---------------------------------------------------------------- Digitise
            with gr.Tab("Digitise", id="digitise"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        photo = gr.Image(label="1. Upload a photo of a textile", type="numpy", height=260)
                        corner_view = gr.Image(label="2. (Optional) click the 4 corners of the cloth", type="numpy",
                                               interactive=False, **_no_buttons(gr.Image))
                        corner_note = gr.Markdown("Corners: none (the whole photo is used)")
                        clear_btn = gr.Button("Clear corners", size="sm")
                        with gr.Row():
                            cells_x = gr.Number(value=0, precision=0, label="Cells across (0 = auto)")
                            cells_y = gr.Number(value=0, precision=0, label="Cells down (0 = auto)")
                        dig_colors = gr.Slider(0, 12, value=0, step=1, label="Colours (0 = automatic)")
                        with gr.Row():
                            dig_rep = gr.Checkbox(value=True, label="Clean up repeats")
                            dig_mir = gr.Checkbox(value=True, label="Clean up symmetry")
                            dig_desk = gr.Checkbox(value=False, label="Straighten")
                        dig_style = gr.Dropdown(["", *STYLE_LABELS], value="", label="Style label (for datasets)")
                        dig_btn = gr.Button("Digitise", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Row():
                            grid_img = gr.Image(label="Rectified photo + fitted grid", type="numpy", interactive=False)
                            dig_preview = gr.Image(label="Digitised design", type="numpy", interactive=False)
                        dig_info = gr.Markdown()
                        dig_files = gr.File(label="Downloads", file_count="multiple")
                        dig_to_weave = gr.Button("Weaving instructions for this design →")

            with gr.Tab("About", id="about"):
                gr.Markdown(ABOUT.format(version=__version__))

        # ---------------------------------------------------------------- events
        style.change(lambda s: STYLE_DESCRIPTIONS.get(s, ""), style, style_info)

        def on_generate(*args):
            previews, states = studio.generate(*args)
            first = states[0] if states else None
            sel = _select(first)
            return (previews, states, first, *sel)

        def _select(state):
            if not state:
                return None, None, ""
            from manitex.render import render_design_cloth

            d = state_to_design(state)
            img, dr = render_design_cloth(d, ends_per_cell=2)
            note = (f"{d.height} x {d.width} cells · {len(d.used_colors())} colours · "
                    f"{dr.summary()['supplementary_picks']} pattern picks<br>" + _palette_md(d))
            return _preview(d), img, note

        gen_inputs = [style, n, rows, cols, n_colors, mirror_x, mirror_y, period_x, stripes, preset, palette_text,
                      seed, steps, temperature, guidance]
        gen_btn.click(on_generate, gen_inputs, [gallery, generated, current, sel_preview, sel_cloth, sel_info])

        def on_pick(states, evt: gr.SelectData):
            state = states[evt.index] if states and evt.index is not None and evt.index < len(states) else None
            return (state, *_select(state))

        gallery.select(on_pick, generated, [current, sel_preview, sel_cloth, sel_info])

        weave_inputs = [current, technique, ground_weave, ground_stripes, supplementary, epc_cell, ppc_cell, max_float,
                        epc, ppc]

        def on_weave(*args):
            try:
                return studio.weave(*args)
            except ValueError as e:
                raise gr.Error(str(e)) from e

        weave_btn.click(on_weave, weave_inputs, [cloth, summary, files])

        def go_weave(state, *rest):
            if not state:
                raise gr.Error("Select a design first.")
            img, md, fl = studio.weave(state, *rest)
            return gr.Tabs(selected="weave"), img, md, fl

        to_weave.click(go_weave, weave_inputs, [tabs, cloth, summary, files])

        def on_upload(f):
            if f is None:
                return None
            path = f if isinstance(f, str) else getattr(f, "name", None) or f["name"]
            try:
                return design_to_state(Design.load(path))
            except Exception as e:
                raise gr.Error(f"Could not read the design: {e}") from e

        upload.change(on_upload, upload, current)

        def on_photo_click(img, pts, evt: gr.SelectData):
            pts = list(pts or [])
            if len(pts) >= 4:
                pts = []
            pts.append([float(evt.index[0]), float(evt.index[1])])
            note = f"Corners: {len(pts)}/4" + (" - ready" if len(pts) == 4 else "")
            return pts, note, _mark_corners(img, pts)

        corner_view.select(on_photo_click, [photo, corners], [corners, corner_note, corner_view])
        no_corners = "Corners: none (the whole photo is used)"
        clear_btn.click(lambda img: ([], no_corners, img), photo, [corners, corner_note, corner_view])
        photo.change(lambda img: ([], no_corners, img), photo, [corners, corner_note, corner_view])

        def on_digitise(img, pts, cx, cy, k, rep, mir, desk, st):
            try:
                ov, prev, info, fl, state = studio.digitise(img, pts, cx or 0, cy or 0, k, rep, mir, desk, st)
            except ValueError as e:
                raise gr.Error(str(e)) from e
            ov = _mark_corners(ov, [])
            return ov, prev, info, fl, state

        dig_btn.click(on_digitise, [photo, corners, cells_x, cells_y, dig_colors, dig_rep, dig_mir, dig_desk, dig_style],
                      [grid_img, dig_preview, dig_info, dig_files, current])
        dig_to_weave.click(go_weave, weave_inputs, [tabs, cloth, summary, files])
    demo.queue(default_concurrency_limit=2)
    return demo


def launch(checkpoint: str | None = None, share: bool = False, port: int = 7860, server_name: str | None = None) -> None:
    import gradio as gr

    demo = build_app(checkpoint or os.environ.get("MANITEX_CHECKPOINT"))
    kwargs: dict[str, Any] = dict(share=share, server_port=port, server_name=server_name)
    theme = gr.themes.Soft(primary_hue="rose", secondary_hue="amber")
    if "theme" in inspect.signature(demo.launch).parameters:  # Gradio >= 6
        kwargs["theme"] = theme
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch()
