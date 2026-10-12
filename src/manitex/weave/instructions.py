"""Weaver-facing instructions: materials, warping plan, loom set-up and pick-by-pick steps.

The instructions are generated from a :class:`~manitex.weave.draft.WeaveDraft`, so they
are always consistent with the WIF file and the charts.  Supplementary (pattern) picks
are written for **pick-up** weaving, which needs no extra shafts:

* ``L n`` - lift ``n`` ends with the pick-up stick (the pattern yarn passes *under* them
  and is hidden on the back),
* ``D n`` - leave ``n`` ends down (the pattern yarn passes *over* them and shows).

Single lifted ends inside a long ``D`` run are tie-downs that bind long floats.
"""

from __future__ import annotations

import base64
import io
from dataclasses import asdict, dataclass
from html import escape
from typing import Any

import numpy as np
from PIL import Image

from manitex.color import rgb_to_hex, yarn_name
from manitex.design import Design
from manitex.weave.chart import design_chart_svg
from manitex.weave.draft import PICK_GROUND, PICK_SUPPLEMENTARY, WeaveDraft


@dataclass
class Sett:
    """Cloth density and allowances used for size and yarn estimates."""

    ends_per_cm: float = 20.0
    picks_per_cm: float = 18.0  # ground picks per cm; pattern picks count by their thickness
    loom_waste_cm: float = 60.0
    warp_take_up: float = 0.10
    weft_take_up: float = 0.08


def run_length(seq: np.ndarray | list) -> list[tuple[int, int]]:
    """``[(value, count), ...]`` for consecutive equal values."""
    a = np.asarray(seq)
    if a.size == 0:
        return []
    change = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [a.size]])
    return [(int(a[s]), int(e - s)) for s, e in zip(starts, ends)]


def colour_label(palette: np.ndarray, idx: int) -> str:
    return f"{idx + 1}:{yarn_name(palette[idx])}"


def cloth_size_cm(draft: WeaveDraft, sett: Sett) -> tuple[float, float]:
    width = draft.n_ends / sett.ends_per_cm
    length = float(draft.weft_thickness.sum()) / sett.picks_per_cm
    return width, length


def yarn_requirements(draft: WeaveDraft, sett: Sett | None = None) -> list[dict[str, Any]]:
    """Per-colour yarn lengths (metres), including take-up and loom waste."""
    sett = sett or Sett()
    width_cm, length_cm = cloth_size_cm(draft, sett)
    warp_len_m = (length_cm * (1 + sett.warp_take_up) + sett.loom_waste_cm) / 100.0
    spacing_cm = 1.0 / sett.ends_per_cm
    res = []
    k = draft.palette.shape[0]
    warp_counts = np.bincount(draft.warp_colors, minlength=k)
    weft_m = np.zeros(k)
    weft_picks = np.zeros(k, dtype=np.int64)
    for p in range(draft.n_picks):
        if draft.inlay is not None and draft.pick_kind[p] == PICK_SUPPLEMENTARY and (draft.inlay[p] >= 0).any():
            row = draft.inlay[p]
            for c in np.unique(row[row >= 0]):
                n_ends = int((row == c).sum())
                # inlay yarn covers its region plus ~2 cm of turn-back allowance per pick
                weft_m[c] += (n_ends * spacing_cm * (1 + sett.weft_take_up) + 2.0) / 100.0
                weft_picks[c] += 1
        else:
            c = draft.weft_colors[p]
            weft_m[c] += width_cm * (1 + sett.weft_take_up) / 100.0
            weft_picks[c] += 1
    for c in range(k):
        if warp_counts[c] == 0 and weft_picks[c] == 0:
            continue
        res.append({
            "index": c + 1,
            "name": yarn_name(draft.palette[c]),
            "hex": rgb_to_hex(draft.palette[c]),
            "warp_ends": int(warp_counts[c]),
            "warp_m": round(float(warp_counts[c] * warp_len_m), 1),
            "weft_picks": int(weft_picks[c]),
            "weft_m": round(float(weft_m[c]), 1),
            "total_m": round(float(warp_counts[c] * warp_len_m + weft_m[c]), 1),
        })
    return res


def pattern_pick_code(lift: np.ndarray) -> str:
    """Run-length pick-up code for one supplementary pick (see module docstring)."""
    return " ".join(("L" if v else "D") + str(n) for v, n in run_length(lift.astype(np.int8)))


def _row_signature(draft: WeaveDraft, picks: list[int]) -> tuple:
    sig = []
    for p in picks:
        if draft.pick_kind[p] == PICK_GROUND:
            sig.append(("G", int(draft.weft_colors[p])))
        else:
            inl = None if draft.inlay is None else tuple(run_length(draft.inlay[p]))
            sig.append(("S", int(draft.weft_colors[p]), draft.liftplan[p].tobytes(), inl))
    return tuple(sig)


def weaving_steps(draft: WeaveDraft) -> list[dict[str, Any]]:
    """Group picks by design row (in weaving order) and merge identical consecutive rows.

    Each group has ``rows`` (1-based chart rows, inclusive), ``picks`` (1-based, inclusive),
    ``pick_list`` (0-based picks of the first row in the group) and a ``signature``.
    Drafts without design-row information (e.g. read from WIF) use one row per pick.
    """
    groups: list[dict[str, Any]] = []
    if draft.n_picks == 0:
        return groups
    row_ids = draft.pick_row
    if np.all(row_ids < 0):
        bounds = np.arange(1, draft.n_picks)
    else:
        bounds = np.flatnonzero(np.diff(row_ids) != 0) + 1
    starts = np.concatenate([[0], bounds]).astype(int)
    ends = np.concatenate([bounds, [draft.n_picks]]).astype(int)
    for chart_row, (s, e) in enumerate(zip(starts.tolist(), ends.tolist()), 1):
        picks = list(range(s, e))
        sig = _row_signature(draft, picks)
        if groups and groups[-1]["signature"] == sig:
            groups[-1]["rows"][1] = chart_row
            groups[-1]["picks"][1] = e
            continue
        groups.append({"rows": [chart_row, chart_row], "picks": [s + 1, e], "signature": sig, "pick_list": picks})
    return groups


def _describe_group(draft: WeaveDraft, g: dict[str, Any]) -> list[str]:
    pal = draft.palette
    lines = []
    for p in g["pick_list"]:
        if draft.pick_kind[p] == PICK_GROUND:
            lines.append(f"ground pick, next ground shed, weft {colour_label(pal, int(draft.weft_colors[p]))}")
            continue
        code = pattern_pick_code(draft.liftplan[p])
        if draft.inlay is not None and (draft.inlay[p] >= 0).any():
            regions = ", ".join(f"{colour_label(pal, c)}x{n}" for c, n in run_length(draft.inlay[p]) if c >= 0)
            lines.append(f"pattern pick (inlay: {regions}): {code}")
        else:
            lines.append(f"pattern pick, yarn {colour_label(pal, int(draft.weft_colors[p]))}: {code}")
    return lines


def loom_setup(draft: WeaveDraft) -> dict[str, Any]:
    threading, lifts = draft.threading()
    ground = draft.pick_kind == PICK_GROUND
    sub = WeaveDraft(draft.liftplan[ground], draft.warp_colors, draft.weft_colors[ground], draft.palette) \
        if ground.any() else None
    ground_shafts = sub.n_shafts if sub is not None else 0
    return {
        "shafts_for_full_design": int(lifts.shape[1]),
        "treadles_for_full_design": int(draft.treadling()[1].shape[0]) if draft.n_picks else 0,
        "shafts_for_ground_only": int(ground_shafts),
        "threading_rle": [(s + 1, n) for s, n in run_length(threading)],
    }


def to_markdown(design: Design, draft: WeaveDraft, sett: Sett | None = None, title: str = "Weaving instructions",
                max_groups: int | None = None) -> str:
    sett = sett or Sett()
    pal = draft.palette
    w_cm, l_cm = cloth_size_cm(draft, sett)
    floats = draft.float_report()
    setup = loom_setup(draft)
    opts = draft.meta.get("options", {})
    out: list[str] = [f"# {title}", ""]
    out.append("## Overview")
    out.append(f"- Design: {design.width} x {design.height} cells, {len(design.used_colors())} colours")
    out.append(f"- Technique: {draft.meta.get('technique', 'custom')}"
               + (f", ground weave {opts.get('ground_weave')}, ground stripes: {draft.meta.get('ground_mode')}"
                  if draft.meta.get("technique") == "brocade" else ""))
    out.append(f"- Cloth: {draft.n_ends} ends x {draft.n_picks} picks "
               f"({int((draft.pick_kind == PICK_GROUND).sum())} ground, "
               f"{int((draft.pick_kind == PICK_SUPPLEMENTARY).sum())} pattern)")
    out.append(f"- Estimated size at {sett.ends_per_cm:g} ends/cm and {sett.picks_per_cm:g} picks/cm: "
               f"{w_cm:.1f} cm wide x {l_cm:.1f} cm long (before finishing)")
    if opts:
        out.append(f"- Each design cell = {opts.get('ends_per_cell')} ends x {opts.get('picks_per_cell')} ground pick(s)")
    out.append(f"- Longest pattern float on the face: {floats['supp_weft_face']} ends"
               + (f" (limit {opts.get('max_float')}; {draft.meta.get('tie_downs', 0)} tie-downs added)" if opts else ""))
    if floats["supp_weft_back"]:
        out.append(f"- Longest pattern float on the back: {floats['supp_weft_back']} ends "
                   "(continuous pattern weft; secure or trim long back floats if the cloth will be handled roughly)")
    out.append("")
    out.append("## Yarns")
    out.append("| # | colour | hex | warp ends | warp (m) | weft picks | weft (m) | total (m) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for y in yarn_requirements(draft, sett):
        out.append(f"| {y['index']} | {y['name']} | {y['hex']} | {y['warp_ends']} | {y['warp_m']} | "
                   f"{y['weft_picks']} | {y['weft_m']} | {y['total_m']} |")
    out.append("")
    out.append(f"Lengths include {sett.warp_take_up:.0%} warp take-up, {sett.weft_take_up:.0%} weft take-up and "
               f"{sett.loom_waste_cm:g} cm loom waste. Add 10% for safety.")
    out.append("")
    out.append("## Warping (colour order, left to right)")
    out.append(", ".join(f"{n} x {colour_label(pal, c)}" for c, n in run_length(draft.warp_colors)))
    out.append("")
    out.append("## Loom set-up")
    out.append(f"- Ground weave needs {setup['shafts_for_ground_only']} shafts (or heddle + shed rod on a loin loom).")
    out.append("- Pattern picks are written for pick-up: no extra shafts needed.")
    out.append(f"- To weave everything on a dobby / jacquard instead: {setup['shafts_for_full_design']} shafts "
               f"({setup['treadles_for_full_design']} distinct sheds). Use the WIF file for the full lift plan.")
    out.append("")
    out.append("## Pick-by-pick (row 1 = first row woven = bottom of the chart)")
    out.append("Pick-up code: `L n` = lift n ends (pattern yarn hidden), `D n` = leave n ends down "
               "(pattern yarn shows). Ends are counted from the left edge of the face.")
    out.append("")
    groups = weaving_steps(draft)
    for gi, g in enumerate(groups):
        if max_groups is not None and gi >= max_groups:
            out.append(f"... {len(groups) - max_groups} more row groups (see the HTML booklet or WIF file)")
            break
        r0, r1 = g["rows"]
        label = f"Row {r0}" if r0 == r1 else f"Rows {r0}-{r1} (repeat each row {r1 - r0 + 1} times)"
        out.append(f"**{label}** - picks {g['picks'][0]}-{g['picks'][1]}")
        for line in _describe_group(draft, g):
            out.append(f"  - {line}")
    out.append("")
    return "\n".join(out)


def _png_data_uri(img: np.ndarray | Image.Image, max_side: int = 1400) -> str:
    im = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img, dtype=np.uint8))
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_CSS = """
body{font-family:'DejaVu Sans',Arial,sans-serif;color:#1d1d1d;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.45}
h1{font-size:24px;margin-bottom:4px} h2{font-size:18px;border-bottom:2px solid #8b1e2b;padding-bottom:3px;margin-top:28px}
table{border-collapse:collapse;margin:8px 0} td,th{border:1px solid #bbb;padding:4px 8px;font-size:13px;text-align:left}
.sw{display:inline-block;width:14px;height:14px;border:1px solid #333;vertical-align:middle;margin-right:4px}
.imgs{display:flex;gap:16px;flex-wrap:wrap} .imgs figure{margin:0} .imgs img{max-width:520px;width:100%;image-rendering:pixelated;border:1px solid #ccc}
figcaption{font-size:12px;color:#555} code{background:#f3f0ea;padding:1px 4px;border-radius:3px;font-size:12px;word-break:break-all}
.small{font-size:12px;color:#555} .chart{overflow:auto;border:1px solid #ddd;padding:4px}
@media print{.chart{overflow:visible} h2{page-break-after:avoid} .pg{page-break-before:always}}
"""


def to_html(
    design: Design,
    draft: WeaveDraft,
    sett: Sett | None = None,
    title: str = "Weaving instructions",
    fabric_render: np.ndarray | Image.Image | None = None,
    chart_cell: int | None = None,
    notes: list[str] | None = None,
) -> str:
    """A self-contained, printable HTML weaving booklet."""
    sett = sett or Sett()
    pal = draft.palette
    w_cm, l_cm = cloth_size_cm(draft, sett)
    floats = draft.float_report()
    setup = loom_setup(draft)
    opts = draft.meta.get("options", {})
    if chart_cell is None:
        chart_cell = int(np.clip(900 // max(design.width, 1), 6, 16))
    chart = design_chart_svg(design, cell=chart_cell, title=None)
    h: list[str] = ["<!doctype html><html><head><meta charset='utf-8'>",
                    "<meta name='viewport' content='width=device-width, initial-scale=1'>",
                    f"<title>{escape(title)}</title><style>{_CSS}</style></head><body>"]
    h.append(f"<h1>{escape(title)}</h1>")
    h.append(f"<p class='small'>Generated with manitex. Technique: <b>{escape(str(draft.meta.get('technique', 'custom')))}</b>"
             + (f", ground weave <b>{escape(str(opts.get('ground_weave')))}</b>" if opts else "") + "</p>")
    h.append("<div class='imgs'>")
    h.append(f"<figure><img src='{_png_data_uri(design.to_rgb(cell=max(1, 480 // max(design.width, 1))))}' alt='design'>"
             "<figcaption>Design (one square = one design cell)</figcaption></figure>")
    if fabric_render is not None:
        h.append(f"<figure><img src='{_png_data_uri(fabric_render)}' alt='simulated cloth'>"
                 "<figcaption>Simulated woven cloth (face)</figcaption></figure>")
    h.append("</div>")
    h.append("<h2>Overview</h2><table>")
    rows = [
        ("Design", f"{design.width} x {design.height} cells, {len(design.used_colors())} colours"),
        ("Cloth", f"{draft.n_ends} ends x {draft.n_picks} picks ({int((draft.pick_kind == PICK_GROUND).sum())} ground, "
                  f"{int((draft.pick_kind == PICK_SUPPLEMENTARY).sum())} pattern)"),
        ("Estimated size", f"{w_cm:.1f} cm x {l_cm:.1f} cm at {sett.ends_per_cm:g} ends/cm, {sett.picks_per_cm:g} picks/cm"),
        ("Pattern floats", f"face max {floats['supp_weft_face']} ends, back max {floats['supp_weft_back']} ends; "
                           f"{draft.meta.get('tie_downs', 0)} tie-downs"),
        ("Loom", f"ground: {setup['shafts_for_ground_only']} shafts; everything on dobby/jacquard: "
                 f"{setup['shafts_for_full_design']} shafts / {setup['treadles_for_full_design']} sheds"),
    ]
    if opts:
        rows.append(("Cell size", f"{opts.get('ends_per_cell')} ends x {opts.get('picks_per_cell')} ground pick(s)"))
    for k, v in rows:
        h.append(f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>")
    h.append("</table>")
    h.append("<h2>Yarns</h2><table><tr><th>#</th><th>colour</th><th>warp ends</th><th>warp (m)</th>"
             "<th>weft picks</th><th>weft (m)</th><th>total (m)</th></tr>")
    for y in yarn_requirements(draft, sett):
        h.append(f"<tr><td>{y['index']}</td><td><span class='sw' style='background:{y['hex']}'></span>"
                 f"{escape(y['name'])} <span class='small'>{y['hex']}</span></td><td>{y['warp_ends']}</td>"
                 f"<td>{y['warp_m']}</td><td>{y['weft_picks']}</td><td>{y['weft_m']}</td><td>{y['total_m']}</td></tr>")
    h.append("</table>")
    h.append(f"<p class='small'>Includes {sett.warp_take_up:.0%} warp / {sett.weft_take_up:.0%} weft take-up and "
             f"{sett.loom_waste_cm:g} cm loom waste; add ~10% for safety.</p>")
    h.append("<h2>Warping order (left to right)</h2><p>")
    h.append(", ".join(f"<span class='sw' style='background:{rgb_to_hex(pal[c])}'></span>{n}&times;{escape(colour_label(pal, c))}"
                       for c, n in run_length(draft.warp_colors)))
    h.append("</p>")
    h.append("<h2 class='pg'>Chart</h2><p class='small'>Row 1 (bottom) is woven first. Bold lines every 10 cells.</p>")
    h.append(f"<div class='chart'>{chart}</div>")
    h.append("<h2 class='pg'>Pick-by-pick</h2>")
    h.append("<p class='small'>Pick-up code: <code>L n</code> = lift n ends (pattern yarn hidden underneath), "
             "<code>D n</code> = leave n ends down (pattern yarn floats on top and shows). Ends are counted from the "
             "left edge of the face. Single <code>L1</code> inside a long <code>D</code> run is a tie-down.</p>")
    h.append("<table><tr><th>rows</th><th>picks</th><th>sequence for each row</th></tr>")
    for g in weaving_steps(draft):
        r0, r1 = g["rows"]
        rr = f"{r0}" if r0 == r1 else f"{r0}&ndash;{r1}"
        lines = "<br>".join(f"<code>{escape(x)}</code>" if ": L" in x or ": D" in x else escape(x)
                            for x in _describe_group(draft, g))
        h.append(f"<tr><td>{rr}</td><td>{g['picks'][0]}&ndash;{g['picks'][1]}</td><td>{lines}</td></tr>")
    h.append("</table>")
    if notes:
        h.append("<h2>Notes</h2><ul>" + "".join(f"<li>{escape(n)}</li>" for n in notes) + "</ul>")
    h.append("<p class='small'>Options: " + escape(str({k: v for k, v in opts.items() if k != 'extra'})) + "</p>")
    h.append("</body></html>")
    return "\n".join(h)


def sett_from_dict(d: dict[str, Any] | None) -> Sett:
    if not d:
        return Sett()
    base = asdict(Sett())
    base.update({k: float(v) for k, v in d.items() if k in base})
    return Sett(**base)
