"""Printable weaving charts (SVG).

Charts follow the usual hand-weaving convention: **row 1 is at the bottom** (the first
row woven) and cell/end numbering starts at the left as seen on the face.  Each colour
gets a symbol as well as its fill so that charts stay readable in black-and-white prints.
"""

from __future__ import annotations

from html import escape

import numpy as np

from manitex.color import rgb_to_hex, srgb_to_lab, yarn_name
from manitex.design import Design

SYMBOLS = ["", "×", "●", "▲", "■", "◆", "○", "+", "★", "▼",
           "◇", "□", "△", "/", "\\", "≡"]


def symbol_map(colors, ground: int) -> dict[int, str]:
    """Chart symbol per palette index: none for the ground, then symbols in index order."""
    out: dict[int, str] = {}
    i = 0
    for c in sorted(int(x) for x in colors):
        if c == ground:
            out[c] = ""
            continue
        out[c] = SYMBOLS[1 + i % (len(SYMBOLS) - 1)]
        i += 1
    return out


def _ink(rgb: np.ndarray) -> str:
    return "#000000" if srgb_to_lab(np.asarray(rgb, float))[0] > 55 else "#ffffff"


def design_chart_svg(
    design: Design,
    cell: int = 14,
    symbols: bool = True,
    number_every: int = 10,
    title: str | None = None,
    ground: int | None = None,
    max_symbol_cells: int = 40000,
    legend: bool = True,
) -> str:
    """Render the design as an SVG weaving chart (row 1 at the bottom)."""
    grid = design.grid
    H, W = grid.shape
    pal = design.palette
    if ground is None:
        ground = int(design.counts().argmax())
    left, right, top, bottom = 44, 12, 34 if title else 12, 30
    used = design.used_colors()
    symbols_of = symbol_map(used.tolist(), ground)
    legend_h = (22 * ((len(used) + 2) // 3) + 16) if legend else 0
    gw, gh = W * cell, H * cell
    width, height = left + gw + right, top + gh + bottom + legend_h
    out: list[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
               f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, Arial, sans-serif">')
    out.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    if title:
        out.append(f'<text x="{left}" y="22" font-size="15" font-weight="bold">{escape(title)}</text>')
    # cells, merged into horizontal runs
    out.append(f'<g transform="translate({left},{top})" shape-rendering="crispEdges">')
    for r in range(H):
        row = grid[r]
        change = np.flatnonzero(np.diff(row)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [W]])
        for a, b in zip(starts.tolist(), ends.tolist()):
            out.append(f'<rect x="{a * cell}" y="{r * cell}" width="{(b - a) * cell}" height="{cell}" '
                       f'fill="{rgb_to_hex(pal[row[a]])}"/>')
    # grid lines
    if cell >= 6:
        thin, bold = [], []
        for i in range(W + 1):
            (bold if i % number_every == 0 or i == W else thin).append(f"M{i * cell},0V{gh}")
        for j in range(H + 1):
            # bold lines are counted from the bottom row (row 1 at the bottom)
            from_bottom = H - j
            (bold if from_bottom % number_every == 0 or j == 0 else thin).append(f"M0,{j * cell}H{gw}")
        out.append(f'<path d="{"".join(thin)}" stroke="#9a9a9a" stroke-width="0.5" fill="none"/>')
        out.append(f'<path d="{"".join(bold)}" stroke="#222222" stroke-width="1.2" fill="none"/>')
    # symbols
    if symbols and cell >= 9 and H * W <= max_symbol_cells:
        fs = cell * 0.72
        out.append(f'<g font-size="{fs:.1f}" text-anchor="middle" dominant-baseline="central">')
        for c in used.tolist():
            sym = symbols_of[c]
            if not sym:
                continue
            ink = _ink(pal[c])
            ys, xs = np.nonzero(grid == c)
            for y, x in zip(ys.tolist(), xs.tolist()):
                out.append(f'<text x="{x * cell + cell / 2:.1f}" y="{y * cell + cell / 2:.1f}" fill="{ink}">'
                           f'{escape(sym)}</text>')
        out.append("</g>")
    out.append("</g>")
    # numbering: rows from the bottom on the left, columns from the left at the bottom
    out.append('<g font-size="10" fill="#333333">')
    for n in range(number_every, H + 1, number_every):
        y = top + (H - n) * cell + cell / 2
        out.append(f'<text x="{left - 6}" y="{y:.1f}" text-anchor="end" dominant-baseline="central">{n}</text>')
    out.append(f'<text x="{left - 6}" y="{top + gh - cell / 2:.1f}" text-anchor="end" dominant-baseline="central">1</text>')
    for n in range(number_every, W + 1, number_every):
        x = left + (n - 1) * cell + cell / 2
        out.append(f'<text x="{x:.1f}" y="{top + gh + 14}" text-anchor="middle">{n}</text>')
    out.append(f'<text x="{left + cell / 2:.1f}" y="{top + gh + 14}" text-anchor="middle">1</text>')
    out.append("</g>")
    if legend:
        y0 = top + gh + bottom + 4
        out.append('<g font-size="11">')
        for i, c in enumerate(used.tolist()):
            x = left + (i % 3) * 210
            y = y0 + (i // 3) * 22
            hexc = rgb_to_hex(pal[c])
            out.append(f'<rect x="{x}" y="{y}" width="16" height="16" fill="{hexc}" stroke="#333" stroke-width="0.6"/>')
            sym = symbols_of[c]
            if sym:
                out.append(f'<text x="{x + 8}" y="{y + 8}" font-size="11" text-anchor="middle" '
                           f'dominant-baseline="central" fill="{_ink(pal[c])}">{escape(sym)}</text>')
            label = f"{c + 1}: {yarn_name(pal[c])} {hexc}" + (" (ground)" if c == ground else "")
            out.append(f'<text x="{x + 22}" y="{y + 12}">{escape(label)}</text>')
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)
