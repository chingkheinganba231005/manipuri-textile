"""Horizontal band builders.

Each builder returns an ``int32`` array of palette indices with shape ``(height, width)``.
Textile layouts in this package are vertical stacks of such bands (stripes, motif rows,
zigzags, ...), which matches how supplementary-weft borders are woven pick by pick.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from manitex.procedural.motifs import place


@dataclass
class BandStack:
    """Accumulates bands top-to-bottom and records their layout for metadata."""

    width: int
    bands: list[np.ndarray] = field(default_factory=list)
    info: list[dict] = field(default_factory=list)

    def add(self, arr: np.ndarray, kind: str, **params) -> BandStack:
        arr = np.asarray(arr, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != self.width:
            raise ValueError(f"band must have shape (h, {self.width}), got {arr.shape}")
        if arr.shape[0] > 0:
            self.bands.append(arr)
            self.info.append({"kind": kind, "height": int(arr.shape[0]), **params})
        return self

    def extend(self, other: BandStack) -> BandStack:
        for arr, inf in zip(other.bands, other.info):
            self.bands.append(arr)
            self.info.append(dict(inf))
        return self

    @property
    def height(self) -> int:
        return int(sum(b.shape[0] for b in self.bands))

    def array(self) -> np.ndarray:
        if not self.bands:
            return np.zeros((0, self.width), dtype=np.int32)
        return np.vstack(self.bands)

    def flipped(self) -> BandStack:
        """Upside-down copy (band order reversed and each band flipped)."""
        out = BandStack(self.width)
        for arr, inf in zip(self.bands[::-1], self.info[::-1]):
            out.bands.append(arr[::-1].copy())
            out.info.append(dict(inf))
        return out

    def layout(self, top: int = 0) -> list[dict]:
        rows, y = [], top
        for inf in self.info:
            rows.append({**inf, "row0": y, "row1": y + inf["height"]})
            y += inf["height"]
        return rows


def solid(height: int, width: int, color: int) -> np.ndarray:
    return np.full((max(int(height), 0), width), int(color), dtype=np.int32)


def stripes(seq: Sequence[tuple[int, int]], width: int) -> np.ndarray:
    """Stack of full-width stripes given as ``(height, colour)`` pairs."""
    rows = [np.full((int(h), width), int(c), dtype=np.int32) for h, c in seq if int(h) > 0]
    return np.vstack(rows) if rows else np.zeros((0, width), dtype=np.int32)


def colorize(slots: np.ndarray, slot_colors: Sequence[int], bg: int) -> np.ndarray:
    """Map motif slots (0 = background) to palette indices."""
    lut = np.array([bg, *[int(c) for c in slot_colors]], dtype=np.int32)
    idx = np.clip(slots, 0, len(lut) - 1)
    return lut[idx]


def paste(out: np.ndarray, values: np.ndarray, mask: np.ndarray, top: int, left: int) -> np.ndarray:
    """Write ``values[mask]`` into ``out`` at (top, left), clipped to the array bounds."""
    h, w = values.shape
    H, W = out.shape
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + h, H), min(left + w, W)
    if y0 < y1 and x0 < x1:
        sv = values[y0 - top : y1 - top, x0 - left : x1 - left]
        sm = mask[y0 - top : y1 - top, x0 - left : x1 - left]
        region = out[y0:y1, x0:x1]
        region[sm] = sv[sm]
    return out


def motif_row(
    width: int,
    motif: np.ndarray,
    slot_colors: Sequence[Sequence[int]],
    bg: int,
    gap: int = 0,
    pad_top: int = 0,
    pad_bottom: int = 0,
    align: str = "center",
    offset: int = 0,
) -> np.ndarray:
    """A horizontal row of repeated motifs.

    ``slot_colors[n % len(slot_colors)]`` colours instance ``n`` (allows alternating colours).
    With ``align='center'`` the repeat is symmetric about the vertical centre line
    (exactly so when the colour alternation is symmetric too).
    """
    mh, mw = motif.shape
    h = mh + max(int(pad_top), 0) + max(int(pad_bottom), 0)
    out = np.full((h, width), bg, dtype=np.int32)
    period = mw + max(int(gap), 0)
    x0 = ((width - mw) // 2 if align == "center" else 0) + int(offset)
    n_min = -((x0 + mw) // period) - 1
    n_max = (width - x0) // period + 1
    mask = motif != 0
    for n in range(n_min, n_max + 1):
        x = x0 + n * period
        if x >= width or x + mw <= 0:
            continue
        vals = colorize(motif, slot_colors[n % len(slot_colors)], bg)
        paste(out, vals, mask, int(pad_top), x)
    return out


def zigzag(
    height: int,
    width: int,
    period: int,
    thickness: int,
    colors: Sequence[int],
    bg: int,
    n_lines: int = 1,
    spacing: int | None = None,
    phase: int = 0,
) -> np.ndarray:
    """Stepped zigzag line(s): a triangle wave of the given period (in cells)."""
    period = max(int(period), 2)
    thickness = max(int(thickness), 1)
    spacing = spacing if spacing is not None else thickness + 1
    amp = max(height - thickness - (n_lines - 1) * spacing, 0)
    x = np.arange(width) + phase
    half = period / 2.0
    tri = np.abs(((x + (width // 2)) % period) - half) / half  # 0..1, symmetric about centre
    y_top = np.rint(tri * amp).astype(int)
    out = np.full((height, width), bg, dtype=np.int32)
    rows = np.arange(height)[:, None]
    for li in range(n_lines):
        y0 = y_top[None, :] + li * spacing
        on = (rows >= y0) & (rows < y0 + thickness)
        out[on] = colors[li % len(colors)]
    return out


def chevron(height: int, width: int, period: int, colors: Sequence[int], band: int = 1) -> np.ndarray:
    """Filled nested chevrons: zigzag bands of ``band`` rows cycling through ``colors``."""
    period = max(int(period), 2)
    x = np.arange(width)
    half = period / 2.0
    tri = np.abs(((x + width // 2) % period) - half)
    rows = np.arange(height)[:, None]
    level = (rows + np.rint(tri)[None, :].astype(int)) // max(int(band), 1)
    return np.asarray(colors, dtype=np.int32)[level % len(colors)]


def checker(height: int, width: int, cell: int, colors: Sequence[int]) -> np.ndarray:
    cell = max(int(cell), 1)
    yy, xx = np.mgrid[0:height, 0:width]
    xx = xx + (cell - (width // 2) % cell)  # centre-symmetric phase
    return np.asarray(colors, dtype=np.int32)[((yy // cell) + (xx // cell)) % len(colors)]


def dots(height: int, width: int, spacing: int, motif: np.ndarray, color: int, bg: int, stagger: bool = True) -> np.ndarray:
    """Rows of small motifs; with ``stagger`` alternate rows are shifted by half a period."""
    out = np.full((height, width), bg, dtype=np.int32)
    mh, mw = motif.shape
    spacing = max(int(spacing), mw + 1)
    row_step = mh + max(1, spacing - mw)
    canvas = np.zeros((height, width), dtype=np.int32)
    r = 0
    k = 0
    while r < height:
        shift = (spacing // 2) if (stagger and k % 2) else 0
        x = (width // 2 - mw // 2 + shift) % spacing - spacing
        while x < width:
            place(canvas, motif, r, x)
            x += spacing
        r += row_step
        k += 1
    out[canvas != 0] = color
    return out


def allover(
    height: int,
    width: int,
    motif: np.ndarray,
    slot_colors: Sequence[int],
    bg: int,
    px: int,
    py: int,
    half_drop: bool = True,
    phase_y: int = 0,
) -> np.ndarray:
    """Motif repeated on a (half-drop) lattice covering the whole area."""
    mh, mw = motif.shape
    px, py = max(int(px), mw), max(int(py), mh)
    canvas = np.zeros((height, width), dtype=np.int32)
    x_c = width // 2 - mw // 2
    col = 0
    for x in range(x_c - px * (x_c // px + 2), width + px, px):
        dy = (py // 2) if (half_drop and col % 2) else 0
        for y in range(phase_y + dy - py * 2, height + py, py):
            place(canvas, motif, y, x)
        col += 1
    return colorize(canvas, slot_colors, bg)
