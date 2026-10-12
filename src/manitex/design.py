"""Core design representation.

A :class:`Design` is a 2-D grid of palette indices plus an RGB palette.  One grid cell
is one *design cell*: the smallest unit of the pattern, which the weave compiler later
maps onto ``ends_per_cell`` warp ends and ``picks_per_cell`` weft picks.

Orientation conventions (used consistently across the package):

* ``grid[0, 0]`` is the top-left cell of the image as seen from the **face** of the cloth.
* Columns run along the weft, i.e. column ``j`` is (a group of) warp end(s) ``j``.
* Rows run along the warp.  Weaving starts at the **bottom** row of the image (the edge
  closest to the weaver) and proceeds upwards, so pick 1 belongs to row ``H - 1``.

Palette indices carry no meaning by themselves; :func:`canonicalize_grid` renumbers them
by decreasing frequency (index 0 = the most frequent colour, usually the ground).  The
generative model works on canonical grids, which separates *structure* from *colour*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, PngImagePlugin

from manitex.color import hex_to_rgb, rgb_to_hex

#: Number of colour slots used by the neural model (canonical indices 0..MAX_COLORS-1).
MAX_COLORS = 16

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PNG_META_KEY = "manitex"
_FORMAT = "manitex.design"


def canonical_order(grid: np.ndarray) -> np.ndarray:
    """Return ``order`` with ``order[new_index] = old_index``.

    Colours are sorted by decreasing cell count; ties are broken by the first occurrence
    in row-major order.  Colours that do not occur in ``grid`` are omitted.
    """
    flat = np.asarray(grid).ravel()
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64)
    if flat.min() < 0:
        raise ValueError("palette indices must be non-negative")
    k = int(flat.max()) + 1
    counts = np.bincount(flat, minlength=k)
    uniq, first = np.unique(flat, return_index=True)
    first_occ = np.full(k, flat.size, dtype=np.int64)
    first_occ[uniq] = first
    used = np.flatnonzero(counts)
    # np.lexsort sorts by the last key first.
    return used[np.lexsort((first_occ[used], -counts[used]))]


def canonicalize_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renumber a grid canonically.

    Returns ``(canonical_grid, order)`` where ``canonical_grid = lut[grid]`` and
    ``order[new] = old``, so a palette is permuted with ``palette[order]``.
    """
    grid = np.asarray(grid)
    order = canonical_order(grid)
    if order.size == 0:
        return grid.astype(np.int64), order
    lut = np.zeros(int(grid.max()) + 1, dtype=np.int64)
    lut[order] = np.arange(order.size)
    return lut[grid], order


def _as_palette(palette: Any) -> np.ndarray:
    if isinstance(palette, (list, tuple)) and palette and isinstance(palette[0], str):
        palette = [hex_to_rgb(p) for p in palette]
    arr = np.asarray(palette)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"palette must have shape (K, 3), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("palette must contain at least one colour")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return np.clip(arr, 0, 255).astype(np.uint8)


@dataclass
class Design:
    """A textile design: ``grid`` (H, W) of palette indices and ``palette`` (K, 3) RGB."""

    grid: np.ndarray
    palette: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        if grid.dtype == bool:
            grid = grid.astype(np.int32)
        if grid.ndim != 2:
            raise ValueError(f"grid must be 2-D, got shape {grid.shape}")
        if not np.issubdtype(grid.dtype, np.integer):
            raise TypeError(f"grid must contain integers, got {grid.dtype}")
        palette = _as_palette(self.palette)
        if palette.shape[0] > 256:
            raise ValueError("at most 256 colours are supported")
        if grid.size and (grid.min() < 0 or grid.max() >= palette.shape[0]):
            raise ValueError(
                f"grid values must lie in [0, {palette.shape[0]}), got [{grid.min()}, {grid.max()}]"
            )
        self.grid = np.ascontiguousarray(grid, dtype=np.int32)
        self.palette = palette
        self.meta = dict(self.meta) if self.meta else {}

    # ------------------------------------------------------------------ properties
    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def n_colors(self) -> int:
        """Number of palette entries (some may be unused, see :meth:`used_colors`)."""
        return int(self.palette.shape[0])

    def used_colors(self) -> np.ndarray:
        return np.flatnonzero(np.bincount(self.grid.ravel(), minlength=self.n_colors))

    def counts(self) -> np.ndarray:
        return np.bincount(self.grid.ravel(), minlength=self.n_colors)

    # ------------------------------------------------------------------ transforms
    def copy(self) -> Design:
        return Design(self.grid.copy(), self.palette.copy(), dict(self.meta))

    def _new(self, grid: np.ndarray, palette: np.ndarray | None = None, **meta: Any) -> Design:
        m = dict(self.meta)
        m.update(meta)
        return Design(grid, self.palette if palette is None else palette, m)

    def canonical(self) -> Design:
        """Palette sorted by decreasing frequency; unused colours removed."""
        grid, order = canonicalize_grid(self.grid)
        return self._new(grid, self.palette[order])

    def with_palette(self, palette: Any) -> Design:
        palette = _as_palette(palette)
        if palette.shape[0] <= int(self.grid.max()):
            raise ValueError("palette has fewer colours than the grid uses")
        return self._new(self.grid, palette)

    def crop(self, top: int, left: int, height: int, width: int) -> Design:
        if top < 0 or left < 0 or top + height > self.height or left + width > self.width:
            raise ValueError("crop window out of bounds")
        return self._new(self.grid[top : top + height, left : left + width].copy())

    def flip_lr(self) -> Design:
        return self._new(self.grid[:, ::-1].copy())

    def flip_ud(self) -> Design:
        return self._new(self.grid[::-1, :].copy())

    def transpose(self) -> Design:
        """Swap rows and columns (i.e. swap the roles of warp and weft)."""
        return self._new(self.grid.T.copy())

    def tile(self, reps_y: int, reps_x: int) -> Design:
        return self._new(np.tile(self.grid, (reps_y, reps_x)))

    # ------------------------------------------------------------------ images
    def to_rgb(self, cell: int = 1) -> np.ndarray:
        """RGB array (H*cell, W*cell, 3) with nearest-neighbour upscaling."""
        rgb = self.palette[self.grid]
        if cell > 1:
            rgb = np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)
        return rgb

    def to_image(self, cell: int = 1, grid_lines: bool = False, line_every: int = 10) -> Image.Image:
        """A viewable PIL image; optional thin grid lines (bold every ``line_every`` cells)."""
        rgb = self.to_rgb(cell).copy()
        if grid_lines and cell >= 4:
            light = np.array([200, 200, 200], dtype=np.int32)
            dark = np.array([70, 70, 70], dtype=np.int32)
            for axis in (0, 1):
                n = self.height if axis == 0 else self.width
                for i in range(1, n):
                    p = i * cell
                    sl = (slice(p, p + 1), slice(None)) if axis == 0 else (slice(None), slice(p, p + 1))
                    col = dark if i % line_every == 0 else light
                    rgb[sl] = ((rgb[sl].astype(np.int32) + col) // 2).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(rgb))

    @classmethod
    def from_rgb(cls, rgb: np.ndarray | Image.Image, max_colors: int = 256, **meta: Any) -> Design:
        """Build a design from an image whose pixels use a small set of exact colours."""
        arr = np.asarray(rgb.convert("RGB") if isinstance(rgb, Image.Image) else rgb)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr[..., :3].astype(np.uint8)
        flat = arr.reshape(-1, 3)
        colors, inverse = np.unique(flat, axis=0, return_inverse=True)
        if colors.shape[0] > max_colors:
            raise ValueError(
                f"image has {colors.shape[0]} distinct colours (> {max_colors}); "
                "this looks like a photo - use `manitex digitise` instead"
            )
        grid = inverse.reshape(arr.shape[:2])
        return cls(grid, colors, meta).canonical()

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "format": _FORMAT,
            "version": 1,
            "height": self.height,
            "width": self.width,
            "palette": [rgb_to_hex(c) for c in self.palette],
            "meta": self.meta,
        }
        if self.n_colors <= len(_ALPHABET):
            d["rows"] = ["".join(_ALPHABET[v] for v in row) for row in self.grid.tolist()]
        else:
            d["grid"] = self.grid.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Design:
        if d.get("format") != _FORMAT:
            raise ValueError("not a manitex design")
        palette = [hex_to_rgb(h) for h in d["palette"]]
        if "rows" in d:
            lut = {ch: i for i, ch in enumerate(_ALPHABET)}
            grid = np.array([[lut[ch] for ch in row] for row in d["rows"]], dtype=np.int32)
        else:
            grid = np.array(d["grid"], dtype=np.int32)
        grid = grid.reshape(int(d["height"]), int(d["width"]))
        return cls(grid, palette, d.get("meta", {}))

    def save(self, path: str | Path, cell: int = 1) -> Path:
        """Save as ``.png`` (indexed, lossless), ``.json`` or ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".png":
            grid = self.grid
            if cell > 1:
                grid = np.repeat(np.repeat(grid, cell, axis=0), cell, axis=1)
            img = Image.frombytes("P", (grid.shape[1], grid.shape[0]), grid.astype(np.uint8).tobytes())
            img.putpalette(self.palette.reshape(-1).tolist())
            info = PngImagePlugin.PngInfo()
            info.add_text(_PNG_META_KEY, json.dumps({"meta": self.meta, "cell": int(cell), "n_colors": self.n_colors}))
            img.save(path, pnginfo=info)
        elif suffix == ".json":
            path.write_text(json.dumps(self.to_dict(), indent=1))
        elif suffix == ".npz":
            np.savez_compressed(path, grid=self.grid.astype(np.uint8), palette=self.palette, meta=np.array(json.dumps(self.meta)))
        else:
            raise ValueError(f"unsupported design format: {suffix}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Design:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            return cls.from_dict(json.loads(path.read_text()))
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"])) if "meta" in z else {}
                return cls(z["grid"].astype(np.int32), z["palette"], meta)
        with Image.open(path) as img:
            img.load()
            text = img.info.get(_PNG_META_KEY) or getattr(img, "text", {}).get(_PNG_META_KEY)
            if img.mode == "P" and text:
                info = json.loads(text)
                grid = np.asarray(img, dtype=np.int32)
                cell = int(info.get("cell", 1))
                if cell > 1:
                    grid = grid[cell // 2 :: cell, cell // 2 :: cell]
                k = int(info.get("n_colors", grid.max() + 1))
                pal = np.array(img.getpalette()[: 3 * max(k, int(grid.max()) + 1)], dtype=np.uint8).reshape(-1, 3)
                return cls(grid, pal, info.get("meta", {}))
            return cls.from_rgb(img, source=str(path.name))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Design({self.height}x{self.width}, {self.n_colors} colours, meta={list(self.meta)})"
