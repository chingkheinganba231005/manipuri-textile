"""Structural constraints as *orbits*: groups of cells that must share one colour.

Mirror symmetry, horizontal / vertical repeats, 180-degree rotation and stripes are all
expressed as an orbit-id map over the grid.  The sampler pools the model's per-cell
predictions over each orbit and samples one colour per orbit, so the constraints hold
exactly (see :func:`manitex.models.sampler.sample`).  The same idea regularises digitised
grids in :mod:`manitex.digitise.structure`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Symmetry:
    mirror_x: bool = False  # left-right mirror (about the vertical centre line)
    mirror_y: bool = False  # top-bottom mirror
    rotate180: bool = False
    period_x: int | None = None  # horizontal repeat (cells)
    period_y: int | None = None  # vertical repeat (cells)
    stripes: str | None = None  # 'rows' (every row uniform) | 'cols'

    def any(self) -> bool:
        return bool(self.mirror_x or self.mirror_y or self.rotate180 or self.period_x or self.period_y or self.stripes)

    def to_dict(self) -> dict:
        return asdict(self)


def make_orbits(height: int, width: int, sym: Symmetry | None = None) -> np.ndarray:
    """Orbit id (the smallest flat index in the orbit) for every cell, shape (H, W)."""
    H, W = int(height), int(width)
    ids = np.arange(H * W, dtype=np.int64).reshape(H, W)
    if sym is None or not sym.any():
        return ids
    rr, cc = np.mgrid[0:H, 0:W]
    partners: list[np.ndarray] = []
    if sym.mirror_x:
        partners.append(rr * W + (W - 1 - cc))
    if sym.mirror_y:
        partners.append((H - 1 - rr) * W + cc)
    if sym.rotate180:
        partners.append((H - 1 - rr) * W + (W - 1 - cc))
    if sym.period_x and 0 < sym.period_x < W:
        p = int(sym.period_x)
        partners.append(rr * W + (cc % p))
    if sym.period_y and 0 < sym.period_y < H:
        p = int(sym.period_y)
        partners.append((rr % p) * W + cc)
    if sym.stripes == "rows":
        partners.append(rr * W)
    elif sym.stripes == "cols":
        partners.append(cc)
    elif sym.stripes not in (None, ""):
        raise ValueError(f"unknown stripes mode {sym.stripes!r}")
    flat = ids.ravel().copy()
    pts = [p.ravel() for p in partners]
    # label propagation with pointer jumping until every relation is satisfied
    for _ in range(4 * (H + W) + 8):
        old = flat.copy()
        for p in pts:
            m = np.minimum(flat, flat[p])
            np.minimum.at(flat, p, m)
            flat = np.minimum(flat, m)
        flat = flat[flat]  # pointer jumping
        if np.array_equal(flat, old):
            break
    return flat.reshape(H, W)


def orbit_index(orbits: np.ndarray) -> tuple[np.ndarray, int]:
    """Compact orbit ids 0..n-1 (row-major order of first appearance) and their count."""
    uniq, inv = np.unique(orbits.ravel(), return_inverse=True)
    return inv.reshape(orbits.shape), int(uniq.size)


def satisfies(grid: np.ndarray, orbits: np.ndarray) -> bool:
    """True if every orbit is monochrome."""
    idx, n = orbit_index(orbits)
    first = np.full(n, -1, dtype=np.int64)
    flat_idx, flat_val = idx.ravel(), grid.ravel()
    first[flat_idx[::-1]] = flat_val[::-1]
    return bool(np.all(first[flat_idx] == flat_val))


def project(grid: np.ndarray, orbits: np.ndarray) -> np.ndarray:
    """Majority vote within every orbit (makes any grid satisfy the constraints)."""
    idx, n = orbit_index(orbits)
    k = int(grid.max()) + 1
    counts = np.zeros((n, k), dtype=np.int64)
    np.add.at(counts, (idx.ravel(), grid.ravel()), 1)
    return counts.argmax(axis=1)[idx]
