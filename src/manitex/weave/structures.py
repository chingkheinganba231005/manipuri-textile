"""Weave structures as binary repeat matrices.

Convention: ``S[pick, end] == 1`` means the warp end is **raised** for that pick, so the
warp lies on top of the weft and is visible on the face of the cloth.
"""

from __future__ import annotations

import math
import re

import numpy as np


def plain() -> np.ndarray:
    return np.array([[1, 0], [0, 1]], dtype=np.uint8)


def twill(up: int, down: int, direction: str = "Z") -> np.ndarray:
    """``up``/``down`` twill (e.g. 2/1).  ``direction`` Z rises to the right, S to the left."""
    if up < 1 or down < 1:
        raise ValueError("twill needs up >= 1 and down >= 1")
    n = up + down
    base = np.array([1] * up + [0] * down, dtype=np.uint8)
    sign = 1 if direction.upper() == "Z" else -1
    return np.stack([np.roll(base, sign * p) for p in range(n)])


def satin(n: int, move: int | None = None, warp_faced: bool = True) -> np.ndarray:
    """``n``-end satin with counter ``move`` (coprime with n, not 1 or n-1)."""
    if n < 5:
        raise ValueError("satins need at least 5 ends")
    if move is None:
        move = next(m for m in range(2, n - 1) if math.gcd(m, n) == 1)
    if math.gcd(move, n) != 1 or move in (1, n - 1):
        raise ValueError(f"invalid satin counter {move} for n={n}")
    s = np.zeros((n, n), dtype=np.uint8)
    for p in range(n):
        s[p, (p * move) % n] = 1  # single warp-up point: weft-faced satin
    return (1 - s) if warp_faced else s


def basket(k: int = 2) -> np.ndarray:
    a = np.ones((k, k), dtype=np.uint8)
    z = np.zeros((k, k), dtype=np.uint8)
    return np.block([[a, z], [z, a]])


def warp_rib(k: int = 2) -> np.ndarray:
    """Plain weave with each pick repeated ``k`` times (ribs run across the weft)."""
    return np.repeat(plain(), k, axis=0)


def weft_rib(k: int = 2) -> np.ndarray:
    return np.repeat(plain(), k, axis=1)


#: Structure names offered in the CLI / demo.
STRUCTURE_NAMES = (
    "plain",
    "twill_2_1",
    "twill_1_2",
    "twill_2_2",
    "twill_3_1",
    "twill_1_3",
    "satin_5_warp",
    "satin_5_weft",
    "basket_2",
    "warp_rib_2",
    "weft_rib_2",
)


def get_structure(name: str) -> np.ndarray:
    """Parse a structure name such as ``plain``, ``twill_2_1``, ``twill_2_2_s``,
    ``satin_5_warp``, ``satin_8_3_weft``, ``basket_2``, ``warp_rib_2``."""
    key = name.strip().lower()
    if key == "plain":
        return plain()
    m = re.fullmatch(r"twill_(\d+)_(\d+)(?:_([zs]))?", key)
    if m:
        return twill(int(m.group(1)), int(m.group(2)), (m.group(3) or "z").upper())
    m = re.fullmatch(r"satin_(\d+)(?:_(\d+))?(?:_(warp|weft))?", key)
    if m:
        return satin(int(m.group(1)), int(m.group(2)) if m.group(2) else None, (m.group(3) or "warp") == "warp")
    m = re.fullmatch(r"basket_(\d+)", key)
    if m:
        return basket(int(m.group(1)))
    m = re.fullmatch(r"(warp|weft)_rib_(\d+)", key)
    if m:
        return warp_rib(int(m.group(2))) if m.group(1) == "warp" else weft_rib(int(m.group(2)))
    raise ValueError(f"unknown weave structure {name!r}")


def tile_structure(s: np.ndarray, n_picks: int, n_ends: int, pick_offset: int = 0, end_offset: int = 0) -> np.ndarray:
    sp, se = s.shape
    rows = (np.arange(n_picks) + pick_offset) % sp
    cols = (np.arange(n_ends) + end_offset) % se
    return s[np.ix_(rows, cols)]
