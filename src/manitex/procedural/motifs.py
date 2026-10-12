"""Parametric motif library.

Every function returns an ``int32`` array of *slots*: 0 = background (transparent), and
1..m = motif colour slots that the caller maps to palette indices.  Shapes are built on the
design-cell grid, so diagonals are staircases ("stepped") exactly as they appear in woven
supplementary-weft motifs.  All motifs are left-right mirror symmetric unless stated
otherwise.

The vocabulary is intentionally generic (triangles / temple spires, stepped pyramids,
diamonds, crosses, stars, hourglasses).  It is a *structural prior* for pretraining, not a
catalogue of authenticated motifs - authentic motif statistics must come from real data.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- primitives


def rows_from_widths(widths: list[int] | np.ndarray, width: int | None = None) -> np.ndarray:
    """Stack centred horizontal spans (row 0 = top).  Span parity is matched to ``width``."""
    widths = [int(w) for w in widths]
    width = int(width if width is not None else max(widths))
    out = np.zeros((len(widths), width), dtype=bool)
    for i, w in enumerate(widths):
        w = min(max(w, 0), width)
        if w == 0:
            continue
        if (width - w) % 2:
            w = w + 1 if w + 1 <= width else w - 1
        left = (width - w) // 2
        out[i, left : left + w] = True
    return out


def triangle(base: int, rise: float = 1.0) -> np.ndarray:
    """Stepped isosceles triangle, apex at the top, base on the bottom row.

    Each level up removes one cell on each side; a level is ``rise`` rows tall on average
    (``rise`` may be fractional, e.g. 1.5 alternates 1- and 2-row steps; values < 1 give
    squat triangles).  Odd bases give a 1-cell apex, even bases a 2-cell apex.
    """
    base = int(base)
    if base < 1:
        raise ValueError("base must be >= 1")
    n_levels = (base + 1) // 2
    # round half up so that the step pattern is regular (1.5 -> 2,1,2,1 ...)
    bounds = np.floor(np.arange(n_levels + 1) * float(rise) + 0.5).astype(int)
    counts = np.diff(bounds)
    counts[0] = max(counts[0], 1)  # the base row always exists
    counts[-1] = max(counts[-1], 1)  # ... and so does the apex
    widths = [base - 2 * k for k in range(n_levels) for _ in range(int(counts[k]))]
    return rows_from_widths(widths[::-1], base)


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary erosion with a 4-neighbourhood (cells outside the array count as empty)."""
    m = mask.astype(bool)
    for _ in range(iterations):
        p = np.pad(m, 1, constant_values=False)
        m = m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def erosion_levels(mask: np.ndarray) -> np.ndarray:
    """Distance-to-boundary level map: 0 outside, 1 on the outer ring, 2 next ring, ..."""
    level = np.zeros(mask.shape, dtype=np.int32)
    m = mask.astype(bool)
    k = 0
    while m.any():
        k += 1
        level[m] = k
        m = erode(m)
    return level


def rings(mask: np.ndarray, n_slots: int = 2, ring_width: int = 1, start_slot: int = 1) -> np.ndarray:
    """Colour a shape with concentric rings cycling through ``n_slots`` slots."""
    lv = erosion_levels(mask)
    out = np.zeros(mask.shape, dtype=np.int32)
    inside = lv > 0
    out[inside] = start_slot + ((lv[inside] - 1) // max(ring_width, 1)) % max(n_slots, 1)
    return out


def place(canvas: np.ndarray, motif: np.ndarray, top: int, left: int) -> np.ndarray:
    """Paint non-zero motif cells onto ``canvas`` in place (clipped at the borders)."""
    h, w = motif.shape
    H, W = canvas.shape
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + h, H), min(left + w, W)
    if y0 >= y1 or x0 >= x1:
        return canvas
    sub = motif[y0 - top : y1 - top, x0 - left : x1 - left]
    region = canvas[y0:y1, x0:x1]
    region[sub != 0] = sub[sub != 0]
    return canvas


# --------------------------------------------------------------------------- motifs


def temple(base: int, rise: float, variant: str = "solid", rng: np.random.Generator | None = None) -> np.ndarray:
    """Temple-spire / tooth motif (apex up) with decorative variants.

    variants: ``solid``, ``outline``, ``nested``, ``spine``, ``split``, ``banded``, ``eyes``.
    """
    rng = rng or np.random.default_rng()
    tri = triangle(base, rise)
    h, w = tri.shape
    out = tri.astype(np.int32)
    if variant == "solid":
        return out
    if variant == "outline":
        inner = erode(tri, 1)
        out[inner] = 0 if rng.random() < 0.5 else 2
        return out
    if variant == "nested":
        return rings(tri, n_slots=2, ring_width=int(rng.integers(1, 3)))
    if variant == "spine":
        cx = (w - 1) / 2.0
        cols = np.abs(np.arange(w) - cx) <= (0.0 if w % 2 else 0.5)
        spine = np.zeros_like(tri)
        top_skip = max(1, int(round(rise)))
        spine[top_skip:, cols] = True
        out[spine & tri] = 2
        return out
    if variant == "split":
        out[:, w // 2 + (w % 2) :][tri[:, w // 2 + (w % 2) :]] = 2
        return out
    if variant == "banded":
        band = int(rng.integers(1, 4))
        rows = (np.arange(h)[::-1] // band) % 2
        out[tri & (rows[:, None] == 1)] = 2
        return out
    if variant == "eyes":
        inner = erode(tri, 1)
        ys, xs = np.nonzero(inner)
        if ys.size:
            yc = int(np.median(ys))
            for dx in (-1, 1):
                x = int(round((w - 1) / 2 + dx * max(1, w // 6)))
                if 0 <= x < w and inner[yc, x]:
                    out[yc, x] = 0 if rng.random() < 0.5 else 2
        return out
    raise ValueError(f"unknown temple variant {variant!r}")


TEMPLE_VARIANTS = ("solid", "outline", "nested", "spine", "split", "banded", "eyes")


def pagoda(base: int, rng: np.random.Generator, n_tiers: int | None = None) -> np.ndarray:
    """Stacked, tapering tiers separated by notches and topped by a finial (temple tower)."""
    base = max(int(base), 5)
    widths: list[int] = []
    w = base
    tiers = 0
    max_tiers = n_tiers if n_tiers is not None else int(rng.integers(2, 5))
    while w >= 3 and tiers < max_tiers:
        th = int(rng.integers(2, 5))
        taper = int(rng.integers(0, 2))
        for r in range(th):
            widths.append(max(w - 2 * taper * r // max(th - 1, 1), 1))
        w = widths[-1] - 2 * int(rng.integers(1, 3))
        if w >= 1:
            widths.append(max(w - 2, 1))  # notch row
        tiers += 1
    finial = int(rng.integers(1, 4))
    widths += [1 if base % 2 else 2] * finial
    mask = rows_from_widths(widths[::-1], base)
    out = mask.astype(np.int32)
    if rng.random() < 0.5:
        out[mask & (erosion_levels(mask) >= 2)] = 2
    return out


def diamond(width: int, rise: float = 1.0) -> np.ndarray:
    """Stepped diamond (lozenge) mask built from two triangles sharing the middle row."""
    top = triangle(width, rise)
    return np.vstack([top, top[::-1][1:]])


def diamond_motif(width: int, rise: float, variant: str, rng: np.random.Generator) -> np.ndarray:
    m = diamond(width, rise)
    if variant == "solid":
        return m.astype(np.int32)
    if variant == "outline":
        out = m.astype(np.int32)
        out[erode(m)] = 0
        return out
    if variant == "nested":
        return rings(m, n_slots=int(rng.integers(2, 4)), ring_width=int(rng.integers(1, 3)))
    if variant == "eye":
        out = m.astype(np.int32)
        lv = erosion_levels(m)
        out[lv >= 2] = 0
        out[lv >= max(3, lv.max() - 1)] = 2
        return out
    raise ValueError(f"unknown diamond variant {variant!r}")


DIAMOND_VARIANTS = ("solid", "outline", "nested", "eye")


def plus(size: int, thickness: int = 1) -> np.ndarray:
    size = max(int(size), 3)
    if (size - thickness) % 2:
        thickness += 1
    m = np.zeros((size, size), dtype=bool)
    a = (size - thickness) // 2
    m[a : a + thickness, :] = True
    m[:, a : a + thickness] = True
    return m


def star(size: int) -> np.ndarray:
    """Eight-pointed star: union of a square and a diamond."""
    size = max(int(size) | 1, 5)
    d = diamond(size, 1.0)[:size, :size]
    sq = np.zeros((size, size), dtype=bool)
    q = size // 4
    sq[q : size - q, q : size - q] = True
    return d | sq


def hourglass(width: int, rise: float = 1.0) -> np.ndarray:
    tri = triangle(width, rise)
    return np.vstack([tri[::-1], tri[1:]])


def small_motif(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Tiny filler motifs (dots, seeds, small crosses / diamonds)."""
    if kind == "dot":
        s = int(rng.integers(1, 3))
        return np.ones((s, s), dtype=np.int32)
    if kind == "seed":
        return diamond(3, 1.0).astype(np.int32)
    if kind == "cross":
        return plus(int(rng.choice([3, 5])), 1).astype(np.int32)
    if kind == "diamond":
        return diamond(int(rng.choice([3, 5])), 1.0).astype(np.int32)
    if kind == "tri":
        return triangle(int(rng.choice([3, 5])), 1.0).astype(np.int32)
    raise ValueError(kind)


SMALL_KINDS = ("dot", "seed", "cross", "diamond", "tri")


def random_motif(rng: np.random.Generator, max_h: int, max_w: int, family: str | None = None) -> np.ndarray:
    """Sample a motif (slots) that fits in ``max_h`` x ``max_w``."""
    family = family or str(rng.choice(["temple", "pagoda", "diamond", "star", "plus", "hourglass"]))
    for _ in range(20):
        if family == "temple":
            base = int(rng.integers(3, max(4, min(max_w, 2 * max_h) + 1)))
            rise = float(rng.choice([0.5, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0]))
            m = temple(base, rise, str(rng.choice(TEMPLE_VARIANTS)), rng)
        elif family == "pagoda":
            m = pagoda(int(rng.integers(5, max(6, max_w + 1))), rng)
        elif family == "diamond":
            w = int(rng.integers(3, max(4, max_w + 1)))
            m = diamond_motif(w, float(rng.choice([1.0, 1.0, 1.5, 2.0])), str(rng.choice(DIAMOND_VARIANTS)), rng)
        elif family == "star":
            m = star(int(rng.integers(5, max(6, min(max_w, max_h) + 1)))).astype(np.int32)
        elif family == "plus":
            m = plus(int(rng.integers(3, max(4, min(max_w, max_h) + 1))), int(rng.integers(1, 3))).astype(np.int32)
        elif family == "hourglass":
            m = hourglass(int(rng.integers(3, max(4, max_w + 1))), float(rng.choice([1.0, 1.5]))).astype(np.int32)
        else:
            raise ValueError(family)
        if m.shape[0] <= max_h and m.shape[1] <= max_w:
            return m.astype(np.int32)
    # fall back to something that always fits
    return np.ones((max(1, min(max_h, 2)), max(1, min(max_w, 2))), dtype=np.int32)
