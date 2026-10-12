"""Procedural style generators: a small probabilistic grammar of banded textiles.

Each generator returns ``(grid, palette, params)``.  Generators encode *structural*
conventions that are visible in published photographs of these textiles:

* ``phanek_mayek_naibi`` - striped phanek: groups of thin coloured stripes (often
  palindromic) on a base colour, optionally with a broader border band (mapan).
* ``moirang_phee`` - plain (usually light) ground with border bands made of a row of
  stepped temple / tooth motifs framed by thin lines, mirrored at both ends.
* ``banded_motif`` - generic supplementary-weft shawl layout: a symmetric stack of motif
  rows, zigzags, chevrons and stripe groups.
* ``check`` - checked cloth from crossing warp / weft stripe sequences (optical mixtures
  at crossings).
* ``motif_tile`` - a single large motif or an all-over (half-drop) motif repeat.

The parameters are deliberately broad.  They are a prior for synthetic pretraining and a
fallback for the demo; they are not claims about authentic designs.  Domain experts can
narrow them, and real digitised data should be used to fine-tune the neural model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from manitex.color import mix_rgb, palette_for_style
from manitex.design import Design
from manitex.procedural import bands as B
from manitex.procedural import motifs as M

GeneratorOutput = tuple[np.ndarray, np.ndarray, dict]


def _pick(rng: np.random.Generator, items: Sequence, p: Sequence[float] | None = None):
    if p is not None:
        p = np.asarray(p, dtype=float)
        p = p / p.sum()
    return items[int(rng.choice(len(items), p=p))]


def _colour_seq(rng: np.random.Generator, pool: Sequence[int], n: int) -> list[int]:
    """n colours from ``pool`` with no two consecutive equal (when the pool allows it)."""
    out: list[int] = []
    for _ in range(n):
        choices = [c for c in pool if not out or c != out[-1]] or list(pool)
        out.append(int(_pick(rng, choices)))
    return out


def _fit_temple(rng: np.random.Generator, max_h: int, max_w: int) -> tuple[np.ndarray, dict]:
    """Sample a temple / pagoda motif no taller than ``max_h`` and no wider than ``max_w``."""
    max_h, max_w = max(int(max_h), 2), max(int(max_w), 3)
    for _ in range(30):
        base = int(rng.integers(5, max(6, min(18, max_w + 1))))
        if rng.random() < 0.2:
            m = M.pagoda(base, rng)
            info = {"motif": "pagoda", "base": base}
        else:
            rise = float(_pick(rng, [1.0, 1.5, 2.0, 2.0, 2.5, 3.0]))
            variant = str(_pick(rng, M.TEMPLE_VARIANTS, p=[0.3, 0.15, 0.2, 0.1, 0.05, 0.1, 0.1]))
            m = M.temple(base, rise, variant, rng)
            info = {"motif": "temple", "base": base, "rise": rise, "variant": variant}
        if m.shape[0] <= max_h and m.shape[1] <= max_w:
            return m, info
    base = min(max_w, 2 * max_h - 1)
    base = max(base, 1)
    m = M.temple(base, 1.0, "solid", rng)[-max_h:]
    return m, {"motif": "temple", "base": int(base), "rise": 1.0, "variant": "solid"}


def _body(rng: np.random.Generator, height: int, width: int, ground: int, cols: Sequence[int]) -> tuple[np.ndarray, str]:
    kind = str(_pick(rng, ["plain", "scatter", "pinstripe"], p=[0.55, 0.3, 0.15]))
    if height <= 0:
        return np.zeros((0, width), np.int32), kind
    if kind == "plain" or not cols:
        return B.solid(height, width, ground), "plain"
    if kind == "scatter":
        sm = M.small_motif(str(_pick(rng, M.SMALL_KINDS)), rng)
        px, py = int(rng.integers(7, 17)), int(rng.integers(7, 17))
        arr = B.allover(height, width, sm, [int(_pick(rng, cols))], ground, px, py, half_drop=rng.random() < 0.7,
                        phase_y=int(rng.integers(0, py)))
        return arr, kind
    step = int(rng.integers(5, 13))
    arr = B.solid(height, width, ground)
    c = int(_pick(rng, cols))
    arr[int(rng.integers(0, step)) :: step] = c
    return arr, kind


# --------------------------------------------------------------------------- Moirang phee


def moirang_phee(height: int, width: int, rng: np.random.Generator, n_colors: int | None = None) -> GeneratorOutput:
    k = int(n_colors) if n_colors else int(_pick(rng, [2, 3, 4], p=[0.3, 0.45, 0.25]))
    k = max(k, 2)
    palette, preset = palette_for_style("moirang_phee", k, rng)
    G = 0
    primary = 1
    secondary = 2 if k >= 3 else 1
    accent = 3 if k >= 4 else secondary

    block = B.BandStack(width)  # edge first: row 0 is the cloth edge
    block.add(B.solid(int(_pick(rng, [1, 2, 2, 3, 4])), width, int(_pick(rng, [primary, secondary]))), "edge")
    block.add(B.solid(int(_pick(rng, [0, 1, 1, 2])), width, G), "gap")
    if rng.random() < 0.7:
        block.add(B.solid(1, width, secondary), "line")
        block.add(B.solid(int(_pick(rng, [1, 1, 2])), width, G), "gap")

    max_motif_h = max(3, int(0.42 * height) if height >= 24 else height - block.height - 2)
    motif, minfo = _fit_temple(rng, max_motif_h, max(3, min(width // 2, 17)))
    gap = int(_pick(rng, [0, 0, 0, 1, 2, 3]))
    if k >= 3 and rng.random() < 0.5:
        slot_sets = [[primary, secondary], [secondary, primary]]
        alternating = True
    else:
        slot_sets = [[primary, accent if accent != primary else secondary]]
        alternating = False
    inward = bool(rng.random() < 0.8)
    oriented = motif[::-1] if inward else motif  # in edge-first coordinates "inward" = downwards
    block.add(B.motif_row(width, oriented, slot_sets, G, gap=gap), "temple_row", gap=gap, inward=inward,
              alternating=alternating, **minfo)
    block.add(B.solid(int(_pick(rng, [0, 1, 1])), width, G), "gap")
    if rng.random() < 0.8:
        block.add(B.solid(1, width, primary), "line")
    if rng.random() < 0.5 and height >= 32:
        kind = str(_pick(rng, ["small_tri", *M.SMALL_KINDS]))
        sm = M.triangle(int(_pick(rng, [3, 5])), 1.0).astype(np.int32)[::-1] if kind == "small_tri" else M.small_motif(kind, rng)
        block.add(B.solid(1, width, G), "gap")
        block.add(B.motif_row(width, sm, [[secondary]], G, gap=int(rng.integers(1, 5))), "small_row", motif=kind)
        block.add(B.solid(1, width, G), "gap")
        if rng.random() < 0.6:
            block.add(B.solid(1, width, secondary), "line")

    bh = block.height
    if height >= 2 * bh + 6 and rng.random() < 0.65:
        mode = "full"
    elif height >= bh + 3:
        mode = "single"
    else:
        mode = "band"

    cols = [c for c in (primary, secondary, accent)]
    if mode == "full":
        body, body_kind = _body(rng, height - 2 * bh, width, G, cols)
        grid = np.vstack([block.array(), body, block.flipped().array()])
        layout = block.layout(0) + [{"kind": "body", "body": body_kind, "row0": bh, "row1": height - bh}]
    elif mode == "single":
        body, body_kind = _body(rng, height - bh, width, G, cols)
        if rng.random() < 0.5:  # border at the bottom (edge = last row)
            grid = np.vstack([body, block.flipped().array()])
            layout = [{"kind": "body", "body": body_kind, "row0": 0, "row1": height - bh}] + block.flipped().layout(height - bh)
        else:
            grid = np.vstack([block.array(), body])
            layout = block.layout(0) + [{"kind": "body", "body": body_kind, "row0": bh, "row1": height}]
    else:
        arr = block.array()
        if arr.shape[0] < height:
            arr = np.vstack([arr, B.solid(height - arr.shape[0], width, G)])
        start = int(rng.integers(0, arr.shape[0] - height + 1))
        grid = arr[start : start + height]
        if rng.random() < 0.5:
            grid = grid[::-1]
        layout = [{"kind": "border_crop", "row0": 0, "row1": height}]

    params = {"n_colors": k, "preset": preset, "mode": mode, "layout": layout}
    return grid, palette, params


# --------------------------------------------------------------------------- Phanek mayek naibi


def phanek_mayek_naibi(height: int, width: int, rng: np.random.Generator, n_colors: int | None = None) -> GeneratorOutput:
    k = int(n_colors) if n_colors else int(_pick(rng, [2, 3, 4, 5, 6], p=[0.1, 0.25, 0.3, 0.25, 0.1]))
    k = max(k, 2)
    palette, preset = palette_for_style("phanek_mayek_naibi", k, rng)
    base = 0
    stripe_cols = list(range(1, k))

    def make_group() -> list[tuple[int, int]]:
        n = int(rng.integers(1, 5))
        seq: list[tuple[int, int]] = []
        colours = _colour_seq(rng, stripe_cols, n)
        for i, c in enumerate(colours):
            seq.append((int(_pick(rng, [1, 1, 1, 2, 2, 3, 4])), c))
            if i < n - 1 and rng.random() < 0.5:
                seq.append((int(_pick(rng, [1, 1, 2])), base))
        if rng.random() < 0.65 and len(seq) > 1:
            seq = seq + seq[-2::-1]
        return seq

    groups = [make_group()]
    if rng.random() < 0.4:
        groups.append(make_group())
    gap = int(rng.integers(3, 15))
    seq: list[tuple[int, int]] = []
    gi = 0
    while sum(h for h, _ in seq) < 2 * height + 40:
        seq.append((gap, base))
        seq += groups[gi % len(groups)]
        gi += 1
    body = B.stripes(seq, width)

    has_border = rng.random() < 0.55 and height >= 24
    layout: list[dict] = []
    if has_border:
        border = B.BandStack(width)  # edge first
        edge_c = int(_pick(rng, stripe_cols))
        border.add(B.solid(int(rng.integers(2, 7)), width, edge_c), "mapan_edge")
        border.add(B.solid(int(_pick(rng, [1, 1, 2])), width, base), "gap")
        for c in _colour_seq(rng, stripe_cols, int(rng.integers(1, 4))):
            border.add(B.solid(int(_pick(rng, [1, 1, 2])), width, c), "line")
            border.add(B.solid(1, width, base), "gap")
        if rng.random() < 0.4:
            fam = str(_pick(rng, ["tri", "diamond"]))
            w = int(_pick(rng, [5, 7, 9]))
            sm = M.triangle(w, 1.0)[::-1].astype(np.int32) if fam == "tri" else M.diamond(w, 1.0).astype(np.int32)
            border.add(B.motif_row(width, sm, [[int(_pick(rng, stripe_cols))]], base, gap=int(rng.integers(0, 4))),
                       "mapan_motif", motif=fam, base=w)
            border.add(B.solid(1, width, base), "gap")
            border.add(B.solid(1, width, edge_c), "line")
        border.add(B.solid(int(rng.integers(1, 4)), width, base), "gap")
        barr = border.flipped().array()  # edge at the bottom
        full_bh = barr.shape[0]
        bh = min(full_bh, height - 4)
        barr = barr[-bh:]
        start = int(rng.integers(0, body.shape[0] - (height - bh) + 1))
        grid = np.vstack([body[start : start + height - bh], barr])
        border_layout = [
            {**b, "row0": max(b["row0"], height - bh)}
            for b in border.flipped().layout(height - full_bh)
            if b["row1"] > height - bh
        ]
        layout = [{"kind": "stripes", "row0": 0, "row1": height - bh}] + border_layout
    else:
        for _ in range(50):  # avoid windows that fall entirely inside a base-colour gap
            start = int(rng.integers(0, body.shape[0] - height + 1))
            grid = body[start : start + height]
            if np.unique(grid).size >= 2:
                break
        layout = [{"kind": "stripes", "row0": 0, "row1": height}]

    params = {
        "n_colors": k,
        "preset": preset,
        "gap": gap,
        "groups": [[[int(h), int(c)] for h, c in g] for g in groups],
        "border": bool(has_border),
        "layout": layout,
    }
    return grid.astype(np.int32), palette, params


# --------------------------------------------------------------------------- banded motif


def _random_band(rng: np.random.Generator, width: int, max_h: int, ground: int, cols: list[int]) -> tuple[np.ndarray, dict]:
    kind = str(_pick(rng, ["stripes", "motif", "zigzag", "chevron", "diamond_chain", "dots", "checker"],
                     p=[0.22, 0.34, 0.12, 0.08, 0.1, 0.08, 0.06]))
    max_h = max(int(max_h), 3)
    if kind == "stripes":
        n = int(rng.integers(1, 5))
        seq = []
        for c in _colour_seq(rng, cols, n):
            seq += [(int(_pick(rng, [1, 1, 2, 3])), c), (int(_pick(rng, [0, 1, 1])), ground)]
        return B.stripes(seq, width)[:max_h], {"kind": "stripes"}
    if kind == "motif":
        fam = str(_pick(rng, ["temple", "diamond", "star", "plus", "hourglass", "pagoda"], p=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1]))
        m = M.random_motif(rng, min(max_h, 21), min(max(width // 3, 3), 21), fam)
        if fam == "temple" and rng.random() < 0.5:
            m = m[::-1]
        n_slots = int(m.max())
        c1 = _colour_seq(rng, cols, max(n_slots, 1))
        sets = [c1]
        if rng.random() < 0.35 and len(cols) > 1:
            sets.append(_colour_seq(rng, cols, max(n_slots, 1)))
        pad = int(_pick(rng, [0, 1, 1, 2]))
        arr = B.motif_row(width, m, sets, ground, gap=int(_pick(rng, [0, 1, 2, 3, 4, 6])), pad_top=pad, pad_bottom=pad)
        return arr[:max_h] if arr.shape[0] > max_h else arr, {"kind": "motif_row", "family": fam}
    if kind == "zigzag":
        h = int(rng.integers(3, min(max_h, 12) + 1))
        n_lines = int(rng.integers(1, 3))
        arr = B.zigzag(h, width, int(rng.integers(4, 17)), int(rng.integers(1, 3)), _colour_seq(rng, cols, n_lines), ground,
                       n_lines=n_lines)
        return arr, {"kind": "zigzag"}
    if kind == "chevron":
        h = int(rng.integers(4, min(max_h, 14) + 1))
        palette_cycle = [ground, *_colour_seq(rng, cols, int(rng.integers(1, 3)))]
        return B.chevron(h, width, int(rng.integers(6, 17)), palette_cycle, band=int(rng.integers(1, 3))), {"kind": "chevron"}
    if kind == "diamond_chain":
        w = int(_pick(rng, [5, 7, 9, 11, 13]))
        w = min(w, max(3, 2 * max_h - 1) | 1)
        m = M.diamond_motif(w, 1.0, str(_pick(rng, M.DIAMOND_VARIANTS)), rng)
        m = m[: max_h]
        sets = [_colour_seq(rng, cols, max(int(m.max()), 1))]
        return B.motif_row(width, m, sets, ground, gap=0), {"kind": "diamond_chain"}
    if kind == "dots":
        h = int(rng.integers(3, min(max_h, 10) + 1))
        sm = M.small_motif(str(_pick(rng, M.SMALL_KINDS)), rng)
        return B.dots(h, width, int(rng.integers(4, 10)), sm, int(_pick(rng, cols)), ground), {"kind": "dots"}
    cell = int(rng.integers(1, 4))
    h = min(max_h, cell * int(rng.integers(1, 4)))
    return B.checker(h, width, cell, [ground, int(_pick(rng, cols))]), {"kind": "checker"}


def banded_motif(height: int, width: int, rng: np.random.Generator, n_colors: int | None = None) -> GeneratorOutput:
    k = int(n_colors) if n_colors else int(_pick(rng, [2, 3, 4, 5, 6], p=[0.1, 0.3, 0.3, 0.2, 0.1]))
    k = max(k, 2)
    palette, preset = palette_for_style("banded_motif", k, rng)
    G = 0
    cols = list(range(1, k))
    symmetric = bool(rng.random() < 0.7)
    target = (height + 1) // 2 + 2 if symmetric else height + 2
    stack = B.BandStack(width)
    while stack.height < target:
        arr, info = _random_band(rng, width, max(4, height // 2), G, cols)
        stack.add(arr, info.pop("kind"), **info)
        stack.add(B.solid(int(_pick(rng, [0, 1, 1, 2, 3])), width, G), "gap")
        if rng.random() < 0.35:
            stack.add(B.solid(1, width, int(_pick(rng, cols))), "line")
            stack.add(B.solid(int(_pick(rng, [0, 1, 1])), width, G), "gap")
    if symmetric:
        full = np.vstack([stack.flipped().array(), stack.array()])
        c = full.shape[0] // 2
        top = c - height // 2
        grid = full[top : top + height]
    else:
        grid = stack.array()[:height]
    params = {"n_colors": k, "preset": preset, "symmetric": symmetric, "bands": [i["kind"] for i in stack.info]}
    return grid.astype(np.int32), palette, params


# --------------------------------------------------------------------------- check


def check(height: int, width: int, rng: np.random.Generator, n_colors: int | None = None) -> GeneratorOutput:
    max_m = 3
    if n_colors:
        max_m = 3 if n_colors >= 6 else 2
    m = int(_pick(rng, [2, 3], p=[0.6, 0.4])) if max_m >= 3 else 2
    base_pal, preset = palette_for_style("check", m, rng)

    def sequence() -> list[tuple[int, int]]:
        n = int(rng.integers(2, 6))
        colours = _colour_seq(rng, list(range(m)), n)
        seq = [(int(rng.integers(1, 7)), c) for c in colours]
        if rng.random() < 0.5:
            seq = seq + seq[-2:0:-1]  # palindromic repeat (mirror check)
        return seq

    warp = sequence()
    weft = warp if rng.random() < 0.7 else sequence()

    def expand(seq: list[tuple[int, int]], n: int, phase: int) -> np.ndarray:
        unit = np.concatenate([np.full(h, c) for h, c in seq])
        reps = -(-(n + phase) // unit.size) + 1
        return np.tile(unit, reps)[phase : phase + n].astype(np.int32)

    cols_c = expand(warp, width, int(rng.integers(0, 50)))
    rows_c = expand(weft, height, int(rng.integers(0, 50)))
    R = rows_c[:, None]
    C = cols_c[None, :]
    mix_index = -np.ones((m, m), dtype=np.int32)
    palette = [base_pal[i] for i in range(m)]
    for a in range(m):
        for b in range(a + 1, m):
            mix_index[a, b] = mix_index[b, a] = len(palette)
            palette.append(np.clip(np.rint(mix_rgb(base_pal[a], base_pal[b])), 0, 255).astype(np.uint8))
    grid = np.where(R == C, R, mix_index[R, C])
    params = {"n_colors": int(len(palette)), "preset": preset, "stripe_colors": m,
              "warp": [[int(h), int(c)] for h, c in warp], "weft": [[int(h), int(c)] for h, c in weft]}
    return grid.astype(np.int32), np.array(palette, dtype=np.uint8), params


# --------------------------------------------------------------------------- motif tile


def motif_tile(height: int, width: int, rng: np.random.Generator, n_colors: int | None = None) -> GeneratorOutput:
    k = int(n_colors) if n_colors else int(_pick(rng, [2, 3, 4, 5], p=[0.2, 0.35, 0.3, 0.15]))
    k = max(k, 2)
    palette, preset = palette_for_style("motif_tile", k, rng)
    G = 0
    cols = list(range(1, k))
    mode = str(_pick(rng, ["single", "allover"]))
    grid = B.solid(height, width, G)
    if mode == "single":
        fam = str(_pick(rng, ["temple", "pagoda", "diamond", "star", "plus", "hourglass"], p=[0.25, 0.15, 0.3, 0.1, 0.1, 0.1]))
        lo = max(3, int(0.45 * min(height, width)))
        m = None
        for _ in range(40):
            cand = M.random_motif(rng, height - 2, width - 2, fam)
            if max(cand.shape) >= lo:
                m = cand
                break
        if m is None:
            m = M.diamond_motif(min(height, width) - 2 | 1, 1.0, "nested", rng)[: height - 2, : width - 2]
        slot_cols = _colour_seq(rng, cols, max(int(m.max()), 1))
        vals = B.colorize(m, slot_cols, G)
        B.paste(grid, vals, m != 0, (height - m.shape[0]) // 2, (width - m.shape[1]) // 2)
        if rng.random() < 0.3:
            fc = int(_pick(rng, cols))
            grid[0, :] = grid[-1, :] = fc
            grid[:, 0] = grid[:, -1] = fc
        info = {"family": fam, "motif_shape": [int(s) for s in m.shape]}
    else:
        size_cap = max(3, min(height, width) // 3)
        fam = str(_pick(rng, ["diamond", "star", "plus", "temple", "hourglass"]))
        m = M.random_motif(rng, size_cap, size_cap, fam)
        px = m.shape[1] + int(rng.integers(1, max(2, m.shape[1]) + 2))
        py = m.shape[0] + int(rng.integers(1, max(2, m.shape[0]) + 2))
        slot_cols = _colour_seq(rng, cols, max(int(m.max()), 1))
        grid = B.allover(height, width, m, slot_cols, G, px, py, half_drop=rng.random() < 0.6, phase_y=int(rng.integers(0, py)))
        info = {"family": fam, "px": px, "py": py, "motif_shape": [int(s) for s in m.shape]}
    params = {"n_colors": k, "preset": preset, "mode": mode, **info}
    return grid.astype(np.int32), palette, params


# --------------------------------------------------------------------------- registry

STYLE_GENERATORS: dict[str, Callable[..., GeneratorOutput]] = {
    "phanek_mayek_naibi": phanek_mayek_naibi,
    "moirang_phee": moirang_phee,
    "banded_motif": banded_motif,
    "check": check,
    "motif_tile": motif_tile,
}
PROCEDURAL_STYLES: tuple[str, ...] = tuple(STYLE_GENERATORS)

#: Human-readable descriptions (used by the CLI and the demo).
STYLE_DESCRIPTIONS: dict[str, str] = {
    "phanek_mayek_naibi": "Striped phanek: groups of thin stripes on a base colour, optional border band.",
    "moirang_phee": "Moirang phee: light ground with borders of stepped temple / tooth motifs framed by lines.",
    "banded_motif": "Banded shawl layout: symmetric stack of motif rows, zigzags, chevrons and stripes.",
    "check": "Checked cloth from crossing warp and weft stripe sequences.",
    "motif_tile": "A single large motif, or an all-over half-drop motif repeat.",
}


def generate(
    style: str = "moirang_phee",
    height: int = 64,
    width: int = 64,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    n_colors: int | None = None,
    palette: np.ndarray | Sequence | None = None,
) -> Design:
    """Generate one procedural design.

    Args:
        style: one of :data:`PROCEDURAL_STYLES` or ``"random"``.
        height, width: grid size in design cells (>= 8 each).
        seed / rng: randomness (``rng`` wins when both are given).
        n_colors: requested number of colours (an upper bound for ``check``).
        palette: optional palette overriding the sampled colours (by role index; it is
            extended automatically if too short).
    """
    if height < 8 or width < 8:
        raise ValueError("height and width must be >= 8 design cells")
    rng = rng if rng is not None else np.random.default_rng(seed)
    if style == "random":
        style = str(_pick(rng, PROCEDURAL_STYLES))
    if style not in STYLE_GENERATORS:
        raise ValueError(f"unknown style {style!r}; choose from {PROCEDURAL_STYLES}")
    grid, pal, params = STYLE_GENERATORS[style](height, width, rng, n_colors)
    if grid.shape != (height, width):  # pragma: no cover - internal invariant
        raise AssertionError(f"{style} produced {grid.shape}, expected {(height, width)}")
    if palette is not None:
        from manitex.color import extend_palette
        from manitex.design import _as_palette

        user = _as_palette(palette)
        if user.shape[0] < pal.shape[0]:
            user = extend_palette(user, pal.shape[0], rng)
        pal = user[: max(pal.shape[0], int(grid.max()) + 1)]
    meta = {"style": style, "source": "procedural", "seed": seed, "params": params}
    return Design(grid, pal, meta)
