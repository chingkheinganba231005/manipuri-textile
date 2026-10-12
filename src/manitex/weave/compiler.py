"""Design -> weave draft compiler.

Two techniques are supported:

``brocade`` (default)
    A ground cloth (plain weave by default) whose colours come from the design's ground
    colour, optionally with weft-wise or warp-wise colour stripes, plus a
    **supplementary weft** for every other cell.  This is how the motifs of borders such
    as the Moirang phee temple row are commonly woven, and pure stripe designs (e.g. a
    striped phanek) compile to stripes with no supplementary picks at all.

    For every design row (in weaving order) the compiler emits ``picks_per_cell``
    repetitions of: one ground pick, followed by one supplementary pick per pattern colour
    in that row (``continuous`` supplementary weft, selvedge to selvedge, floating on the
    back where hidden) or a single pick carrying all colours as separate inlay yarns
    (``discontinuous``).  Face floats longer than ``max_float`` ends are bound by
    **tie-down** ends placed on a staggered lattice.

``damask``
    Two-colour designs as a jacquard damask: warp-faced satin where the ground colour is,
    weft-faced satin elsewhere (warp = ground colour, weft = figure colour).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from manitex.color import delta_e2000, mix_rgb, srgb_to_lab
from manitex.design import Design
from manitex.weave.draft import PICK_GROUND, PICK_SUPPLEMENTARY, WeaveDraft, runs
from manitex.weave.structures import get_structure, satin


@dataclass
class CompileOptions:
    technique: str = "brocade"  # 'brocade' | 'damask'
    ground_weave: str = "plain"
    ground_stripes: str = "auto"  # 'auto' | 'none' | 'weft' | 'warp' | 'plaid'
    stripe_threshold: float = 0.9  # a row/column counts as a stripe if >= this fraction is one colour
    ground_color: int | None = None  # default: the most frequent colour
    supplementary: str = "continuous"  # 'continuous' | 'discontinuous'
    ends_per_cell: int = 2
    picks_per_cell: int = 1  # ground picks per design row (brocade) / picks per row (damask)
    max_float: int = 8  # longest allowed supplementary face float, in ends
    tie_stagger: str = "brick"  # 'brick' | 'twill' | 'aligned'
    pattern_thickness: float = 1.5  # relative thickness of the supplementary yarn
    damask_satin: int = 5
    bottom_up: bool = True  # weave the bottom image row first
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.technique not in ("brocade", "damask"):
            raise ValueError(f"unknown technique {self.technique!r}")
        if self.ground_stripes not in ("auto", "none", "weft", "warp", "plaid"):
            raise ValueError(f"unknown ground_stripes {self.ground_stripes!r}")
        if self.supplementary not in ("continuous", "discontinuous"):
            raise ValueError(f"unknown supplementary mode {self.supplementary!r}")
        if self.tie_stagger not in ("brick", "twill", "aligned"):
            raise ValueError(f"unknown tie_stagger {self.tie_stagger!r}")
        if self.ends_per_cell < 1 or self.picks_per_cell < 1:
            raise ValueError("ends_per_cell and picks_per_cell must be >= 1")
        if self.max_float < 2:
            raise ValueError("max_float must be >= 2")
        if not 0.5 <= self.stripe_threshold <= 1.0:
            raise ValueError("stripe_threshold must be in [0.5, 1]")


@dataclass
class GroundPlan:
    mode: str  # 'none' | 'weft' | 'warp' | 'plaid'
    weft: np.ndarray  # (H,) ground weft colour per design row
    warp: np.ndarray  # (W,) warp colour per design column
    appearance: np.ndarray  # (H, W) colour the ground shows
    coverage: float  # fraction of cells explained by the ground alone


def _modes(a: np.ndarray, k: int, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Most frequent value and its fraction along ``axis`` of an int array with values < k."""
    a2 = a if axis == 1 else a.T
    counts = np.stack([(a2 == c).sum(axis=1) for c in range(k)], axis=1)
    mode = counts.argmax(axis=1)
    frac = counts.max(axis=1) / max(a2.shape[1], 1)
    return mode, frac


def mix_table(palette: np.ndarray, colors: np.ndarray, tol: float = 10.0) -> np.ndarray:
    """``mt[a, b]`` = palette index that looks like the optical mixture of a and b (or -1).

    Only colours in ``colors`` are considered.  ``mt[a, a] = a``.
    """
    k = palette.shape[0]
    mt = -np.ones((k, k), dtype=np.int64)
    colors = np.asarray(colors, dtype=np.int64)
    mt[colors, colors] = colors
    lab = srgb_to_lab(palette.astype(np.float64))
    for i, a in enumerate(colors.tolist()):
        for b in colors[i + 1 :].tolist():
            others = colors[(colors != a) & (colors != b)]
            if others.size == 0:
                continue
            m = srgb_to_lab(mix_rgb(palette[a], palette[b]))
            d = delta_e2000(lab[others], np.broadcast_to(m, (others.size, 3)))
            j = int(np.argmin(d))
            if d[j] < tol:
                mt[a, b] = mt[b, a] = int(others[j])
    return mt


def infer_plaid(grid: np.ndarray, palette: np.ndarray, tol: float = 10.0) -> GroundPlan | None:
    """Explain a design as crossing warp / weft stripes in a balanced weave.

    Colours that look like the optical mixture of two other colours are treated as
    crossings; the weft colour of every row and the warp colour of every column are then
    recovered by voting.  Returns ``None`` when the design has no mixture colours.
    """
    used = np.unique(grid)
    if used.size < 3:
        return None
    # every palette colour is a candidate: a stripe colour may only appear in mixtures
    cand = np.arange(palette.shape[0])
    mt = mix_table(palette, cand, tol)
    # Mark mixtures, visiting parent pairs from the most to the least distinct; a pair whose
    # parent is already known to be a mixture is skipped, so a stripe colour is never
    # mistaken for the mixture of another colour with a mixture.
    lab = srgb_to_lab(palette.astype(np.float64))
    pairs = [(a, b) for a in cand.tolist() for b in cand.tolist() if a < b and mt[a, b] >= 0]
    pairs.sort(key=lambda ab: -float(delta_e2000(lab[ab[0]], lab[ab[1]])))
    is_mix = np.zeros(palette.shape[0], dtype=bool)
    for a, b in pairs:
        if is_mix[a] or is_mix[b]:
            continue
        is_mix[mt[a, b]] = True
    base = [int(c) for c in cand if not is_mix[c]]
    if len(base) < 2 or not is_mix[used].any():
        return None
    H, W = grid.shape
    k = palette.shape[0]
    inverse = {}
    for f in base:
        inv = -np.ones(k, dtype=np.int64)
        for x in base:
            if x != f and mt[f, x] >= 0 and inv[mt[f, x]] < 0:
                inv[mt[f, x]] = x
        inv[f] = f
        inverse[f] = inv
    weft = np.zeros(H, dtype=np.int32)
    votes = np.zeros((W, k), dtype=np.int64)
    for r in range(H):
        row = grid[r]
        best = max(base, key=lambda f: int((inverse[f][row] >= 0).sum()))
        weft[r] = best
        w = inverse[best][row]
        ok = w >= 0
        votes[np.flatnonzero(ok), w[ok]] += 1
    warp = votes.argmax(axis=1).astype(np.int32)
    no_vote = votes.sum(axis=1) == 0
    if no_vote.any():
        warp[no_vote] = int(np.bincount(warp[~no_vote]).argmax()) if (~no_vote).any() else base[0]
    app = np.where(weft[:, None] == warp[None, :], weft[:, None], mt[weft[:, None], warp[None, :]])
    return GroundPlan("plaid", weft, warp, app.astype(np.int32), float((app == grid).mean()))


def plan_ground(grid: np.ndarray, k: int, mode: str = "auto", threshold: float = 0.9,
                ground_color: int | None = None, palette: np.ndarray | None = None) -> GroundPlan:
    """Decide which cells the ground cloth itself can show.

    ``none``: one ground colour.  ``weft`` / ``warp``: rows / columns that are (nearly)
    uniform become weft / warp colour stripes.  ``plaid``: crossing warp and weft stripes
    whose crossings show optical mixtures (needs ``palette``).  ``auto`` picks the plan
    that explains most cells (plaid must win by a clear margin).
    """
    H, W = grid.shape
    g = int(np.bincount(grid.ravel(), minlength=k).argmax()) if ground_color is None else int(ground_color)
    row_mode, row_frac = _modes(grid, k, axis=1)
    col_mode, col_frac = _modes(grid, k, axis=0)

    def build(m: str) -> GroundPlan:
        weft = np.full(H, g, np.int32)
        warp = np.full(W, g, np.int32)
        if m == "weft":
            weft = np.where(row_frac >= threshold, row_mode, g).astype(np.int32)
            app = np.broadcast_to(weft[:, None], (H, W))
        elif m == "warp":
            warp = np.where(col_frac >= threshold, col_mode, g).astype(np.int32)
            app = np.broadcast_to(warp[None, :], (H, W))
        else:
            app = np.full((H, W), g, np.int32)
        cov = float((grid == app).mean())
        return GroundPlan(m, weft, warp, np.array(app, np.int32), cov)

    if mode == "plaid":
        if palette is None:
            raise ValueError("plaid ground planning needs the palette")
        plan = infer_plaid(grid, palette)
        return plan if plan is not None else build("none")
    if mode != "auto":
        return build(mode)
    best = None
    for m in ("none", "weft", "warp"):  # ties resolved in this order (simplest set-up first)
        cand = build(m)
        if best is None or cand.coverage > best.coverage + 1e-12:
            best = cand
    assert best is not None
    if palette is not None:
        plaid = infer_plaid(grid, palette)
        if plaid is not None and plaid.coverage >= best.coverage + 0.05:
            best = plaid
    return best


def add_tiedowns(lift: np.ndarray, visible: np.ndarray, max_float: int, phase: int) -> tuple[np.ndarray, int]:
    """Raise tie-down ends inside visible runs longer than ``max_float``.

    Tie-downs sit on the absolute lattice ``(end - phase) % max_float == 0`` (so they line
    up in a regular pattern across picks), never on the first or last end of a run, and
    extra mid-points are added if a gap still exceeds ``max_float``.
    Returns the new lift vector and the number of tie-downs added.
    """
    lift = lift.copy()
    s = int(max_float)
    _, starts, lengths = runs(visible)
    n_ties = 0
    for a, L in zip(starts.tolist(), lengths.tolist()):
        if L <= s:
            continue
        b = a + L  # run covers [a, b)
        first = a + 1 + ((phase - (a + 1)) % s)
        ties = list(range(first, b - 1, s))
        # repair any gap that is still too long (short runs whose lattice points hit the edges)
        changed = True
        while changed:
            changed = False
            pts = [a - 1, *ties, b]
            for i in range(len(pts) - 1):
                gap = pts[i + 1] - pts[i] - 1
                if gap > s:
                    ties.append((pts[i] + pts[i + 1]) // 2)
                    ties.sort()
                    changed = True
                    break
        lift[ties] = 1
        n_ties += len(ties)
    return lift, n_ties


def _stagger_phase(counter: int, s: int, mode: str) -> int:
    if mode == "aligned":
        return 0
    if mode == "twill":
        return counter % s
    return (s // 2) * (counter % 2)  # brick / half-drop


def compile_brocade(design: Design, opt: CompileOptions) -> WeaveDraft:
    grid = design.grid
    H, W = grid.shape
    k = design.n_colors
    epc, ppc = opt.ends_per_cell, opt.picks_per_cell
    E = W * epc
    S = get_structure(opt.ground_weave)
    sp, se = S.shape
    plan = plan_ground(grid, k, opt.ground_stripes, opt.stripe_threshold, opt.ground_color, palette=design.palette)
    pattern = grid != plan.appearance
    end_cols = np.repeat(np.arange(W), epc)
    ends = np.arange(E)

    lifts: list[np.ndarray] = []
    kinds: list[int] = []
    colors: list[int] = []
    rows: list[int] = []
    thick: list[float] = []
    inlays: list[np.ndarray | None] = []
    n_ties = 0
    ground_count = 0
    supp_count = 0
    row_order = range(H - 1, -1, -1) if opt.bottom_up else range(H)
    for r in row_order:
        row_cols = grid[r]
        row_pat = pattern[r]
        pat_colors = [int(c) for c in np.unique(row_cols[row_pat])]
        for _ in range(ppc):
            lifts.append(S[ground_count % sp, ends % se].astype(np.uint8))
            kinds.append(PICK_GROUND)
            colors.append(int(plan.weft[r]))
            rows.append(r)
            thick.append(1.0)
            inlays.append(None)
            ground_count += 1
            if not pat_colors:
                continue
            if opt.supplementary == "continuous":
                for c in pat_colors:
                    vis = np.repeat(row_pat & (row_cols == c), epc)
                    lift = (~vis).astype(np.uint8)
                    lift, t = add_tiedowns(lift, vis, opt.max_float, _stagger_phase(supp_count, opt.max_float, opt.tie_stagger))
                    n_ties += t
                    lifts.append(lift)
                    kinds.append(PICK_SUPPLEMENTARY)
                    colors.append(c)
                    rows.append(r)
                    thick.append(opt.pattern_thickness)
                    inlays.append(None)
                    supp_count += 1
            else:
                vis_all = np.repeat(row_pat, epc)
                lift = (~vis_all).astype(np.uint8)
                phase = _stagger_phase(supp_count, opt.max_float, opt.tie_stagger)
                inlay = np.full(E, -1, np.int32)
                for c in pat_colors:
                    vis = np.repeat(row_pat & (row_cols == c), epc)
                    lift_c, t = add_tiedowns(lift, vis, opt.max_float, phase)
                    lift = np.where(vis, lift_c, lift).astype(np.uint8)
                    n_ties += t
                    inlay[vis] = c
                counts = np.bincount(row_cols[row_pat], minlength=k)
                lifts.append(lift)
                kinds.append(PICK_SUPPLEMENTARY)
                colors.append(int(counts.argmax()))
                rows.append(r)
                thick.append(opt.pattern_thickness)
                inlays.append(inlay)
                supp_count += 1

    liftplan = np.stack(lifts) if lifts else np.zeros((0, E), np.uint8)
    inlay_arr = None
    if any(x is not None for x in inlays):
        inlay_arr = np.stack([x if x is not None else np.full(E, -1, np.int32) for x in inlays])
    meta = {
        "technique": "brocade",
        "options": asdict(opt),
        "ground_mode": plan.mode,
        "ground_coverage": plan.coverage,
        "pattern_cells": int(pattern.sum()),
        "tie_downs": int(n_ties),
        "design_shape": [H, W],
    }
    return WeaveDraft(
        liftplan=liftplan,
        warp_colors=plan.warp[end_cols],
        weft_colors=np.array(colors, np.int32),
        palette=design.palette,
        pick_kind=np.array(kinds, np.uint8),
        pick_row=np.array(rows, np.int32),
        end_col=end_cols.astype(np.int32),
        weft_thickness=np.array(thick, np.float64),
        inlay=inlay_arr,
        meta=meta,
    )


def compile_damask(design: Design, opt: CompileOptions) -> WeaveDraft:
    grid = design.grid
    used = design.used_colors()
    if used.size > 2:
        raise ValueError(f"damask needs a design with at most 2 colours (has {used.size}); use 'brocade'")
    H, W = grid.shape
    epc, ppc = opt.ends_per_cell, opt.picks_per_cell
    counts = design.counts()
    a = int(counts.argmax()) if opt.ground_color is None else int(opt.ground_color)
    b = int(next((c for c in used if c != a), a))
    n = int(opt.damask_satin)
    warp_faced = satin(n, warp_faced=True)
    weft_faced = satin(n, warp_faced=False)
    row_order = np.arange(H - 1, -1, -1) if opt.bottom_up else np.arange(H)
    pick_rows = np.repeat(row_order, ppc)
    end_cols = np.repeat(np.arange(W), epc)
    P, E = pick_rows.size, end_cols.size
    cell_is_ground = grid[np.ix_(pick_rows, end_cols)] == a
    pp = np.arange(P)[:, None] % n
    ee = np.arange(E)[None, :] % n
    liftplan = np.where(cell_is_ground, warp_faced[pp, ee], weft_faced[pp, ee]).astype(np.uint8)
    meta = {"technique": "damask", "options": asdict(opt), "design_shape": [H, W], "ground": a, "figure": b}
    return WeaveDraft(
        liftplan=liftplan,
        warp_colors=np.full(E, a, np.int32),
        weft_colors=np.full(P, b, np.int32),
        palette=design.palette,
        pick_kind=np.zeros(P, np.uint8),
        pick_row=pick_rows.astype(np.int32),
        end_col=end_cols.astype(np.int32),
        meta=meta,
    )


def compile_design(design: Design, options: CompileOptions | None = None, **overrides: Any) -> WeaveDraft:
    """Compile a :class:`~manitex.design.Design` into a :class:`~manitex.weave.draft.WeaveDraft`."""
    opt = options or CompileOptions()
    if overrides:
        opt = CompileOptions(**{**asdict(opt), **overrides})
    opt.validate()
    if opt.technique == "damask":
        return compile_damask(design, opt)
    return compile_brocade(design, opt)


def face_matches_design(design: Design, draft: WeaveDraft) -> float:
    """Fraction of supplementary-pattern cells whose colour shows on the face.

    For every pattern cell, checks that at least one supplementary crossing of that cell
    shows the cell's colour (tie-downs hide individual crossings by design).
    """
    plan = plan_ground(design.grid, design.n_colors, draft.meta.get("ground_mode", "none"),
                       draft.meta.get("options", {}).get("stripe_threshold", 0.9),
                       draft.meta.get("options", {}).get("ground_color"), palette=design.palette)
    pattern = design.grid != plan.appearance
    if not pattern.any():
        return 1.0
    face = draft.face()
    shown = np.zeros_like(pattern)
    supp = np.flatnonzero(draft.pick_kind == PICK_SUPPLEMENTARY)
    for p in supp:
        r = draft.pick_row[p]
        cols = draft.end_col
        hit = face[p] == design.grid[r, cols]
        hit &= draft.liftplan[p] == 0
        np.logical_or.at(shown[r], cols[hit], True)
    return float(shown[pattern].mean())
