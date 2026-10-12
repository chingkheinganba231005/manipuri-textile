"""Yarn-level fabric simulation renders.

Every crossing of a draft is drawn as the yarn that is on top, with a cylindrical
cross-section shading, darkening where the yarn dives under a crossing thread, a twist
texture and per-thread irregularity.  Shading is computed in linear light.

Two effects matter for supplementary-weft cloth and are modelled explicitly:

* **Float spread** - pattern yarns are thicker/softer than the ground weft and spread over
  the thin ground picks next to them, so a motif reads as a solid shape (not stripes).
* **Face dominance** - a cloth can be balanced, weft-faced (weft stripes look solid) or
  warp-faced (warp stripes look solid).  ``face_mode='auto'`` follows the ground plan that
  the compiler chose.

The renderer is vectorised with numpy and deterministic for a given ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from manitex.color import linear_to_srgb, srgb_to_linear
from manitex.weave.draft import PICK_SUPPLEMENTARY, WeaveDraft

#: (warp_cover, weft_cover, weft_through_warp_gap, warp_through_weft_gap)
FACE_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "balanced": (0.86, 0.90, 0.30, 0.30),
    "weft": (0.34, 1.00, 0.97, 0.30),
    "warp": (1.00, 0.34, 0.30, 0.97),
}


@dataclass
class RenderOptions:
    end_px: int = 6  # pixels per warp end
    pick_px: int = 6  # pixels per ground pick (pattern picks scale with their thickness)
    face_mode: str = "auto"  # 'auto' | 'balanced' | 'weft' | 'warp'
    warp_cover: float | None = None  # override the face preset
    weft_cover: float | None = None
    spread: float = 0.65  # pattern-float spread over neighbouring picks, in ground-pick heights
    ambient: float = 0.42
    diffuse: float = 0.68
    specular: float = 0.10
    dive_darkening: float = 0.45
    twist_amp: float = 0.07
    twist_period_px: float = 5.0
    jitter: float = 0.035
    noise: float = 0.025
    background: tuple[int, int, int] = (28, 24, 22)
    origin: str = "bottom"  # put pick 1 at the bottom (matches the design image)
    max_pixels: int = 6_000_000
    seed: int = 0


def _profile(coord: np.ndarray, cover: float) -> tuple[np.ndarray, np.ndarray]:
    """Cross-section of a yarn: returns (inside mask, normalised position in [-1, 1])."""
    n = (coord - 0.5) / (max(cover, 1e-6) / 2.0)
    return np.abs(n) <= 1.0, np.clip(n, -1.0, 1.0)


def _dive_curve(down: np.ndarray, rows: np.ndarray, cols: np.ndarray, u: np.ndarray, a: float) -> np.ndarray:
    """Darkening of a weft yarn near the ends of its floats (where it dives under a warp).

    ``down[p, e]`` is True where the weft of pick p is on top at end e; ``rows``/``cols``
    map pixels to picks/ends and ``u`` is the position within an end.
    """
    left_down = np.hstack([down[:, :1], down[:, :-1]])
    right_down = np.hstack([down[:, 1:], down[:, -1:]])
    dive_l = (~left_down & down)[rows[:, None], cols[None, :]]
    dive_r = (~right_down & down)[rows[:, None], cols[None, :]]
    uu = u[None, :].astype(np.float32)
    fall_l = np.clip(1.0 - uu / 0.5, 0, 1) ** 2
    fall_r = np.clip((uu - 0.5) / 0.5, 0, 1) ** 2
    return (1.0 - a * dive_l * fall_l) * (1.0 - a * dive_r * fall_r)


def render_draft(draft: WeaveDraft, options: RenderOptions | None = None, **kw) -> np.ndarray:
    """Render the face of a draft as an RGB ``uint8`` image."""
    opt = options or RenderOptions()
    if kw:
        opt = replace(opt, **kw)
    P, E = draft.n_picks, draft.n_ends
    if P == 0 or E == 0:
        return np.zeros((1, 1, 3), np.uint8)
    mode = opt.face_mode
    if mode == "auto":
        mode = {"weft": "weft", "warp": "warp"}.get(str(draft.meta.get("ground_mode")), "balanced")
    if mode not in FACE_PRESETS:
        raise ValueError(f"unknown face_mode {opt.face_mode!r}")
    warp_cover, weft_cover, gap_weft, gap_warp = FACE_PRESETS[mode]
    warp_cover = opt.warp_cover if opt.warp_cover is not None else warp_cover
    weft_cover = opt.weft_cover if opt.weft_cover is not None else weft_cover

    end_px, pick_px = max(int(opt.end_px), 2), max(int(opt.pick_px), 2)
    heights = np.maximum(np.rint(pick_px * draft.weft_thickness).astype(int), 2)
    total = int(heights.sum()) * E * end_px
    if total > opt.max_pixels:  # keep the image within budget
        scale = np.sqrt(opt.max_pixels / total)
        end_px = max(2, int(end_px * scale))
        pick_px = max(2, int(pick_px * scale))
        heights = np.maximum(np.rint(pick_px * draft.weft_thickness).astype(int), 2)
    rng = np.random.default_rng(opt.seed)
    a = opt.dive_darkening

    # Raster layout: every pick (ground and pattern) occupies its own rows.
    pick_of_y = np.repeat(np.arange(P), heights)
    end_of_x = np.repeat(np.arange(E), end_px)
    u = ((np.arange(end_of_x.size) % end_px) + 0.5) / end_px
    Hpx, Wpx = pick_of_y.size, end_of_x.size
    yc = np.arange(Hpx) + 0.5
    centers_all = np.cumsum(heights) - heights / 2.0

    # Base layer = the ground cloth.  Hidden pattern yarns lie behind it, so every pixel row
    # belongs to the nearest *ground* pick (rows of pattern picks are shared out between the
    # ground picks around them) and the ground weave is drawn continuously.
    ground_idx = np.flatnonzero(draft.pick_kind != PICK_SUPPLEMENTARY)
    if ground_idx.size == 0:
        ground_idx = np.arange(P)
    gc = centers_all[ground_idx]
    pos = np.searchsorted(gc, yc)
    lo = np.clip(pos - 1, 0, gc.size - 1)
    hi = np.clip(pos, 0, gc.size - 1)
    g_of_y = np.where(np.abs(yc - gc[lo]) <= np.abs(yc - gc[hi]), lo, hi)  # index into ground_idx
    bounds = np.concatenate([[0.0], (gc[1:] + gc[:-1]) / 2.0, [float(Hpx)]])
    v = (yc - bounds[g_of_y]) / np.maximum(bounds[g_of_y + 1] - bounds[g_of_y], 1e-6)

    top = draft.liftplan[ground_idx].astype(bool)  # warp on top, ground picks only
    T = top[g_of_y[:, None], end_of_x[None, :]]

    def shade(h: np.ndarray, n: np.ndarray) -> np.ndarray:
        spec = opt.specular * np.exp(-((n + 0.35) ** 2) / 0.04)
        return (opt.ambient + opt.diffuse * h + spec).astype(np.float32)

    warp_in, nu = _profile(u, warp_cover)
    weft_in, nv = _profile(v, weft_cover)
    warp_shade_x = shade(np.sqrt(1.0 - nu**2), nu)  # (Wpx,)
    weft_shade_y = shade(np.sqrt(1.0 - nv**2), nv)  # (Hpx,)

    # warp darkening where it dives under the weft (along the warp, i.e. across picks)
    up_prev = np.vstack([top[:1], top[:-1]])
    up_next = np.vstack([top[1:], top[-1:]])
    dive_lo = (~up_prev & top)[g_of_y[:, None], end_of_x[None, :]]
    dive_hi = (~up_next & top)[g_of_y[:, None], end_of_x[None, :]]
    vv = v[:, None].astype(np.float32)
    warp_curve = (1.0 - a * dive_lo * np.clip(1.0 - vv / 0.5, 0, 1) ** 2) * \
                 (1.0 - a * dive_hi * np.clip((vv - 0.5) / 0.5, 0, 1) ** 2)
    weft_curve = _dive_curve(~top, g_of_y, end_of_x, u, a)

    yy = np.arange(Hpx, dtype=np.float32)[:, None]
    xx = np.arange(Wpx, dtype=np.float32)[None, :]
    warp_tex = 1.0 + opt.twist_amp * np.sin(2 * np.pi * (yy / opt.twist_period_px + nu[None, :] * 0.9))
    warp_jit = (1.0 + opt.jitter * rng.standard_normal(E)).astype(np.float32)[end_of_x][None, :]
    weft_jit_p = (1.0 + opt.jitter * rng.standard_normal(P)).astype(np.float32)

    warp_lin = srgb_to_linear(draft.palette[draft.warp_colors]).astype(np.float32)  # (E, 3)
    weft_lin = srgb_to_linear(draft.palette[draft.weft_color_grid()]).astype(np.float32)  # (P, E, 3)
    gpick_of_y = ground_idx[g_of_y]
    warp_val = (warp_shade_x[None, :] * warp_curve * warp_tex * warp_jit)[..., None] * warp_lin[end_of_x][None, :, :]
    weft_tex = 1.0 + opt.twist_amp * np.sin(2 * np.pi * (xx / opt.twist_period_px - nv[:, None] * 0.9))
    weft_val = (weft_shade_y[:, None] * weft_curve * weft_tex * weft_jit_p[gpick_of_y][:, None])[..., None] * \
        weft_lin[gpick_of_y[:, None], end_of_x[None, :]]

    in_warp = warp_in[None, :]
    in_weft = weft_in[:, None]
    out = np.empty((Hpx, Wpx, 3), dtype=np.float32)
    out[:] = srgb_to_linear(np.array(opt.background, dtype=float)).astype(np.float32)
    out = np.where((T & ~in_warp & in_weft)[..., None], weft_val * gap_weft, out)
    out = np.where((~T & ~in_weft & in_warp)[..., None], warp_val * gap_warp, out)
    out = np.where((T & in_warp)[..., None], warp_val, out)
    out = np.where((~T & in_weft)[..., None], weft_val, out)

    # ---- pattern floats spreading over neighbouring picks
    supp_idx = np.flatnonzero(draft.pick_kind == PICK_SUPPLEMENTARY)
    if supp_idx.size and opt.spread > 0:
        centers = centers_all[supp_idx]
        pos = np.searchsorted(centers, yc)
        lo = np.clip(pos - 1, 0, supp_idx.size - 1)
        hi = np.clip(pos, 0, supp_idx.size - 1)
        d_lo, d_hi = np.abs(yc - centers[lo]), np.abs(yc - centers[hi])
        pick_near = np.where(d_lo <= d_hi, supp_idx[lo], supp_idx[hi])
        dist = np.minimum(d_lo, d_hi)
        half = heights[pick_near] / 2.0 + opt.spread * pick_px
        within = dist <= half
        if within.any():
            rows_y = np.flatnonzero(within)
            q = pick_near[rows_y]
            vis_q = draft.liftplan[q] == 0  # (n_rows, E) pattern yarn on the face
            covered = vis_q[:, end_of_x]
            n_s = np.clip(dist[rows_y] / half[rows_y], 0, 1)
            sh = shade(np.sqrt(1.0 - n_s**2), (yc[rows_y] - np.where(d_lo <= d_hi, centers[lo], centers[hi])[rows_y])
                       / half[rows_y])
            # dive darkening at float ends, computed on the pattern pick itself
            curve = _dive_curve(vis_q, np.arange(rows_y.size), end_of_x, u, a)
            tex = 1.0 + opt.twist_amp * np.sin(2 * np.pi * (xx / opt.twist_period_px - n_s[:, None] * 0.9))
            col = weft_lin[q[:, None], end_of_x[None, :]]
            val = (sh[:, None] * curve * tex * weft_jit_p[q][:, None])[..., None] * col
            sub = out[rows_y]
            out[rows_y] = np.where(covered[..., None], val, sub)

    if opt.noise > 0:
        out *= (1.0 + opt.noise * rng.standard_normal((Hpx, Wpx, 1))).astype(np.float32)
    img = np.clip(np.rint(linear_to_srgb(np.clip(out, 0.0, 1.0))), 0, 255).astype(np.uint8)
    if opt.origin == "bottom":
        img = img[::-1]
    return np.ascontiguousarray(img)


def render_design_cloth(design, compile_options=None, render_options: RenderOptions | None = None, **compile_kw):
    """Compile a design (brocade by default) and render it.  Returns ``(image, draft)``."""
    from manitex.weave.compiler import compile_design

    draft = compile_design(design, compile_options, **compile_kw)
    return render_draft(draft, render_options), draft
