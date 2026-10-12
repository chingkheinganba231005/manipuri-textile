"""The :class:`WeaveDraft`: a complete, loom-agnostic description of a woven cloth.

A draft is stored as a **lift plan** (``P`` picks x ``E`` ends, 1 = end raised) plus the
colour of every warp end and every weft pick.  Threading / tie-up / treadling for shaft
looms are *derived* from the lift plan (:meth:`WeaveDraft.threading`,
:meth:`WeaveDraft.treadling`), so the same draft serves jacquard, dobby, treadle and
pick-up (loin loom) weaving.

Pick order is weaving order: ``liftplan[0]`` is the first pick woven (the edge closest to
the weaver).  End order is left-to-right as seen on the face of the cloth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PICK_GROUND = 0
PICK_SUPPLEMENTARY = 1


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs of ``True`` along the last axis of a 2-D boolean array.

    Returns ``(row, start, length)`` arrays (row-major order).
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 1:
        m = m[None, :]
    padded = np.pad(m.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(padded, axis=1)
    r_s, c_s = np.nonzero(d == 1)
    r_e, c_e = np.nonzero(d == -1)
    return r_s, c_s, c_e - c_s


def max_equal_run(keys: np.ndarray) -> np.ndarray:
    """Longest run of identical non-negative values in each row (-1 entries never count)."""
    keys = np.asarray(keys)
    n, m = keys.shape
    if m == 0 or n == 0:
        return np.zeros(n, dtype=np.int64)
    start = np.ones((n, m), dtype=bool)
    start[:, 1:] = keys[:, 1:] != keys[:, :-1]
    run_id = np.cumsum(start, axis=1)
    valid = keys >= 0
    flat = (np.arange(n)[:, None] * (m + 1) + run_id)[valid]
    counts = np.bincount(flat, minlength=n * (m + 1)).reshape(n, m + 1)
    return counts.max(axis=1)


def max_run(mask: np.ndarray, axis: int = -1) -> np.ndarray:
    """Longest run of ``True`` in each 1-D slice along ``axis`` (0 if none)."""
    m = np.asarray(mask, dtype=bool)
    if axis in (0, -2) and m.ndim == 2:
        m = m.T
    r, _, length = runs(m)
    out = np.zeros(m.shape[0], dtype=np.int64)
    if r.size:
        np.maximum.at(out, r, length)
    return out


@dataclass
class WeaveDraft:
    liftplan: np.ndarray
    warp_colors: np.ndarray
    weft_colors: np.ndarray
    palette: np.ndarray
    pick_kind: np.ndarray | None = None
    pick_row: np.ndarray | None = None
    end_col: np.ndarray | None = None
    weft_thickness: np.ndarray | None = None
    inlay: np.ndarray | None = None  # (P, E) per-crossing weft colour for inlay picks, -1 = weft_colors[p]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lp = np.asarray(self.liftplan)
        if lp.ndim != 2:
            raise ValueError("liftplan must be 2-D (picks x ends)")
        self.liftplan = (lp != 0).astype(np.uint8)
        P, E = self.liftplan.shape
        self.warp_colors = np.asarray(self.warp_colors, dtype=np.int32).reshape(-1)
        self.weft_colors = np.asarray(self.weft_colors, dtype=np.int32).reshape(-1)
        if self.warp_colors.shape != (E,) or self.weft_colors.shape != (P,):
            raise ValueError("warp_colors must have one entry per end and weft_colors one per pick")
        self.palette = np.asarray(self.palette, dtype=np.uint8).reshape(-1, 3)
        k = self.palette.shape[0]
        for name, arr in (("warp_colors", self.warp_colors), ("weft_colors", self.weft_colors)):
            if arr.size and (arr.min() < 0 or arr.max() >= k):
                raise ValueError(f"{name} out of palette range")
        self.pick_kind = np.zeros(P, np.uint8) if self.pick_kind is None else np.asarray(self.pick_kind, np.uint8)
        self.pick_row = np.full(P, -1, np.int32) if self.pick_row is None else np.asarray(self.pick_row, np.int32)
        self.end_col = np.full(E, -1, np.int32) if self.end_col is None else np.asarray(self.end_col, np.int32)
        if self.weft_thickness is None:
            self.weft_thickness = np.where(self.pick_kind == PICK_SUPPLEMENTARY, 1.5, 1.0)
        self.weft_thickness = np.asarray(self.weft_thickness, dtype=np.float64)
        if self.inlay is not None:
            self.inlay = np.asarray(self.inlay, dtype=np.int32)
            if self.inlay.shape != (P, E):
                raise ValueError("inlay must have the same shape as the lift plan")
        for name, arr, n in (("pick_kind", self.pick_kind, P), ("pick_row", self.pick_row, P),
                             ("weft_thickness", self.weft_thickness, P), ("end_col", self.end_col, E)):
            if arr.shape != (n,):
                raise ValueError(f"{name} has wrong length")
        self.meta = dict(self.meta) if self.meta else {}

    # ------------------------------------------------------------------ sizes
    @property
    def n_picks(self) -> int:
        return int(self.liftplan.shape[0])

    @property
    def n_ends(self) -> int:
        return int(self.liftplan.shape[1])

    # ------------------------------------------------------------------ appearance
    def weft_color_grid(self) -> np.ndarray:
        """(P, E) colour of the weft at every crossing (handles inlay picks)."""
        wc = np.broadcast_to(self.weft_colors[:, None], self.liftplan.shape)
        if self.inlay is None:
            return np.array(wc)
        return np.where(self.inlay >= 0, self.inlay, wc)

    def face(self) -> np.ndarray:
        """(P, E) palette index visible on the face at each crossing (the drawdown)."""
        return np.where(self.liftplan == 1, self.warp_colors[None, :], self.weft_color_grid())

    def back(self) -> np.ndarray:
        return np.where(self.liftplan == 1, self.weft_color_grid(), self.warp_colors[None, :])

    def face_rgb(self) -> np.ndarray:
        return self.palette[self.face()]

    # ------------------------------------------------------------------ loom set-up
    def threading(self) -> tuple[np.ndarray, np.ndarray]:
        """Minimal threading: one shaft per distinct end behaviour.

        Returns ``(threading, shaft_lifts)``: ``threading[e]`` is the 0-based shaft of end
        ``e`` and ``shaft_lifts[p, s]`` says whether shaft ``s`` is raised on pick ``p``.
        Shafts are numbered in order of first use from the left.
        """
        cols = self.liftplan.T
        uniq, first, inverse = np.unique(cols, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        threading = rank[inverse.reshape(-1)]
        shaft_lifts = uniq[order].T.copy()
        return threading.astype(np.int32), shaft_lifts.astype(np.uint8)

    def treadling(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single-treadle-per-pick decomposition of the shaft lift plan.

        Returns ``(threading, tieup, treadling)`` where ``tieup[t, s]`` ties treadle ``t``
        to shaft ``s`` (rising shed) and ``treadling[p]`` is the treadle for pick ``p``.
        """
        threading, lifts = self.threading()
        uniq, first, inverse = np.unique(lifts, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        return threading, uniq[order].astype(np.uint8), rank[inverse.reshape(-1)].astype(np.int32)

    @property
    def n_shafts(self) -> int:
        return int(np.unique(self.liftplan.T, axis=0).shape[0]) if self.n_ends else 0

    @property
    def n_treadles(self) -> int:
        return int(self.treadling()[1].shape[0]) if self.n_picks else 0

    # ------------------------------------------------------------------ checks
    def float_report(self) -> dict[str, int]:
        """Longest floats (in ends for weft floats, in picks for warp floats)."""
        lp = self.liftplan.astype(bool)
        ground = self.pick_kind == PICK_GROUND
        supp = ~ground
        rep: dict[str, int] = {}

        def mx(a: np.ndarray) -> int:
            return int(a.max()) if a.size else 0

        face_weft = max_run(~lp, axis=1)
        back_weft = max_run(lp, axis=1)
        if self.inlay is not None:
            # each inlay colour is a separate yarn that only floats inside its own region
            inl = (self.inlay >= 0).any(axis=1)
            if inl.any():
                keys = np.where(~lp & (self.inlay >= 0), self.inlay, -1)[inl]
                face_weft[inl] = max_equal_run(keys)
                back_weft[inl] = 0
        rep["weft_face"] = mx(face_weft)
        rep["weft_back"] = mx(back_weft)
        rep["ground_weft_face"] = mx(face_weft[ground])
        rep["ground_weft_back"] = mx(back_weft[ground])
        rep["supp_weft_face"] = mx(face_weft[supp])
        rep["supp_weft_back"] = mx(back_weft[supp])
        # warp floats measured over ground picks only would hide the effect of pattern
        # picks, so they are measured over all picks.
        rep["warp_face"] = mx(max_run(lp, axis=0))
        rep["warp_back"] = mx(max_run(~lp, axis=0))
        return rep

    def summary(self) -> dict[str, Any]:
        used = np.unique(np.concatenate([self.warp_colors, self.weft_colors] +
                                        ([self.inlay[self.inlay >= 0]] if self.inlay is not None else [])))
        return {
            "ends": self.n_ends,
            "picks": self.n_picks,
            "ground_picks": int((self.pick_kind == PICK_GROUND).sum()),
            "supplementary_picks": int((self.pick_kind == PICK_SUPPLEMENTARY).sum()),
            "colours": int(used.size),
            "shafts_needed": self.n_shafts,
            "floats": self.float_report(),
        }
