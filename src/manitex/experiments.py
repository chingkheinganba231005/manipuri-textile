"""Shared helpers for the experiment scripts in ``scripts/``: reference designs, constraint
specifications, model likelihoods and result tables."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from manitex.constraints import Symmetry
from manitex.design import MAX_COLORS, Design
from manitex.procedural import PROCEDURAL_STYLES, generate


def _hw(size: int | tuple[int, int]) -> tuple[int, int]:
    return (int(size), int(size)) if np.isscalar(size) else (int(size[0]), int(size[1]))


def crop_design(d: Design, size: int | tuple[int, int], rng: np.random.Generator) -> Design:
    """Random crop of ``size`` cells (an int for squares or ``(height, width)``; designs smaller
    than that wrap around, as in training) in canonical colour order, with at most
    ``MAX_COLORS`` colours (the rarest are merged into the nearest kept colour)."""
    h, w = _hw(size)
    g = d.grid
    if g.shape[0] < h or g.shape[1] < w:
        g = np.pad(g, ((0, max(0, h - g.shape[0])), (0, max(0, w - g.shape[1]))), mode="wrap")
    y, x = int(rng.integers(0, g.shape[0] - h + 1)), int(rng.integers(0, g.shape[1] - w + 1))
    c = Design(np.ascontiguousarray(g[y : y + h, x : x + w]), d.palette, dict(d.meta)).canonical()
    if c.n_colors > MAX_COLORS:
        pal = c.palette.astype(float)
        lut = np.arange(c.n_colors)
        for j in range(MAX_COLORS, c.n_colors):
            lut[j] = int(np.argmin(((pal[:MAX_COLORS] - pal[j]) ** 2).sum(axis=1)))
        c = Design(lut[c.grid], c.palette[:MAX_COLORS], c.meta).canonical()
    return c


def reference_designs(style: str, n: int, size: int | tuple[int, int], shards: Sequence | None = None,
                      seed: int = 0) -> list[Design]:
    """``n`` reference designs of ``size`` cells: procedural designs of ``style``, or random crops
    of the designs of that style in ``shards``."""
    if shards is None:
        if style not in PROCEDURAL_STYLES:
            raise ValueError(f"no procedural reference for style {style!r}; pass reference shards")
        h, w = _hw(size)
        return [generate(style, h, w, seed=seed + i) for i in range(n)]
    rng = np.random.default_rng(seed)
    pool = [(sh, j) for sh in shards for j in range(len(sh)) if str(sh.styles[j]) == style]
    if not pool:
        raise ValueError(f"no designs of style {style!r} in the reference shards")
    out = []
    for _ in range(n):
        sh, j = pool[int(rng.integers(len(pool)))]
        out.append(crop_design(sh.design(j), size, rng))
    return out


def reference_grids(style: str, n: int, size: int | tuple[int, int], shards: Sequence | None = None,
                    seed: int = 0) -> list[np.ndarray]:
    """The grids of :func:`reference_designs` (int64)."""
    return [d.grid.astype(np.int64) for d in reference_designs(style, n, size, shards, seed)]


def parse_symmetry(spec: str) -> Symmetry:
    """``'mirror_x+period_x=16'``, ``'stripes=rows'`` or ``'none'`` -> :class:`Symmetry`."""
    kw: dict[str, Any] = {}
    if spec.strip().lower() not in ("", "none"):
        for tok in spec.split("+"):
            key, _, val = tok.strip().partition("=")
            if key not in Symmetry.__dataclass_fields__:
                raise ValueError(f"unknown symmetry {key!r}; use {list(Symmetry.__dataclass_fields__)}")
            kw[key] = True if not val else (val if key == "stripes" else int(val))
    return Symmetry(**kw)


def model_nll(model, grids: Sequence[np.ndarray], style: int | None = None, domain: int | None = None,
              batch: int = 64, n_t: int = 16, seed: int = 0) -> np.ndarray:
    """Per-grid NLL in bits per cell under a trained model: the ELBO bound of the masked model or
    the exact NLL of the autoregressive one (conditions not given use their null entries).  The
    masking randomness of the bound is seeded, so equal inputs give equal values."""
    import torch

    from manitex.models.factory import objective_for

    obj = objective_for(model)
    dev = next(model.parameters()).device
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(seed)
        return _model_nll(obj, model, grids, style, domain, batch, n_t, dev)


def _model_nll(obj, model, grids, style, domain, batch, n_t, dev) -> np.ndarray:
    import torch

    from manitex.models.diffusion import Batch

    out = []
    for i in range(0, len(grids), batch):
        g = torch.as_tensor(np.stack(grids[i : i + batch]), dtype=torch.long, device=dev)
        B = g.shape[0]
        conds = [torch.full((B,), null if v is None else int(v), dtype=torch.long, device=dev)
                 for v, null in ((style, model.null_style), (None, model.null_ncolors), (domain, model.null_domain))]
        out.append(obj.nll_bound(Batch(g, *conds), n_t=n_t).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return "" if v is None else str(v)


def write_tables(results: dict[str, dict[str, dict[str, Any]]], out: str | Path, columns: Sequence[str],
                 row_label: str = "") -> tuple[Path, Path]:
    """Write ``results[group][row][column]`` as ``<out>.json`` and one Markdown table per group
    in ``<out>.md``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    js, md = out.with_suffix(".json"), out.with_suffix(".md")
    js.write_text(json.dumps(results, indent=1))
    lines: list[str] = []
    for group, rows in results.items():
        lines += [f"### {group}", "", f"| {row_label} | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        for name, m in rows.items():
            lines.append(f"| {name} | " + " | ".join(_fmt(m.get(k)) for k in columns) + " |")
        lines.append("")
    md.write_text("\n".join(lines))
    return js, md
