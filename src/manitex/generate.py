"""High-level generation API.

:class:`TextileGenerator` wraps a trained model (or, if none is available, the procedural
grammar) and returns ready-to-weave :class:`~manitex.design.Design` objects:

* ``generate``: new designs for a style, colour budget, size and symmetry;
* ``inpaint``: regenerate a masked region of an existing design;
* palettes: canonical colour slots are coloured by *frequency-rank transfer* from real
  palettes seen in training (``palette_bank``), from presets, or from a user palette.

Large canvases are produced by tiled outpainting (each window conditions on the cells
already generated).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manitex.color import PALETTE_PRESETS, extend_palette, hex_to_rgb, jitter_palette, palette_for_style, preset_palette
from manitex.constraints import Symmetry, make_orbits, project
from manitex.design import Design, canonicalize_grid
from manitex.procedural import PROCEDURAL_STYLES
from manitex.procedural import generate as procedural_generate


@dataclass
class GenerationOptions:
    steps: int = 24
    temperature: float = 1.0
    cfg: float = 1.0
    strategy: str = "confidence"
    domain: str = "real"  # condition on real-textile statistics when the model has seen real data


def assign_palette(grid: np.ndarray, style: str, rng: np.random.Generator,
                   palette: str | Sequence | np.ndarray | None = None,
                   palette_bank: dict[str, list[list[str]]] | None = None) -> np.ndarray:
    """Colours for a canonical grid (slot 0 = most frequent)."""
    k = int(grid.max()) + 1
    if palette is not None and not (isinstance(palette, str) and palette in ("auto", "")):
        if isinstance(palette, str):
            if palette not in PALETTE_PRESETS:
                raise ValueError(f"unknown palette preset {palette!r}")
            pal = preset_palette(palette)
        else:
            pal = np.array([hex_to_rgb(c) if isinstance(c, str) else c for c in palette], dtype=np.uint8)
        return pal[:k] if pal.shape[0] >= k else extend_palette(pal, k, rng)
    bank = (palette_bank or {}).get(style) or []
    usable = [p for p in bank if len(p) >= k]
    if usable:
        pal = np.array([hex_to_rgb(c) for c in usable[int(rng.integers(len(usable)))][:k]], dtype=np.uint8)
        return jitter_palette(pal, rng, 1.5, 2.0)
    pal, _ = palette_for_style(style, k, rng)
    return pal


class TextileGenerator:
    """Generate designs with a trained model, or procedurally when no model is loaded."""

    def __init__(self, model=None, info: dict[str, Any] | None = None, device: str = "cpu"):
        self.model = model
        self.info = info or {}
        self.device = device

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path: str | Path | None, device: str = "auto") -> TextileGenerator:
        """Load from a checkpoint (``.pt``), a slim export directory, or ``hf://user/repo[/subdir]``.

        Falls back to the procedural generator when ``path`` is None/empty.
        """
        if not path:
            return cls()
        import torch

        from manitex.models.checkpoint import load_model

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        p = str(path)
        if p.startswith("hf://"):
            from huggingface_hub import snapshot_download

            repo, _, sub = p[5:].partition("/")
            owner, _, rest = sub.partition("/")
            repo_id = f"{repo}/{owner}"
            local = Path(snapshot_download(repo_id=repo_id))
            p = str(local / rest) if rest else str(local)
        model, info = load_model(p, device=device)
        return cls(model, info, device)

    @property
    def has_model(self) -> bool:
        return self.model is not None

    @property
    def styles(self) -> list[str]:
        if self.has_model:
            return [s for s in self.info["vocab"].names if s != "unknown"]
        return list(PROCEDURAL_STYLES)

    @property
    def train_crop(self) -> int:
        return int(self.info.get("train_crop", 64))

    # ------------------------------------------------------------------ generation
    def generate(
        self,
        style: str = "moirang_phee",
        n: int = 4,
        height: int = 64,
        width: int = 64,
        n_colors: int | None = None,
        symmetry: Symmetry | None = None,
        palette: str | Sequence | None = "auto",
        seed: int | None = None,
        options: GenerationOptions | None = None,
        known: np.ndarray | None = None,
    ) -> list[Design]:
        """Generate ``n`` designs.  ``known``: (H, W) colour slots (-1 = free) for inpainting."""
        opt = options or GenerationOptions()
        rng = np.random.default_rng(seed)
        if not self.has_model:
            designs = []
            for i in range(n):
                d = procedural_generate(style if style in PROCEDURAL_STYLES else "random", height, width,
                                        seed=None if seed is None else seed + i, n_colors=n_colors)
                if symmetry is not None and symmetry.any():
                    d = Design(project(d.grid, make_orbits(height, width, symmetry)), d.palette, d.meta)
                d = d.canonical()
                if palette not in (None, "auto"):
                    d = d.with_palette(assign_palette(d.grid, style, rng, palette))
                d.meta.update({"generator": "procedural"})
                designs.append(d)
            return designs
        grids = self._sample_grids(style, n, height, width, n_colors, symmetry, seed, opt, known)
        out = []
        for g in grids:
            cg, _ = canonicalize_grid(g)
            pal = assign_palette(cg, style, rng, palette, self.info.get("palette_bank"))
            meta = {"style": style, "source": "generated", "generator": "manitex-mdm", "seed": seed,
                    "step": self.info.get("step"), "symmetry": symmetry.to_dict() if symmetry else None,
                    "sampler": opt.__dict__}
            out.append(Design(cg, pal, meta))
        return out

    def inpaint(self, design: Design, mask: np.ndarray, n: int = 4, style: str | None = None,
                seed: int | None = None, options: GenerationOptions | None = None,
                symmetry: Symmetry | None = None) -> list[Design]:
        """Regenerate the cells where ``mask`` is True, keeping the rest of ``design``."""
        cd = design.canonical()
        if cd.n_colors > (self.model.cfg.vocab if self.has_model else 16):
            raise ValueError("design has too many colours for the model")
        known = np.where(np.asarray(mask, bool), -1, cd.grid)
        style = style or str(design.meta.get("style", "moirang_phee"))
        if not self.has_model:
            raise RuntimeError("inpainting needs a trained model")
        grids = self._sample_grids(style, n, cd.height, cd.width, cd.n_colors, symmetry, seed,
                                   options or GenerationOptions(), known)
        out = []
        for g in grids:
            k = int(g.max()) + 1
            pal = cd.palette if k <= cd.n_colors else extend_palette(cd.palette, k, np.random.default_rng(seed))
            out.append(Design(g, pal, {**design.meta, "source": "inpainted"}))
        return out

    # ------------------------------------------------------------------ internals
    def _sample_grids(self, style, n, height, width, n_colors, symmetry, seed, opt, known) -> np.ndarray:
        import torch

        from manitex.models.config import DOMAINS
        from manitex.models.sampler import SamplerConfig, sample

        vocab = self.info["vocab"]
        sid = vocab.id(style)
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        scfg = SamplerConfig(steps=opt.steps, temperature=opt.temperature, cfg=opt.cfg, strategy=opt.strategy)
        dom = DOMAINS.get(opt.domain, None)
        window = 2 * self.train_crop
        if height <= window and width <= window:
            out = sample(self.model, n, height, width, style=sid, ncolors=n_colors, domain=dom, symmetry=symmetry,
                         known=known, config=scfg, generator=gen, device=self.device)
            return out.numpy()
        # tiled outpainting for large canvases
        tile = self.train_crop
        overlap = tile // 2
        canvas = np.full((n, height, width), -1, dtype=np.int64)
        if known is not None:
            canvas[:] = np.asarray(known)[None]
        ys = list(range(0, max(1, height - tile), tile - overlap)) + [max(0, height - tile)]
        xs = list(range(0, max(1, width - tile), tile - overlap)) + [max(0, width - tile)]
        for y0 in sorted(set(ys)):
            for x0 in sorted(set(xs)):
                h, w = min(tile, height), min(tile, width)
                kn = canvas[:, y0 : y0 + h, x0 : x0 + w]
                if (kn >= 0).all():
                    continue
                out = sample(self.model, n, h, w, style=sid, ncolors=n_colors, domain=dom, known=kn,
                             config=scfg, generator=gen, device=self.device).numpy()
                canvas[:, y0 : y0 + h, x0 : x0 + w] = out
        if symmetry is not None and symmetry.any():
            orb = make_orbits(height, width, symmetry)
            canvas = np.stack([project(g, orb) for g in canvas])
        return canvas
