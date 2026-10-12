"""Training data streams: procedural designs, shards of real / digitised designs, mixtures.

Every sample is a random ``crop x crop`` window (wrap-padded if the design is smaller),
randomly flipped, and **canonicalised** (colour slots renumbered by frequency), together
with its style id, colour count and domain id.  All streams are infinite and seeded per
(rank, worker), so training is step-based and reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from manitex.data.shards import ShardData
from manitex.design import canonicalize_grid
from manitex.models.config import DOMAINS, StyleVocab


@dataclass
class CropSpec:
    size: int = 64
    flip_lr: bool = True
    flip_ud: bool = True
    max_colors: int = 16


def prepare_crop(grid: np.ndarray, spec: CropSpec, rng: np.random.Generator,
                 palette: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Random crop + flips + canonical colour order.  Returns (uint8 grid, n_colors)."""
    s = spec.size
    H, W = grid.shape
    if H < s or W < s:
        grid = np.pad(grid, ((0, max(0, s - H)), (0, max(0, s - W))), mode="wrap")
        H, W = grid.shape
    y = int(rng.integers(0, H - s + 1))
    x = int(rng.integers(0, W - s + 1))
    g = grid[y : y + s, x : x + s]
    if spec.flip_lr and rng.random() < 0.5:
        g = g[:, ::-1]
    if spec.flip_ud and rng.random() < 0.5:
        g = g[::-1, :]
    cg, order = canonicalize_grid(g)
    k = int(order.size)
    if k > spec.max_colors:
        # merge the rarest colours into the most similar kept colour (or the ground colour)
        lut = np.arange(k)
        if palette is not None:
            pal = palette[order].astype(float)
            for j in range(spec.max_colors, k):
                lut[j] = int(np.argmin(((pal[: spec.max_colors] - pal[j]) ** 2).sum(axis=1)))
        else:
            lut[spec.max_colors :] = 0
        cg, order2 = canonicalize_grid(lut[cg])
        k = int(order2.size)
    return cg.astype(np.uint8), k


def _worker_rng(seed: int, salt: int = 0) -> np.random.Generator:
    info = get_worker_info()
    wid = info.id if info is not None else 0
    rank = int(torch.distributed.get_rank()) if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
    return np.random.default_rng([seed, rank, wid, salt])


class ProceduralStream(IterableDataset):
    """Infinite stream of procedural designs (see :mod:`manitex.procedural`)."""

    def __init__(self, vocab: StyleVocab, crop: CropSpec, styles: dict[str, float] | None = None,
                 canvas_scale: tuple[float, float] = (1.0, 2.5), seed: int = 0):
        from manitex.procedural import PROCEDURAL_STYLES

        super().__init__()
        self.vocab = vocab
        self.crop = crop
        self.styles = styles or {s: 1.0 for s in PROCEDURAL_STYLES}
        self.canvas_scale = canvas_scale
        self.seed = seed

    def sample(self, rng: np.random.Generator) -> dict:
        from manitex.procedural import generate

        names = list(self.styles)
        p = np.array([self.styles[n] for n in names], dtype=float)
        style = names[int(rng.choice(len(names), p=p / p.sum()))]
        s = self.crop.size
        h = max(8, int(round(s * rng.uniform(*self.canvas_scale))))
        w = max(8, int(round(s * rng.uniform(*self.canvas_scale))))
        d = generate(style, h, w, rng=rng)
        g, k = prepare_crop(d.grid, self.crop, rng, d.palette)
        return {"grid": g, "style": self.vocab.id(style), "ncolors": k, "domain": DOMAINS["synthetic"]}

    def __iter__(self) -> Iterator[dict]:
        rng = _worker_rng(self.seed, 1)
        while True:
            yield self.sample(rng)


class ShardStream(IterableDataset):
    """Infinite stream of random crops from shards (designs drawn uniformly or by weight)."""

    def __init__(self, shards: Sequence[ShardData], vocab: StyleVocab, crop: CropSpec, seed: int = 0,
                 domain: str | None = None, add_styles: bool = True):
        super().__init__()
        self.shards = list(shards)
        self.vocab = vocab
        self.crop = crop
        self.seed = seed
        self.domain = domain
        self.index = [(i, j) for i, sh in enumerate(self.shards) for j in range(len(sh))]
        if not self.index:
            raise ValueError("no designs in the given shards")
        if add_styles:
            for sh in self.shards:
                for name in np.unique(sh.styles):
                    self.vocab.id(str(name), add=True)

    def __len__(self) -> int:
        return len(self.index)

    def sample(self, rng: np.random.Generator) -> dict:
        i, j = self.index[int(rng.integers(len(self.index)))]
        sh = self.shards[i]
        g, k = prepare_crop(sh.grid(j), self.crop, rng, sh.palette(j))
        dom = self.domain or str(sh.domains[j])
        return {"grid": g, "style": self.vocab.id(str(sh.styles[j])), "ncolors": k,
                "domain": DOMAINS.get(dom, DOMAINS["real"])}

    def __iter__(self) -> Iterator[dict]:
        rng = _worker_rng(self.seed, 2)
        while True:
            yield self.sample(rng)


class MixedStream(IterableDataset):
    """Draw each sample from one of several streams with the given probabilities."""

    def __init__(self, sources: Sequence, weights: Sequence[float], seed: int = 0):
        super().__init__()
        if len(sources) != len(weights) or not sources:
            raise ValueError("need one weight per source")
        self.sources = list(sources)
        self.weights = [float(w) for w in weights]
        self.seed = seed

    def sample(self, rng: np.random.Generator) -> dict:
        p = np.asarray(self.weights) / sum(self.weights)
        return self.sources[int(rng.choice(len(self.sources), p=p))].sample(rng)

    def __iter__(self) -> Iterator[dict]:
        rng = _worker_rng(self.seed, 3)
        p = np.asarray(self.weights) / sum(self.weights)
        rngs = [_worker_rng(self.seed, 10 + i) for i in range(len(self.sources))]
        while True:
            i = int(rng.choice(len(self.sources), p=p))
            yield self.sources[i].sample(rngs[i])


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "grid": torch.from_numpy(np.stack([it["grid"] for it in items])).long(),
        "style": torch.tensor([it["style"] for it in items], dtype=torch.long),
        "ncolors": torch.tensor([it["ncolors"] for it in items], dtype=torch.long),
        "domain": torch.tensor([it["domain"] for it in items], dtype=torch.long),
    }


def fixed_batch(stream, n: int, seed: int) -> dict[str, torch.Tensor]:
    """A deterministic held-out batch drawn from a stream's ``sample`` method."""
    rng = np.random.default_rng([seed, 999_983])
    return collate([stream.sample(rng) for _ in range(n)])


def steps_per_epoch(n_items: int, batch_size: int) -> int:
    return max(1, math.ceil(n_items / batch_size))
