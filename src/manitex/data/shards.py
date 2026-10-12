"""Compact on-disk storage for many variable-size designs (``.npz`` shards).

A shard stores all grids concatenated in one ``uint8`` array plus shapes, palettes and
per-design metadata, so millions of small designs load in a few array reads and need no
pickle.  Use :func:`write_shard` / :func:`read_shard`, or the CLI
``manitex data build-synthetic`` / ``manitex data import``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manitex.design import Design


@dataclass
class ShardData:
    grids: np.ndarray  # flat uint8
    offsets: np.ndarray  # (N + 1,) int64
    shapes: np.ndarray  # (N, 2) int32
    palettes: np.ndarray  # flat (sum K, 3) uint8
    pal_offsets: np.ndarray  # (N + 1,) int64
    styles: np.ndarray  # (N,) str
    domains: np.ndarray  # (N,) str
    meta: np.ndarray  # (N,) str (JSON)

    def __len__(self) -> int:
        return int(self.shapes.shape[0])

    def grid(self, i: int) -> np.ndarray:
        h, w = self.shapes[i]
        return self.grids[self.offsets[i] : self.offsets[i + 1]].reshape(h, w)

    def palette(self, i: int) -> np.ndarray:
        return self.palettes[self.pal_offsets[i] : self.pal_offsets[i + 1]]

    def design(self, i: int) -> Design:
        meta = json.loads(str(self.meta[i])) if self.meta.size else {}
        meta.setdefault("style", str(self.styles[i]))
        return Design(self.grid(i).astype(np.int32), self.palette(i), meta)


def write_shard(path: str | Path, designs: Sequence[Design], styles: Sequence[str] | None = None,
                domains: Sequence[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(designs)
    styles = list(styles) if styles is not None else [str(d.meta.get("style", "unknown")) for d in designs]
    domains = list(domains) if domains is not None else [
        "synthetic" if d.meta.get("source") == "procedural" else "real" for d in designs]
    if any(d.n_colors > 256 for d in designs):
        raise ValueError("designs must have at most 256 colours")
    grids = [d.grid.astype(np.uint8).ravel() for d in designs]
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([g.size for g in grids])
    pals = [d.palette for d in designs]
    pal_offsets = np.zeros(n + 1, dtype=np.int64)
    pal_offsets[1:] = np.cumsum([p.shape[0] for p in pals])
    meta = np.array([json.dumps({k: v for k, v in d.meta.items() if k != "params"}) for d in designs])
    np.savez_compressed(
        path,
        grids=np.concatenate(grids) if grids else np.zeros(0, np.uint8),
        offsets=offsets,
        shapes=np.array([d.shape for d in designs], dtype=np.int32).reshape(n, 2),
        palettes=np.concatenate(pals) if pals else np.zeros((0, 3), np.uint8),
        pal_offsets=pal_offsets,
        styles=np.array(styles, dtype=str),
        domains=np.array(domains, dtype=str),
        meta=meta,
    )
    return path


def read_shard(path: str | Path) -> ShardData:
    with np.load(Path(path), allow_pickle=False) as z:
        return ShardData(
            grids=z["grids"], offsets=z["offsets"], shapes=z["shapes"], palettes=z["palettes"],
            pal_offsets=z["pal_offsets"], styles=z["styles"], domains=z["domains"],
            meta=z["meta"] if "meta" in z else np.zeros(0, dtype=str),
        )


def read_shards(paths: Iterable[str | Path]) -> list[ShardData]:
    out = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.npz")) if p.is_dir() else [p]
        out.extend(read_shard(f) for f in files)
    return out


def build_procedural_shards(out_dir: str | Path, n: int, height: tuple[int, int] = (64, 160),
                            width: tuple[int, int] = (64, 160), styles: dict[str, float] | None = None,
                            seed: int = 0, shard_size: int = 10000) -> list[Path]:
    """Pre-generate procedural designs into shards (reproducible from ``seed``)."""
    from manitex.procedural import PROCEDURAL_STYLES, generate

    styles = styles or {s: 1.0 for s in PROCEDURAL_STYLES}
    names = list(styles)
    probs = np.array([styles[s] for s in names], dtype=float)
    probs /= probs.sum()
    out_dir = Path(out_dir)
    paths = []
    buf: list[Design] = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        style = names[int(rng.choice(len(names), p=probs))]
        h = int(rng.integers(height[0], height[1] + 1))
        w = int(rng.integers(width[0], width[1] + 1))
        d = generate(style, h, w, rng=rng)
        d.meta["seed"] = [seed, i]
        buf.append(d)
        if len(buf) == shard_size or i == n - 1:
            paths.append(write_shard(out_dir / f"synthetic-{len(paths):05d}.npz", buf))
            buf = []
    return paths


#: Files that ``manitex digitise`` writes next to a design but that are not designs.
NON_DESIGN_SUFFIXES = ("_preview.png", "_grid.png")
#: Metadata files that may sit in a design folder.
NON_DESIGN_NAMES = ("meta.json", "splits.json")


def find_design_files(src: str | Path) -> list[Path]:
    """Design files (``.png`` / ``.json`` / ``.npz``) under ``src``.

    The preview and grid-overlay images written by ``manitex digitise`` and ``meta.json`` files
    are skipped, and a design saved in several formats (same path without the extension) is
    counted once, preferring the ``.png``: that is the file curated in an image editor.
    """
    pref = {".png": 0, ".json": 1, ".npz": 2}
    best: dict[Path, Path] = {}
    for f in sorted(Path(src).rglob("*")):
        name = f.name.lower()
        if not f.is_file() or f.suffix.lower() not in pref or name.endswith(NON_DESIGN_SUFFIXES) or name in NON_DESIGN_NAMES:
            continue
        key = f.with_suffix("")
        if key not in best or pref[f.suffix.lower()] < pref[best[key].suffix.lower()]:
            best[key] = f
    return sorted(best.values())


def limit_colors(d: Design, max_colors: int = 16) -> Design:
    """Canonical design with at most ``max_colors`` colours: the least frequent colours are merged
    into their nearest (CIEDE2000) frequent colour."""
    from manitex.color import delta_e2000, srgb_to_lab

    d = d.canonical()
    if d.n_colors <= max_colors:
        return d
    lab = srgb_to_lab(d.palette.astype(float))
    keep = lab[:max_colors]
    lut = np.arange(d.n_colors)
    for j in range(max_colors, d.n_colors):
        lut[j] = int(np.argmin(delta_e2000(keep, np.broadcast_to(lab[j], keep.shape))))
    return Design(lut[d.grid], d.palette[:max_colors], d.meta).canonical()


def load_design_folder(src: str | Path, max_colors: int = 16) -> list[Design]:
    """Load every design under ``src`` (see :func:`find_design_files`) with ``source_file`` and
    ``piece_id`` (the first folder below ``src``, else the file stem) in its metadata."""
    src = Path(src)
    out = []
    for f in find_design_files(src):
        try:
            d = Design.load(f)
        except Exception as e:  # e.g. a photo or other image in the design folder
            raise ValueError(f"cannot read {f} as a design ({e}); remove it from {src}") from e
        d = limit_colors(d, max_colors)
        rel = f.relative_to(src)
        d.meta.setdefault("source_file", rel.as_posix())
        d.meta.setdefault("piece_id", rel.parts[0] if len(rel.parts) > 1 else f.stem)
        out.append(d)
    return out


def import_designs(src: str | Path, out: str | Path, style: str | None = None, domain: str = "real",
                   max_colors: int = 16) -> Path:
    """Import a folder of design files into one shard (see :func:`load_design_folder`).

    The style is taken from ``style`` if given, else from each design's metadata (``manitex
    digitise --style``), else from the name of the file's parent folder.  Designs with more than
    ``max_colors`` colours are reduced by merging the least frequent colours into their nearest
    frequent colour.
    """
    designs = load_design_folder(src, max_colors)
    if not designs:
        raise ValueError(f"no design files found under {src}")
    styles = [style or str(d.meta.get("style") or (Path(src) / d.meta["source_file"]).parent.name) for d in designs]
    return write_shard(out, designs, styles, [domain] * len(designs))
