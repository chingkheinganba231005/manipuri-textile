"""Model configuration, presets and the style vocabulary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from manitex.design import MAX_COLORS


@dataclass
class ModelConfig:
    vocab: int = MAX_COLORS  # number of colour slots (canonical palette indices)
    patch: int = 2  # cells per patch side
    d_model: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    cell_dim: int = 32  # per-cell embedding size before patch projection
    n_styles: int = 64  # style-embedding capacity (room for real-data categories)
    n_domains: int = 4  # 0 = synthetic, 1 = real; rest reserved
    qk_norm: bool = True
    rope_base: float = 100.0
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


PRESETS: dict[str, dict[str, Any]] = {
    # ~1.3M parameters: unit tests and CPU smoke runs
    "tiny": dict(d_model=128, depth=4, heads=4, cell_dim=16),
    # ~32M (DiT-S scale): quick experiments / demo model
    "small": dict(d_model=384, depth=12, heads=6, cell_dim=32),
    # ~129M (DiT-B scale): main model on an A100
    "base": dict(d_model=768, depth=12, heads=12, cell_dim=48),
    # ~457M (DiT-L scale): scaling experiments
    "large": dict(d_model=1024, depth=24, heads=16, cell_dim=64),
}


def make_config(spec: str | dict[str, Any] | ModelConfig) -> ModelConfig:
    if isinstance(spec, ModelConfig):
        return spec
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"unknown model preset {spec!r}; choose from {list(PRESETS)}")
        return ModelConfig(**PRESETS[spec])
    spec = dict(spec)
    base = spec.pop("preset", None)
    cfg = ModelConfig(**PRESETS[base]) if base else ModelConfig()
    for k, v in spec.items():
        if k not in ModelConfig.__dataclass_fields__:
            raise ValueError(f"unknown model option {k!r}")
        setattr(cfg, k, v)
    return cfg


DOMAINS = {"synthetic": 0, "real": 1}


@dataclass
class StyleVocab:
    """Maps style names to embedding ids.  Id 0 is ``unknown``; new names are appended."""

    names: list[str] = field(default_factory=lambda: ["unknown"])
    capacity: int = 64

    def id(self, name: str | None, add: bool = False) -> int:
        if name is None:
            return 0
        if name in self.names:
            return self.names.index(name)
        if add:
            if len(self.names) >= self.capacity:
                raise ValueError("style vocabulary is full; increase ModelConfig.n_styles")
            self.names.append(name)
            return len(self.names) - 1
        return 0

    def name(self, i: int) -> str:
        return self.names[i] if 0 <= i < len(self.names) else "unknown"

    def to_list(self) -> list[str]:
        return list(self.names)

    @classmethod
    def default(cls, capacity: int = 64) -> StyleVocab:
        from manitex.procedural import PROCEDURAL_STYLES

        return cls(["unknown", *PROCEDURAL_STYLES], capacity)
