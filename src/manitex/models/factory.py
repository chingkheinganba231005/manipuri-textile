"""Build the network for a model configuration and pick the matching objective and sampler."""

from __future__ import annotations

import torch

from manitex.models.autoregressive import ARGridTransformer, AutoregressiveLM, ar_sample
from manitex.models.config import ModelConfig
from manitex.models.diffusion import MaskedDiffusion
from manitex.models.sampler import sample
from manitex.models.transformer import GridTransformer

ARCHS = ("mdm", "ar")


def build_model(cfg: ModelConfig) -> GridTransformer:
    """``arch='mdm'``: masked diffusion transformer; ``arch='ar'``: autoregressive baseline."""
    if cfg.arch == "mdm":
        return GridTransformer(cfg)
    if cfg.arch == "ar":
        return ARGridTransformer(cfg)
    raise ValueError(f"unknown model architecture {cfg.arch!r}; choose from {ARCHS}")


def is_autoregressive(model: torch.nn.Module) -> bool:
    return isinstance(model, ARGridTransformer)


def sampler_for(model: torch.nn.Module):
    """The sampling function for a model (same signature for both architectures)."""
    return ar_sample if is_autoregressive(model) else sample


def objective_for(model: GridTransformer, schedule: str = "cosine", weighting: str = "mean", p_uncond: float = 0.1,
                  p_drop_field: float = 0.1, net: torch.nn.Module | None = None):
    """Training objective with ``loss(batch)`` and ``nll_bound(batch)`` (bits per cell)."""
    if is_autoregressive(model):
        return AutoregressiveLM(model, p_uncond, p_drop_field, net=net)
    return MaskedDiffusion(model, schedule, weighting, p_uncond, p_drop_field, net=net)
