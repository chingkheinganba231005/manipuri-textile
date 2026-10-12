"""Masked (absorbing-state) discrete diffusion over design grids.

Forward process: every cell is independently replaced by ``MASK`` with probability
``r(t)`` at time ``t in [0, 1]`` (``r(0) = 0``, ``r(1) = 1``).  The network predicts the
original colour of masked cells.

* ``schedule='cosine'``: ``r(t) = sin(pi t / 2)`` (the MaskGIT schedule, which spends more
  training on heavily masked grids); ``'linear'``: ``r(t) = t``.
* ``weighting='mean'``: average cross-entropy over masked cells (MaskGIT objective, best
  sample quality); ``'elbo'``: the continuous-time ELBO weight ``r'(t) / r(t)``.

:meth:`MaskedDiffusion.nll_bound` returns an upper bound on the negative log-likelihood
in bits per cell (linear schedule ELBO), valid for any trained denoiser and comparable
across models - the likelihood metric reported in the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from manitex.models.transformer import GridTransformer


@dataclass
class Batch:
    grid: torch.Tensor  # (B, H, W) long, colour slots
    style: torch.Tensor  # (B,)
    ncolors: torch.Tensor  # (B,)
    domain: torch.Tensor  # (B,)

    def to(self, device) -> Batch:
        return Batch(*(t.to(device, non_blocking=True) for t in (self.grid, self.style, self.ncolors, self.domain)))


def mask_ratio(t: torch.Tensor, schedule: str) -> torch.Tensor:
    if schedule == "linear":
        return t
    if schedule == "cosine":
        return torch.sin(0.5 * math.pi * t)
    raise ValueError(f"unknown schedule {schedule!r}")


def mask_ratio_rate(t: torch.Tensor, schedule: str) -> torch.Tensor:
    """d r / d t"""
    if schedule == "linear":
        return torch.ones_like(t)
    return 0.5 * math.pi * torch.cos(0.5 * math.pi * t)


def drop_conditions(b: Batch, model: GridTransformer, p_uncond: float, p_drop_field: float) -> Batch:
    """Classifier-free guidance dropout: all conditions null with ``p_uncond``, else each field
    independently null with ``p_drop_field``."""
    B = b.grid.shape[0]
    dev = b.grid.device
    all_null = torch.rand(B, device=dev) < p_uncond

    def drop(x: torch.Tensor, null: int) -> torch.Tensor:
        d = all_null | (torch.rand(B, device=dev) < p_drop_field)
        return torch.where(d, torch.full_like(x, null), x)

    return Batch(b.grid, drop(b.style, model.null_style), drop(b.ncolors, model.null_ncolors),
                 drop(b.domain, model.null_domain))


class MaskedDiffusion:
    def __init__(self, model: GridTransformer, schedule: str = "cosine", weighting: str = "mean",
                 p_uncond: float = 0.1, p_drop_field: float = 0.1, net: torch.nn.Module | None = None):
        self.model = model  # the GridTransformer (mask / null ids)
        self.net = net if net is not None else model  # module used for forward passes (DDP / compiled)
        self.schedule = schedule
        self.weighting = weighting
        self.p_uncond = p_uncond
        self.p_drop_field = p_drop_field
        if weighting not in ("mean", "elbo"):
            raise ValueError(f"unknown weighting {weighting!r}")

    @property
    def mask_id(self) -> int:
        return self.model.mask_id

    def drop_conditions(self, b: Batch) -> Batch:
        return drop_conditions(b, self.model, self.p_uncond, self.p_drop_field)

    def corrupt(self, x0: torch.Tensor, t: torch.Tensor, schedule: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask each cell with probability r(t); every sample keeps >= 1 masked cell."""
        B = x0.shape[0]
        r = mask_ratio(t, schedule or self.schedule).view(B, 1, 1)
        u = torch.rand(x0.shape, device=x0.device)
        mask = u < r
        none = ~mask.view(B, -1).any(dim=1)
        if none.any():
            flat = u.view(B, -1)
            idx = flat.argmin(dim=1)
            mask.view(B, -1)[none, idx[none]] = True
        return torch.where(mask, torch.full_like(x0, self.mask_id), x0), mask

    def loss(self, b: Batch, train: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
        x0 = b.grid
        B = x0.shape[0]
        dev = x0.device
        if train:
            b = self.drop_conditions(b)
        # stratified times reduce the variance of the loss estimate
        t = ((torch.rand(1, device=dev) + torch.arange(B, device=dev) / B) % 1.0).clamp_min(1e-3)
        xt, mask = self.corrupt(x0, t)
        logits = self.net(xt, b.style, b.ncolors, b.domain)
        ce = F.cross_entropy(logits.float().permute(0, 3, 1, 2), x0, reduction="none")  # (B, H, W)
        m = mask.float()
        n_masked = m.sum(dim=(1, 2)).clamp_min(1.0)
        if self.weighting == "mean":
            per = (ce * m).sum(dim=(1, 2)) / n_masked
        else:
            w = mask_ratio_rate(t, self.schedule) / mask_ratio(t, self.schedule).clamp_min(1e-4)
            per = w * (ce * m).sum(dim=(1, 2)) / (x0.shape[1] * x0.shape[2])
        loss = per.mean()
        with torch.no_grad():
            acc = ((logits.argmax(-1) == x0).float() * m).sum() / m.sum().clamp_min(1.0)
        return loss, {"loss": float(loss.detach()), "masked_acc": float(acc), "mask_frac": float(m.mean())}

    @torch.no_grad()
    def nll_bound(self, b: Batch, n_t: int = 16) -> torch.Tensor:
        """Per-sample NELBO in bits per cell (linear-schedule ELBO, ``n_t`` stratified times)."""
        x0 = b.grid
        B, H, W = x0.shape
        total = torch.zeros(B, device=x0.device)
        for i in range(n_t):
            t = torch.full((B,), (i + 0.5) / n_t, device=x0.device)
            xt, mask = self.corrupt(x0, t, schedule="linear")
            logits = self.net(xt, b.style, b.ncolors, b.domain)
            ce = F.cross_entropy(logits.float().permute(0, 3, 1, 2), x0, reduction="none")
            total += (ce * mask.float()).sum(dim=(1, 2)) / t
        return total / n_t / (H * W) / math.log(2.0)
