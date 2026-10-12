"""Constrained iterative unmasking sampler.

Generation starts from an all-``MASK`` grid (or a partially known grid for inpainting /
outpainting) and unmasks cells over ``steps`` iterations.  Structural constraints are
enforced **exactly** with *orbit pooling*: cells that must share a colour (mirror images,
repeats of a motif, all cells of a stripe) form an orbit; the per-cell log-probabilities
are averaged over the orbit, one colour is sampled per orbit and the whole orbit is
unmasked at once.  For a factorised per-step proposal this is the product-of-experts
conditional given the equality constraint (with ``pool='sum'``) or its tempered version
(``pool='mean'``, the default, which avoids over-confidence from correlated cells).

Unmasking order: ``'random'`` (default) is the ancestral reverse process of the masking
diffusion -- every masked orbit is revealed with probability ``(r_t - r_s) / r_t`` and its
colour drawn from the model.  ``'confidence'`` is MaskGIT's ordering (most confident first,
with annealed Gumbel noise); it is mode-seeking and, with little noise, collapses sparse-motif
styles to plain ground, so it is kept for ablations.

Other controls: classifier-free guidance, a colour budget (``ncolors`` restricts the
palette slots), temperature and nucleus (top-p) filtering of each orbit's colour
distribution, which removes the stray low-probability colours that cells revealed in the
same step would otherwise pick up independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from manitex.constraints import Symmetry, make_orbits
from manitex.models.diffusion import mask_ratio
from manitex.models.transformer import GridTransformer


@dataclass
class SamplerConfig:
    steps: int = 64
    temperature: float = 1.0
    cfg: float = 1.0  # guidance weight w: logits = l_u + (1 + w) (l_c - l_u)
    strategy: str = "random"  # 'random' (ancestral) | 'confidence' (MaskGIT)
    noise: float = 4.5  # 'confidence' only: Gumbel noise scale on confidences, annealed to 0
    top_p: float = 0.95  # nucleus filtering of each orbit's colour distribution (1 = off)
    schedule: str = "cosine"
    restrict_colors: bool = True
    pool: str = "mean"  # 'mean' | 'sum'


def _as_vec(v, n: int, default: int, device) -> torch.Tensor:
    if v is None:
        return torch.full((n,), default, dtype=torch.long, device=device)
    if isinstance(v, int | np.integer):
        return torch.full((n,), int(v), dtype=torch.long, device=device)
    t = torch.as_tensor(v, dtype=torch.long, device=device).reshape(-1)
    return t.expand(n).clone() if t.numel() == 1 else t


@torch.no_grad()
def sample(
    model: GridTransformer,
    n: int,
    height: int,
    width: int,
    *,
    style: int | torch.Tensor | None = None,
    ncolors: int | torch.Tensor | None = None,
    domain: int | torch.Tensor | None = None,
    symmetry: Symmetry | None = None,
    orbits: np.ndarray | None = None,
    known: np.ndarray | torch.Tensor | None = None,
    config: SamplerConfig | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample ``n`` grids of size ``height x width`` (colour slots, long tensor on CPU).

    ``known``: optional (H, W) or (n, H, W) array of colour slots with -1 for unknown cells
    (inpainting).  ``orbits``: explicit orbit map (overrides ``symmetry``).
    """
    cfg = config or SamplerConfig()
    model.eval()
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    if generator is not None and generator.device.type != dev.type:
        # random ops on CUDA tensors need a CUDA generator: re-seed one on the sampling device
        generator = torch.Generator(device=dev).manual_seed(int(generator.initial_seed()))
    V = model.cfg.vocab
    MASK = model.mask_id
    p = model.cfg.patch
    H, W = int(height), int(width)
    Hp, Wp = -(-H // p) * p, -(-W // p) * p

    # orbit ids on the padded canvas (padding cells are free and cropped away)
    orb = orbits if orbits is not None else make_orbits(H, W, symmetry)
    full = np.arange(Hp * Wp, dtype=np.int64).reshape(Hp, Wp) + Hp * Wp  # unique ids for padding cells
    rr, cc = np.divmod(orb, W)
    full[:H, :W] = rr * Wp + cc
    uniq, oid_np = np.unique(full.ravel(), return_inverse=True)
    n_orb = int(uniq.size)
    oid = torch.as_tensor(oid_np.reshape(-1), device=dev)
    counts = torch.bincount(oid, minlength=n_orb).float()

    style_v = _as_vec(style, n, model.null_style, dev)
    ncol_v = _as_vec(ncolors, n, model.null_ncolors, dev)
    dom_v = _as_vec(domain, n, model.null_domain, dev)
    use_cfg = cfg.cfg > 0 and (style is not None or ncolors is not None or domain is not None)

    orbit_known = torch.zeros(n, n_orb, dtype=torch.bool, device=dev)
    orbit_val = torch.zeros(n, n_orb, dtype=torch.long, device=dev)
    if known is not None:
        kn = torch.as_tensor(np.asarray(known), dtype=torch.long, device=dev)
        if kn.dim() == 2:
            kn = kn.unsqueeze(0).expand(n, -1, -1)
        kfull = torch.full((n, Hp, Wp), -1, dtype=torch.long, device=dev)
        kfull[:, :H, :W] = kn[:, :H, :W]
        kflat = kfull.view(n, -1)
        is_k = kflat >= 0
        if is_k.any():
            if int(kflat.max()) >= V:
                raise ValueError("known colour slots must be < vocab")
            # majority known value per orbit
            votes = torch.zeros(n, n_orb * V, device=dev)
            idx = oid.unsqueeze(0).expand(n, -1)
            votes.scatter_add_(1, idx * V + kflat.clamp_min(0), is_k.float())
            votes = votes.view(n, n_orb, V)
            orbit_known = votes.sum(-1) > 0
            orbit_val = votes.argmax(-1)

    colour_ok = torch.arange(V, device=dev).unsqueeze(0) < ncol_v.clamp(1, V).unsqueeze(1)  # (n, V)
    if not cfg.restrict_colors or ncolors is None:
        colour_ok = torch.ones_like(colour_ok)

    def canvas() -> torch.Tensor:
        vals = torch.where(orbit_known, orbit_val, torch.full_like(orbit_val, MASK))
        return vals[:, oid].view(n, Hp, Wp)

    M0 = (~orbit_known).sum(dim=1).float()
    steps = max(1, int(cfg.steps))
    for s in range(steps):
        masked = ~orbit_known
        if not masked.any():
            break
        x = canvas()
        if use_cfg:
            null = (torch.full_like(style_v, model.null_style), torch.full_like(ncol_v, model.null_ncolors),
                    torch.full_like(dom_v, model.null_domain))
            logits = model(torch.cat([x, x]), torch.cat([style_v, null[0]]), torch.cat([ncol_v, null[1]]),
                           torch.cat([dom_v, null[2]])).float()
            lc, lu = logits[:n], logits[n:]
            logits = lu + (1.0 + cfg.cfg) * (lc - lu)
        else:
            logits = model(x, style_v, ncol_v, dom_v).float()
        logits = logits.view(n, Hp * Wp, V)
        logits = logits.masked_fill(~colour_ok.unsqueeze(1), float("-inf"))
        logp = F.log_softmax(logits / max(cfg.temperature, 1e-4), dim=-1)
        pooled = torch.zeros(n, n_orb, V, device=dev).index_add_(1, oid, logp.nan_to_num(neginf=-1e9))
        if cfg.pool == "mean":
            pooled = pooled / counts.view(1, -1, 1)
        pooled = pooled.masked_fill(~colour_ok.unsqueeze(1), float("-inf"))
        logq = F.log_softmax(pooled, dim=-1)
        probs = logq.exp()
        if cfg.top_p < 1.0:  # keep the most probable colours covering top_p of the mass
            ps, pi = probs.sort(dim=-1, descending=True)
            ps = ps.masked_fill(ps.cumsum(-1) - ps >= cfg.top_p, 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, pi, ps)
        draw = torch.multinomial(probs.view(-1, V), 1, generator=generator).view(n, n_orb)
        conf = logq.gather(-1, draw.unsqueeze(-1)).squeeze(-1)

        last = s == steps - 1
        if cfg.strategy not in ("random", "confidence"):
            raise ValueError(f"unknown sampling strategy {cfg.strategy!r}")
        if cfg.strategy == "random":
            r_now = mask_ratio(torch.tensor(1.0 - s / steps), cfg.schedule).item()
            r_next = mask_ratio(torch.tensor(1.0 - (s + 1) / steps), cfg.schedule).item()
            p_unmask = 1.0 if last or r_now <= 0 else (r_now - r_next) / r_now
            u = torch.rand(n, n_orb, device=dev, generator=generator)
            chosen = masked & (u < p_unmask)
        else:
            r_next = mask_ratio(torch.tensor(1.0 - (s + 1) / steps), cfg.schedule).item()
            target = torch.floor(M0 * r_next) if not last else torch.zeros_like(M0)
            n_masked = masked.sum(dim=1).float()
            k = (n_masked - target).clamp_min(1.0).minimum(n_masked).long()
            if cfg.noise > 0:
                g = -torch.log(-torch.log(torch.rand(n, n_orb, device=dev, generator=generator).clamp(1e-9, 1 - 1e-9)))
                conf = conf + cfg.noise * (1.0 - (s + 1) / steps) * g
            conf = conf.masked_fill(~masked, float("-inf"))
            order = conf.argsort(dim=1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(n_orb, device=dev).unsqueeze(0).expand(n, -1))
            chosen = masked & (rank < k.unsqueeze(1))
        orbit_val = torch.where(chosen, draw, orbit_val)
        orbit_known = orbit_known | chosen

    out = canvas()
    if (out == MASK).any():  # pragma: no cover - only if steps ran out unexpectedly
        out = torch.where(out == MASK, torch.zeros_like(out), out)
    return out[:, :H, :W].cpu()
