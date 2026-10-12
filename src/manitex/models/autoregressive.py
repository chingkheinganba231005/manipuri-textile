"""Autoregressive baseline: the same backbone with causal attention over cells in raster order.

The grid is read row by row.  Position ``i`` receives the colour of cell ``i - 1`` (``MASK``
is the start token) and the 2-D rotary position of cell ``i``; with causal attention its
output predicts cell ``i`` from all earlier cells, so ``log p(x) = sum_i log p(x_i | x_<i)``
is exact.  Cells are not grouped into patches (``patch = 1``): a patch token would predict
several cells at once.

Sampling generates one cell per step with a key/value cache (``H * W`` sequential steps).
Structural constraints are imposed by copying: a cell whose orbit already has a generated
member takes its colour, so mirrors, repeats and stripes hold exactly - but, unlike orbit
pooling in the masked model, each orbit's colour is chosen before its later members are seen.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from manitex.constraints import Symmetry, make_orbits
from manitex.models.config import ModelConfig
from manitex.models.diffusion import Batch, drop_conditions
from manitex.models.sampler import SamplerConfig, _as_vec, filter_top_p
from manitex.models.transformer import GridTransformer, apply_rope, modulate


class ARGridTransformer(GridTransformer):
    """Raster-order autoregressive model over design cells (teacher-forced ``forward``)."""

    def __init__(self, cfg: ModelConfig):
        if cfg.patch != 1:
            raise ValueError("the autoregressive model predicts single cells: set patch=1")
        super().__init__(cfg)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor | None = None,
        ncolors: torch.Tensor | None = None,
        domain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, H, W) colour slots.  ``logits[:, r, c]`` predict ``x[:, r, c]`` from the cells
        before it in raster order."""
        B, H, W = x.shape
        flat = x.reshape(B, H * W)
        shifted = torch.cat([torch.full_like(flat[:, :1], self.mask_id), flat[:, :-1]], dim=1)
        return self.backbone(shifted.view(B, H, W), self.conditioning(B, x.device, style, ncolors, domain), causal=True)


class AutoregressiveLM:
    """Training objective of the autoregressive baseline (same interface as ``MaskedDiffusion``)."""

    def __init__(self, model: ARGridTransformer, p_uncond: float = 0.1, p_drop_field: float = 0.1,
                 net: torch.nn.Module | None = None):
        self.model = model
        self.net = net if net is not None else model  # module used for forward passes (DDP / compiled)
        self.p_uncond = p_uncond
        self.p_drop_field = p_drop_field

    def loss(self, b: Batch, train: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
        if train:
            b = drop_conditions(b, self.model, self.p_uncond, self.p_drop_field)
        logits = self.net(b.grid, b.style, b.ncolors, b.domain)
        loss = F.cross_entropy(logits.float().permute(0, 3, 1, 2), b.grid)
        with torch.no_grad():
            acc = (logits.argmax(-1) == b.grid).float().mean()
        return loss, {"loss": float(loss.detach()), "next_cell_acc": float(acc)}

    @torch.no_grad()
    def nll_bound(self, b: Batch, n_t: int = 0) -> torch.Tensor:
        """Per-sample negative log-likelihood in bits per cell - exact for this model (the name
        matches ``MaskedDiffusion.nll_bound``; ``n_t`` is unused)."""
        logits = self.net(b.grid, b.style, b.ncolors, b.domain)
        ce = F.cross_entropy(logits.float().permute(0, 3, 1, 2), b.grid, reduction="none")
        return ce.mean(dim=(1, 2)) / math.log(2.0)


def _spread_known(known: torch.Tensor, rep: torch.Tensor, V: int) -> torch.Tensor:
    """(n, T) known slots (-1 = free) -> majority known slot of each cell's orbit (or -1)."""
    n, T = known.shape
    is_k = known >= 0
    votes = torch.zeros(n, T * V, device=known.device)
    votes.scatter_add_(1, rep.unsqueeze(0).expand(n, -1) * V + known.clamp_min(0), is_k.float())
    votes = votes.view(n, T, V)
    val = torch.where(votes.sum(-1) > 0, votes.argmax(-1), torch.full((n, T), -1, device=known.device))
    return val[:, rep]


class KVCacheDecoder:
    """Evaluates an :class:`ARGridTransformer` one cell at a time with a key/value cache.

    ``step(prev)`` takes the colour of the previous cell (``MASK`` before the first cell) and
    returns the logits of the next cell in raster order - the same values as the teacher-forced
    ``forward`` at that position.
    """

    def __init__(self, model: ARGridTransformer, height: int, width: int, c: torch.Tensor):
        self.model = model
        self.T = int(height) * int(width)
        self.B = c.shape[0]
        self.mods = [blk.ada(F.silu(c)).chunk(6, dim=-1) for blk in model.blocks]
        self.f_shift, self.f_scale = model.final_ada(F.silu(c)).chunk(2, dim=-1)
        self.cos, self.sin = model._rope(int(height), int(width), c.device)
        self.heads, self.D = model.cfg.heads, model.cfg.d_model
        self.hd = self.D // self.heads
        shape = (len(model.blocks), self.B, self.heads, self.T, self.hd)
        dtype = next(model.parameters()).dtype
        self.k_cache = torch.zeros(shape, device=c.device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=c.device, dtype=dtype)
        self.i = 0

    @torch.no_grad()
    def step(self, prev: torch.Tensor) -> torch.Tensor:
        if self.i >= self.T:
            raise IndexError("all cells have been decoded")
        m, i, B, D, heads, hd = self.model, self.i, self.B, self.D, self.heads, self.hd
        tok = m.patch_proj(m.cell_emb(prev)).unsqueeze(1)  # (B, 1, D)
        ci, si = self.cos[i : i + 1], self.sin[i : i + 1]
        for li, blk in enumerate(m.blocks):
            s1, sc1, g1, s2, sc2, g2 = self.mods[li]
            att = blk.attn
            q, k, v = att.qkv(modulate(blk.norm1(tok), s1, sc1)).view(B, 1, 3, heads, hd).permute(2, 0, 3, 1, 4)
            q, k = att.q_norm(q), att.k_norm(k)
            q = apply_rope(q, ci.to(q.dtype), si.to(q.dtype))
            k = apply_rope(k, ci.to(k.dtype), si.to(k.dtype))
            self.k_cache[li, :, :, i] = k[:, :, 0]
            self.v_cache[li, :, :, i] = v[:, :, 0]
            o = F.scaled_dot_product_attention(q, self.k_cache[li, :, :, : i + 1], self.v_cache[li, :, :, : i + 1])
            tok = tok + g1.unsqueeze(1) * att.proj(o.transpose(1, 2).reshape(B, 1, D))
            tok = tok + g2.unsqueeze(1) * blk.mlp(modulate(blk.norm2(tok), s2, sc2))
        self.i += 1
        return m.head(modulate(m.final_norm(tok), self.f_shift, self.f_scale))[:, 0]  # (B, V)


@torch.no_grad()
def ar_sample(
    model: ARGridTransformer,
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
    """Generate ``n`` grids cell by cell; arguments as in :func:`manitex.models.sampler.sample`
    (``steps``, ``strategy`` and ``noise`` do not apply).  Returns (n, H, W) long on the CPU."""
    cfg = config or SamplerConfig()
    if not 0.0 < cfg.top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {cfg.top_p}")
    model.eval()
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    if generator is not None and generator.device.type != dev.type:
        generator = torch.Generator(device=dev).manual_seed(int(generator.initial_seed()))
    V, H, W = model.cfg.vocab, int(height), int(width)
    T = H * W
    orb = orbits if orbits is not None else make_orbits(H, W, symmetry)
    _, first, inv = np.unique(np.asarray(orb).reshape(-1), return_index=True, return_inverse=True)
    rep = torch.as_tensor(first[inv.reshape(-1)], device=dev)  # earliest cell (raster order) of each orbit
    kn = torch.full((n, T), -1, dtype=torch.long, device=dev)
    if known is not None:
        k = torch.as_tensor(np.asarray(known), dtype=torch.long, device=dev)
        k = (k.unsqueeze(0).expand(n, -1, -1) if k.dim() == 2 else k)[:, :H, :W].reshape(n, T)
        if int(k.max()) >= V:
            raise ValueError("known colour slots must be < vocab")
        kn = _spread_known(k, rep, V)  # a known cell fixes its whole orbit

    style_v = _as_vec(style, n, model.null_style, dev)
    ncol_v = _as_vec(ncolors, n, model.null_ncolors, dev)
    dom_v = _as_vec(domain, n, model.null_domain, dev)
    use_cfg = cfg.cfg > 0 and (style is not None or ncolors is not None or domain is not None)
    colour_ok = torch.arange(V, device=dev).unsqueeze(0) < ncol_v.clamp(1, V).unsqueeze(1)  # (n, V)
    if not cfg.restrict_colors or ncolors is None:
        colour_ok = torch.ones_like(colour_ok)
    if use_cfg:  # conditional and unconditional sequences side by side
        style_v = torch.cat([style_v, torch.full_like(style_v, model.null_style)])
        ncol_v = torch.cat([ncol_v, torch.full_like(ncol_v, model.null_ncolors)])
        dom_v = torch.cat([dom_v, torch.full_like(dom_v, model.null_domain)])
    B = style_v.shape[0]

    decoder = KVCacheDecoder(model, H, W, model.conditioning(B, dev, style_v, ncol_v, dom_v))
    out = torch.zeros(n, T, dtype=torch.long, device=dev)
    prev = torch.full((B,), model.mask_id, dtype=torch.long, device=dev)  # start token
    for i in range(T):
        logits = decoder.step(prev).float()  # (B, V)
        if use_cfg:
            logits = logits[n:] + (1.0 + cfg.cfg) * (logits[:n] - logits[n:])
        logits = logits.masked_fill(~colour_ok, float("-inf"))
        probs = filter_top_p(F.softmax(logits / max(cfg.temperature, 1e-4), dim=-1), cfg.top_p)
        draw = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        r = int(rep[i])
        val = torch.where(kn[:, i] >= 0, kn[:, i], out[:, r] if r < i else draw)
        out[:, i] = val
        prev = torch.cat([val, val]) if use_cfg else val
    return out.view(n, H, W).cpu()
