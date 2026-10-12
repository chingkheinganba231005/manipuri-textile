"""Grid transformer: a DiT-style backbone over design-cell grids.

* Each cell (a colour slot or the ``MASK`` token) is embedded; ``p x p`` patches of cell
  embeddings are concatenated and projected to one token, so the token sequence is
  ``(H/p) * (W/p)`` long while every cell stays individually addressable.
* 2-D rotary position embeddings (half of each head for rows, half for columns) make the
  model translation-equivariant and usable at sizes other than the training size.
* Conditioning (style, number of colours, domain) enters through adaLN-Zero modulation;
  each condition has a learned *null* entry for classifier-free guidance.
* Output: per-cell logits over the colour slots.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from manitex.models.config import ModelConfig


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def rope_tables(h: int, w: int, head_dim: int, base: float, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin tables of shape (h*w, head_dim/2) for 2-D axial rotary embeddings."""
    if head_dim % 4:
        raise ValueError("head_dim must be divisible by 4 for 2-D RoPE")
    n = head_dim // 4
    inv = base ** (-torch.arange(n, device=device, dtype=torch.float32) / n)
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    ang = torch.cat([yy.reshape(-1, 1) * inv, xx.reshape(-1, 1) * inv], dim=1)  # (h*w, head_dim/2)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive feature pairs of x (B, heads, N, D) by the given angles."""
    x2 = x.reshape(*x.shape[:-1], -1, 2)
    a, b = x2[..., 0], x2[..., 1]
    out = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return out.reshape(x.shape)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, qk_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        if d % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos.to(q.dtype), sin.to(q.dtype)), apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class SwiGLU(nn.Module):
    def __init__(self, d: int, mlp_ratio: float):
        super().__init__()
        hidden = int(2 * mlp_ratio * d / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.w12 = nn.Linear(d, 2 * hidden)
        self.w3 = nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, cfg.heads, cfg.qk_norm, cfg.dropout)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(d, cfg.mlp_ratio)
        self.ada = nn.Linear(d, 6 * d)

    def forward(self, x: torch.Tensor, c: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.ada(F.silu(c)).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, sc1), cos, sin)
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class GridTransformer(nn.Module):
    """Predicts per-cell colour logits for a (partially masked) design grid."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, p, V = cfg.d_model, cfg.patch, cfg.vocab
        self.mask_id = V
        self.cell_emb = nn.Embedding(V + 1, cfg.cell_dim)
        self.patch_proj = nn.Linear(p * p * cfg.cell_dim, d)
        self.style_emb = nn.Embedding(cfg.n_styles + 1, d)  # last = null
        self.ncol_emb = nn.Embedding(V + 2, d)  # 1..V colours, V+1 = null (0 unused)
        self.domain_emb = nn.Embedding(cfg.n_domains + 1, d)  # last = null
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Linear(d, 2 * d)
        self.head = nn.Linear(d, p * p * V)
        self._rope_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.reset_parameters()

    # null ids for classifier-free guidance
    @property
    def null_style(self) -> int:
        return self.cfg.n_styles

    @property
    def null_ncolors(self) -> int:
        return self.cfg.vocab + 1

    @property
    def null_domain(self) -> int:
        return self.cfg.n_domains

    def reset_parameters(self) -> None:
        def init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        self.apply(init)
        for blk in self.blocks:
            nn.init.zeros_(blk.ada.weight)
            nn.init.zeros_(blk.ada.bias)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope(self, h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (h, w, str(device))
        if key not in self._rope_cache:
            self._rope_cache[key] = rope_tables(h, w, self.cfg.d_model // self.cfg.heads, self.cfg.rope_base, device)
        return self._rope_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor | None = None,
        ncolors: torch.Tensor | None = None,
        domain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, H, W) long in [0, vocab] (vocab = MASK).  Returns logits (B, H, W, vocab)."""
        B, H, W = x.shape
        p = self.cfg.patch
        if H % p or W % p:
            raise ValueError(f"grid size {H}x{W} must be divisible by the patch size {p}")
        dev = x.device
        if style is None:
            style = torch.full((B,), self.null_style, device=dev, dtype=torch.long)
        if ncolors is None:
            ncolors = torch.full((B,), self.null_ncolors, device=dev, dtype=torch.long)
        if domain is None:
            domain = torch.full((B,), self.null_domain, device=dev, dtype=torch.long)
        h, w = H // p, W // p
        e = self.cell_emb(x)  # (B, H, W, c)
        e = e.view(B, h, p, w, p, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, h * w, -1)
        tok = self.patch_proj(e)
        c = self.style_emb(style) + self.ncol_emb(ncolors) + self.domain_emb(domain)
        cos, sin = self._rope(h, w, dev)
        for blk in self.blocks:
            tok = blk(tok, c, cos, sin)
        shift, scale = self.final_ada(F.silu(c)).chunk(2, dim=-1)
        tok = modulate(self.final_norm(tok), shift, scale)
        out = self.head(tok)  # (B, h*w, p*p*V)
        V = self.cfg.vocab
        out = out.view(B, h, w, p, p, V).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, V)
        return out


def count_params_human(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"
