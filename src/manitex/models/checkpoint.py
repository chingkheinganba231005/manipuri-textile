"""Checkpoint saving / loading and slim export for inference (safetensors + JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from manitex.models.config import ModelConfig, StyleVocab
from manitex.models.transformer import GridTransformer


def save_checkpoint(path: str | Path, *, model: torch.nn.Module, ema: torch.nn.Module | None,
                    optimizer: torch.optim.Optimizer | None, step: int, model_cfg: ModelConfig,
                    train_cfg: dict[str, Any], vocab: StyleVocab, palette_bank: dict[str, list[list[str]]] | None = None,
                    extra: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "manitex.ckpt",
        "version": 1,
        "step": int(step),
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg,
        "vocab": vocab.to_list(),
        "palette_bank": palette_bank or {},
        "model": _unwrap(model).state_dict(),
        "ema": _unwrap(ema).state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def _unwrap(m: torch.nn.Module | None) -> torch.nn.Module | None:
    if m is None:
        return None
    m = getattr(m, "module", m)  # DDP
    return getattr(m, "_orig_mod", m)  # torch.compile


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state = torch.load(Path(path), map_location=map_location, weights_only=True)
    if state.get("format") != "manitex.ckpt":
        raise ValueError(f"{path} is not a manitex checkpoint")
    return state


def export_slim(ckpt: str | Path, out_dir: str | Path, use_ema: bool = True, dtype: str = "float16") -> Path:
    """Write ``model.safetensors`` + ``config.json`` (model config, vocab, palette bank)."""
    from safetensors.torch import save_file

    state = load_checkpoint(ckpt)
    weights = state["ema"] if (use_ema and state.get("ema")) else state["model"]
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    weights = {k: v.to(dt).contiguous() for k, v in weights.items()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out / "model.safetensors"))
    meta = {
        "format": "manitex.slim",
        "version": 1,
        "step": state["step"],
        "model_config": state["model_config"],
        "vocab": state["vocab"],
        "palette_bank": state.get("palette_bank", {}),
        "weights": "ema" if (use_ema and state.get("ema")) else "model",
        "dtype": dtype,
        "train_crop": int((state.get("train_config") or {}).get("crop", 64)),
    }
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    return out


def load_model(path: str | Path, device: str | torch.device = "cpu", use_ema: bool = True
               ) -> tuple[GridTransformer, dict[str, Any]]:
    """Load a model from a full checkpoint (``.pt``) or a slim export directory.

    Returns ``(model in eval mode, meta)`` with ``meta`` holding ``vocab`` (a
    :class:`StyleVocab`), ``palette_bank`` and ``step``.
    """
    path = Path(path)
    if path.is_dir():
        from safetensors.torch import load_file

        meta = json.loads((path / "config.json").read_text())
        cfg = ModelConfig.from_dict(meta["model_config"])
        model = GridTransformer(cfg)
        sd = {k: v.float() for k, v in load_file(str(path / "model.safetensors")).items()}
        model.load_state_dict(sd)
    else:
        state = load_checkpoint(path)
        meta = state
        cfg = ModelConfig.from_dict(state["model_config"])
        model = GridTransformer(cfg)
        sd = state["ema"] if (use_ema and state.get("ema")) else state["model"]
        model.load_state_dict(sd)
    model.to(device).eval()
    info = {
        "vocab": StyleVocab(list(meta["vocab"]), cfg.n_styles),
        "palette_bank": meta.get("palette_bank", {}),
        "step": meta.get("step", 0),
        "model_config": cfg,
        "train_crop": int(meta.get("train_crop") or (meta.get("train_config") or {}).get("crop", 64)),
    }
    return model, info
