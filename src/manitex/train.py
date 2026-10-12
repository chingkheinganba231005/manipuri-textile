"""Training loop for the masked discrete diffusion model.

Single GPU::

    python -m manitex.train --config configs/train/pretrain_small.yaml

Multi-GPU (DDP)::

    torchrun --nproc_per_node=8 -m manitex.train --config ...

Any config key can be overridden on the command line, e.g. ``steps=2000 batch_size=64``.
Runs auto-resume from ``<out_dir>/last.pt``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from manitex.color import palette_for_style
from manitex.data.dataset import CropSpec, MixedStream, ProceduralStream, ShardStream, collate, fixed_batch
from manitex.data.shards import read_shards
from manitex.design import Design
from manitex.models.checkpoint import load_checkpoint, save_checkpoint
from manitex.models.config import DOMAINS, StyleVocab, make_config
from manitex.models.diffusion import Batch, MaskedDiffusion
from manitex.models.sampler import SamplerConfig, sample
from manitex.models.transformer import GridTransformer, count_params_human


@dataclass
class TrainConfig:
    out_dir: str = "runs/debug"
    model: Any = "small"  # preset name or dict (e.g. {preset: base, patch: 2})
    crop: int = 64
    batch_size: int = 128  # per GPU, per micro-step
    grad_accum: int = 1
    steps: int = 100_000
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup: int = 2000
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    precision: str = "bf16"  # 'bf16' | 'fp16' | 'fp32'
    compile: bool = False
    num_workers: int = 8
    seed: int = 0
    schedule: str = "cosine"
    weighting: str = "mean"
    p_uncond: float = 0.1
    p_drop_field: float = 0.1
    flip_lr: bool = True
    flip_ud: bool = True
    data: list[dict] = field(default_factory=lambda: [{"type": "procedural", "weight": 1.0}])
    log_every: int = 50
    eval_every: int = 2000
    eval_size: int = 256
    sample_every: int = 5000
    save_every: int = 5000
    init_from: str | None = None  # fine-tune: initialise weights from a checkpoint
    resume: bool = True
    wandb: bool = False
    wandb_project: str = "manitex"

    @classmethod
    def load(cls, path: str | Path | None, overrides: list[str] | None = None) -> TrainConfig:
        d: dict[str, Any] = {}
        if path:
            d = yaml.safe_load(Path(path).read_text()) or {}
        for ov in overrides or []:
            key, _, val = ov.partition("=")
            if not _:
                raise ValueError(f"override must look like key=value, got {ov!r}")
            d[key.strip()] = yaml.safe_load(val)
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**{k: _coerce(cls, k, v) for k, v in d.items()})


# ----------------------------------------------------------------------------- helpers


def _coerce(cls, key: str, value: Any) -> Any:
    """Coerce YAML values to the field's type (PyYAML reads ``1e-3`` as a string)."""
    default = cls.__dataclass_fields__[key].default
    if isinstance(value, str):
        if isinstance(default, bool):
            low = value.strip().lower()
            if low in ("true", "yes", "on", "1"):
                return True
            if low in ("false", "no", "off", "0"):
                return False
            raise ValueError(f"{key} expects a boolean, got {value!r}")
        if isinstance(default, int):
            return int(float(value))
        if isinstance(default, float):
            return float(value)
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return torch.distributed.get_rank() if _is_dist() else 0


def _world() -> int:
    return torch.distributed.get_world_size() if _is_dist() else 1


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(1, cfg.warmup)
    prog = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm.detach().float(), 1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def build_stream(cfg: TrainConfig, vocab: StyleVocab, split: str = "train"):
    crop = CropSpec(cfg.crop, cfg.flip_lr, cfg.flip_ud)
    sources, weights = [], []
    for src in cfg.data:
        kind = src.get("type", "procedural")
        if kind == "procedural":
            seed = cfg.seed + (0 if split == "train" else 7_777_777)
            sources.append(ProceduralStream(vocab, crop, src.get("styles"), seed=seed))
        elif kind == "shards":
            shards = read_shards(src["paths"] if isinstance(src["paths"], list) else [src["paths"]])
            if split == "val" and src.get("val_paths"):
                shards = read_shards(src["val_paths"] if isinstance(src["val_paths"], list) else [src["val_paths"]])
            sources.append(ShardStream(shards, vocab, crop, seed=cfg.seed, domain=src.get("domain")))
        else:
            raise ValueError(f"unknown data source type {kind!r}")
        weights.append(float(src.get("weight", 1.0)))
    return sources[0] if len(sources) == 1 else MixedStream(sources, weights, seed=cfg.seed)


def data_domains(cfg: TrainConfig) -> list[str]:
    """Domain tags present in the training data (saved in checkpoints so that generation
    conditions only on a domain the model has actually seen)."""
    seen: set[str] = set()
    for src in cfg.data:
        if src.get("type", "procedural") == "procedural":
            seen.add("synthetic")
        elif src.get("domain"):
            seen.add(str(src["domain"]))
        else:
            for sh in read_shards(src["paths"] if isinstance(src["paths"], list) else [src["paths"]]):
                seen.update(str(d) for d in np.unique(sh.domains))
    return sorted({d if d in DOMAINS else "real" for d in seen})  # unknown tags train as 'real'


def palette_bank_from_config(cfg: TrainConfig) -> dict[str, list[list[str]]]:
    """Canonical palettes of the real designs in the training shards, grouped by style."""
    from manitex.color import rgb_to_hex

    bank: dict[str, list[list[str]]] = {}
    for src in cfg.data:
        if src.get("type") != "shards":
            continue
        for sh in read_shards(src["paths"] if isinstance(src["paths"], list) else [src["paths"]]):
            for i in range(len(sh)):
                d = sh.design(i).canonical()
                bank.setdefault(str(sh.styles[i]), []).append([rgb_to_hex(c) for c in d.palette])
    return bank


def sample_sheet(model: GridTransformer, vocab: StyleVocab, size: int, device, step: int, per_style: int = 4,
                 domain: str = "synthetic") -> np.ndarray:
    """A grid image with ``per_style`` samples for every known style (rows)."""
    rows = []
    rng = np.random.default_rng(step)
    for sid, name in enumerate(vocab.names):
        if name == "unknown":
            continue
        out = sample(model, per_style, size, size, style=sid, domain=DOMAINS.get(domain, 0),
                     config=SamplerConfig(cfg=1.0), generator=torch.Generator().manual_seed(step + sid),
                     device=device).numpy()
        tiles = []
        for g in out:
            k = int(g.max()) + 1
            pal, _ = palette_for_style(name, k, rng)
            img = Design(g, pal).to_rgb(cell=3)
            tiles.append(np.pad(img, ((2, 2), (2, 2), (0, 0)), constant_values=255))
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0) if rows else np.zeros((1, 1, 3), np.uint8)


# ----------------------------------------------------------------------------- main loop


def train(cfg: TrainConfig) -> Path:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1 and not _is_dist():
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world = _rank(), _world()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    out = Path(cfg.out_dir)
    if rank == 0:
        (out / "samples").mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    model_cfg = make_config(cfg.model)
    vocab = StyleVocab.default(model_cfg.n_styles)
    init_state = None
    if cfg.init_from:
        init_state = load_checkpoint(cfg.init_from)
        model_cfg = make_config(init_state["model_config"] | {})
        vocab = StyleVocab(list(init_state["vocab"]), model_cfg.n_styles)
    stream = build_stream(cfg, vocab, "train")  # may add real-data styles to the vocabulary
    val_stream = build_stream(cfg, vocab, "val")

    model = GridTransformer(model_cfg).to(device)
    if init_state is not None:
        model.load_state_dict(init_state["ema"] or init_state["model"])
    ema = copy.deepcopy(model).float().eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            fused=device.type == "cuda")
    step = 0
    last = out / "last.pt"
    if cfg.resume and last.exists():
        st = load_checkpoint(last, map_location="cpu")
        model.load_state_dict(st["model"])
        if st.get("ema"):
            ema.load_state_dict(st["ema"])
        if st.get("optimizer"):
            opt.load_state_dict(st["optimizer"])
        vocab = StyleVocab(list(st["vocab"]), model_cfg.n_styles)
        step = int(st["step"])
        if rank == 0:
            print(f"resumed from {last} at step {step}")

    net: torch.nn.Module = model
    if cfg.compile:
        net = torch.compile(model)
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank] if device.type == "cuda" else None)
    diff = MaskedDiffusion(model, cfg.schedule, cfg.weighting, cfg.p_uncond, cfg.p_drop_field, net=net)
    diff_ema = MaskedDiffusion(ema, cfg.schedule, cfg.weighting)

    loader = torch.utils.data.DataLoader(stream, batch_size=cfg.batch_size, collate_fn=collate,
                                         num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                                         persistent_workers=cfg.num_workers > 0,
                                         prefetch_factor=4 if cfg.num_workers > 0 else None)
    batches = iter(loader)
    held_out = fixed_batch(val_stream, cfg.eval_size, cfg.seed)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.precision]
    use_amp = cfg.precision != "fp32" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16" and device.type == "cuda")
    palette_bank = palette_bank_from_config(cfg) if rank == 0 else {}

    wb = None
    if cfg.wandb and rank == 0:
        try:
            import wandb

            wb = wandb.init(project=cfg.wandb_project, config=asdict(cfg), dir=str(out))
        except ImportError:
            print("wandb not installed; logging to log.jsonl only")
    if rank == 0:
        print(f"model {count_params_human(model.num_params())} params | device {device} | world {world} | "
              f"styles {vocab.names}")
    log_f = open(out / "log.jsonl", "a") if rank == 0 else None
    t0, seen = time.time(), 0

    domains_seen = data_domains(cfg)
    if init_state is not None:
        domains_seen = sorted(set(domains_seen) | set((init_state.get("extra") or {}).get("domains_seen", [])))

    def _save(path: Path) -> None:
        save_checkpoint(path, model=model, ema=ema, optimizer=opt, step=step, model_cfg=model_cfg,
                        train_cfg=asdict(cfg), vocab=vocab, palette_bank=palette_bank,
                        extra={"domains_seen": domains_seen})

    while step < cfg.steps:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        stats_acc: dict[str, float] = {}
        for _ in range(cfg.grad_accum):
            raw = next(batches)
            b = Batch(raw["grid"], raw["style"], raw["ncolors"], raw["domain"]).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, stats = diff.loss(b)
            scaler.scale(loss / cfg.grad_accum).backward()
            for k, v in stats.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v / cfg.grad_accum
            seen += b.grid.shape[0] * world
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        ema_update(ema, model, min(cfg.ema_decay, (1 + step) / (10 + step)))

        if rank == 0 and (step % cfg.log_every == 0 or step == 1):
            dt = time.time() - t0
            rec = {"step": step, "lr": lr, "grad_norm": gnorm, "samples_per_s": seen / max(dt, 1e-9), **stats_acc}
            if device.type == "cuda":
                rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if wb:
                wb.log(rec, step=step)
            t0, seen = time.time(), 0
        if rank == 0 and (step % cfg.eval_every == 0 or step == cfg.steps):
            hb = Batch(held_out["grid"], held_out["style"], held_out["ncolors"], held_out["domain"]).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                bpc = float(diff_ema.nll_bound(hb, n_t=8).mean())
            rec = {"step": step, "eval_bits_per_cell_ema": bpc}
            print(json.dumps(rec))
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if wb:
                wb.log(rec, step=step)
        if rank == 0 and (step % cfg.sample_every == 0 or step == cfg.steps):
            sheet = sample_sheet(ema, vocab, cfg.crop, device, step)
            Image.fromarray(sheet).save(out / "samples" / f"step_{step:07d}.png")
        if rank == 0 and (step % cfg.save_every == 0 or step == cfg.steps):
            _save(last)
    if rank == 0:
        _save(out / "final.pt")
        log_f.close()
    if _is_dist():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return out / "final.pt"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the manitex masked diffusion model")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("overrides", nargs="*", help="key=value overrides")
    args = ap.parse_args(argv)
    train(TrainConfig.load(args.config, args.overrides))


if __name__ == "__main__":
    main()
