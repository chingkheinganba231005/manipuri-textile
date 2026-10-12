#!/usr/bin/env python
"""Check the GPU set-up and measure training speed and memory before a long run.

    python scripts/benchmark_gpu.py                                    # set-up report only
    python scripts/benchmark_gpu.py --config configs/train/pretrain_base.yaml --steps 30 --probe

Reports the GPU, its memory and compute capability, whether the installed PyTorch build contains
kernels for it (Blackwell GPUs such as the RTX PRO 6000 of Google Cloud G4 machines need a CUDA
12.8 or newer build) and the achieved bf16 matrix-multiply throughput. With ``--config`` it runs
``--steps`` real training micro-steps of that config (same model, data stream, precision and
``torch.compile`` setting as ``manitex train``) and reports seconds per optimiser step, peak
memory and the projected time for the config's ``steps``. ``--probe`` also tries larger
micro-batches (without ``torch.compile``, which only lowers memory) and suggests ``batch_size`` /
``grad_accum`` for the same effective batch. Config keys can be overridden as in training:
``... --config configs/train/pretrain_small.yaml crop=32 compile=false``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

MEMORY_FRACTION = 0.85  # largest share of the GPU's memory a suggested micro-batch may use (headroom for fragmentation)


def environment() -> dict:
    info: dict = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return info
    p = torch.cuda.get_device_properties(0)
    arch = torch.cuda.get_arch_list()
    sm = f"sm_{p.major}{p.minor}"
    info.update({"gpu": p.name, "memory_gb": round(p.total_memory / 1e9, 1), "capability": sm,
                 "kernels_for_gpu": sm in arch, "arch_list": arch, "bf16": torch.cuda.is_bf16_supported()})
    try:
        out = os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null").read().strip()
        info["driver"] = out.splitlines()[0] if out else None
    except OSError:
        info["driver"] = None
    return info


def matmul_tflops(dtype: torch.dtype, n: int = 8192, iters: int = 20) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * iters / (time.perf_counter() - t) / 1e12


def _setup(cfg, device: torch.device, compile_model: bool):
    from manitex.models import StyleVocab, build_model, make_config, objective_for
    from manitex.train import build_stream, param_groups

    model_cfg = make_config(cfg.model)
    vocab = StyleVocab.default(model_cfg.n_styles)
    stream = build_stream(cfg, vocab, "train")
    model = build_model(model_cfg).to(device)
    net = torch.compile(model) if compile_model else model
    obj = objective_for(model, cfg.schedule, cfg.weighting, cfg.p_uncond, cfg.p_drop_field, net=net)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, fused=device.type == "cuda")
    return model, stream, obj, opt


def run_steps(cfg, model, stream, obj, opt, device: torch.device, micro: int, steps: int, warmup: int) -> dict:
    """Time ``steps`` micro-steps (forward, backward, optimiser step) after ``warmup`` untimed ones."""
    from manitex.data.dataset import collate
    from manitex.models.diffusion import Batch

    workers = min(cfg.num_workers, os.cpu_count() or 1)
    loader = torch.utils.data.DataLoader(stream, batch_size=micro, collate_fn=collate, num_workers=workers,
                                         pin_memory=device.type == "cuda", prefetch_factor=4 if workers else None)
    it = iter(loader)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.precision]
    use_amp = cfg.precision != "fp32" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp == torch.float16)

    def step() -> None:
        raw = next(it)
        b = Batch(raw["grid"], raw["style"], raw["ncolors"], raw["domain"]).to(device)
        with torch.autocast(device_type=device.type, dtype=amp, enabled=use_amp):
            loss, _ = obj.loss(b)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(warmup):
        step()
    sync()
    t1 = time.perf_counter()
    for _ in range(steps):
        step()
    sync()
    dt = (time.perf_counter() - t1) / max(steps, 1)
    out = {"micro_batch": micro, "warmup_s": round(t1 - t0, 1), "sec_per_micro_step": dt, "samples_per_s": micro / dt}
    if device.type == "cuda":
        out["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    return out


def probe(cfg, device: torch.device, steps: int = 2) -> list[dict]:
    """Largest micro-batches that fit (eager mode, a conservative memory estimate)."""
    model, stream, obj, opt = _setup(cfg, device, compile_model=False)
    eff = cfg.batch_size * cfg.grad_accum
    results = []
    for micro in sorted({m for m in (cfg.batch_size, eff // 4, eff // 2, eff) if m >= 1 and eff % m == 0}):
        try:
            results.append(run_steps(cfg, model, stream, obj, opt, device, micro, steps, warmup=1))
        except torch.cuda.OutOfMemoryError:
            results.append({"micro_batch": micro, "out_of_memory": True})
            break
        finally:
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


def suggest(results: list[dict], effective_batch: int, total_gb: float | None) -> dict | None:
    """The largest probed micro-batch whose measured peak memory leaves headroom, as the config
    overrides ``{"batch_size", "grad_accum"}`` that keep the effective batch; None if none fits."""
    fits = [p for p in results if not p.get("out_of_memory")
            and (total_gb is None or p.get("peak_memory_gb", 0) <= MEMORY_FRACTION * total_gb)]
    if not fits:
        return None
    micro = max(p["micro_batch"] for p in fits)
    return {"batch_size": micro, "grad_accum": effective_batch // micro}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="training config to benchmark")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    ap.add_argument("--steps", type=int, default=20, help="timed micro-steps")
    ap.add_argument("--warmup", type=int, default=5, help="untimed micro-steps (includes torch.compile)")
    ap.add_argument("--probe", action="store_true", help="find the largest micro-batch that fits")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", default=None, help="also write the report to this file")
    args = ap.parse_args()

    report: dict = {"environment": environment()}
    env = report["environment"]
    print(json.dumps({k: v for k, v in env.items() if k != "arch_list"}, indent=1))
    if env.get("cuda_available"):
        if not env["kernels_for_gpu"]:
            print(f"WARNING: this PyTorch build has no kernels for {env['capability']} ({env['gpu']}); install a "
                  "build for your GPU (Blackwell needs CUDA 12.8+, see docs/TRAINING.md)")
        report["matmul_tflops_bf16"] = matmul_tflops(torch.bfloat16)
        print(f"bf16 matmul: {report['matmul_tflops_bf16']:.0f} TFLOP/s achieved (8192^3)")
    if args.config:
        from manitex.train import TrainConfig

        cfg = TrainConfig.load(args.config, args.overrides)
        device = torch.device(args.device)
        model, stream, obj, opt = _setup(cfg, device, compile_model=cfg.compile and device.type == "cuda")
        r = run_steps(cfg, model, stream, obj, opt, device, cfg.batch_size, args.steps, args.warmup)
        per_step = r["sec_per_micro_step"] * cfg.grad_accum
        r.update({"sec_per_optimizer_step": per_step, "grad_accum": cfg.grad_accum, "config_steps": cfg.steps,
                  "projected_hours": per_step * cfg.steps / 3600})
        report["benchmark"] = r
        hours = r["projected_hours"]
        span = f"{hours:.1f} h" if hours >= 1 else f"{60 * hours:.0f} min"
        print(f"{args.config}: micro-batch {cfg.batch_size} x accum {cfg.grad_accum}: {per_step:.3f} s per step, "
              f"{r['samples_per_s']:.0f} samples/s" + (f", peak {r['peak_memory_gb']:.1f} GB" if "peak_memory_gb" in r else "")
              + f" -> {cfg.steps} steps ~ {span} (plus evaluation and sample sheets)")
        del model, stream, obj, opt
        if args.probe:
            report["probe"] = probe(cfg, device)
            for p in report["probe"]:
                print("  probe", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}))
            eff = cfg.batch_size * cfg.grad_accum
            best = suggest(report["probe"], eff, env.get("memory_gb")) if device.type == "cuda" else None
            if best:
                report["suggestion"] = best
                print(f"suggestion: batch_size={best['batch_size']} grad_accum={best['grad_accum']} "
                      f"(same effective batch {eff}, peak memory <= {MEMORY_FRACTION:.0%} of the GPU in eager "
                      "mode; torch.compile uses less)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
