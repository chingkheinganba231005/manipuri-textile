"""Image-space baseline (experiment E1): a latent diffusion model fine-tuned with LoRA on rendered
designs, whose samples are turned back into design grids by the digitiser.

This is how a general image generator would be used for textile design. Comparing it with the
grid-native model shows what is lost when cells, colours and structure are only implicit in
pixels. A ``cells x cells`` crop is rendered as flat colour blocks of ``cell_px`` pixels
(64 x 8 = 512 px, the native size of Stable Diffusion 1.x), and generated images are digitised
with the cell count known, so the comparison does not depend on grid detection.

Requires the optional dependencies ``diffusers``, ``transformers`` and ``peft``
(``pip install -e ".[image-baseline]"``) and, for real use, a GPU and an open Stable Diffusion
checkpoint, e.g. ``stable-diffusion-v1-5/stable-diffusion-v1-5`` (check its licence before
publishing results or weights).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from manitex.design import Design
from manitex.procedural import PROCEDURAL_STYLES, generate

LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def prompt_for(style: str) -> str:
    """Text prompt used for a style during fine-tuning and sampling."""
    return f"{style.replace('_', ' ')} handloom textile design, flat colours on a square grid of cells"


def render_design(design: Design, cell_px: int) -> np.ndarray:
    """Flat-colour rendering, ``cell_px`` pixels per cell (uint8, H x W x 3)."""
    return design.to_rgb(cell=cell_px)


@dataclass
class ImageBaselineConfig:
    model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    out_dir: str = "runs/image_baseline"
    cells: int = 64  # design cells per side of a training image
    cell_px: int = 8  # pixels per cell (64 x 8 = 512 px)
    shards: list[str] = field(default_factory=list)  # real designs (manitex data import); empty = procedural only
    procedural_weight: float = 0.0  # fraction of procedural designs when shards are given
    styles: list[str] | None = None  # restrict to these styles (default: all in the data)
    steps: int = 5000
    batch_size: int = 8
    lr: float = 1e-4
    rank: int = 16  # LoRA rank
    p_uncond: float = 0.1  # empty-prompt dropout, keeps classifier-free guidance working
    precision: str = "bf16"  # 'bf16' | 'fp16' | 'fp32' (CUDA only; CPU runs in fp32)
    log_every: int = 50
    save_every: int = 1000
    seed: int = 0


def design_stream(cfg: ImageBaselineConfig, seed: int) -> Iterator[tuple[Design, str]]:
    """Endless (design crop, style) pairs from the configured data."""
    from manitex.experiments import crop_design

    rng = np.random.default_rng(seed)
    pool: list[tuple[Any, int]] = []
    if cfg.shards:
        from manitex.data import read_shards

        for sh in read_shards(cfg.shards):
            pool += [(sh, j) for j in range(len(sh)) if cfg.styles is None or str(sh.styles[j]) in cfg.styles]
        if not pool:
            raise ValueError("no designs of the requested styles in the shards")
    proc_styles = [s for s in (cfg.styles or PROCEDURAL_STYLES) if s in PROCEDURAL_STYLES]
    p_proc = 1.0 if not pool else cfg.procedural_weight
    if p_proc > 0 and not proc_styles:
        raise ValueError("no procedural styles to sample from")
    while True:
        if rng.random() < p_proc:
            style = proc_styles[int(rng.integers(len(proc_styles)))]
            size = int(round(cfg.cells * rng.uniform(1.0, 2.0)))
            yield crop_design(generate(style, size, size, rng=rng), cfg.cells, rng), style
        else:
            sh, j = pool[int(rng.integers(len(pool)))]
            yield crop_design(sh.design(j), cfg.cells, rng), str(sh.styles[j])


def load_components(model: str) -> dict[str, Any]:
    """tokenizer, text_encoder, vae, unet and the training noise scheduler of a pretrained model."""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    return {
        "tokenizer": CLIPTokenizer.from_pretrained(model, subfolder="tokenizer"),
        "text_encoder": CLIPTextModel.from_pretrained(model, subfolder="text_encoder"),
        "vae": AutoencoderKL.from_pretrained(model, subfolder="vae"),
        "unet": UNet2DConditionModel.from_pretrained(model, subfolder="unet"),
        "scheduler": DDPMScheduler.from_pretrained(model, subfolder="scheduler"),
    }


def _encode_prompts(components: dict[str, Any], prompts: Sequence[str], device) -> Any:
    import torch

    tok = components["tokenizer"]
    ids = tok(list(prompts), padding="max_length", max_length=tok.model_max_length, truncation=True,
              return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        return components["text_encoder"](ids)[0]


def train_lora(cfg: ImageBaselineConfig, components: dict[str, Any] | None = None, device: str | None = None) -> Path:
    """Fine-tune LoRA adapters on the UNet's attention projections; returns the LoRA directory."""
    import torch
    import torch.nn.functional as F
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    expected_image_size(cfg)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    comp = components or load_components(cfg.model)
    vae, text_encoder, unet, sched = comp["vae"], comp["text_encoder"], comp["unet"], comp["scheduler"]
    for m in (vae, text_encoder, unet):
        m.requires_grad_(False)
        m.to(dev)
    vae.eval()
    text_encoder.eval()
    unet.add_adapter(LoraConfig(r=cfg.rank, lora_alpha=cfg.rank, init_lora_weights="gaussian",
                                target_modules=list(LORA_TARGETS)))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # adapters train in fp32
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-2)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.precision)
    use_amp = amp is not None and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp == torch.float16)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    emb_cache: dict[str, Any] = {"": _encode_prompts(comp, [""], dev)}
    stream = design_stream(cfg, cfg.seed)
    scaling = getattr(vae.config, "scaling_factor", 0.18215)
    lora_dir = out / "lora"
    log = open(out / "log.jsonl", "a")
    t0 = time.time()

    def save() -> None:
        layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(str(lora_dir), unet_lora_layers=layers, safe_serialization=True)

    unet.train()
    for step in range(1, cfg.steps + 1):
        items = [next(stream) for _ in range(cfg.batch_size)]
        pix = np.stack([render_design(d, cfg.cell_px) for d, _ in items]).astype(np.float32) / 127.5 - 1.0
        pixels = torch.from_numpy(pix).permute(0, 3, 1, 2).to(dev)
        prompts = ["" if np.random.default_rng([cfg.seed, step, i]).random() < cfg.p_uncond else prompt_for(s)
                   for i, (_, s) in enumerate(items)]
        for pr in set(prompts) - set(emb_cache):
            emb_cache[pr] = _encode_prompts(comp, [pr], dev)
        cond = torch.cat([emb_cache[pr] for pr in prompts])
        with torch.no_grad():
            latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample() * scaling
        noise = torch.randn_like(latents)
        t = torch.randint(0, sched.config.num_train_timesteps, (latents.shape[0],), device=dev).long()
        noisy = sched.add_noise(latents, noise, t)
        if sched.config.prediction_type == "v_prediction":
            target = sched.get_velocity(latents, noise, t)
        else:
            target = noise
        with torch.autocast(device_type=dev.type, dtype=amp or torch.float32, enabled=use_amp):
            pred = unet(noisy, t, cond).sample
        loss = F.mse_loss(pred.float(), target.float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % cfg.log_every == 0 or step == 1:
            rec = {"step": step, "loss": float(loss), "samples_per_s": cfg.batch_size * (1 if step == 1 else
                   cfg.log_every) / max(time.time() - t0, 1e-9)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            t0 = time.time()
        if step % cfg.save_every == 0 or step == cfg.steps:
            save()
    log.close()
    return lora_dir


def build_pipeline(model: str, lora_dir: str | Path | None, components: dict[str, Any] | None = None,
                   device: str | None = None):
    """A Stable Diffusion pipeline with the fine-tuned LoRA weights loaded."""
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if components is not None:
        pipe = StableDiffusionPipeline(vae=components["vae"], text_encoder=components["text_encoder"],
                                       tokenizer=components["tokenizer"], unet=components["unet"],
                                       scheduler=components["scheduler"], safety_checker=None,
                                       feature_extractor=None, requires_safety_checker=False)
    else:
        dtype = torch.float16 if dev == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype, safety_checker=None,
                                                       requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if lora_dir is not None:
        pipe.load_lora_weights(str(lora_dir))
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(dev)


def digitise_image(image: np.ndarray, cells: int, regularise: bool = False, max_colors: int = 16) -> Design:
    """Generated image -> design grid with the known cell count (no structural clean-up unless
    ``regularise``, so the grid reflects what the image model produced)."""
    from manitex.digitise import DigitiseOptions, digitise

    opt = DigitiseOptions(cells_x=cells, cells_y=cells, min_cell_px=1.0, k_max=max_colors,
                          stripes=regularise, repeats=regularise, mirror=regularise)
    return digitise(image, opt).design.canonical()


def sample_designs(pipe, style: str, n: int, cells: int, cell_px: int, steps: int = 30, guidance: float = 5.0,
                   batch: int = 8, seed: int = 0, regularise: bool = False,
                   keep_images: list | None = None) -> list[Design]:
    """Sample ``n`` images for ``style`` and digitise them into ``cells x cells`` designs."""
    import torch

    size = cells * cell_px
    designs: list[Design] = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        gen = torch.Generator(device=pipe.device).manual_seed(seed + i)
        images = pipe(prompt=[prompt_for(style)] * b, num_inference_steps=steps, guidance_scale=guidance,
                      height=size, width=size, generator=gen, output_type="np").images
        for img in images:
            u8 = np.clip(np.round(np.asarray(img) * 255.0), 0, 255).astype(np.uint8)
            if keep_images is not None:
                keep_images.append(u8)
            d = digitise_image(u8, cells, regularise=regularise)
            d.meta.update({"style": style, "generator": "image-ldm-lora"})
            designs.append(d)
    return designs


def lora_parameter_count(unet) -> int:
    return sum(p.numel() for n, p in unet.named_parameters() if "lora" in n)


def expected_image_size(cfg: ImageBaselineConfig) -> int:
    size = cfg.cells * cfg.cell_px
    if size % 8:
        raise ValueError("cells * cell_px must be divisible by 8 (latent diffusion downsampling)")
    return size


__all__ = ["ImageBaselineConfig", "build_pipeline", "design_stream", "digitise_image", "expected_image_size",
           "load_components", "lora_parameter_count", "prompt_for", "render_design", "sample_designs", "train_lora"]
