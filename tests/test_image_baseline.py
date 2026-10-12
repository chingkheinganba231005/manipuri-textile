import json
import string

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from manitex.baselines.image_diffusion import (  # noqa: E402
    ImageBaselineConfig,
    build_pipeline,
    design_stream,
    digitise_image,
    expected_image_size,
    lora_parameter_count,
    prompt_for,
    render_design,
    sample_designs,
    train_lora,
)
from manitex.procedural import generate  # noqa: E402


def tiny_components(tmp_path):
    """Randomly initialised miniature Stable Diffusion parts (nothing is downloaded)."""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for c in string.ascii_lowercase + string.digits + ",.-":
        vocab.setdefault(c, len(vocab))
        vocab.setdefault(c + "</w>", len(vocab))
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    torch.manual_seed(0)
    return {
        "tokenizer": CLIPTokenizer(str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"), model_max_length=16),
        "text_encoder": CLIPTextModel(CLIPTextConfig(bos_token_id=0, eos_token_id=1, pad_token_id=1, hidden_size=32,
                                                     intermediate_size=64, num_attention_heads=4, num_hidden_layers=2,
                                                     vocab_size=128, max_position_embeddings=16)),
        "vae": AutoencoderKL(block_out_channels=[8, 16], in_channels=3, out_channels=3, latent_channels=4,
                             down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2,
                             norm_num_groups=4),
        "unet": UNet2DConditionModel(block_out_channels=(8, 16), layers_per_block=1, sample_size=8, in_channels=4,
                                     out_channels=4, down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                                     up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=32,
                                     norm_num_groups=4),
        "scheduler": DDPMScheduler(num_train_timesteps=50),
    }


def test_render_prompt_and_stream(tmp_path):
    cfg = ImageBaselineConfig(cells=8, cell_px=2, styles=["check", "moirang_phee"])
    it = design_stream(cfg, 0)
    pairs = [next(it) for _ in range(6)]
    assert all(d.shape == (8, 8) and s in ("check", "moirang_phee") for d, s in pairs)
    assert render_design(pairs[0][0], 2).shape == (16, 16, 3)
    assert "moirang phee" in prompt_for("moirang_phee")
    assert expected_image_size(ImageBaselineConfig()) == 512
    with pytest.raises(ValueError):
        expected_image_size(ImageBaselineConfig(cells=10, cell_px=3))
    # real designs from shards, mixed with procedural ones
    from manitex.data import write_shard

    shard = write_shard(tmp_path / "s.npz", [generate("check", 20, 20, seed=i) for i in range(2)], styles=["leirum"] * 2)
    it = design_stream(ImageBaselineConfig(cells=8, cell_px=2, shards=[str(shard)], procedural_weight=0.5), 1)
    styles = {next(it)[1] for _ in range(30)}
    assert "leirum" in styles and len(styles) > 1


def test_clean_render_digitises_exactly():
    d = generate("check", 16, 16, seed=3).canonical()
    back = digitise_image(render_design(d, 8), 16)
    assert back.shape == (16, 16) and (back.grid == d.grid).all()


def test_lora_training_sampling_and_digitising(tmp_path):
    from safetensors.torch import load_file

    comp = tiny_components(tmp_path)
    cfg = ImageBaselineConfig(model="unused", out_dir=str(tmp_path / "run"), cells=8, cell_px=2, steps=3, batch_size=2,
                              rank=4, log_every=1, save_every=3, precision="fp32", styles=["check"])
    lora = train_lora(cfg, components=comp, device="cpu")
    assert lora_parameter_count(comp["unet"]) > 0
    weights = load_file(str(lora / "pytorch_lora_weights.safetensors"))
    up = [v for k, v in weights.items() if "lora.up" in k or "lora_B" in k]
    assert up and any(float(v.abs().sum()) > 0 for v in up)  # adapters moved away from their zero init
    assert len((tmp_path / "run" / "log.jsonl").read_text().splitlines()) == 3
    pipe = build_pipeline("unused", lora, components=tiny_components(tmp_path), device="cpu")
    images: list = []
    designs = sample_designs(pipe, "check", n=3, cells=8, cell_px=2, steps=2, batch=2, keep_images=images)
    assert len(designs) == 3 and all(d.shape == (8, 8) for d in designs)
    assert images[0].shape == (16, 16, 3) and images[0].dtype == np.uint8
    assert all(d.meta["generator"] == "image-ldm-lora" and d.meta["style"] == "check" for d in designs)
