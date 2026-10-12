# Training guide (one 80–96 GB GPU)

This guide takes you from a fresh machine to a trained, exported model and a running demo on a
single GPU with 80–96 GB of memory: an A100 80 GB, an H100, or the RTX PRO 6000 Blackwell (96 GB)
of Google Cloud G4 machines. Time estimates below assume an A100; measure your own GPU with
`scripts/benchmark_gpu.py` (step 2) before starting long runs.

## 1. Environment

```bash
nvidia-smi                       # GPU, driver, and the newest CUDA version the driver supports
python -m venv .venv && source .venv/bin/activate
pip install torch                # Linux: the default wheel ships CUDA 13 (driver must support CUDA 13)
# if nvidia-smi shows CUDA 12.x, use a CUDA 12.8 build instead (the oldest that supports Blackwell):
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[all]"
pytest -q                        # everything should pass
```

Blackwell GPUs (RTX PRO 6000, B200, RTX 50xx) need a PyTorch build with CUDA 12.8 or newer; the
older `cu124` / `cu126` builds have no kernels for them. The A100 and H100 work with any recent
build. PyTorch 2.4 or newer is required.

## 2. Check the GPU and measure speed (10 minutes)

```bash
python scripts/benchmark_gpu.py
python scripts/benchmark_gpu.py --config configs/train/pretrain_small.yaml --steps 30 --probe
manitex train --config configs/train/debug_cpu.yaml precision=bf16 num_workers=8   # 5-minute sanity run
```

The first command checks that the installed PyTorch has kernels for your GPU (it warns
otherwise) and measures the achieved bf16 matrix-multiply throughput. The second runs real
training steps of a config and prints seconds per step, peak memory and the projected time for
the whole run. `--probe` tries larger micro-batches and suggests `batch_size` / `grad_accum` for
the same effective batch: a 96 GB card usually fits larger ones than the defaults, which are
sized for 80 GB. For the sanity run, look at `runs/debug_cpu/log.jsonl` (loss and
`eval_bits_per_cell_ema` fall) and `runs/debug_cpu/samples/*.png` (one row per style).

## 3. Pretraining on the procedural grammar

The procedural stream is generated on the fly by the DataLoader workers (no dataset to
download). Two presets:

| config | params | tokens / sample | batch (micro × accum) | steps | compute / step | A100 estimate |
|---|---|---|---|---|---|---|
| `configs/train/pretrain_small.yaml` | 32 M | 1024 (64×64 cells, patch 2) | 256 (128 × 2) | 50 k | 48 TFLOP | 4–9 h |
| `configs/train/pretrain_base.yaml` | 129 M | 1024 | 256 (64 × 4) | 100 k | 164 TFLOP | 1–2.5 days |

Compute per step is counted exactly (forward + backward including attention, with
`torch.utils.flop_counter` and the math attention backend); the times assume 25–55 % of the
A100's 312 TFLOP/s bf16 peak. For other GPUs use the measured projection of
`scripts/benchmark_gpu.py` rather than scaling by peak figures (the RTX PRO 6000's datasheet, for
example, quotes only an FP4 peak). During training, `samples_per_s` in `log.jsonl` gives the real
speed.

Memory: the activations kept for the backward pass measure ≈ 322 MB per 64×64 sample for `small`
and ≈ 643 MB for `base` (bf16), so a full batch of 256 would need ≈ 82 GB or ≈ 165 GB. The configs
split it into micro-batches (`batch_size`) with gradient accumulation (`grad_accum`), staying near
41 GB, which fits 80 GB cards; on a 96 GB card `--probe` (step 2) shows how far `batch_size` can go
up (with `grad_accum` lowered to keep the effective batch).

Run `pretrain_small` first: if its `eval_bits_per_cell_ema` flattens long before the end, shorten
the base run (`steps=...`); the learning-rate schedule follows `steps`, so set it before starting.

```bash
manitex train --config configs/train/pretrain_small.yaml
# or, for the main model
manitex train --config configs/train/pretrain_base.yaml
```

* Any key can be overridden: `manitex train --config ... steps=20000 eval_every=1000`.
* Runs **auto-resume** from `<out_dir>/last.pt` (saved every `save_every` steps).
* `wandb=true` logs to Weights & Biases (if installed); otherwise use `log.jsonl`.
* If the GPU is under-utilised, the CPU workers cannot keep up: increase `num_workers` (≈ number
  of CPU cores) or pre-generate shards and train from them:

  ```bash
  manitex data build-synthetic --out data/synthetic --n 1000000 --min-size 64 --max-size 160
  manitex train --config configs/train/pretrain_base.yaml \
      "data=[{type: shards, paths: [data/synthetic], domain: synthetic}]"
  ```
* Out of memory: halve `batch_size` and double `grad_accum` (same effective batch). With spare
  memory, do the opposite for slightly higher throughput.
* `torch.compile` problems: set `compile=false` (slower, otherwise identical).
* Multi-GPU: `torchrun --nproc_per_node=4 -m manitex.train --config ...` (batch size is per GPU).

What to watch: `eval_bits_per_cell_ema` should keep falling (the procedural data has low entropy:
a 4 M-parameter CPU sanity model reached 0.13 bits/cell after 3.5 k steps of batch 32) and
`masked_acc` should rise (it stays below 1 because heavily masked inputs are genuinely ambiguous). Sample
sheets should show clean stripes for phanek, framed temple rows for Moirang phee, and exact
plaids for checks.

### Autoregressive baseline (experiment E1)

```bash
manitex train --config configs/train/baseline_ar_small.yaml
```

The same `small` backbone with causal attention over single cells in raster order. It sees the
same data as `pretrain_small`, but its 4096-token sequences cost about 5× the compute (≈ 990 vs
189 GFLOP per sample with causal attention kernels) and ≈ 1.5 GB of activations per sample, hence
micro-batches of 32 × 8: roughly 20–45 h on an A100. Its `eval_bits_per_cell_ema` is an exact NLL; the masked model's is an
upper bound. Sampling takes one step per cell (with a key/value cache).

### Image-space baseline (experiment E1)

```bash
pip install -e ".[image-baseline]"          # diffusers, transformers, peft
python scripts/image_baseline.py train --shards data/real/train.npz --procedural-weight 0.2 \
    --steps 5000 --out runs/image_baseline
python scripts/image_baseline.py sample --lora runs/image_baseline/lora --n 256 \
    --out results/image_baseline
python scripts/eval_generators.py --designs image=results/image_baseline/designs.npz ...
```

LoRA adapters of an open latent diffusion model (default
`stable-diffusion-v1-5/stable-diffusion-v1-5`, downloaded from the Hugging Face Hub; check its
licence) are fine-tuned on designs rendered at 8 px per cell; generated 512 px images are
digitised back into 64×64-cell designs. Use the same training designs as the grid models (with
`--shards` only, or with procedural designs mixed in) for a fair comparison.

## 4. Real data

Follow [DATA_COLLECTION.md](DATA_COLLECTION.md) to photograph textiles with consent, then:

```bash
# digitise each photo (inspect the *_grid.png overlay; re-run with --cells-x/--cells-y if needed)
manitex digitise raw/moirang_001.jpg --corners ... --cells-x 120 --style moirang_phee --out data/designs/moirang_phee
# curate: delete bad digitisations, fix small errors in any pixel editor (indexed PNG keeps the palette)
# split by *textile* (never put crops of the same piece in train and val)
manitex data import data/designs_train --out data/real/train.npz
manitex data import data/designs_val   --out data/real/val.npz
```

The style label comes from `--style`, from each design's metadata, or from its folder name.
New style names (e.g. `leirum_phee`) are added to the model's style vocabulary automatically.

## 5. Fine-tuning

```bash
manitex train --config configs/train/finetune_real.yaml init_from=runs/pretrain_base/final.pt
```

For experiment E1, fine-tune models of equal size the same way, e.g. the two `small` models
(the autoregressive one needs smaller micro-batches):

```bash
manitex train --config configs/train/finetune_real.yaml init_from=runs/pretrain_small/final.pt \
    out_dir=runs/finetune_real_small
manitex train --config configs/train/finetune_real.yaml init_from=runs/baseline_ar_small/final.pt \
    out_dir=runs/finetune_real_ar batch_size=16 grad_accum=8
```

For experiment E3, fine-tune on the nested subsets written by `scripts/make_splits.py`, e.g.

```bash
manitex train --config configs/train/finetune_real.yaml out_dir=runs/e3_25 \
    "data=[{type: shards, paths: [data/real/train_25.npz], val_paths: [data/real/val.npz], domain: real, weight: 0.8}, {type: procedural, weight: 0.2}]"
```

and add `init_from=null model=base` for the curve without pretraining (same architecture, random
initialisation).

The fine-tuning mix keeps 20 % procedural data to avoid forgetting the structural prior. Real
designs are tagged with the `real` domain, so at generation time `--domain real` asks for
real-textile statistics and `--domain synthetic` for grammar-like ones. Checkpoints record the
domains they were trained on, and the default `--domain auto` picks `real` when the model has seen
real data and the unconditional (null) domain otherwise.

## 6. Evaluation

```bash
# metric suite for one model and style
manitex eval --checkpoint runs/finetune_real/final.pt --style moirang_phee --n 256 \
             --reference data/real/val.npz --train data/real/train.npz > results/moirang_eval.json
# E1: generators side by side for every style (reference floor, grammar, retrieval, models)
python scripts/eval_generators.py --model mdm=runs/finetune_real_small/final.pt \
    --model ar=runs/finetune_real_ar/final.pt --reference data/real/val.npz \
    --train data/real/train.npz --n 256 --out results/e1
# E2: orbit pooling vs post-hoc projection for mirror / repeat / stripe constraints
python scripts/eval_constraints.py --checkpoint runs/finetune_real/final.pt --n 128 --out results/e2
```

Metrics: pattern-JS (2×2, 3×3), run-length distances, colour-count TV, symmetry statistics,
diversity, novelty / memorisation (with `--train`: structural distance to the closest window of
any training design, over all translations and flips; below 0.02 counts as memorised) and
weavability (tie-down density, dobby shafts, float limits). The scripts add the models' NLL of
the reference designs (bits/cell) and write JSON and Markdown tables. The held-out NLL
(`eval_bits_per_cell_ema`) is also logged during training. Sampler settings
(order, steps, guidance, top-p) are compared on a finished checkpoint with
`scripts/ablate_sampler.py` (see [configs/ablations/README.md](../configs/ablations/README.md));
re-tune the defaults (64 steps, top-p 0.95, guidance 1) on your final model. See
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the full protocol.

## 7. Export and publish

```bash
manitex export runs/finetune_real/final.pt --out exports/manitex-v1        # fp16 safetensors + config.json
hf auth login
hf upload <user>/manitex-v1 exports/manitex-v1 .                         # model repo
```

Load anywhere with `TextileGenerator.from_pretrained("hf://<user>/manitex-v1")`.

## 8. Demo

```bash
manitex app --checkpoint exports/manitex-v1                               # local
python scripts/build_space.py --out space_build --model exports/manitex-v1 --hardware zerogpu
hf repos create <user>/manitex --type space --sdk gradio
hf upload <user>/manitex space_build . --type space    # then select ZeroGPU in the Space settings
```

`--hardware` must match the Space hardware: `zerogpu` (free for two Spaces per personal account;
the app borrows half of an RTX PRO 6000 Blackwell while it samples), `cpu` or `gpu` (paid T4 to
A100); see the README. On the Space you can instead set the variable
`MANITEX_CHECKPOINT=hf://<user>/manitex-v1` (and build without `--model`). Sampling speed: a 64×64
design with the `small` model takes ≈ 30 s on two CPU threads (32 steps, guidance); the app
detects a CPU-only machine and starts with lighter settings.
