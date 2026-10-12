# Training guide (single A100 80 GB)

This guide takes you from a fresh machine to a trained, exported model and a running demo.
Timings are rough estimates; measure on your machine with the first few hundred steps.

## 1. Environment

```bash
nvidia-smi                                   # check the GPU and the CUDA driver
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[all]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
pytest -q                                    # everything should pass
```

## 2. Sanity run (5 minutes)

```bash
manitex train --config configs/train/debug_cpu.yaml precision=bf16 num_workers=8
```

Look at `runs/debug_cpu/log.jsonl` (loss falls, `eval_bits_per_cell_ema` falls) and
`runs/debug_cpu/samples/*.png` (one row per style).

## 3. Pretraining on the procedural grammar

The procedural stream is generated on the fly by the DataLoader workers (no dataset to
download). Two presets:

| config | params | tokens / sample | batch | steps | rough A100 time |
|---|---|---|---|---|---|
| `configs/train/pretrain_small.yaml` | ~23 M | 1024 (64×64 cells, patch 2) | 256 | 50 k | 2–4 h |
| `configs/train/pretrain_base.yaml` | ~100 M | 1024 | 256 | 150 k | 12–24 h |

```bash
manitex train --config configs/train/pretrain_small.yaml
# or, for the main model
manitex train --config configs/train/pretrain_base.yaml
```

* Any key can be overridden: `manitex train --config ... steps=20000 batch_size=128`.
* Runs **auto-resume** from `<out_dir>/last.pt` (saved every `save_every` steps).
* `wandb=true` logs to Weights & Biases (if installed); otherwise use `log.jsonl`.
* If the GPU is under-utilised, the CPU workers cannot keep up: increase `num_workers` (≈ number
  of CPU cores) or pre-generate shards and train from them:

  ```bash
  manitex data build-synthetic --out data/synthetic --n 1000000 --min-size 64 --max-size 160
  manitex train --config configs/train/pretrain_base.yaml \
      "data=[{type: shards, paths: [data/synthetic], domain: synthetic}]"
  ```
* Out of memory: halve `batch_size` and set `grad_accum=2` (same effective batch).
* `torch.compile` problems: set `compile=false` (slower, otherwise identical).
* Multi-GPU: `torchrun --nproc_per_node=4 -m manitex.train --config ...` (batch size is per GPU).

What to watch: `masked_acc` should exceed 0.9 and `eval_bits_per_cell_ema` should keep falling
(the procedural data has low entropy, so values well below 0.2 bits/cell are normal). Sample
sheets should show clean stripes for phanek, framed temple rows for Moirang phee, and exact
plaids for checks.

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

The fine-tuning mix keeps 20 % procedural data to avoid forgetting the structural prior. Real
designs are tagged with the `real` domain, so at generation time `--domain real` asks for
real-textile statistics and `--domain synthetic` for grammar-like ones. Checkpoints record the
domains they were trained on, and the default `--domain auto` picks `real` when the model has seen
real data and the unconditional (null) domain otherwise.

## 6. Evaluation

```bash
manitex eval --checkpoint runs/finetune_real/final.pt --style moirang_phee --n 256 \
             --reference data/real/val.npz > results/moirang_eval.json
```

Reports pattern-JS (2×2, 3×3), run-length distances, colour-count TV, symmetry statistics,
diversity, novelty / memorisation and weavability (tie-down density, dobby shafts, float limits).
The held-out NLL bound (`eval_bits_per_cell_ema`) is logged during training. See
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the full protocol.

## 7. Export and publish

```bash
manitex export runs/finetune_real/final.pt --out exports/manitex-v1        # fp16 safetensors + config.json
huggingface-cli login
huggingface-cli upload <user>/manitex-v1 exports/manitex-v1 .            # model repo
```

Load anywhere with `TextileGenerator.from_pretrained("hf://<user>/manitex-v1")`.

## 8. Demo

```bash
manitex app --checkpoint exports/manitex-v1                               # local
python scripts/build_space.py --out space_build --model exports/manitex-v1  # Hugging Face Space
```

On the Space you can instead set the variable `MANITEX_CHECKPOINT=hf://<user>/manitex-v1`.
