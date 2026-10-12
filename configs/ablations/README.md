# Ablation configs (paper section "Ablations")

Each training ablation is a command-line override of a base config.  Run e.g.

    manitex train --config configs/train/pretrain_small.yaml out_dir=runs/abl_patch4 "model={preset: small, patch: 4}"

| ablation | override |
|---|---|
| patch size 4 (fewer tokens) | `model={preset: small, patch: 4}` |
| ELBO-weighted loss | `weighting=elbo` |
| linear masking schedule | `schedule=linear` |
| no classifier-free guidance training | `p_uncond=0.0 p_drop_field=0.0` |
| no horizontal flips | `flip_lr=false` |
| real data only (no procedural pretraining) | fine-tune config without `init_from` and without the procedural source |
| model size | `model=tiny` / `small` / `base` / `large` |

Sampling-time ablations need no retraining.  `scripts/ablate_sampler.py` compares sampler
settings on one checkpoint against a procedural "floor" (two disjoint reference sets).  Each
setting overrides the library defaults (ancestral order, 64 steps, guidance 1, top-p 0.95):

    python scripts/ablate_sampler.py --checkpoint runs/pretrain_base/final.pt --n 64 --size 64 \
        --settings "steps=32,top_p=1;steps=64,top_p=1;steps=64;strategy=confidence,noise=4.5,top_p=1"

| ablation | settings |
|---|---|
| unmasking order | `steps=64` (ancestral, default) vs `strategy=confidence,noise=4.5` (MaskGIT) |
| Gumbel noise for confidence order | `strategy=confidence,noise=1` vs `strategy=confidence,noise=4.5` |
| steps | `steps=16`, `steps=32`, `steps=64`, `steps=128` |
| nucleus filtering | `top_p=1`, `top_p=0.95`, `top_p=0.9` |
| guidance | `cfg=0`, `cfg=0.5`, `cfg=1`, `cfg=2` |

Without `--settings` the script runs a default grid covering these axes.
