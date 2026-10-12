# Ablation configs (paper section "Ablations")

Each file overrides the base config.  Run e.g.

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
