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

Sampling-time ablations need no retraining.  `scripts/ablate_sampler.py` compares unmasking order,
steps and guidance on one checkpoint, against a procedural "floor" (two disjoint reference sets):

    python scripts/ablate_sampler.py --checkpoint runs/pretrain_base/final.pt --n 64 --size 64 \
        --settings random:16:1,random:32:1,random:64:1,random:32:0,confidence:16:1:1,confidence:16:1:4.5

| ablation | setting (`strategy:steps:cfg[:noise]`) |
|---|---|
| unmasking order | `random:32:1` (ancestral, default) vs `confidence:32:1:4.5` (MaskGIT) |
| Gumbel noise for confidence order | `confidence:16:1:1` vs `confidence:16:1:4.5` |
| steps | `random:16:1`, `random:32:1`, `random:64:1` |
| guidance | `random:32:0`, `random:32:0.5`, `random:32:1`, `random:32:2` |
