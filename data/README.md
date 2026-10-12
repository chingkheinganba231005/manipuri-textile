# data/ (not tracked by git)

Everything in this folder except this file is ignored by git: raw photographs and community
designs must only be shared according to the consent obtained (see docs/DATA_COLLECTION.md).

Suggested layout:

```
data/
  raw/<piece_id>/          photographs + meta.json (consent, provenance)
  designs/<split>/<style>/ digitised, curated design files (.png / .json)
  real/                    train.npz, val.npz, test.npz  (manitex data import ...)
  synthetic/               optional pre-generated procedural shards (manitex data build-synthetic ...)
```
