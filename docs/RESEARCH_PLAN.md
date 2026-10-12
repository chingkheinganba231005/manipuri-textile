# Research plan (target: *The Visual Computer*, Springer)

## Working titles

* *From Photograph to Loom: Structure-Constrained Generation and Digitisation of Manipuri Textile Patterns*
* *Weaving the Grid: Discrete Diffusion with Exact Structural Constraints for Handloom Design*

## Contributions

1. **Dataset** — the first curated set of digitised Manipuri textile designs (phanek, Moirang phee,
   …) with provenance metadata, collected with informed consent, plus the collection protocol.
2. **Grid-native generator** — masked discrete diffusion on *design cells* (the units a weaver
   actually weaves) with a palette-agnostic canonical representation, style / colour-count /
   domain conditioning, and **exact** structural control (mirror, repeat, stripes) via *orbit
   pooling* at sampling time; inpainting and outpainting for free.
3. **Design-to-loom compiler** — supplementary-weft brocade on single-colour, striped or plaid
   grounds (plaids inferred from optical-mixture colours), float-bounded tie-down placement,
   row balancing, WIF export, pick-up instructions, yarn estimates and dobby-loom analysis.
4. **Digitisation** — photo → grid with lattice fitting, robust cell colours, Potts denoising and
   orbit-pooled repeat/symmetry regularisation; a synthetic-photo benchmark with exact ground
   truth.
5. **Evaluation** — palette-independent pattern statistics, weavability metrics, memorisation
   checks, a study with Manipuri weavers, and a physical weaving case study.

## Research questions

* **RQ1 Fidelity.** Does a grid-native generator reproduce the structure of real Manipuri textiles
  better than the procedural grammar and than image-space generators followed by digitisation?
* **RQ2 Control.** Does exact orbit pooling keep sample quality while guaranteeing constraints,
  compared with unconstrained sampling + post-hoc projection?
* **RQ3 Data efficiency.** How much real data is needed when pretraining on the grammar?
* **RQ4 Weavability & acceptance.** Are generated designs efficient to weave and judged authentic
  and useful by weavers?
* **RQ5 Digitisation.** How accurately can casual photos be turned into design grids?

## Method notes (for the paper)

* **Canonical representation.** Colours are renumbered by decreasing frequency; the model predicts
  colour *slots*, the palette is assigned afterwards by frequency-rank transfer from real palettes.
* **Forward process.** Each cell is masked independently with probability `r(t)`; training uses
  the cosine schedule `r(t) = sin(πt/2)` and the mean cross-entropy over masked cells.
* **Likelihood.** With the linear schedule, `NELBO = E_t[(1/t) Σ_{masked i} −log p(x_i | x_t)]`,
  reported in bits per cell on held-out data (valid for any trained denoiser).
* **Orbit pooling.** A symmetry group action partitions cells into orbits `O`. At each step the
  pooled distribution of an orbit is `q_O(c) ∝ exp( (1/|O|) Σ_{i∈O} log p_i(c) )` (mean pooling;
  `sum` gives the product-of-experts conditional of the factorised proposal given equality). One
  colour is sampled per orbit and the orbit is unmasked as a unit, so every sample satisfies the
  constraints by construction.
* **Sampling order.** Default: ancestral (random-order) unmasking, each masked orbit revealed with
  probability `(r_t − r_s)/r_t`, 64 steps, guidance 1 and top-p 0.95 on each orbit's colour
  distribution. MaskGIT confidence ordering is kept for ablations: with little Gumbel noise it
  collapsed sparse-motif styles to plain ground in the CPU sanity run (75 % of Moirang phee samples
  single-coloured vs 0 % for ancestral sampling). On that run's final checkpoint, 64 instead of 32
  steps fixed most of the structure error (3×3 pattern-JS 0.150 → 0.100) and top-p 0.95 halved the
  colour-count error (0.53 → 0.28, −10 % diversity). Re-tune on the final models with
  `scripts/ablate_sampler.py`.
* **Tie-downs.** For a visible run `[a, b)` longer than `F`, ties are placed at
  `{x ∈ (a, b−1) : (x − φ) mod F = 0}` with a brick-staggered phase `φ`; any residual gap > `F` is
  split at its midpoint. Every remaining float is ≤ `F` and no motif edge is hidden.

## Experiments

| ID | Question | Setup | Measures |
|---|---|---|---|
| E1 | RQ1 | Train on real train split (with/without grammar pretraining); compare with grammar, retrieval of training crops, AR transformer baseline, image-space LDM + digitiser (`scripts/eval_generators.py`) | pattern-JS 2×2/3×3, run-length W1 (rows/cols), colour-count TV, symmetry stats, NLL of held-out designs (bits/cell), diversity, novelty, weaving cost |
| E2 | RQ2 | Same model: unconstrained / post-hoc projection / orbit pooling (mean, sum) for mirror, repeat, stripes (`scripts/eval_constraints.py`) | constraint satisfaction, NLL of samples under the model, pattern-JS, diversity, expert preference |
| E3 | RQ3 | Fine-tune with 0/10/25/50/100 % of the real data, with and without pretraining | NLL and pattern-JS vs data fraction |
| E4 | RQ4 | Compile all samples | tie-downs / 100 cells, max floats, dobby shafts, pattern-pick fraction; weaver ratings of booklets |
| E5 | RQ4 | Blind study with 8–15 weavers/designers: real vs procedural vs generated (Likert authenticity, pairwise preference, "would you weave it?") | ratings, preference rates, qualitative comments |
| E6 | RQ4 | Weave 1–3 generated borders from the booklet on a real loom; photograph and digitise the result | design-vs-woven agreement, time, weaver notes |
| E7 | RQ5 | `scripts/benchmark_digitise.py` (synthetic) + a real subset with manually corrected ground truth | cell accuracy (assisted), pixel accuracy (automatic), colour ΔE00, exact-grid rate |

**Ablations:** patch size (2 vs 4), loss weighting (mean vs ELBO), schedule (cosine vs linear),
condition dropout, domain tag on/off, flips, model size (tiny → large); at sampling time, unmasking
order (ancestral vs MaskGIT confidence), steps and guidance (`scripts/ablate_sampler.py`).
See `configs/ablations/README.md`.

**Baselines** (all implemented): the grammar; retrieval of training crops; an autoregressive
raster-order transformer on cells (same backbone with causal attention, patch 1, exact likelihood,
KV-cached sampling; `configs/train/baseline_ar_small.yaml`, ~5× the compute of the masked model
per sample at equal data); and the image-space pipeline (`scripts/image_baseline.py`): LoRA
fine-tuning of an open latent diffusion model on designs rendered at 8 px per cell (64 cells =
512 px), sampling, then digitisation with the cell count known and no structural clean-up, so
the grids reflect what the image model produced.

**Likelihoods.** The masked model reports an upper bound (ELBO) on the NLL, the autoregressive
model the exact NLL, so a lower AR number can partly reflect the gap of the bound; report both and
say so.

**Novelty.** `metrics.novelty` is the fraction of neighbouring-cell pairs whose colour-change
status differs from the closest window of any training design, over all translations and flips
(palette-independent, exact). Retrieval of training crops scores ≈ 0; designs below 0.02 count
as memorised.

## Current results (synthetic, for orientation only)

Digitisation benchmark (`python scripts/benchmark_digitise.py --seeds 6`, 30 synthetic photos with
perspective, blur, uneven light, noise and JPEG; 40×48-cell designs):

| style | assisted cell acc. | automatic pixel acc. | clean image, automatic | clean image, assisted exact |
|---|---|---|---|---|
| phanek mayek naibi | 1.000 | 0.965 | 1.000 | 100 % |
| Moirang phee | 0.999 | 0.960 | 1.000 | 100 % |
| banded motif | 0.946 | 0.962 | 1.000 | 100 % |
| check | 1.000 | 0.873 | 0.990 | 100 % |
| motif tile | 0.994 | 0.957 | 0.987 | 100 % |
| **overall** | **0.988** | **0.943** | **0.995** | **100 %** |

(Automatic mode also recovers the exact grid size in 20 % of cases; otherwise it returns an
equivalent finer or coarser grid, which is why pixel accuracy is the relevant measure. Mean time
≈ 2 s per photo on one CPU core.)

Known weakness: automatic grid fitting can lock onto yarn-level structure when ground picks are
clearly visible between pattern picks (it then returns a finer, still faithful grid). A
*texton-based* grid inference (the coarsest grid whose cells share a repeating internal texture)
is planned; the assisted mode (cells across/down given) is exact on clean inputs.

Model sanity check (CPU, 4.1 M parameters, 32×32 crops, batch 32, procedural data only): the
held-out NLL bound (EMA weights) was 0.294 bits/cell after 500 steps, 0.218 after 1 k and 0.131
after 4 k. Samples show recognisable stripes, temple rows and plaids, with stray cells and extra
colours expected from such a small, short run; exact symmetry constraints hold. Real results come
from the A100 runs (`configs/train/*.yaml`).

## Figures

1. Teaser: photo → digitised grid → generated variations → weaving booklet → woven sample.
2. Pipeline overview. 3. Orbit pooling illustration. 4. Samples per style and per colour budget.
5. Structural control (mirror / repeat / stripes) and inpainting. 6. Compiler: draft, tie-downs,
render. 7. Digitisation examples and failure cases. 8. User-study results.
(`scripts/make_figures.py` produces the procedural/compiler/render panels.)

## Timeline (example, 16 weeks)

| weeks | work |
|---|---|
| 1–2 | Pretrain small and base models on the A100; ablations on procedural data |
| 2–6 | Ethics/consent, photography, digitisation and curation of real textiles |
| 6–8 | Fine-tuning, baselines, metric evaluation (E1–E4, E7) |
| 8–10 | Weaver study (E5) and physical weaving (E6) |
| 10–14 | Writing, figures, supplementary material, public demo + model release |

## Venue notes

Check the current *The Visual Computer* author guidelines (scope, length, the Springer Nature
LaTeX template) before formatting. Related venues if needed: ACM Journal on Computing and Cultural
Heritage, Computers & Graphics, Heritage Science; for the HCI angle (weaver study), CHI/DIS.

## Related work to cover (verify each reference)

Masked generative transformers (MaskGIT), discrete / masked diffusion (D3PM, MDLM, MD4),
diffusion transformers (DiT), classifier-free guidance, rotary embeddings; computational knitting
and weaving (machine-knitting compilers, neural inverse knitting, yarn-level cloth rendering,
weave pattern design tools); texture and pattern synthesis; generative models for traditional
textiles and crafts (e.g. batik, kente, songket, Indian handloom studies); digital cultural heritage
and ethics of computational craft.
