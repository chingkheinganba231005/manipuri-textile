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
* **Tie-downs.** For a visible run `[a, b)` longer than `F`, ties are placed at
  `{x ∈ (a, b−1) : (x − φ) mod F = 0}` with a brick-staggered phase `φ`; any residual gap > `F` is
  split at its midpoint. Every remaining float is ≤ `F` and no motif edge is hidden.

## Experiments

| ID | Question | Setup | Measures |
|---|---|---|---|
| E1 | RQ1 | Train on real train split (with/without grammar pretraining); compare with grammar, nearest-neighbour retrieval, AR transformer baseline, image-space LDM + digitiser | pattern-JS 2×2/3×3, run-length W1 (rows/cols), colour-count TV, symmetry stats, NLL (bits/cell), diversity, novelty |
| E2 | RQ2 | Same model: unconstrained / post-hoc projection / orbit pooling (mean, sum) for mirror, repeat, stripes | constraint satisfaction, NLL of samples under the model, pattern-JS, expert preference |
| E3 | RQ3 | Fine-tune with 0/10/25/50/100 % of the real data, with and without pretraining | NLL and pattern-JS vs data fraction |
| E4 | RQ4 | Compile all samples | tie-downs / 100 cells, max floats, dobby shafts, pattern-pick fraction; weaver ratings of booklets |
| E5 | RQ4 | Blind study with 8–15 weavers/designers: real vs procedural vs generated (Likert authenticity, pairwise preference, "would you weave it?") | ratings, preference rates, qualitative comments |
| E6 | RQ4 | Weave 1–3 generated borders from the booklet on a real loom; photograph and digitise the result | design-vs-woven agreement, time, weaver notes |
| E7 | RQ5 | `scripts/benchmark_digitise.py` (synthetic) + a real subset with manually corrected ground truth | cell accuracy (assisted), pixel accuracy (automatic), colour ΔE00, exact-grid rate |

**Ablations:** patch size (2 vs 4), loss weighting (mean vs ELBO), schedule (cosine vs linear),
condition dropout, domain tag on/off, flips, model size (tiny → large), sampling steps and guidance.
See `configs/ablations/README.md`.

**Baselines still to implement:** an autoregressive raster-order transformer on cells (same
backbone with causal attention), and the image-space pipeline (fine-tune an open latent diffusion
model on rendered designs, then digitise its outputs with `manitex digitise`).

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

Model sanity check (CPU, 4 M parameters, 32×32 crops, procedural data only, 4 k steps): held-out
NLL bound fell from 3.2 to ≈ 0.2 bits/cell; samples show correct stripe and check structure.
Real results come from the A100 runs (`configs/train/*.yaml`).

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
