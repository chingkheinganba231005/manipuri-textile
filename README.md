# manitex — generating, digitising and weaving Manipuri textile patterns

**manitex** is a research codebase for *cultural computing* on the handloom textiles of Manipur:
the striped **phanek** (e.g. *phanek mayek naibi*) and the **Moirang phee** with its rows of
temple-like motifs. It covers the whole loop from photograph to loom:

```
 photo of a textile ──► digitise ──► design grid ──► generative model ──► new designs
                                         │                                     │
                                         └──────────────► weave compiler ◄─────┘
                                                              │
                     WIF draft · pick-up instructions · chart · yarn list · simulated cloth
```

| Component | What it does |
|---|---|
| **Design representation** (`manitex.design`) | A design is a grid of *design cells* (palette indices) + a palette. Colours are numbered by frequency (*canonical* order), which separates **structure** from **colour**. |
| **Procedural grammar** (`manitex.procedural`) | Stepped motif library (temple spires, pagodas, diamonds, stars…) and band layouts generating phanek, Moirang phee, banded-shawl, check and motif-tile designs — a structural prior for pretraining and a fallback generator. |
| **Generative model** (`manitex.models`) | A *masked discrete diffusion* transformer that works directly on the weavable grid. Mirror symmetry, repeats and stripes are imposed **exactly** at sampling time by *orbit pooling*; inpainting, colour budgets and classifier-free guidance are built in. An autoregressive (raster-order) baseline on the same backbone is included for comparison. |
| **Weave compiler** (`manitex.weave`) | Turns any design into a loom-agnostic draft: ground cloth (single colour, weft/warp stripes, or a plaid inferred from optical-mixture colours) + supplementary-weft pattern picks with **tie-downs** that bound float lengths. Exports **WIF** (weaving software, computer looms), run-length **pick-up instructions** for loin-loom weavers, SVG charts, yarn quantities and a printable HTML booklet. |
| **Fabric renderer** (`manitex.render`) | Yarn-level simulation of the woven face (float spread, warp-/weft-faced cloth, twist texture). |
| **Digitiser** (`manitex.digitise`) | Photo → clean design grid: perspective rectification, grid fitting, robust cell colours, CIELAB clustering, Potts denoising and repeat/symmetry regularisation. |
| **Metrics** (`manitex.metrics`) | Palette-independent pattern statistics, run-length distances, symmetry, diversity, novelty (memorisation, checked over all translations and flips) and weavability. |
| **Demo** (`manitex.app`) | A Gradio studio (generate / weave / digitise), ready for Hugging Face Spaces. |

> **Status.** The full pipeline is implemented and tested (CPU). No trained weights are shipped:
> train on your GPU (an A100 80 GB is plenty — see [docs/TRAINING.md](docs/TRAINING.md)), and
> collect real textile data with the protocol in [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)
> before making claims about authenticity. Until a checkpoint is configured, the demo uses the
> procedural grammar.

---

## Install

```bash
git clone <this repo> && cd manipuri-textile
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or /cpu
pip install -e ".[all]"
pytest -q                                                             # ~130 tests
```

Python ≥ 3.10. Only `numpy`, `pillow`, `scipy`, `pyyaml` are needed for the grammar, compiler,
renderer and digitiser; `torch` for the model; `gradio` for the demo.

## Quick start

```bash
# procedural designs + weaving booklets (WIF, HTML, Markdown, cloth render)
manitex procedural --style moirang_phee --size 64x96 --n 4 --render --booklet --out outputs/demo

# compile any design (PNG/JSON) into weaving instructions
manitex compile outputs/demo/moirang_phee_00000.png --ends-per-cell 2 --max-float 8 --epc 20 --ppc 18

# digitise a photo (optionally give the 4 cloth corners and the number of cells)
manitex digitise photo.jpg --corners 40,30,980,25,1000,760,30,770 --cells-x 120 --cells-y 90

# train (see docs/TRAINING.md), export, generate with exact symmetry
manitex train --config configs/train/pretrain_small.yaml
manitex export runs/pretrain_small/final.pt --out exports/manitex-small
manitex generate --checkpoint exports/manitex-small --style moirang_phee --mirror-x --period-x 16 --booklet

# demo
manitex app --checkpoint exports/manitex-small
```

From Python:

```python
from manitex.procedural import generate
from manitex.weave import compile_design, to_html, save_wif
from manitex.render import render_draft

design = generate("moirang_phee", height=64, width=96, seed=7)
draft = compile_design(design, ends_per_cell=2, max_float=8)      # brocade on a plain ground
save_wif(draft, "moirang.wif")
open("moirang.html", "w").write(to_html(design, draft, fabric_render=render_draft(draft)))
```

```python
from manitex.generate import TextileGenerator
from manitex.constraints import Symmetry

gen = TextileGenerator.from_pretrained("exports/manitex-small")   # or "hf://<user>/<repo>"
designs = gen.generate("moirang_phee", n=8, height=64, width=64, n_colors=3,
                       symmetry=Symmetry(mirror_x=True, period_x=16), seed=0)
```

## How it works (short)

* **Representation.** Rows run along the warp, columns along the weft; row 1 of the chart is the
  *bottom* row (woven first). Each design cell becomes `ends_per_cell` warp ends and one or more
  picks. See [docs/WEAVING.md](docs/WEAVING.md).
* **Model.** Cells are embedded, `p×p` patches become tokens, a DiT-style transformer with 2-D
  rotary positions and adaLN-Zero conditioning (style, colour count, synthetic/real domain)
  predicts the colour of masked cells. Training uses the MaskGIT (cosine) schedule; the
  linear-schedule ELBO gives a bits-per-cell likelihood for evaluation.
* **Exact structure.** A symmetry (mirror, repeat, stripes, rotation) partitions cells into
  *orbits*. At every sampling step the per-cell log-probabilities are pooled over each orbit and
  one colour is sampled per orbit, so constraints hold exactly — no post-hoc projection.
* **Weavability.** Pattern floats longer than `max_float` ends are bound by tie-downs on a
  staggered lattice; the compiler reports shafts/treadles needed for a dobby loom and the float
  lengths on both faces.

## Repository layout

```
src/manitex/        design, color, constraints, procedural/, weave/, render/, digitise/,
                    models/, data/, train.py, generate.py, metrics.py, cli.py, app.py
configs/            training configs (A100) and ablation notes
scripts/            experiments (eval_generators.py: E1, eval_constraints.py: E2, ablate_sampler.py,
                    benchmark_digitise.py: E7), make_figures.py, build_space.py
docs/               TRAINING, WEAVING, DATA_COLLECTION, RESEARCH_PLAN, ETHICS
paper/              LaTeX skeleton for the journal paper (The Visual Computer)
space/              Hugging Face Space entry point
tests/              pytest suite
```

## Demo deployment (Hugging Face Spaces)

```bash
manitex export runs/pretrain_small/final.pt --out exports/manitex-small
python scripts/build_space.py --out space_build --model exports/manitex-small
hf auth login                                                  # huggingface_hub CLI
hf repos create <user>/manitex --type space --sdk gradio
hf upload <user>/manitex space_build . --type space
```

The procedural generator, weave compiler and digitiser run comfortably on the free CPU tier.
Neural sampling is heavier: with the `small` model (32 M parameters) one 64×64 design takes
≈ 30 s at 32 steps with guidance on two CPU threads, so on CPU the app starts with lighter
settings (two 48×48 designs, 16 steps: ≈ 23 s). For a responsive public demo, give the Space GPU
hardware (a paid upgrade in the Space settings).

## Research

The research plan (questions, experiments, baselines, metrics, ablations, user study with
weavers) and current benchmark numbers are in [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md);
the paper skeleton is in [paper/](paper/).

## Cultural and ethical note

These patterns belong to living weaving communities. *Moirang Phee* is reported to be a
registered Geographical Indication of India — verify its current status before publication or
commercial use. Work **with** weavers: obtain consent for photographs, credit makers, share
results back. The colour presets are approximations, not authenticated colourways. See
[docs/ETHICS.md](docs/ETHICS.md).

## License

Code: MIT (see [LICENSE](LICENSE)). Data you collect remains subject to the consent and licences
agreed with its owners; do not redistribute community data without permission.

## Citation

```bibtex
@misc{manitex2026,
  title  = {manitex: Structure-Aware Generation, Digitisation and Weaving of Manipuri Textile Patterns},
  author = {<author names>},
  year   = {2026},
  note   = {Software, https://github.com/chingkheinganba231005/manipuri-textile}
}
```
