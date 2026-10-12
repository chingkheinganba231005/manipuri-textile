# Paper draft

`main.tex` is a working draft for *The Visual Computer*. The method sections describe the code in
this repository; all results are placeholders (`\todo{}`) until the experiments in
[`docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md) are run.

```bash
python scripts/make_figures.py --out paper/figures                 # procedural / compiler / digitisation figures
python scripts/make_figures.py --out paper/figures --checkpoint exports/manitex-v1   # + model figures
cd paper && latexmk -pdf main.tex
```

Before submission:

1. Port to the current Springer Nature LaTeX template and check the journal's author guidelines.
2. Verify every entry in `references.bib` against the publisher's page; add domain sources on
   Manipuri textiles (GI registry, ethnographic literature) and related work on traditional-textile
   generation.
3. Fill the declarations (ethics approval and consent for the dataset and the weaver study).
