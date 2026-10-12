# Paper draft

`main.tex` is a working draft for *The Visual Computer*. The method sections describe the code in
this repository. The digitisation table (synthetic benchmark) and the sampling-order paragraph
(4 M-parameter CPU model) report real but preliminary measurements; everything marked `\todo{}`
waits for the experiments in [`docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md).

```bash
python scripts/make_figures.py --out paper/figures                 # procedural / compiler / digitisation figures
python scripts/make_figures.py --out paper/figures --checkpoint exports/manitex-v1   # + model figures
cd paper && latexmk -pdf main.tex     # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The draft compiles without errors or warnings with TeX Live 2023 (packages: `latex-extra`,
`science` for the algorithm environments, `bibtex-extra`).

Before submission:

1. Port to the current Springer Nature LaTeX template and check the journal's author guidelines.
2. Verify every entry in `references.bib` against the publisher's page; add domain sources on
   Manipuri textiles (GI registry, ethnographic literature) and related work on traditional-textile
   generation.
3. Fill the declarations (ethics approval and consent for the dataset and the weaver study).
