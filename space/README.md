---
title: Manitex - Manipuri Textile Studio
emoji: 🧵
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Generate, digitise and weave Manipuri textile patterns
---

# manitex · Manipuri textile design studio

Generate designs inspired by Manipuri handloom textiles (striped *phanek*, *Moirang phee* temple
borders), turn any design into **weaving instructions** (WIF file, printable booklet, pick-up
instructions, simulated cloth) and **digitise** photos of textiles into clean design grids.

Set the Space variable `MANITEX_CHECKPOINT` to a trained model (e.g. `hf://<user>/<model-repo>`) or
bundle an exported model in `model/`; otherwise the procedural generator is used. Neural sampling
is slow on CPU hardware (about 30 s per 64×64 design with the small model), so the app starts with
lighter settings there; GPU hardware makes it interactive.

Source code and documentation: see the project repository.
