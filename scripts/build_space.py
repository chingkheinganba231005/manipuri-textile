#!/usr/bin/env python
"""Assemble a ready-to-push Hugging Face Space folder.

    python scripts/build_space.py --out space_build [--model exports/manitex-small]

Then:  hf auth login
       hf repos create <user>/<space> --type space --sdk gradio
       hf upload <user>/<space> space_build . --type space
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="space_build")
    ap.add_argument("--model", default=None, help="slim export directory to bundle (model.safetensors + config.json)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("README.md", "app.py", "requirements.txt"):
        shutil.copy(ROOT / "space" / name, out / name)
    shutil.copytree(ROOT / "src" / "manitex", out / "manitex", ignore=shutil.ignore_patterns("__pycache__"))
    if args.model:
        shutil.copytree(args.model, out / "model")
    print(f"Space folder ready: {out.resolve()}")


if __name__ == "__main__":
    main()
