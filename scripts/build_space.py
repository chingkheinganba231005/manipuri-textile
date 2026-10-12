#!/usr/bin/env python
"""Assemble a ready-to-push Hugging Face Space folder.

    python scripts/build_space.py --out space_build [--model exports/manitex-small] [--hardware zerogpu]

Then:  hf auth login
       hf repos create <user>/<space> --type space --sdk gradio
       hf upload <user>/<space> space_build . --type space

and pick the matching hardware in the Space settings. ``--hardware`` decides which PyTorch build
the Space installs:

* ``cpu`` (CPU Basic / CPU Upgrade): the small CPU-only build.
* ``zerogpu`` (ZeroGPU, free for a few Spaces per account; shared RTX PRO 6000 Blackwell GPUs): a
  PyTorch version ZeroGPU supports, plus its ``spaces`` package. The app requests the GPU only
  while it samples.
* ``gpu`` (paid T4, L4, L40S, A10G or A100): the CUDA 12.8 build, which runs on all of these GPUs
  and on drivers from 525 on (PyPI's default wheel needs a CUDA 13 driver, 580 or newer).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# requirement lines replacing the CPU index and the torch line of space/requirements.txt
TORCH_REQUIREMENTS = {
    "cpu": None,  # keep space/requirements.txt as it is
    # versions ZeroGPU lists as supported: 2.8.0, 2.9.1, 2.10.0, 2.11.0, 2.12.1, 2.13.0
    "zerogpu": ["torch==2.13.0", "spaces"],
    # 2.11 is the newest PyTorch built for CUDA 12.8 (the +cu128 wheel wins over PyPI's of the same version)
    "gpu": ["--extra-index-url https://download.pytorch.org/whl/cu128", "torch>=2.4,<2.12"],
}


def requirements_for(text: str, hardware: str) -> str:
    """``space/requirements.txt`` adapted to the Space hardware."""
    lines = TORCH_REQUIREMENTS[hardware]
    if lines is None:
        return text
    keep = [r for r in text.splitlines()
            if "download.pytorch.org/whl/cpu" not in r and not re.match(r"torch\s*([<>=!~;\[]|$)", r)]
    return "\n".join([*lines, *keep]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="space_build")
    ap.add_argument("--model", default=None, help="slim export directory to bundle (model.safetensors + config.json)")
    ap.add_argument("--hardware", default="cpu", choices=list(TORCH_REQUIREMENTS),
                    help="Space hardware the build is for (decides the PyTorch build, see above)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("README.md", "app.py"):
        shutil.copy(ROOT / "space" / name, out / name)
    req = (ROOT / "space" / "requirements.txt").read_text()
    (out / "requirements.txt").write_text(requirements_for(req, args.hardware))
    shutil.copytree(ROOT / "src" / "manitex", out / "manitex", ignore=shutil.ignore_patterns("__pycache__"))
    if args.model:
        shutil.copytree(args.model, out / "model")
    print(f"Space folder ready for {args.hardware} hardware: {out.resolve()}")


if __name__ == "__main__":
    main()
