"""Hugging Face Spaces entry point for the manitex demo."""

import os
from pathlib import Path

from manitex.app import launch

checkpoint = os.environ.get("MANITEX_CHECKPOINT")
if not checkpoint and (Path(__file__).parent / "model" / "config.json").exists():
    checkpoint = str(Path(__file__).parent / "model")

launch(checkpoint=checkpoint, server_name="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
