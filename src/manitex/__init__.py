"""manitex: generative modelling, digitisation and weaving instructions for Manipuri textiles.

The package is organised in layers that only depend "downwards":

* :mod:`manitex.design`      - the core :class:`~manitex.design.Design` representation
* :mod:`manitex.color`       - colour science, yarn colour names and palette presets
* :mod:`manitex.procedural`  - a grammar of phanek / Moirang phee style designs
* :mod:`manitex.weave`       - weave structures, the design-to-draft compiler, WIF IO,
  weaver-facing instructions and charts
* :mod:`manitex.render`      - yarn-level fabric simulation renders
* :mod:`manitex.digitise`    - photo -> clean design grid
* :mod:`manitex.models`      - masked discrete diffusion over design grids (needs torch)
* :mod:`manitex.generate`    - high level generation API (model or procedural fallback)

Only numpy / Pillow / scipy are needed for everything except the neural model.
"""

from manitex.design import MAX_COLORS, Design, canonicalize_grid

__all__ = ["Design", "MAX_COLORS", "canonicalize_grid", "__version__"]

__version__ = "0.1.0"
