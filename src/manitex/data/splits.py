"""Dataset splits by textile (piece): no crop of one textile in two splits."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def split_pieces(categories: dict[str, str], val: float, test: float, seed: int) -> dict[str, list[str]]:
    """Stratified piece split: per category, ``round(test * n)`` pieces to test and
    ``round(val * n)`` to validation (at least one each when the category has >= 3 pieces)."""
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for piece, cat in sorted(categories.items()):
        by_cat[cat].append(piece)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for cat in sorted(by_cat):
        pieces = [by_cat[cat][i] for i in rng.permutation(len(by_cat[cat]))]
        n = len(pieces)
        n_test, n_val = int(round(test * n)), int(round(val * n))
        if n >= 3:
            n_test, n_val = max(n_test, 1 if test > 0 else 0), max(n_val, 1 if val > 0 else 0)
        n_test, n_val = min(n_test, max(n - 1, 0)), min(n_val, max(n - 1 - n_test, 0))
        out["test"] += pieces[:n_test]
        out["val"] += pieces[n_test : n_test + n_val]
        out["train"] += pieces[n_test + n_val :]
    return out


def nested_fractions(train: list[str], categories: dict[str, str], fractions: list[float],
                     seed: int) -> dict[float, list[str]]:
    """Per category, the first ``ceil(f * n)`` pieces of one fixed shuffle (nested in ``f``)."""
    rng = np.random.default_rng(seed + 1)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for piece in sorted(train):
        by_cat[categories[piece]].append(piece)
    order = {cat: [ps[i] for i in rng.permutation(len(ps))] for cat, ps in by_cat.items()}
    return {f: sorted(p for ps in order.values() for p in ps[: math.ceil(f * len(ps))]) for f in fractions}
