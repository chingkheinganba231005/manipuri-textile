"""Colour science helpers, yarn colour names and (illustrative) palette presets.

All functions are vectorised over leading dimensions.  RGB values are sRGB in 0..255.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# --------------------------------------------------------------------------- hex helpers


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    s = h.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"invalid hex colour: {h!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def rgb_to_hex(rgb: Sequence[int] | np.ndarray) -> str:
    r, g, b = (int(np.clip(round(float(v)), 0, 255)) for v in list(rgb)[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------- sRGB <-> CIELAB (D65)

_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_EPS = (6.0 / 29.0) ** 3
_KAPPA = 3.0 * (6.0 / 29.0) ** 2


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return c * 255.0


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255) -> CIELAB (D65)."""
    xyz = srgb_to_linear(rgb) @ _M_RGB2XYZ.T
    t = xyz / _WHITE_D65
    f = np.where(t > _EPS, np.cbrt(t), t / _KAPPA + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_srgb(lab: np.ndarray, clip: bool = True) -> np.ndarray:
    """CIELAB (D65) -> sRGB (0..255, float).  Out-of-gamut values are clipped."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    t = np.where(f > 6.0 / 29.0, f**3, _KAPPA * (f - 4.0 / 29.0))
    lin = (t * _WHITE_D65) @ _M_XYZ2RGB.T
    if clip:
        lin = np.clip(lin, 0.0, 1.0)
    return linear_to_srgb(lin)


def in_gamut(lab: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """True where a Lab colour maps inside the sRGB gamut."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    f = np.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], axis=-1)
    t = np.where(f > 6.0 / 29.0, f**3, _KAPPA * (f - 4.0 / 29.0))
    lin = (t * _WHITE_D65) @ _M_XYZ2RGB.T
    return np.all((lin >= -tol) & (lin <= 1 + tol), axis=-1)


# --------------------------------------------------------------------------- colour differences


def delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(lab1, float) - np.asarray(lab2, float), axis=-1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (kL = kC = kH = 1), following Sharma et al. (2005)."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0**7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, h1p)
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p
    zero = (C1p * C2p) == 0
    dh = h2p - h1p
    dhp = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dhp = np.where(zero, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    habs = np.abs(h1p - h2p)
    hbarp = np.where(
        zero,
        hsum,
        np.where(habs <= 180.0, hsum / 2.0, np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0)),
    )
    T = (
        1.0
        - 0.17 * np.cos(np.radians(hbarp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hbarp))
        + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0))
    )
    dtheta = 30.0 * np.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp**7
    RC = 2.0 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0**7))
    SL = 1.0 + 0.015 * (Lbarp - 50.0) ** 2 / np.sqrt(20.0 + (Lbarp - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cbarp
    SH = 1.0 + 0.015 * Cbarp * T
    RT = -np.sin(np.radians(2.0 * dtheta)) * RC
    tL, tC, tH = dLp / SL, dCp / SC, dHp / SH
    return np.sqrt(np.maximum(tL**2 + tC**2 + tH**2 + RT * tC * tH, 0.0))


def mix_rgb(c1: np.ndarray, c2: np.ndarray, w: float = 0.5) -> np.ndarray:
    """Optical mixture of two colours (averaged in linear light), as seen in a fine weave."""
    lin = (1.0 - w) * srgb_to_linear(c1) + w * srgb_to_linear(c2)
    return linear_to_srgb(lin)


# --------------------------------------------------------------------------- yarn colour names

#: A small vocabulary of yarn shade names used in weaver-facing instructions.
YARN_COLORS: dict[str, str] = {
    "white": "#F7F5F0",
    "cream": "#EFE6D2",
    "ivory": "#F3EBD7",
    "beige": "#D8C3A0",
    "gold": "#C9A227",
    "mustard": "#C99A2E",
    "yellow": "#E8C547",
    "orange": "#D9772B",
    "rust": "#A4471F",
    "red": "#B3262B",
    "scarlet": "#D02F2F",
    "crimson": "#9E1B32",
    "maroon": "#6B1E2B",
    "wine": "#4E1625",
    "pink": "#E07A9A",
    "light pink": "#F4C6D3",
    "rose": "#C2577A",
    "magenta": "#A3195B",
    "purple": "#5B2A6E",
    "lavender": "#9C88B8",
    "navy": "#1E2A4A",
    "royal blue": "#2E5A9C",
    "sky blue": "#7FB2D9",
    "teal": "#1F6F6B",
    "green": "#2F6B4F",
    "leaf green": "#4C8C3A",
    "parrot green": "#6DBE45",
    "olive": "#6B6B2A",
    "bottle green": "#1F4D3A",
    "brown": "#5A3A22",
    "tan": "#A87B4F",
    "grey": "#8C8C8C",
    "charcoal": "#3A3A3A",
    "black": "#1A1A1A",
}
_YARN_NAMES = list(YARN_COLORS)
_YARN_LAB = srgb_to_lab(np.array([hex_to_rgb(YARN_COLORS[n]) for n in _YARN_NAMES], dtype=float))


def yarn_name(rgb: Sequence[int] | np.ndarray) -> str:
    """Nearest yarn shade name (CIEDE2000)."""
    lab = srgb_to_lab(np.asarray(rgb, dtype=float)[None, :3])
    d = delta_e2000(np.repeat(lab, len(_YARN_NAMES), axis=0), _YARN_LAB)
    return _YARN_NAMES[int(np.argmin(d))]


# --------------------------------------------------------------------------- palette presets

#: Illustrative colourways.  These are *approximations* chosen by eye to resemble commonly
#: photographed pieces; they are NOT measured from authenticated textiles.  Replace or extend
#: them with palettes fitted from digitised real textiles (see ``manitex palettes``).
#: Index 0 is the ground / base colour; later entries are progressively less frequent.
PALETTE_PRESETS: dict[str, dict] = {
    "moirang_red": {"styles": ["moirang_phee"], "colors": ["#F1EADB", "#A61E22", "#1F1A17", "#C9A227"]},
    "moirang_maroon": {"styles": ["moirang_phee"], "colors": ["#F6F2E9", "#6D1A24", "#C9A227", "#1C1C1C"]},
    "moirang_tricolor": {"styles": ["moirang_phee", "banded_motif"], "colors": ["#EFE7D6", "#B3262B", "#1C1C1C", "#2E5E3E"]},
    "moirang_black": {"styles": ["moirang_phee"], "colors": ["#F4EFE3", "#1E1B1B", "#A61E22", "#C9A227"]},
    "moirang_green": {"styles": ["moirang_phee"], "colors": ["#F3EEE1", "#1F4D3A", "#A61E22", "#E8C547"]},
    "phanek_maroon": {"styles": ["phanek_mayek_naibi"], "colors": ["#5C1A22", "#1B1B1B", "#D9A6B3", "#2F6B4F", "#E8D9B0", "#C9A227"]},
    "phanek_green": {"styles": ["phanek_mayek_naibi"], "colors": ["#1F4D3A", "#111111", "#E9C46A", "#C0392B", "#F2E8CF", "#7FB2D9"]},
    "phanek_black": {"styles": ["phanek_mayek_naibi"], "colors": ["#171515", "#8E1B2A", "#E07A9A", "#3A7D44", "#F0E2B6", "#2E5A9C"]},
    "phanek_pink": {"styles": ["phanek_mayek_naibi"], "colors": ["#C2577A", "#3B1F2B", "#F4D1DC", "#1E5F4B", "#F8F1E7", "#5B2A6E"]},
    "phanek_navy": {"styles": ["phanek_mayek_naibi"], "colors": ["#1E2A4A", "#B3262B", "#E8C547", "#F3EEE2", "#2F6B4F", "#E07A9A"]},
    "shawl_white_gold": {"styles": ["banded_motif", "motif_tile"], "colors": ["#F5F1E6", "#C9A227", "#7A1F2B", "#1F4D3A", "#1A1A1A"]},
    "shawl_red": {"styles": ["banded_motif", "motif_tile"], "colors": ["#9E1B32", "#F3EBD7", "#1A1A1A", "#E8C547", "#2F6B4F"]},
    "shawl_black": {"styles": ["banded_motif", "motif_tile"], "colors": ["#1A1A1A", "#E8C547", "#B3262B", "#F3EBD7", "#2E5A9C"]},
    "check_classic": {"styles": ["check"], "colors": ["#F3EEE2", "#7A1F2B", "#1E3D59", "#2F6B4F"]},
    "check_earth": {"styles": ["check"], "colors": ["#D8C3A0", "#5A3A22", "#1A1A1A", "#A4471F"]},
}


def preset_palette(name: str) -> np.ndarray:
    return np.array([hex_to_rgb(h) for h in PALETTE_PRESETS[name]["colors"]], dtype=np.uint8)


def presets_for_style(style: str) -> list[str]:
    names = [n for n, p in PALETTE_PRESETS.items() if style in p["styles"]]
    return names or list(PALETTE_PRESETS)


def jitter_palette(palette: np.ndarray, rng: np.random.Generator, sigma_l: float = 2.5, sigma_ab: float = 3.0) -> np.ndarray:
    """Small random perturbation in Lab (simulates dye-lot variation)."""
    lab = srgb_to_lab(np.asarray(palette, dtype=float))
    lab = lab + rng.normal(0.0, 1.0, lab.shape) * np.array([sigma_l, sigma_ab, sigma_ab])
    lab[:, 0] = np.clip(lab[:, 0], 3.0, 98.0)
    return np.clip(np.rint(lab_to_srgb(lab)), 0, 255).astype(np.uint8)


def extend_palette(palette: np.ndarray, k: int, rng: np.random.Generator, min_delta_e: float = 18.0) -> np.ndarray:
    """Append random in-gamut colours (at least ``min_delta_e`` from existing ones) up to ``k`` colours."""
    pal = [np.asarray(c, dtype=np.uint8) for c in np.asarray(palette, dtype=np.uint8)]
    tries, thr = 0, min_delta_e
    while len(pal) < k:
        lab = np.array([rng.uniform(15, 92), rng.uniform(-55, 65), rng.uniform(-55, 70)])
        tries += 1
        if tries % 200 == 0:
            thr *= 0.8
        if not in_gamut(lab):
            continue
        existing = srgb_to_lab(np.array(pal, dtype=float)) if pal else np.zeros((0, 3))
        if existing.size and delta_e76(existing, lab[None]).min() < thr:
            continue
        pal.append(np.clip(np.rint(lab_to_srgb(lab)), 0, 255).astype(np.uint8))
    return np.array(pal[:k], dtype=np.uint8) if k > 0 else np.zeros((0, 3), np.uint8)


def palette_for_style(style: str, k: int, rng: np.random.Generator, jitter: bool = True) -> tuple[np.ndarray, str]:
    """Pick a preset suited to ``style`` and return exactly ``k`` colours (+ the preset name)."""
    names = presets_for_style(style)
    name = names[int(rng.integers(len(names)))]
    pal = preset_palette(name)
    if jitter:
        pal = jitter_palette(pal, rng)
    if pal.shape[0] >= k:
        # keep the ground colour, sample the rest in preset order with occasional swaps
        rest = np.arange(1, pal.shape[0])
        if rest.size > 1 and rng.random() < 0.35:
            i, j = rng.choice(rest.size, 2, replace=False)
            rest[[i, j]] = rest[[j, i]]
        pal = pal[np.concatenate([[0], rest])[:k]]
    else:
        pal = extend_palette(pal, k, rng)
    return pal, name
