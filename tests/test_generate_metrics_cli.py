import json

import numpy as np
import pytest

from manitex.constraints import Symmetry
from manitex.design import Design
from manitex.metrics import (
    color_count_distance,
    diversity,
    evaluate,
    hamming,
    novelty,
    partition_codes,
    pattern_js,
    run_length_distance,
    symmetry_profile,
    weavability,
)
from manitex.procedural import generate


def test_partition_codes_palette_invariant():
    g = np.random.default_rng(0).integers(0, 4, (12, 12))
    perm = np.array([3, 1, 0, 2])
    np.testing.assert_array_equal(partition_codes(g, 3), partition_codes(perm[g], 3))
    assert partition_codes(np.zeros((3, 3), int), 3)[0] == (1 << 36) - 1  # all equal


def test_distribution_metrics():
    a = [generate("phanek_mayek_naibi", 32, 32, seed=i).grid for i in range(8)]
    b = [generate("phanek_mayek_naibi", 32, 32, seed=100 + i).grid for i in range(8)]
    c = [generate("motif_tile", 32, 32, seed=i).grid for i in range(8)]
    assert pattern_js(a, a) == pytest.approx(0.0, abs=1e-12)
    assert pattern_js(a, b) < pattern_js(a, c)
    assert run_length_distance(a, b, axis=1) < run_length_distance(a, c, axis=1)
    assert 0 <= color_count_distance(a, c) <= 1


def test_symmetry_diversity_novelty():
    d = generate("moirang_phee", 32, 33, seed=1)
    prof = symmetry_profile(d.grid)
    assert 0 <= prof["mirror_x"] <= 1 and prof["period_x"] >= 0
    g = d.grid
    assert hamming(g, g[:, ::-1]) == 0.0  # flip invariant
    perm = np.random.default_rng(3).permutation(int(g.max()) + 1)
    assert hamming(g, perm[g]) == 0.0  # palette permutations do not count
    grids = [generate("banded_motif", 24, 24, seed=i).grid for i in range(5)]
    assert diversity(grids) > 0
    nov = novelty(grids[:2], grids)  # generated set contained in "training" set -> memorised
    assert np.allclose(nov, 0.0)
    far = novelty([np.zeros((24, 24), int)], grids[2:])
    assert far[0] > 0
    # copies are found at any offset, flipped and recoloured, inside larger training designs
    big = generate("moirang_phee", 40, 56, seed=7).grid
    copy = big[5:29, 11:35][::-1, :]
    copy = np.random.default_rng(0).permutation(int(copy.max()) + 1)[copy]
    noise = np.random.default_rng(1).integers(0, 3, (24, 24))
    nov = novelty([copy, noise], [grids[0], big])
    assert nov[0] == 0.0 and nov[1] > 0.2
    small = generate("check", 10, 12, seed=2).grid  # smaller than the generated size: wraps around
    assert novelty([np.tile(small, (3, 2))[:24, :24]], [small])[0] == 0.0


def test_weavability_and_evaluate():
    ds = [generate("moirang_phee", 24, 32, seed=i) for i in range(3)]
    w = weavability(ds)
    assert w["face_floats_within_limit"] == 1.0
    assert 0 <= w["pattern_cell_frac"] <= 1
    res = evaluate([d.grid for d in ds], [generate("moirang_phee", 24, 32, seed=50 + i).grid for i in range(3)],
                   train=[d.grid for d in ds])
    assert "pattern_js_3x3" in res and res["memorised_frac(<0.02)"] == 1.0


def test_generator_procedural_fallback():
    from manitex.generate import TextileGenerator

    gen = TextileGenerator()
    assert not gen.has_model and "moirang_phee" in gen.styles
    ds = gen.generate("moirang_phee", n=2, height=24, width=30, symmetry=Symmetry(mirror_x=True), seed=3)
    assert len(ds) == 2
    for d in ds:
        assert d.shape == (24, 30)
        assert (d.grid == d.grid[:, ::-1]).all()
    custom = gen.generate("check", n=1, height=16, width=16, palette=["#ffffff", "#000000", "#ff0000"], seed=1)[0]
    assert tuple(custom.palette[0]) == (255, 255, 255)


def test_propagate_orbits():
    from manitex.constraints import make_orbits
    from manitex.generate import propagate_orbits

    orb = make_orbits(3, 4, Symmetry(mirror_x=True))
    g = -np.ones((2, 3, 4), dtype=np.int64)
    g[0, 0, 0] = 2  # mirrored to (0, 3)
    g[0, 1, 1], g[0, 1, 2] = 1, 1
    g[1, 2, 0], g[1, 2, 3], g[1, 2, 1] = 0, 0, 3  # (2,1) has no known partner yet -> (2,2) = 3
    out = propagate_orbits(g, orb)
    assert out[0, 0, 3] == 2 and out[0, 1, 1] == out[0, 1, 2] == 1 and (out[0, 2] == -1).all()
    assert out[1, 2].tolist() == [0, 3, 3, 0] and (out[1, :2] == -1).all()


def test_generator_with_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from manitex.generate import GenerationOptions, TextileGenerator
    from manitex.models import GridTransformer, StyleVocab, make_config
    from manitex.models.checkpoint import export_slim, load_model, save_checkpoint

    torch.manual_seed(0)
    cfg = make_config("tiny")
    model = GridTransformer(cfg)
    vocab = StyleVocab.default(cfg.n_styles)
    ck = save_checkpoint(tmp_path / "m.pt", model=model, ema=model, optimizer=None, step=7, model_cfg=cfg,
                         train_cfg={"crop": 16}, vocab=vocab, palette_bank={"moirang_phee": [["#ffffff", "#aa0000", "#000000"]]})
    slim = export_slim(ck, tmp_path / "slim")
    m2, info = load_model(slim)
    assert info["step"] == 7 and info["train_crop"] == 16
    for k, v in model.state_dict().items():
        assert torch.allclose(v.float(), m2.state_dict()[k].float(), atol=1e-2)
    gen = TextileGenerator.from_pretrained(str(ck), device="cpu")
    assert gen.has_model
    opts = GenerationOptions(steps=3, cfg=0.5)
    ds = gen.generate("moirang_phee", n=2, height=16, width=16, n_colors=3, symmetry=Symmetry(mirror_x=True),
                      seed=0, options=opts)
    assert len(ds) == 2 and all(d.shape == (16, 16) for d in ds)
    assert all((d.grid == d.grid[:, ::-1]).all() for d in ds)
    assert all(len(d.used_colors()) <= 3 for d in ds)
    # palette bank used for colouring (3 colours, frequency-rank transfer)
    big = gen.generate("moirang_phee", n=1, height=40, width=40, seed=1, options=opts)[0]  # tiled path (> 2 x crop)
    assert big.shape == (40, 40)
    # constraints stay exact on the tiled path (orbits span several tiles)
    for sym, check in ((Symmetry(mirror_x=True, period_y=10),
                        lambda g: (g == g[:, ::-1]).all() and (g[:10] == g[10:20]).all() and (g[:10] == g[30:40]).all()),
                       (Symmetry(stripes="rows"), lambda g: all(len(np.unique(r)) == 1 for r in g))):
        for d in gen.generate("moirang_phee", n=2, height=40, width=56, symmetry=sym, seed=3, options=opts):
            assert d.shape == (40, 56) and check(d.grid)
    mask = np.zeros((16, 16), bool)
    mask[4:8, 4:8] = True
    inp = gen.inpaint(ds[0], mask, n=1, seed=2, options=opts)[0]
    keep = ~mask
    assert (inp.to_rgb()[keep] == ds[0].to_rgb()[keep]).all()
    # domain='auto' conditions on 'real' only when the model was trained on real data
    import manitex.models.sampler as sampler_mod

    seen_domains = []
    real_sample = sampler_mod.sample

    def spy(*args, **kwargs):
        seen_domains.append(kwargs.get("domain"))
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(sampler_mod, "sample", spy)
    assert info["domains_seen"] == []
    gen.generate("moirang_phee", n=1, height=8, width=8, seed=0, options=opts)
    ck2 = save_checkpoint(tmp_path / "m2.pt", model=model, ema=model, optimizer=None, step=8, model_cfg=cfg,
                          train_cfg={"crop": 16}, vocab=vocab, extra={"domains_seen": ["real", "synthetic"]})
    gen2 = TextileGenerator.from_pretrained(str(export_slim(ck2, tmp_path / "slim2")), device="cpu")
    assert gen2.info["domains_seen"] == ["real", "synthetic"]
    gen2.generate("moirang_phee", n=1, height=8, width=8, seed=0, options=opts)
    gen2.generate("moirang_phee", n=1, height=8, width=8, seed=0,
                  options=GenerationOptions(steps=3, domain="synthetic"))
    assert seen_domains == [None, 1, 0]


def test_cli_procedural_compile_digitise(tmp_path):
    from manitex.cli import main

    out = tmp_path / "proc"
    main(["procedural", "--style", "moirang_phee", "--size", "24x30", "--n", "1", "--booklet", "--out", str(out)])
    png = next(out.glob("moirang_phee_*[0-9].png"))
    assert (out / (png.stem + ".wif")).exists() and (out / (png.stem + ".html")).exists()
    main(["compile", str(png), "--out", str(tmp_path / "comp"), "--max-float", "6"])
    assert (tmp_path / "comp" / (png.stem + "_chart.svg")).exists()
    main(["render", str(tmp_path / "comp" / (png.stem + ".wif")), "--out", str(tmp_path / "r.png")])
    assert (tmp_path / "r.png").exists()
    d = Design.load(png)
    d.to_image(cell=6).save(tmp_path / "photo.png")
    main(["digitise", str(tmp_path / "photo.png"), "--out", str(tmp_path / "dig"), "--cells-x", str(d.width),
          "--cells-y", str(d.height)])
    back = Design.load(tmp_path / "dig" / "photo_design.png")
    assert back.shape == d.shape
    main(["data", "build-synthetic", "--out", str(tmp_path / "shards"), "--n", "6", "--shard-size", "4",
          "--min-size", "16", "--max-size", "24"])
    assert len(list((tmp_path / "shards").glob("*.npz"))) == 2


def test_cli_eval_with_references_and_training_set(tmp_path, capsys):
    pytest.importorskip("torch")
    from manitex.cli import main
    from manitex.data import write_shard
    from manitex.models import StyleVocab, build_model, make_config
    from manitex.models.checkpoint import save_checkpoint

    m = build_model(make_config("tiny"))
    ck = save_checkpoint(tmp_path / "m.pt", model=m, ema=m, optimizer=None, step=1, model_cfg=m.cfg,
                         train_cfg={"crop": 16}, vocab=StyleVocab.default(m.cfg.n_styles))
    # reference designs smaller than the canvas: crops wrap around instead of being skipped
    shard = write_shard(tmp_path / "ref.npz", [generate("check", 12, 14, seed=i) for i in range(3)], styles=["check"] * 3)
    main(["eval", "--checkpoint", str(ck), "--style", "check", "--size", "16x20", "--n", "3", "--steps", "4",
          "--reference", str(shard), "--train", str(shard)])
    res = json.loads(capsys.readouterr().out)
    assert 0 <= res["pattern_js_3x3"] <= 1 and 0 <= res["novelty_mean"] <= 1 and "tie_downs_per_100_cells" in res


def test_app_studio_actions():
    pytest.importorskip("gradio")
    from manitex.app import Studio, build_app

    build_app(None)
    st = Studio(None)
    prev, states = st.generate("phanek_mayek_naibi", 2, 32, 40, 0, False, False, 0, "none", "auto", "", 1, 8, 1.0, 1.0)
    assert len(prev) == 2 and len(states) == 2
    img, md, files = st.weave(states[1], "brocade", "plain", "auto", "continuous", 2, 1, 8, 20, 18)
    assert img.ndim == 3 and "Yarns" in md and any(f.endswith(".wif") for f in files)
    d = generate("motif_tile", 20, 24, seed=0)
    ov, pv, info, fl, state = st.digitise(d.to_rgb(cell=8), [], d.width, d.height, 0, True, True, False, "motif_tile")
    assert Design.from_dict(state).shape == (20, 24)
    assert json.loads(json.dumps(state))["meta"]["style"] == "motif_tile"
