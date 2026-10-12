import numpy as np
import pytest

torch = pytest.importorskip("torch")

from manitex.constraints import Symmetry, make_orbits, project, satisfies  # noqa: E402
from manitex.data.dataset import CropSpec, MixedStream, ProceduralStream, ShardStream, collate, prepare_crop  # noqa: E402
from manitex.data.shards import read_shard, write_shard  # noqa: E402
from manitex.models import (  # noqa: E402
    Batch,
    GridTransformer,
    MaskedDiffusion,
    SamplerConfig,
    StyleVocab,
    make_config,
    sample,
)
from manitex.models.transformer import apply_rope, rope_tables  # noqa: E402
from manitex.procedural import generate  # noqa: E402


def tiny_model(**kw):
    torch.manual_seed(0)
    cfg = make_config({"preset": "tiny", **kw})
    return GridTransformer(cfg)


# --------------------------------------------------------------------------- constraints


def test_orbits():
    o = make_orbits(6, 8, Symmetry(mirror_x=True))
    assert (o == o[:, ::-1]).all()
    o2 = make_orbits(6, 9, Symmetry(period_x=3))
    assert len(np.unique(o2)) == 6 * 3
    o3 = make_orbits(5, 7, Symmetry(stripes="rows"))
    assert len(np.unique(o3)) == 5
    o4 = make_orbits(8, 8, Symmetry(mirror_x=True, mirror_y=True, period_x=4))
    g = project(np.random.default_rng(0).integers(0, 4, (8, 8)), o4)
    assert satisfies(g, o4)
    assert (g == g[:, ::-1]).all() and (g == g[::-1]).all() and (g[:, :4] == g[:, 4:]).all()
    assert len(np.unique(make_orbits(4, 4, None))) == 16


# --------------------------------------------------------------------------- transformer


def test_rope_is_relative():
    # attention scores between rotated queries/keys depend only on the 2-D offset
    cos, sin = rope_tables(4, 4, 16, 100.0, "cpu")
    q0, k0 = torch.randn(16), torch.randn(16)
    qr = apply_rope(q0.expand(16, 16).reshape(1, 1, 16, 16), cos, sin)[0, 0]
    kr = apply_rope(k0.expand(16, 16).reshape(1, 1, 16, 16), cos, sin)[0, 0]
    s1 = (qr[0] * kr[5]).sum()  # (0,0) -> (1,1)
    s2 = (qr[10] * kr[15]).sum()  # (2,2) -> (3,3)
    s3 = (qr[0] * kr[6]).sum()  # (0,0) -> (1,2): a different offset
    assert torch.allclose(s1, s2, atol=1e-5)
    assert not torch.allclose(s1, s3, atol=1e-5)


def test_forward_shapes_and_null_conditions():
    m = tiny_model()
    x = torch.randint(0, m.cfg.vocab + 1, (3, 16, 24))
    out = m(x)
    assert out.shape == (3, 16, 24, m.cfg.vocab)
    out2 = m(x, torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]), torch.tensor([0, 1, 0]))
    assert out2.shape == out.shape
    with pytest.raises(ValueError):
        m(torch.zeros(1, 15, 16, dtype=torch.long))
    assert m.num_params() > 10_000


def test_patch_cells_are_independent_outputs():
    # zero-initialised head -> uniform logits; after a random head the logits of different
    # cells inside a patch must differ (each cell has its own output slice)
    m = tiny_model()
    torch.nn.init.normal_(m.head.weight, std=0.1)
    torch.nn.init.normal_(m.final_ada.weight, std=0.01)
    x = torch.randint(0, m.cfg.vocab, (1, 8, 8))
    out = m(x)
    assert not torch.allclose(out[0, 0, 0], out[0, 0, 1])


def test_overfit_tiny_batch():
    m = tiny_model()
    diff = MaskedDiffusion(m, p_uncond=0.0, p_drop_field=0.0)
    rng = np.random.default_rng(0)
    grids = []
    for i in range(8):
        d = generate("moirang_phee", 16, 16, seed=i)
        g, _ = prepare_crop(d.grid, CropSpec(16, False, False), rng)
        grids.append(g)
    b = Batch(torch.from_numpy(np.stack(grids)).long(), torch.zeros(8, dtype=torch.long),
              torch.full((8,), 3, dtype=torch.long), torch.zeros(8, dtype=torch.long))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    first = None
    for _ in range(150):
        loss, stats = diff.loss(b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = stats["loss"]
    final = float(np.mean([diff.loss(b, train=False)[1]["loss"] for _ in range(5)]))
    assert final < 0.5 * first
    bpc = diff.nll_bound(b, n_t=4)
    assert bpc.shape == (8,) and torch.isfinite(bpc).all()


# --------------------------------------------------------------------------- sampler


@pytest.mark.parametrize("strategy", ["confidence", "random"])
def test_sampler_constraints(strategy):
    m = tiny_model()
    torch.nn.init.normal_(m.head.weight, std=0.5)  # make predictions non-uniform
    cfg = SamplerConfig(steps=6, strategy=strategy, cfg=1.0)
    sym = Symmetry(mirror_x=True, period_y=5)
    out = sample(m, 3, 15, 13, style=1, ncolors=3, symmetry=sym, config=cfg,
                 generator=torch.Generator().manual_seed(0))
    assert out.shape == (3, 15, 13)
    assert int(out.max()) < 3 and int(out.min()) >= 0
    orb = make_orbits(15, 13, sym)
    for g in out.numpy():
        assert satisfies(g, orb)
        assert (g == g[:, ::-1]).all()


def test_sampler_inpainting_keeps_known_cells():
    m = tiny_model()
    torch.nn.init.normal_(m.head.weight, std=0.5)
    known = -np.ones((12, 12), dtype=np.int64)
    known[:6] = np.random.default_rng(0).integers(0, 4, (6, 12))
    out = sample(m, 2, 12, 12, known=known, config=SamplerConfig(steps=5, cfg=0.0),
                 generator=torch.Generator().manual_seed(1))
    assert (out[:, :6].numpy() == known[:6]).all()


def test_sampler_stripes_rows():
    m = tiny_model()
    torch.nn.init.normal_(m.head.weight, std=0.5)
    out = sample(m, 2, 10, 10, symmetry=Symmetry(stripes="rows"), config=SamplerConfig(steps=4, cfg=0.0))
    for g in out.numpy():
        assert all(len(np.unique(r)) == 1 for r in g)


class FixedMarginals(torch.nn.Module):
    """Stub denoiser: the same colour distribution for every cell, whatever the context."""

    def __init__(self, probs):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(probs).log())
        self.cfg = type("Cfg", (), {"vocab": len(probs), "patch": 1})()
        self.mask_id = len(probs)
        self.null_style = self.null_ncolors = self.null_domain = 0

    def forward(self, x, style, ncolors, domain):
        return self.logits.expand(*x.shape, -1).clone()


def test_ancestral_sampling_is_unbiased_and_confidence_order_is_not():
    m = FixedMarginals([0.8, 0.2])
    g = sample(m, 8, 32, 32, config=SamplerConfig(strategy="random", steps=16, cfg=0.0, top_p=1.0),
               generator=torch.Generator().manual_seed(0))
    assert abs((g == 1).float().mean().item() - 0.2) < 0.02  # 8192 cells: std ~0.004
    gc = sample(m, 8, 32, 32, config=SamplerConfig(strategy="confidence", steps=16, cfg=0.0, noise=0.0),
                generator=torch.Generator().manual_seed(0))
    assert (gc == 1).float().mean().item() < 0.1  # most confident (ground) cells are revealed first
    assert SamplerConfig().strategy == "random"
    with pytest.raises(ValueError):
        sample(m, 1, 4, 4, config=SamplerConfig(strategy="greedy"))


def test_top_p_drops_the_tail():
    m = FixedMarginals([0.7, 0.2, 0.1])
    gen = torch.Generator().manual_seed(0)
    g = sample(m, 8, 32, 32, config=SamplerConfig(steps=8, cfg=0.0, top_p=0.85), generator=gen)
    assert int(g.max()) == 1  # colour 2 lies beyond the nucleus (0.7 + 0.2 >= 0.85)
    assert abs((g == 1).float().mean().item() - 0.2 / 0.9) < 0.02  # renormalised within the nucleus
    g = sample(m, 2, 8, 8, config=SamplerConfig(steps=4, cfg=0.0, top_p=0.6), generator=gen)
    assert (g == 0).all()
    with pytest.raises(ValueError):
        sample(m, 1, 4, 4, config=SamplerConfig(top_p=0.0))


# --------------------------------------------------------------------------- data


def test_streams_and_shards(tmp_path):
    vocab = StyleVocab.default()
    crop = CropSpec(size=16)
    ps = ProceduralStream(vocab, crop, seed=0)
    it = iter(ps)
    items = [next(it) for _ in range(6)]
    b = collate(items)
    assert b["grid"].shape == (6, 16, 16) and b["grid"].max() < 16
    for itm in items:
        assert itm["ncolors"] == len(np.unique(itm["grid"]))
        counts = np.bincount(itm["grid"].ravel())
        assert (counts > 0).all()  # canonical slots are contiguous 0..k-1
        assert (np.diff(counts) <= 0).all()  # ... ordered by decreasing frequency
    designs = [generate("check", 20, 12, seed=i) for i in range(3)] + [generate("moirang_phee", 8, 30, seed=5)]
    path = write_shard(tmp_path / "s.npz", designs, domains=["real"] * 4)
    sh = read_shard(path)
    assert len(sh) == 4
    np.testing.assert_array_equal(sh.grid(3), designs[3].grid)
    ss = ShardStream([sh], vocab, crop)
    x = ss.sample(np.random.default_rng(0))
    assert x["grid"].shape == (16, 16) and x["domain"] == 1
    mix = MixedStream([ps, ss], [0.5, 0.5])
    it2 = iter(mix)
    assert next(it2)["grid"].shape == (16, 16)
    # domains recorded in checkpoints come from the data actually used
    from manitex.train import TrainConfig, data_domains

    assert data_domains(TrainConfig()) == ["synthetic"]
    assert data_domains(TrainConfig(data=[{"type": "procedural"}, {"type": "shards", "paths": [str(path)]}])) == [
        "real", "synthetic"]
    syn = write_shard(tmp_path / "syn.npz", designs, domains=["synthetic"] * 4)
    assert data_domains(TrainConfig(data=[{"type": "shards", "paths": str(syn)}])) == ["synthetic"]
    assert data_domains(TrainConfig(data=[{"type": "shards", "paths": str(syn), "domain": "real"}])) == ["real"]


def test_prepare_crop_canonical_and_colour_cap():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 20, (30, 30))
    pal = rng.integers(0, 256, (20, 3))
    out, k = prepare_crop(g, CropSpec(16, max_colors=8), rng, pal)
    assert out.max() < 8 and k <= 8
    counts = np.bincount(out.ravel())
    assert (np.diff(counts) <= 0).all()
