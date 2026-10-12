import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from manitex.constraints import Symmetry, make_orbits, satisfies  # noqa: E402
from manitex.models import (  # noqa: E402
    ARGridTransformer,
    AutoregressiveLM,
    Batch,
    SamplerConfig,
    ar_sample,
    build_model,
    make_config,
)
from manitex.models.autoregressive import KVCacheDecoder  # noqa: E402


def ar_model(seed: int = 0) -> ARGridTransformer:
    torch.manual_seed(seed)
    m = build_model(make_config({"preset": "tiny", "arch": "ar", "patch": 1}))
    # adaLN-Zero starts every block as the identity: open the gates so attention matters
    for blk in m.blocks:
        torch.nn.init.normal_(blk.ada.bias, std=0.5)
    torch.nn.init.normal_(m.head.weight, std=0.5)
    return m.eval()


def test_ar_needs_single_cell_tokens():
    with pytest.raises(ValueError):
        build_model(make_config({"preset": "tiny", "arch": "ar", "patch": 2}))
    with pytest.raises(ValueError):
        build_model(make_config({"preset": "tiny", "arch": "nope"}))


def test_teacher_forcing_is_causal():
    m = ar_model()
    x = torch.randint(0, 5, (2, 6, 7))
    base = m(x).view(2, 42, -1)
    for i in (0, 9, 41):
        y = x.clone().view(2, 42)
        y[:, i:] = torch.randint(0, 5, (2, 42 - i))  # change cell i and everything after it
        out = m(y.view(2, 6, 7)).view(2, 42, -1)
        assert torch.allclose(out[:, : i + 1], base[:, : i + 1], atol=1e-5)  # prediction of cell i ignores cell i
    assert not torch.allclose(m(torch.zeros_like(x)), m(torch.ones_like(x)))


def test_kv_cache_matches_teacher_forcing():
    m = ar_model(1)
    x = torch.randint(0, 5, (2, 5, 6))
    style = torch.tensor([1, 4])
    full = m(x, style).view(2, 30, -1)
    dec = KVCacheDecoder(m, 5, 6, m.conditioning(2, x.device, style))
    prev = torch.full((2,), m.mask_id)
    for i in range(30):
        assert torch.allclose(dec.step(prev), full[:, i], atol=1e-4, rtol=1e-4), f"cell {i}"
        prev = x.view(2, 30)[:, i]
    with pytest.raises(IndexError):
        dec.step(prev)


def test_sampling_matches_full_forward():
    m = ar_model(1)
    greedy = SamplerConfig(temperature=1e-4, top_p=1.0, cfg=0.0)
    g = ar_sample(m, 3, 5, 6, style=2, config=greedy, generator=torch.Generator().manual_seed(0))
    full = m(g, torch.full((3,), 2))  # teacher-forced logits of the sampled grids
    assert (full.argmax(-1) == g).all()
    # classifier-free guidance combines the conditional and unconditional passes
    w = 1.5
    g = ar_sample(m, 2, 4, 5, style=3, config=SamplerConfig(temperature=1e-4, top_p=1.0, cfg=w),
                  generator=torch.Generator().manual_seed(1))
    lc, lu = m(g, torch.full((2,), 3)), m(g)
    assert ((lu + (1 + w) * (lc - lu)).argmax(-1) == g).all()


def test_constraints_known_cells_and_colour_budget():
    m = ar_model(2)
    sym = Symmetry(mirror_x=True, period_y=3)
    orb = make_orbits(9, 8, sym)
    known = -np.ones((9, 8), dtype=np.int64)
    known[1, 2] = 3  # mirrored to (1, 5) and repeated every 3 rows
    out = ar_sample(m, 4, 9, 8, symmetry=sym, known=known, ncolors=4, config=SamplerConfig(cfg=0.0),
                    generator=torch.Generator().manual_seed(0)).numpy()
    for g in out:
        assert satisfies(g, orb)
        assert g[1, 2] == g[1, 5] == g[4, 2] == g[7, 5] == 3
        assert g.max() < 4
    with pytest.raises(ValueError):
        ar_sample(m, 1, 3, 3, config=SamplerConfig(top_p=0.0))


def test_objective_and_exact_nll():
    m = ar_model(3)
    b = Batch(torch.randint(0, 4, (3, 4, 4)), torch.ones(3, dtype=torch.long), torch.full((3,), 4),
              torch.zeros(3, dtype=torch.long))
    obj = AutoregressiveLM(m)
    loss, stats = obj.loss(b, train=False)
    assert torch.isfinite(loss) and 0.0 <= stats["next_cell_acc"] <= 1.0
    logits = m(b.grid, b.style, b.ncolors, b.domain)
    logp = torch.log_softmax(logits.float(), -1).gather(-1, b.grid.unsqueeze(-1)).squeeze(-1)
    expected = -logp.mean(dim=(1, 2)) / math.log(2.0)
    assert torch.allclose(obj.nll_bound(b), expected, atol=1e-5)
    assert torch.allclose(loss, -logp.mean(), atol=1e-5)


def test_checkpoint_roundtrip_and_generator(tmp_path):
    from manitex.generate import GenerationOptions, TextileGenerator
    from manitex.models import StyleVocab
    from manitex.models.checkpoint import export_slim, load_model, save_checkpoint

    m = ar_model(4)
    vocab = StyleVocab.default(m.cfg.n_styles)
    ck = save_checkpoint(tmp_path / "ar.pt", model=m, ema=m, optimizer=None, step=5, model_cfg=m.cfg,
                         train_cfg={"crop": 8}, vocab=vocab)
    m2, info = load_model(export_slim(ck, tmp_path / "slim", dtype="float32"))
    assert isinstance(m2, ARGridTransformer) and info["model_config"].arch == "ar"
    x = torch.randint(0, 4, (1, 4, 4))
    assert torch.allclose(m(x), m2(x), atol=1e-5)
    gen = TextileGenerator.from_pretrained(str(ck), device="cpu")
    ds = gen.generate("moirang_phee", n=2, height=6, width=10, symmetry=Symmetry(mirror_x=True), seed=0,
                      options=GenerationOptions(cfg=0.5))
    assert all(d.shape == (6, 10) and (d.grid == d.grid[:, ::-1]).all() for d in ds)


@pytest.mark.parametrize("arch", ["mdm", "ar"])
def test_training_loop_end_to_end(tmp_path, arch):
    from manitex.models.checkpoint import load_model
    from manitex.train import TrainConfig, train

    model = {"preset": "tiny", "arch": "ar", "patch": 1} if arch == "ar" else "tiny"
    cfg = TrainConfig(out_dir=str(tmp_path / arch), model=model, crop=8, batch_size=4, steps=3, warmup=1,
                      precision="fp32", num_workers=0, log_every=1, eval_every=3, eval_size=4, sample_every=3,
                      save_every=3, ema_decay=0.9)
    final = train(cfg)
    assert final.exists() and (tmp_path / arch / "samples" / "step_0000003.png").exists()
    log = (tmp_path / arch / "log.jsonl").read_text()
    assert "eval_bits_per_cell_ema" in log
    m, info = load_model(final)
    assert info["step"] == 3 and isinstance(m, ARGridTransformer) == (arch == "ar")
