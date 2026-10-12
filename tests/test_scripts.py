"""End-to-end runs of the experiment scripts on tiny inputs (they must keep working)."""

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from manitex.data import write_shard
from manitex.procedural import generate

ROOT = Path(__file__).resolve().parents[1]


def run(*args, timeout=600):
    res = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    return res.stdout


@pytest.fixture()
def shards(tmp_path):
    out = {}
    for split, n, off in (("train", 4, 0), ("val", 3, 100)):
        ds = [generate(s, 20, 24, seed=off + i) for s in ("moirang_phee", "check") for i in range(n)]
        out[split] = write_shard(tmp_path / f"{split}.npz", ds, styles=[d.meta["style"] for d in ds],
                                 domains=["real"] * len(ds))
    return out


@pytest.fixture()
def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from manitex.models import StyleVocab, build_model, make_config
    from manitex.models.checkpoint import save_checkpoint

    torch.manual_seed(0)
    m = build_model(make_config("tiny"))
    return save_checkpoint(tmp_path / "m.pt", model=m, ema=m, optimizer=None, step=1, model_cfg=m.cfg,
                           train_cfg={"crop": 8}, vocab=StyleVocab.default(m.cfg.n_styles))


def test_model_experiment_scripts(tmp_path, shards, checkpoint):
    run("scripts/eval_generators.py", "--model", f"tiny={checkpoint}", "--designs", f"stored={shards['train']}",
        "--reference", shards["val"], "--train", shards["train"], "--styles", "check", "--n", "2", "--size", "8",
        "--n-weave", "1", "--max-train", "4", "--device", "cpu", "--out", tmp_path / "e1")
    e1 = json.loads((tmp_path / "e1.json").read_text())
    assert set(e1["check"]) == {"floor", "grammar", "retrieval", "stored", "tiny"} and "nll_ref" in e1["check"]["tiny"]
    run("scripts/eval_constraints.py", "--checkpoint", checkpoint, "--styles", "check", "--n", "2", "--size", "8",
        "--steps", "2", "--constraints", "mirror_x", "--device", "cpu", "--out", tmp_path / "e2")
    e2 = json.loads((tmp_path / "e2.json").read_text())["check | mirror_x"]
    assert e2["orbit_mean"]["satisfied"] == 1.0 and e2["projection"]["satisfied"] == 1.0
    run("scripts/ablate_sampler.py", "--checkpoint", checkpoint, "--styles", "check", "--n", "2", "--size", "8",
        "--settings", "steps=2,top_p=1;steps=2", "--device", "cpu", "--out", tmp_path / "abl")
    assert len(json.loads((tmp_path / "abl.json").read_text())["check"]) == 3  # floor + 2 settings


def test_data_and_study_scripts(tmp_path, shards):
    # piece-level splits of a design folder
    for i in range(6):
        piece = tmp_path / "designs" / f"P{i}"
        piece.mkdir(parents=True)
        generate("check", 10, 12, seed=i).save(piece / "a_design.png")
    run("scripts/make_splits.py", tmp_path / "designs", "--out", tmp_path / "real", "--val", "0.2", "--test", "0.2",
        "--fractions", "0.5")
    sp = json.loads((tmp_path / "real" / "splits.json").read_text())
    assert sum(len(sp["pieces"][k]) for k in ("train", "val", "test")) == 6 and sp["pieces"]["train_50"]
    # blind study set, then the analysis of (fake) responses
    run("scripts/make_study_set.py", "build", "--real", shards["val"], "--styles", "check", "--n", "2", "--size",
        "12x14", "--image-px", "96", "--out", tmp_path / "study")
    key = list(csv.DictReader(open(tmp_path / "study" / "key.csv")))
    assert {r["source"] for r in key} == {"real", "grammar"} and (tmp_path / "study" / "sheet.html").exists()
    with open(tmp_path / "study" / "resp.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "item_id", "authenticity", "would_weave", "comment"])
        for j, r in enumerate(key):
            w.writerow(["P1", r["item_id"], 1 + j % 5, "yes", ""])
    run("scripts/make_study_set.py", "analyse", "--key", tmp_path / "study" / "key.csv", "--responses",
        tmp_path / "study" / "resp.csv", "--out", tmp_path / "e5")
    assert "real" in json.loads((tmp_path / "e5.json").read_text())["all styles"]


def test_real_photo_digitisation_script(tmp_path):
    from PIL import Image

    from manitex.digitise.synth import synthetic_photo

    # (on very small designs a yarn colour woven both as a plain-weave stripe and as floats can come
    # out as two colour clusters - real optical mixing, see the paper's error analysis)
    d = generate("moirang_phee", 24, 30, seed=40)
    sp = synthetic_photo(d, np.random.default_rng(0))
    Image.fromarray(sp.photo).save(tmp_path / "p.png")
    (tmp_path / "p.json").write_text(json.dumps({"corners": np.asarray(sp.quad).ravel().tolist(), "cells_x": 30,
                                                 "cells_y": 24}))
    d.save(tmp_path / "p_truth.png")
    run("scripts/eval_digitise_real.py", tmp_path, "--out", tmp_path / "e7")
    res = json.loads((tmp_path / "e7.json").read_text())["real photographs"]
    assert res["p.png"]["assisted_cell_acc"] > 0.95


def test_gpu_benchmark_script_runs_on_cpu(tmp_path):
    out = run("scripts/benchmark_gpu.py", "--config", "configs/train/debug_cpu.yaml", "--steps", "2", "--warmup", "1",
              "--device", "cpu", "--probe", "--json", tmp_path / "b.json", "crop=8", "batch_size=2", "grad_accum=2",
              "num_workers=0")
    rep = json.loads((tmp_path / "b.json").read_text())
    assert rep["benchmark"]["grad_accum"] == 2 and rep["benchmark"]["sec_per_optimizer_step"] > 0
    assert "per step" in out and "probe" not in rep and "skipped" in out  # probing needs a GPU


def test_space_builds(tmp_path):
    for hw in ("cpu", "zerogpu", "gpu"):
        run("scripts/build_space.py", "--out", tmp_path / hw, "--hardware", hw)
        req = (tmp_path / hw / "requirements.txt").read_text().splitlines()
        assert len([r for r in req if r.startswith("torch")]) == 1 and "numpy>=1.24" in req
        assert any("whl/cpu" in r for r in req) == (hw == "cpu") and (tmp_path / hw / "manitex" / "app.py").exists()
    assert "spaces" in (tmp_path / "zerogpu" / "requirements.txt").read_text().split()
    assert "whl/cu128" in (tmp_path / "gpu" / "requirements.txt").read_text()


def test_gpu_benchmark_batch_suggestion():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("benchmark_gpu", ROOT / "scripts" / "benchmark_gpu.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    probe = [{"micro_batch": 128, "peak_memory_gb": 41.0}, {"micro_batch": 256, "peak_memory_gb": 82.0}]
    assert bench.suggest(probe, 256, 96.0) == {"batch_size": 128, "grad_accum": 2}  # 82 GB > 85% of 96 GB
    assert bench.suggest(probe, 256, 102.0) == {"batch_size": 256, "grad_accum": 1}  # a 96 GiB card reports ~102 GB
    probe[1] = {"micro_batch": 256, "out_of_memory": True}
    assert bench.suggest(probe, 256, 102.0) == {"batch_size": 128, "grad_accum": 2}
    assert bench.suggest([{"micro_batch": 64, "peak_memory_gb": 99.0}], 256, 96.0) is None
    assert bench.suggest([{"micro_batch": 64, "peak_memory_gb": 99.0}], 256, None) == {"batch_size": 64, "grad_accum": 4}
