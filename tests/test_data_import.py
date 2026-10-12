import numpy as np
import pytest
from PIL import Image

from manitex.data import find_design_files, import_designs, limit_colors, load_design_folder, read_shard
from manitex.design import Design
from manitex.procedural import generate


def test_import_digitise_output_folders(tmp_path):
    """`manitex digitise` writes a design (.png + .json), a preview and a grid overlay per photo;
    importing the folder must yield each design exactly once."""
    from manitex.cli import main

    designs = {"MP-0001": generate("moirang_phee", 12, 16, seed=1), "MP-0002": generate("check", 10, 14, seed=2)}
    for piece, d in designs.items():
        photo = tmp_path / "raw" / f"{piece}.png"
        photo.parent.mkdir(parents=True, exist_ok=True)
        d.to_image(cell=6).save(photo)
        main(["digitise", str(photo), "--cells-x", str(d.width), "--cells-y", str(d.height), "--style",
              d.meta.get("style", "x"), "--out", str(tmp_path / "designs" / piece)])
    root = tmp_path / "designs"
    assert len(list(root.rglob("*.png"))) == 6  # design, preview, grid per photo
    files = find_design_files(root)
    assert [f.name for f in files] == ["MP-0001_design.png", "MP-0002_design.png"]
    loaded = load_design_folder(root)
    assert [d.meta["piece_id"] for d in loaded] == ["MP-0001", "MP-0002"]
    for d, truth in zip(loaded, designs.values()):
        assert d.shape == truth.shape and (d.grid == truth.canonical().grid).all()
    sh = read_shard(import_designs(root, tmp_path / "real.npz"))
    assert len(sh) == 2 and list(sh.styles) == ["moirang_phee", "check"]
    # a curated .png wins over the stale .json written next to it
    edited = Design(np.zeros((3, 3), int), np.array([[10, 20, 30]], np.uint8), {"style": "x"})
    edited.save(root / "MP-0001" / "MP-0001_design.png")
    assert load_design_folder(root)[0].shape == (3, 3)
    # anything else that is not a design gets a clear error
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (40, 40, 3), np.uint8)).save(root / "stray_photo.png")
    with pytest.raises(ValueError, match="stray_photo.png"):
        load_design_folder(root)


def test_style_falls_back_to_folder_name(tmp_path):
    d = generate("check", 8, 8, seed=0)
    d.meta.pop("style", None)
    (tmp_path / "leirum_phee").mkdir()
    d.save(tmp_path / "leirum_phee" / "a.png")
    sh = read_shard(import_designs(tmp_path / "leirum_phee", tmp_path / "s.npz"))
    assert list(sh.styles) == ["leirum_phee"]


def test_limit_colors():
    rng = np.random.default_rng(0)
    d = Design(np.arange(64).reshape(8, 8) % 20, rng.integers(0, 256, (20, 3)).astype(np.uint8))
    c = limit_colors(d, 16)
    assert c.n_colors == 16 and int(c.grid.max()) == 15


def test_piece_splits_are_disjoint_stratified_and_nested():
    from manitex.data.splits import nested_fractions, split_pieces

    cats = {f"P{i:03d}": ("a" if i % 4 else "b") for i in range(40)}  # 30 x 'a', 10 x 'b'
    sp = split_pieces(cats, val=0.1, test=0.2, seed=3)
    assert sorted(sp["train"] + sp["val"] + sp["test"]) == sorted(cats)
    assert not set(sp["train"]) & set(sp["val"]) and not set(sp["train"]) & set(sp["test"])
    count = {k: sum(cats[p] == "b" for p in v) for k, v in sp.items()}
    assert count == {"train": 7, "val": 1, "test": 2}  # stratified: 10 % / 20 % of the 'b' pieces
    assert split_pieces(cats, 0.1, 0.2, seed=3) == sp  # reproducible
    sub = nested_fractions(sp["train"], cats, [0.1, 0.5, 1.0], seed=3)
    assert set(sub[0.1]) <= set(sub[0.5]) <= set(sub[1.0]) == set(sp["train"])
    assert {cats[p] for p in sub[0.1]} == {"a", "b"}  # every category represented
    tiny = split_pieces({"x": "c", "y": "c"}, 0.1, 0.1, seed=0)  # too few pieces to hold any out
    assert sorted(tiny["train"]) == ["x", "y"] and not tiny["val"] and not tiny["test"]
