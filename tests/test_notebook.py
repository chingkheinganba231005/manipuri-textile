"""The Colab notebook must keep working: it is the way the project is run on a GPU.

Static checks run everywhere; the dry run executes the whole notebook on a tiny model (CPU) when Jupyter's
kernel is installed (``pip install -e ".[notebook]"``).
"""

import argparse
import ast
import json
import os
import re
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "manitex_colab.ipynb"


def sources(kind: str) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == kind]


def shell_lines(code: str, command: str) -> list[list[str]]:
    """The arguments of every ``!<command> ...`` line of the notebook (``{variables}`` stay as words)."""
    lines = re.findall(rf"^\s*!{command}\s+(.*)$", code, re.M)
    return [shlex.split(line) for line in lines]


def test_notebook_is_committed_clean():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4 and nb["metadata"]["accelerator"] == "GPU"
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, "clear the outputs before committing"


def test_notebook_code_is_valid_python():
    code = "\n".join(sources("code"))
    ast.parse(re.sub(r"^(\s*)[!%].*$", r"\1pass", code, flags=re.M))  # shell and magic lines removed


def test_notebook_commands_match_the_project():
    from manitex.cli import build_parser
    from manitex.train import TrainConfig

    code = "\n".join(sources("code"))
    for rel in set(re.findall(r"(?:scripts|configs)/[\w/.-]+\.(?:py|yaml)", code)):
        assert (ROOT / rel).exists(), f"{rel} is used by the notebook but does not exist"
    commands = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices
    for args in shell_lines(code, "manitex"):
        assert args[0] in commands, f"manitex {args[0]}: no such command"
        for flag in (a for a in args if a.startswith("--")):
            assert flag in commands[args[0]]._option_string_actions, f"manitex {args[0]} has no option {flag}"
    for args in shell_lines(code, "python"):
        script = next((a for a in args if a.startswith("scripts/")), None)
        if script:
            text = (ROOT / script).read_text()
            for flag in (a for a in args if a.startswith("--")):
                assert f'"{flag}"' in text, f"{script} has no option {flag}"
    # key=value overrides of the training configs (they are written inside strings)
    keys = set()
    for node in ast.walk(ast.parse(re.sub(r"^(\s*)[!%].*$", r"\1pass", code, flags=re.M))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            keys |= set(re.findall(r"(?:^|\s)([a-z_]+)=", node.value))
    assert keys and keys <= set(TrainConfig.__dataclass_fields__), keys - set(TrainConfig.__dataclass_fields__)


@pytest.mark.slow
def test_notebook_dry_run(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")
    pytest.importorskip("torch")
    bindir = Path(sys.executable).parent
    if not (bindir / "manitex").exists():
        pytest.skip("the manitex command is not installed (pip install -e .)")
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")  # `python` and `manitex` of this environment
    monkeypatch.setenv("MANITEX_OUT", str(tmp_path / "out"))

    nb = nbformat.read(NOTEBOOK, as_version=4)
    flag = next(c for c in nb.cells if c.cell_type == "code" and "SMOKE_TEST = False" in c.source)
    flag.source = flag.source.replace("SMOKE_TEST = False", "SMOKE_TEST = True")
    nbclient.NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()

    out = tmp_path / "out"
    assert json.loads((out / "benchmark.json").read_text())["benchmark"]["sec_per_optimizer_step"] > 0
    assert (out / "runs" / "pretrain_small" / "final.pt").exists() and (out / "exports" / "manitex-small").is_dir()
    assert list((out / "generated").glob("*.wif")) and list((out / "generated").glob("*_cloth.png"))
    summary = json.loads(next(o["text"] for o in reversed(nb.cells[-1].outputs) if o.output_type == "stream"))
    assert summary["training"]["last_step"] == 30
