"""Standalone import and packaging smoke tests (no checkpoints required)."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def test_package_imports():
    import ponyexl3
    from ponyexl3.ref import EXL3Layer, decode_3inst, load_exl3_layer
    from ponyexl3.ref.synthetic import make_exl3_layer

    layer = make_exl3_layer(k=4, in_features=64, out_features=64, seed=0)
    assert isinstance(layer, EXL3Layer)
    assert decode_3inst is not None
    assert load_exl3_layer is not None
    assert ponyexl3.__all__


def test_mlx_subpackage_imports():
    mlx = importlib.import_module("ponyexl3.mlx")
    assert hasattr(mlx, "linear_forward_mlx")


def test_no_pony_dependency_on_path():
    assert not any(m.startswith("pony.") for m in sys.modules)


def test_console_scripts_registered():
    from importlib.metadata import entry_points

    names = {
        ep.name
        for ep in entry_points(group="console_scripts")
        if ep.value.startswith("ponyexl3.")
    }
    expected = {
        "ponyexl3-generate",
        "ponyexl3-generate-bench",
        "ponyexl3-compare-layer",
        "ponyexl3-compare-engines",
    }
    assert expected <= names, f"missing entry points: {expected - names}"


def test_generate_synthetic_layer_cli():
    out = Path(__file__).resolve().parent / "fixtures" / "_smoke_synthetic.npz"
    out.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "ponyexl3.cli.generate_synthetic_layer", "--out", str(out)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out.is_file()
    out.unlink(missing_ok=True)
