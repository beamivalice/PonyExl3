"""MiniCPM5 EXL3 runtime support."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ponyexl3.mlx.model import describe, load_model

_EXL3_DIR = Path(
    os.environ.get("PONYEXL3_MODELS_DIR", Path.home() / "llm/models/exl3")
)
MINICPM_ORACLE = Path(
    os.environ.get(
        "PONYEXL3_MODEL_MINICPM5",
        _EXL3_DIR / "MiniCPM5-1B-exl3-4.00bpw",
    )
)
MINICPM_PONY = Path(
    os.environ.get(
        "PONYEXL3_MODEL_MINICPM5_PONY",
        _EXL3_DIR / "MiniCPM5-1B-PonyExl3-4.00bpw",
    )
)
_HAS_ORACLE = (MINICPM_ORACLE / "quantization_config.json").is_file()
_HAS_PONY = (MINICPM_PONY / "quantization_config.json").is_file()


@pytest.mark.skipif(not _HAS_ORACLE, reason="MiniCPM5 EXL3 oracle checkpoint not present")
def test_minicpm5_oracle_loads_with_llama_mapping():
    model, cfg = load_model(str(MINICPM_ORACLE), warm=False, verbose=False)

    assert cfg["model_type"] == "llama"
    summary = describe(model)
    assert "169 exact EXL3 linears" in summary
    assert "1 huge/GEMV" in summary


@pytest.mark.skipif(not _HAS_PONY, reason="MiniCPM5 PonyExl3 converted checkpoint not present")
def test_minicpm5_pony_checkpoint_loads():
    model, cfg = load_model(str(MINICPM_PONY), warm=False, verbose=False)

    assert cfg["model_type"] == "llama"
    assert "169 exact EXL3 linears" in describe(model)
