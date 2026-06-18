"""MiniCPM5 EXL3 runtime support."""

from __future__ import annotations

from pathlib import Path

import pytest

from ponyexl3.mlx.model import describe, load_model


MINICPM_ORACLE = Path("/Users/beam/llm/models/Exl3/MiniCPM5-1B-exl3-4.00bpw")


@pytest.mark.skipif(
    not (MINICPM_ORACLE / "quantization_config.json").is_file(),
    reason="local MiniCPM5 EXL3 oracle checkpoint not present",
)
def test_minicpm5_oracle_loads_with_llama_mapping():
    model, cfg = load_model(str(MINICPM_ORACLE), warm=False, verbose=False)

    assert cfg["model_type"] == "llama"
    summary = describe(model)
    assert "169 exact EXL3 linears" in summary
