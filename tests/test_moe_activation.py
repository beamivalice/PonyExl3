"""MoE gate activation (SwiGLU vs GeGLU) parity."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from ponyexl3.mlx.exl3_moe import _moe_gate_activation

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]

GEMMA4_MODEL = os.environ.get(
    "PONYEXL3_MODEL_GEMMA4",
    "/Users/beam/llm/models/gemma-4-26B-A4B-exl3-4.10bpw",
)
_HAS_GEMMA4 = Path(GEMMA4_MODEL).joinpath("config.json").is_file()


def test_gelu_activation_matches_mlx():
    g = mlx.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=mlx.float16)
    u = mlx.array([0.25, 1.0, -1.0, 2.0, 0.5], dtype=mlx.float16)
    got = np.array(_moe_gate_activation(g, u, "gelu")).astype(np.float32)
    ref = np.array(nn.gelu_approx(g) * u).astype(np.float32)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-3)


def test_gelu_differs_from_silu():
    g = mlx.array([-1.0, 0.5, 2.0], dtype=mlx.float16)
    u = mlx.array([1.0, -0.5, 0.75], dtype=mlx.float16)
    gelu = np.array(_moe_gate_activation(g, u, "gelu")).astype(np.float32)
    silu = np.array(_moe_gate_activation(g, u, "silu")).astype(np.float32)
    assert not np.allclose(gelu, silu)


@pytest.mark.skipif(not _HAS_GEMMA4, reason="Gemma4 checkpoint not available")
def test_gemma4_gelu_fused_unfused_parity():
    """Fused and unfused GeGLU decode paths agree on one MoE layer."""
    from ponyexl3.mlx.model import load_model

    model, _ = load_model(GEMMA4_MODEL, engine="exl3", warm=True, verbose=False)
    switch = model.layers[0].router.switch_mlp
    assert switch._activation == "gelu"  # pyright: ignore[reportPrivateUsage]

    x = mlx.random.normal((1, switch.input_dims), dtype=mlx.float16) * 0.1
    inds = mlx.array([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=mlx.int32)

    os.environ["EXL3_MOE_FUSED"] = "0"
    y_u = switch(x, inds)
    os.environ["EXL3_MOE_FUSED"] = "1"
    y_f = switch(x, inds)
    mlx.eval(y_u, y_f)

    got = np.array(y_u).astype(np.float32)
    ref = np.array(y_f).astype(np.float32)
    scale = float(np.abs(ref).max()) + 1e-9
    assert float(np.abs(got - ref).max()) / scale < 5e-3
