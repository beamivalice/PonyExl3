"""Parity + smoke tests for EXL3MLPMonolith."""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from ponyexl3.mlx.exl3_fused import FusedEXL3Group
from ponyexl3.mlx.exl3_linear import EXL3Linear
from ponyexl3.mlx.exl3_mlp_monolith import EXL3MLPMonolith
from ponyexl3.mlx.model import load_model
from ponyexl3.ref.synthetic import make_exl3_layer

MODEL_27B = os.environ.get("PONYEXL3_MODEL_27B", "")


def _reference_mlp(group: FusedEXL3Group, down: EXL3Linear, x: mx.array) -> mx.array:
    rows = int(np.prod(x.shape[:-1]))
    x2d = x.reshape(rows, x.shape[-1])
    g, u = group.forward_all(x2d)
    h = nn.silu(g) * u
    return down(h).reshape(x.shape)


def test_mlp_monolith_matches_switch_decode_synthetic():
    os.environ["EXL3_MLP_KERNEL"] = "moe"
    hidden = 2048  # >1024: down uses mapped path (gateup fused still applies)
    gate_l = make_exl3_layer(
        k=4, in_features=512, out_features=hidden, seed=1, mcg=True
    )
    up_l = make_exl3_layer(
        k=4, in_features=512, out_features=hidden, seed=2, mcg=True
    )
    down_l = make_exl3_layer(
        k=4, in_features=hidden, out_features=512, seed=3, mcg=True
    )
    group = FusedEXL3Group([gate_l, up_l])
    down = EXL3Linear(down_l)
    mono = EXL3MLPMonolith.from_fused_gate_up(group, down)

    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((1, 512)).astype(np.float16))
    sel = mx.array([0], dtype=mx.int32)
    y_ref = mono._switch._decode(x.reshape(1, -1), sel).reshape(1, 512)
    y_mono = mono(x)
    mx.eval(y_ref, y_mono)
    np.testing.assert_allclose(
        np.array(y_mono), np.array(y_ref), rtol=0, atol=5e-3
    )
    os.environ.pop("EXL3_MLP_KERNEL", None)


@pytest.mark.skipif(
    not MODEL_27B or not os.path.isdir(MODEL_27B),
    reason="set PONYEXL3_MODEL_27B to a 27B EXL3 checkpoint",
)
def test_mlp_monolith_layer0_parity_27b():
    model_dir = MODEL_27B
    os.environ["EXL3_MLP_MONO"] = "0"
    model_ref, _ = load_model(model_dir, engine="exl3", warm=True, verbose=False)

    os.environ["EXL3_MLP_MONO"] = "1"
    model_mono, _ = load_model(model_dir, engine="exl3", warm=True, verbose=False)
    assert isinstance(model_mono.layers[0].mlp, EXL3MLPMonolith)

    rng = np.random.default_rng(7)
    x = mx.array(rng.standard_normal((1, 1, 5120)).astype(np.float16))
    y_ref = model_ref.layers[0].mlp(x)
    y_mono = model_mono.layers[0].mlp(x)
    mx.eval(y_ref, y_mono)
    diff = float(mx.max(mx.abs(y_ref.astype(mx.float32) - y_mono.astype(mx.float32))))
    assert diff < 0.08, f"layer0 MLP max diff {diff}"
