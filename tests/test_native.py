"""Native-MLX engine conversions: exact fold, fused MLP, lossy requant guards."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")
import mlx.nn as nn_mlx

from ponyexl3.mlx.exl3_linear import EXL3Linear
from ponyexl3.mlx.native import (
    FusedSwiGLU,
    _fuse_two_linears,
    folded_linear_from_exl3,
    layer_error,
    public_weight_chunks,
    public_weight_mlx,
    quantized_linear_from_exl3,
)
from ponyexl3.ref.synthetic import make_exl3_layer

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]


def _layer(key: str, **kw):
    layer = make_exl3_layer(
        k=kw.pop("k", 4),
        in_features=kw.pop("in_features", 256),
        out_features=kw.pop("out_features", 384),
        seed=kw.pop("seed", 21),
        mcg=True,
        **kw,
    )
    layer.key = key
    return layer


def test_public_weight_chunking_is_exact():
    layer = _layer("native.chunk")
    w_full = np.array(public_weight_mlx(layer))
    w_chunked = np.concatenate(
        [np.array(c) for c in public_weight_chunks(layer, chunk_cols=128)], axis=1
    )
    np.testing.assert_array_equal(w_full, w_chunked)


def test_fold16_preserves_exl3_accuracy():
    """The fold may only deviate by fp16 rounding of the folded weight —
    orders of magnitude below trellis quantization noise."""
    layer = _layer("native.fold")
    err = layer_error(layer, folded_linear_from_exl3(layer), "fold16")
    assert err.rel < 2e-3, f"fold16 must be fp16-rounding exact, got {err.rel:.2e}"


def test_requant_is_lossy_and_ranked():
    """w8 must sit well below trellis noise; w4 is expected to be bad — this
    test documents the ranking so nobody ships w4 thinking it is free."""
    layer = _layer("native.requant")
    fold = layer_error(layer, folded_linear_from_exl3(layer), "fold16").rel
    w8 = layer_error(layer, quantized_linear_from_exl3(layer, bits=8), "w8a16").rel
    w4 = layer_error(layer, quantized_linear_from_exl3(layer, bits=4), "w4a16").rel
    assert fold < w8 < w4
    assert w8 < 0.02
    assert w4 > 0.02  # genuinely lossy — never silently treat as exact


def test_fused_swiglu_matches_unfused():
    layer_g = _layer("native.fuse.gate", out_features=512)
    layer_u = _layer("native.fuse.up", out_features=512, seed=22)
    layer_d = _layer("native.fuse.down", in_features=512, out_features=256, seed=23)
    gate = folded_linear_from_exl3(layer_g)
    up = folded_linear_from_exl3(layer_u)
    down = folded_linear_from_exl3(layer_d)

    fused = FusedSwiGLU(_fuse_two_linears(gate, up), down)
    x = (mlx.random.normal((4, 256), key=mlx.random.key(1)) * 0.1).astype(mlx.float16)
    y_ref = down(nn_mlx.silu(gate(x)) * up(x))
    y_fused = fused(x)
    d = np.abs(np.array(y_ref).astype(np.float32) - np.array(y_fused).astype(np.float32))
    assert d.max() < 1e-2, f"fused MLP diverged: {d.max()}"


def test_folded_linear_matches_exl3_module_outputs():
    layer = _layer("native.fwd", bias=True)
    mod_exact = EXL3Linear(layer)
    mod_fold = folded_linear_from_exl3(layer)
    x = (mlx.random.normal((8, 256), key=mlx.random.key(2)) * 0.5).astype(mlx.float16)
    y_a = np.array(mod_exact(x)).astype(np.float32)
    y_b = np.array(mod_fold(x)).astype(np.float32)
    scale = np.abs(y_a).max()
    assert np.abs(y_a - y_b).max() < 0.01 * scale + 0.05
