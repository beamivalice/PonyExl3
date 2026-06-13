"""Fused Metal GEMV/GEMM parity vs full reconstruct forward."""

from __future__ import annotations

import os

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.exl3_qmv import linear_forward_fused_mlx
from ponyexl3.mlx.forward import linear_forward_reconstruct_mlx
from ponyexl3.mlx.gemv_metal import inner_gemm_mlx, inner_gemv_mlx
from ponyexl3.mlx.decode import decode_packed_trellis_mlx
from ponyexl3.ref.codebook import codebook_mode_from_flags
from ponyexl3.ref.synthetic import make_exl3_layer
from ponyexl3.mlx._parity import assert_allclose_np

MODEL_2B = os.environ.get("PONYEXL3_MODEL_2B", "")

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]


def _inner_vs_decode(layer, xh_np: np.ndarray, *, atol: float, rtol: float, label: str) -> None:
    cb = codebook_mode_from_flags(mcg=layer.mcg, mul1=layer.mul1)
    w_ref = np.array(
        decode_packed_trellis_mlx(
            mlx.array(layer.trellis.astype(np.uint16)), layer.k, cb
        )
    )
    y_ref = xh_np.astype(np.float32) @ w_ref.astype(np.float32)
    y_gemv = np.array(
        inner_gemv_mlx(
            mlx.array(xh_np),
            mlx.array(layer.trellis.astype(np.uint16)),
            layer.k,
            cb,
        )
    ).astype(np.float32)
    assert_allclose_np(y_gemv, y_ref, atol=atol, rtol=rtol, label=label)


@pytest.mark.parametrize("k", [2, 3])
def test_inner_gemv_matches_mlx_decode_matmul_small_k(k: int):
    layer = make_exl3_layer(
        k=k, in_features=128, out_features=256, seed=k, mcg=False
    )
    rng = np.random.default_rng(k)
    xh_np = rng.standard_normal(layer.in_features).astype(np.float16)
    _inner_vs_decode(layer, xh_np, atol=0.5, rtol=0.02, label=f"inner gemv k={k}")


@pytest.mark.parametrize("k", [4, 6, 7, 8])
def test_inner_gemv_matches_mlx_decode_matmul_large_k(k: int):
    layer = make_exl3_layer(
        k=k, in_features=128, out_features=256, seed=k, mcg=True
    )
    rng = np.random.default_rng(k)
    xh_np = rng.standard_normal(layer.in_features).astype(np.float16)
    _inner_vs_decode(layer, xh_np, atol=3.0, rtol=0.05, label=f"inner gemv k={k}")


@pytest.mark.parametrize("rows", [4, 16])
@pytest.mark.parametrize("k", [4])
def test_inner_gemm_matches_mlx_decode_matmul_k4(rows: int, k: int):
    layer = make_exl3_layer(
        k=k,
        in_features=128,
        out_features=256,
        seed=rows + k,
        mcg=True,
    )
    rng = np.random.default_rng(rows + k)
    xh_np = rng.standard_normal((rows, layer.in_features)).astype(np.float16)
    cb = codebook_mode_from_flags(mcg=True, mul1=False)
    w_ref = np.array(
        decode_packed_trellis_mlx(
            mlx.array(layer.trellis.astype(np.uint16)), layer.k, cb
        )
    )
    y_ref = xh_np.astype(np.float32) @ w_ref.astype(np.float32)
    y_gemm = np.array(
        inner_gemm_mlx(
            mlx.array(xh_np),
            mlx.array(layer.trellis.astype(np.uint16)),
            layer.k,
            cb,
        )
    ).astype(np.float32)
    assert_allclose_np(
        y_gemm, y_ref, atol=3.0, rtol=0.05, label=f"inner gemm k={k} rows={rows}"
    )


@pytest.mark.parametrize("rows", [16])
def test_inner_gemm_matches_mlx_decode_matmul_k6(rows: int):
    layer = make_exl3_layer(
        k=6,
        in_features=128,
        out_features=256,
        seed=rows + 6,
        mcg=True,
    )
    rng = np.random.default_rng(rows + 6)
    xh_np = rng.standard_normal((rows, layer.in_features)).astype(np.float16)
    cb = codebook_mode_from_flags(mcg=True, mul1=False)
    w_ref = np.array(
        decode_packed_trellis_mlx(
            mlx.array(layer.trellis.astype(np.uint16)), layer.k, cb
        )
    )
    y_ref = xh_np.astype(np.float32) @ w_ref.astype(np.float32)
    y_gemm = np.array(
        inner_gemm_mlx(
            mlx.array(xh_np),
            mlx.array(layer.trellis.astype(np.uint16)),
            layer.k,
            cb,
        )
    ).astype(np.float32)
    assert_allclose_np(
        y_gemm, y_ref, atol=3.0, rtol=0.05, label=f"inner gemm k=6 rows={rows}"
    )


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("rows", [1, 4])
def test_linear_forward_fused_matches_reconstruct_small_k(k: int, rows: int):
    layer = make_exl3_layer(
        k=k, in_features=128, out_features=256, seed=20 + k + rows, mcg=False
    )
    rng = np.random.default_rng(rows)
    x = rng.standard_normal((rows, layer.in_features)).astype(np.float16)
    ref = np.array(linear_forward_reconstruct_mlx(layer, x))
    got = np.array(linear_forward_fused_mlx(layer, x))
    assert_allclose_np(got, ref, atol=2.5, rtol=0.5, label=f"fused k={k} rows={rows}")


@pytest.mark.parametrize("k", [4, 6])
@pytest.mark.parametrize("rows", [1, 4])
def test_linear_forward_fused_matches_reconstruct_mcg(k: int, rows: int):
    layer = make_exl3_layer(
        k=k, in_features=128, out_features=256, seed=20 + k + rows, mcg=True
    )
    rng = np.random.default_rng(rows)
    x = rng.standard_normal((rows, layer.in_features)).astype(np.float16)
    ref = np.array(linear_forward_reconstruct_mlx(layer, x))
    got = np.array(linear_forward_fused_mlx(layer, x))
    assert_allclose_np(got, ref, atol=2.5, rtol=0.5, label=f"fused k={k} rows={rows}")


@pytest.mark.skipif(
    not MODEL_2B or not __import__("pathlib").Path(f"{MODEL_2B}/quantization_config.json").is_file(),
    reason="set PONYEXL3_MODEL_2B to a Qwen3.5-2B EXL3 checkpoint",
)
@pytest.mark.parametrize(
    "module_key",
    [
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "lm_head",
    ],
)
@pytest.mark.parametrize("rows", [1, 4])
def test_fused_on_qwen2b_layers(module_key: str, rows: int):
    from ponyexl3.ref.loader import load_exl3_layer

    layer = load_exl3_layer(MODEL_2B, module_key)
    rng = np.random.default_rng(hash(module_key) % 1000 + rows)
    x = rng.standard_normal((rows, layer.in_features)).astype(np.float16)
    ref = np.array(linear_forward_reconstruct_mlx(layer, x))
    got = np.array(linear_forward_fused_mlx(layer, x))
    diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
    both_nan = np.isnan(ref) & np.isnan(got)
    diff[both_nan] = 0.0
    assert float(np.nanmax(diff)) <= 2.5
