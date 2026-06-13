"""Prefill matmul + decode-once cache parity."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.cache import clear_inner_weight_cache
from ponyexl3.mlx.forward import linear_forward_reconstruct_mlx
from ponyexl3.mlx.prefill import linear_forward_matmul_mlx
from ponyexl3.ref.synthetic import make_exl3_layer
from ponyexl3.mlx._parity import assert_allclose_np

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]


@pytest.mark.parametrize("rows", [2, 8, 16])
def test_matmul_prefill_matches_reconstruct(rows: int):
    clear_inner_weight_cache()
    layer = make_exl3_layer(k=4, in_features=128, out_features=256, seed=rows, mcg=True)
    rng = np.random.default_rng(rows)
    x = rng.standard_normal((rows, layer.in_features)).astype(np.float16)
    ref = np.array(linear_forward_reconstruct_mlx(layer, x))
    got = np.array(linear_forward_matmul_mlx(layer, x))
    assert_allclose_np(got, ref, atol=0.07, rtol=0.1, label=f"prefill rows={rows}")


def test_weight_cache_reused():
    clear_inner_weight_cache()
    layer = make_exl3_layer(k=4, in_features=128, out_features=128, seed=99, mcg=False)
    from ponyexl3.mlx.cache import inner_weight_mlx, stripe_weight_mlx

    w1 = inner_weight_mlx(layer)
    w2 = inner_weight_mlx(layer)
    assert w1 is w2

    s1 = stripe_weight_mlx(layer, 0, 128)
    s2 = stripe_weight_mlx(layer, 0, 128)
    assert s1 is s2


def test_compiled_prefill_block_matches_eager():
    from ponyexl3.mlx.layer_state import layer_runtime_mlx
    from ponyexl3.mlx.ops import prefill_matmul_mlx
    from ponyexl3.mlx.cache import inner_weight_mlx

    layer = make_exl3_layer(k=4, in_features=128, out_features=256, seed=8, mcg=True)
    rt = layer_runtime_mlx(layer)
    rng = np.random.default_rng(8)
    x2d = mlx.array(rng.standard_normal((4, 128)).astype(np.float16))
    w = inner_weight_mlx(layer)
    eager = np.array(prefill_matmul_mlx(x2d, w, rt.suh, rt.svh, use_compile=False))
    compiled = np.array(prefill_matmul_mlx(x2d, w, rt.suh, rt.svh, use_compile=True))
    assert_allclose_np(compiled, eager, atol=0.07, rtol=0.1, label="compiled prefill block")


def test_compiled_matmul_matches_eager():
    from ponyexl3.mlx.layer_state import layer_runtime_mlx
    from ponyexl3.mlx.ops import inner_matmul_mlx

    from ponyexl3.mlx.cache import inner_weight_mlx

    layer = make_exl3_layer(k=4, in_features=128, out_features=256, seed=7, mcg=True)
    rt = layer_runtime_mlx(layer)
    rng = np.random.default_rng(7)
    xh_np = rng.standard_normal((4, 128)).astype(np.float16)
    xh = rt.prepare_xh(mlx.array(xh_np))
    w = inner_weight_mlx(layer)
    eager = np.array(inner_matmul_mlx(xh, w, svh=rt.svh, use_compile=False))
    compiled = np.array(inner_matmul_mlx(xh, w, svh=rt.svh, use_compile=True))
    assert_allclose_np(compiled, eager, atol=0.07, rtol=0.1, label="compile vs eager")
