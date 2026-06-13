"""Striped fast forward parity vs full reconstruct path."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.forward import linear_forward_mlx, linear_forward_reconstruct_mlx
from ponyexl3.mlx.stripe import linear_forward_stripe_mlx
from ponyexl3.ref.synthetic import make_exl3_layer
from ponyexl3.mlx._parity import assert_allclose_np

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]


@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_stripe_forward_matches_reconstruct_mlx(k: int, rows: int):
    layer = make_exl3_layer(k=k, in_features=256, out_features=384, seed=k, mcg=True)
    rng = np.random.default_rng(rows + k)
    x = rng.standard_normal((rows, layer.in_features)).astype(np.float16)
    ref = np.array(linear_forward_reconstruct_mlx(layer, x))
    got = np.array(linear_forward_stripe_mlx(layer, x))
    assert_allclose_np(got, ref, atol=0.07, rtol=0.1, label=f"stripe k={k} rows={rows}")


def test_dispatch_default_is_fast():
    layer = make_exl3_layer(k=4, in_features=128, out_features=256, seed=1, mcg=False)
    x = np.random.default_rng(0).standard_normal((2, 128)).astype(np.float16)
    ref = np.array(linear_forward_reconstruct_mlx(layer, x))
    got = np.array(linear_forward_mlx(layer, x))
    assert_allclose_np(got, ref, atol=0.07, rtol=0.1, label="dispatch default")


def test_warm_layer_caches_inner():
    from ponyexl3.mlx.layer_state import inner_weight_mlx, warm_layer_mlx

    layer = make_exl3_layer(k=4, in_features=128, out_features=128, seed=3, mcg=True)
    warm_layer_mlx(layer)
    w1 = inner_weight_mlx(layer)
    w2 = inner_weight_mlx(layer)
    assert w1 is w2


def test_dispatch_m1_uses_gemv():
    layer = make_exl3_layer(k=4, in_features=128, out_features=256, seed=2, mcg=False)
    x = np.random.default_rng(0).standard_normal((1, 128)).astype(np.float16)
    ref = np.array(linear_forward_reconstruct_mlx(layer, x))
    got = np.array(linear_forward_mlx(layer, x))
    assert_allclose_np(got, ref, atol=2.5, rtol=0.5, label="dispatch gemv m=1")
