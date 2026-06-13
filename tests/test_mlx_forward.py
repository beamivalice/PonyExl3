"""MLX linear forward parity vs CPU golden path."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.forward import linear_forward_mlx, linear_forward_public_mlx
from ponyexl3.ref.forward import linear_forward_reconstruct
from ponyexl3.ref.reconstruct import reconstruct_public_weights
from ponyexl3.mlx._parity import assert_allclose_np
from ponyexl3.ref.synthetic import make_exl3_layer
from ponyexl3.testing import require_finite

pytestmark = pytest.mark.ponyexl3


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("k", [3, 4])
def test_linear_forward_mlx_matches_reconstruct(batch: int, k: int):
    layer = make_exl3_layer(k=k, in_features=128, out_features=128, seed=batch * 10 + k)
    rng = np.random.default_rng(batch + k)
    x = rng.standard_normal((batch, layer.in_features)).astype(np.float16)

    ref = linear_forward_reconstruct(layer, x)
    require_finite(ref, label=f"forward b={batch} k={k}")
    got = np.array(linear_forward_mlx(layer, x))
    rtol = 0.1 if batch == 1 else 0.6
    assert_allclose_np(got, ref, atol=2.5, rtol=rtol, label=f"forward b={batch} k={k}")


@pytest.mark.parametrize("mcg_mul1", [(False, False), (True, False), (False, True)])
def test_linear_forward_mlx_codebook_modes(mcg_mul1: tuple[bool, bool]):
    mcg, mul1 = mcg_mul1
    layer = make_exl3_layer(
        k=4,
        in_features=128,
        out_features=128,
        seed=90 + int(mcg) + 2 * int(mul1),
        mcg=mcg,
        mul1=mul1,
    )
    x = np.random.default_rng(1).standard_normal((1, 128)).astype(np.float16)
    ref = linear_forward_reconstruct(layer, x)
    require_finite(ref, label="codebook mode")
    got = np.array(linear_forward_mlx(layer, x))
    assert_allclose_np(got, ref, atol=2.5, rtol=0.15, label="codebook mode")


def test_linear_forward_public_mlx_matches_public_matmul():
    layer = make_exl3_layer(k=4, in_features=128, out_features=128, seed=55, bias=True)
    x = np.random.default_rng(3).standard_normal((2, 128)).astype(np.float16)

    w_pub = reconstruct_public_weights(
        layer.trellis,
        layer.suh,
        layer.svh,
        layer.k,
        mcg=layer.mcg,
        mul1=layer.mul1,
    )
    ref = (x.astype(np.float32) @ w_pub.astype(np.float32)).astype(np.float16)
    if layer.bias is not None:
        ref = ref + layer.bias.astype(np.float16)
    require_finite(ref, w_pub, label="public matmul")

    got = np.array(linear_forward_public_mlx(layer, x))
    assert_allclose_np(got, ref, atol=0.25, rtol=0.15, label="public matmul")
