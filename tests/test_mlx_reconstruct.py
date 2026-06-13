"""MLX weight reconstruction parity vs CPU reference."""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.reconstruct import reconstruct_inner_mlx, reconstruct_public_mlx
from ponyexl3.ref.reconstruct import reconstruct_inner, reconstruct_public_weights
from ponyexl3.mlx._parity import assert_allclose_mlx
from ponyexl3.ref.synthetic import make_exl3_layer, make_trellis
from ponyexl3.testing import require_finite

pytestmark = pytest.mark.ponyexl3


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("mcg_mul1", [(False, False), (True, False), (False, True)])
def test_reconstruct_inner_mlx_matches_ref(k: int, mcg_mul1: tuple[bool, bool]):
    mcg, mul1 = mcg_mul1
    trellis = make_trellis(k, 64, 128, seed=k + int(mcg) + 2 * int(mul1))
    ref = reconstruct_inner(trellis, k, mcg=mcg, mul1=mul1)
    require_finite(ref, label=f"inner k={k}")
    got = reconstruct_inner_mlx(trellis, k, mcg=mcg, mul1=mul1)
    assert_allclose_mlx(got, ref, atol=1e-7, label=f"inner k={k}")


@pytest.mark.parametrize("k", [3, 4])
def test_reconstruct_public_mlx_matches_ref(k: int):
    layer = make_exl3_layer(k=k, in_features=128, out_features=256, seed=k + 40)
    ref = reconstruct_public_weights(
        layer.trellis,
        layer.suh,
        layer.svh,
        layer.k,
        mcg=layer.mcg,
        mul1=layer.mul1,
    )
    got = reconstruct_public_mlx(layer)
    assert_allclose_mlx(got, ref, atol=0.07, label=f"public k={k}")
