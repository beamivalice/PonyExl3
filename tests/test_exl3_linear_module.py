"""EXL3Linear nn.Module parity vs the functional dispatch forwards."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx._parity import assert_allclose_np
from ponyexl3.mlx.exl3_linear import EXL3Linear
from ponyexl3.mlx.forward import linear_forward_reconstruct_mlx
from ponyexl3.ref.synthetic import make_exl3_layer

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]


@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.parametrize("k", [2, 4])
def test_module_matches_reconstruct(rows: int, k: int):
    layer = make_exl3_layer(
        k=k, in_features=256, out_features=384, seed=11 + k, mcg=True, bias=True
    )
    layer.key = f"test.module.{k}"
    rng = np.random.default_rng(rows)
    x = (rng.standard_normal((rows, 256)) * 0.5).astype(np.float16)
    y_ref = np.array(linear_forward_reconstruct_mlx(layer, x)).astype(np.float32)
    y_mod = np.array(EXL3Linear(layer)(mlx.array(x))).astype(np.float32)
    assert np.isfinite(y_ref).all()
    assert_allclose_np(
        y_mod, y_ref, atol=0.05, rtol=0.02, label=f"EXL3Linear rows={rows} k={k}"
    )


def test_module_keeps_leading_shape():
    layer = make_exl3_layer(k=4, in_features=128, out_features=128, seed=3)
    layer.key = "test.module.shape"
    mod = EXL3Linear(layer)
    x = mlx.zeros((2, 3, 128), dtype=mlx.float16)
    assert mod(x).shape == (2, 3, 128)


def test_module_contributes_no_parameters():
    """Trellis state must stay out of the MLX param tree so strict weight
    loading of the surrounding model skeleton works."""
    from mlx.utils import tree_flatten

    layer = make_exl3_layer(k=4, in_features=128, out_features=128, seed=5)
    layer.key = "test.module.params"
    mod = EXL3Linear(layer)
    assert tree_flatten(mod.parameters()) == []
