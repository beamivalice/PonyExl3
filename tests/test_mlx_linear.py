import numpy as np
import pytest

pytestmark = pytest.mark.ponyexl3

mlx = pytest.importorskip("mlx.core", reason="mlx not installed")

from ponyexl3.mlx.linear import compare_numpy_vs_mlx
from ponyexl3.ref.synthetic import make_exl3_layer


def test_mlx_forward_matches_numpy():
    layer = make_exl3_layer(k=4, in_features=128, out_features=128, seed=1)
    x = np.random.default_rng(1).standard_normal((3, 128)).astype(np.float16)
    stats = compare_numpy_vs_mlx(layer, x, atol=2.5)
    assert stats["ok"], stats
