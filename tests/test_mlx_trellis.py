"""Metal trellis unpack parity vs CPU reference."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.metal_kernels import unpack_trellis_tiles_mlx
from ponyexl3.ref.trellis import pack_trellis_tile, unpack_trellis_tile
from ponyexl3.mlx._parity import assert_allclose_mlx

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
def test_unpack_trellis_tiles_mlx_matches_ref(k: int):
    rng = np.random.default_rng(k)
    mask = (1 << k) - 1
    encoded = (rng.integers(0, 2**16, size=(3, 256), dtype=np.uint32) & mask).astype(np.uint16)
    packed = np.stack([pack_trellis_tile(encoded[i], k) for i in range(3)], axis=0)

    ref = np.stack([unpack_trellis_tile(packed[i], k) for i in range(3)], axis=0)
    got = unpack_trellis_tiles_mlx(mlx.array(packed), k)
    assert_allclose_mlx(got, ref, atol=1e-7, label=f"unpack k={k}")
