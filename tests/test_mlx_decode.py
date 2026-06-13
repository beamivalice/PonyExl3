"""MLX tile + trellis decode parity vs CPU reference."""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.decode import decode_packed_tile_mlx, decode_packed_trellis_mlx
from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.decode import decode_packed_tile, decode_packed_trellis
from ponyexl3.ref.synthetic import make_trellis
from ponyexl3.mlx._parity import assert_allclose_mlx
from ponyexl3.testing import require_finite

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_decode_packed_tile_mlx_matches_ref(k: int):
    trellis = make_trellis(k, 32, 32, seed=k)
    packed = trellis[0, 0]
    ref = decode_packed_tile(packed, k, CodebookMode.DEFAULT)
    require_finite(ref, label="tile ref")
    got = decode_packed_tile_mlx(mlx.array(packed), k, CodebookMode.DEFAULT)
    assert_allclose_mlx(got, ref, atol=0.05, label=f"tile k={k}")


@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("mcg_mul1", [(False, False), (True, False)])
def test_decode_packed_trellis_mlx_matches_ref(k: int, mcg_mul1: tuple[bool, bool]):
    from ponyexl3.ref.codebook import codebook_mode_from_flags

    mcg, mul1 = mcg_mul1
    trellis = make_trellis(k, 64, 64, seed=k + int(mcg))
    cb = codebook_mode_from_flags(mcg=mcg, mul1=mul1)
    ref = decode_packed_trellis(trellis, k, cb)
    require_finite(ref, label="trellis ref")
    got = decode_packed_trellis_mlx(mlx.array(trellis), k, cb)
    assert_allclose_mlx(got, ref, atol=0.05, label=f"trellis k={k}")
