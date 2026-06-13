"""MLX LOP3 + Metal codebook decode parity vs CPU reference."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.codebook import lop3_b32_mlx
from ponyexl3.mlx.metal_kernels import decode_codewords_mlx
from ponyexl3.ref.codebook import CodebookMode, decode_3inst, lop3_b32
from ponyexl3.mlx._parity import assert_allclose_mlx
from ponyexl3.testing import require_finite

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]


def test_lop3_b32_mlx_matches_ref():
    a = np.uint32(0xA5A5A5A5)
    b = np.uint32(0x8FFF8FFF)
    c = np.uint32(0x3B603B60)
    ref = int(lop3_b32(a, b, c, 0x6A))
    got = int(np.array(lop3_b32_mlx(mlx.array(a), mlx.array(b), mlx.array(c), 0x6A)))
    assert got == ref


@pytest.mark.parametrize("cb", list(CodebookMode))
@pytest.mark.parametrize("n", [1, 17, 256])
def test_decode_codewords_mlx_matches_ref(cb: CodebookMode, n: int):
    rng = np.random.default_rng(n + int(cb))
    words = rng.integers(0, 65536, size=n, dtype=np.uint32)
    ref = np.array([decode_3inst(int(w), cb) for w in words], dtype=np.float16)
    require_finite(ref, label="codebook ref")
    got = decode_codewords_mlx(mlx.array(words), cb)
    assert_allclose_mlx(got, ref, atol=0.05, label=f"decode cb={cb.name}")
