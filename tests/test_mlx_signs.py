"""MLX sign unpack parity vs CPU reference."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.signs import unpack_sign_bitfield_mlx, unpack_signs_or_pass_mlx
from ponyexl3.ref.signs import unpack_sign_bitfield, unpack_signs_or_pass
from ponyexl3.mlx._parity import assert_allclose_mlx

pytestmark = pytest.mark.ponyexl3


@pytest.mark.parametrize("n_groups", [1, 4, 16, 128])
def test_unpack_sign_bitfield_mlx_matches_ref(n_groups: int):
    rng = np.random.default_rng(n_groups)
    bitfield = rng.integers(-(2**15), 2**15, size=n_groups, dtype=np.int16)

    ref = unpack_sign_bitfield(bitfield)
    got = unpack_sign_bitfield_mlx(mlx.array(bitfield))
    assert_allclose_mlx(got, ref, atol=1e-7, label="bitfield")


@pytest.mark.parametrize("kind", ["float16", "bitfield", "none"])
def test_unpack_signs_or_pass_mlx_matches_ref(kind: str):
    if kind == "none":
        assert unpack_signs_or_pass_mlx(None) is None
        assert unpack_signs_or_pass(None) is None
        return

    rng = np.random.default_rng(7)
    if kind == "float16":
        values = np.where(rng.random(128) > 0.5, -1.0, 1.0).astype(np.float16)
    else:
        values = rng.integers(-(2**15), 2**15, size=8, dtype=np.int16)

    ref = unpack_signs_or_pass(values)
    got = unpack_signs_or_pass_mlx(mlx.array(values))
    assert ref is not None and got is not None
    assert_allclose_mlx(got, ref, atol=1e-7, label=kind)
