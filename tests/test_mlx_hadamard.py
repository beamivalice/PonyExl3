"""MLX Hadamard parity vs CPU reference."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.hadamard import (
    HAD_DIM,
    had_r_128_mlx,
    preapply_had_left_mlx,
    preapply_had_right_mlx,
)
from ponyexl3.ref.hadamard import had_r_128, preapply_had_left, preapply_had_right
from ponyexl3.mlx._parity import assert_allclose_mlx

pytestmark = pytest.mark.ponyexl3


@pytest.mark.parametrize("rows", [1, 3, 17])
@pytest.mark.parametrize("n_blocks", [1, 2, 4])
def test_had_r_128_mlx_matches_ref(rows: int, n_blocks: int):
    rng = np.random.default_rng(rows * 10 + n_blocks)
    features = n_blocks * HAD_DIM
    x = rng.standard_normal((rows, features)).astype(np.float16)

    ref = had_r_128(x.astype(np.float32), r_scale=1.0).astype(np.float16)
    got = had_r_128_mlx(mlx.array(x), r_scale=1.0)
    assert_allclose_mlx(got, ref, atol=1e-2, label="had_r_128")


@pytest.mark.parametrize("rows", [2, 8])
def test_had_r_128_mlx_with_scales_matches_ref(rows: int):
    rng = np.random.default_rng(rows + 99)
    features = 256
    x = rng.standard_normal((rows, features)).astype(np.float16)
    pre = np.where(rng.random(features) > 0.5, -1.0, 1.0).astype(np.float16)
    post = np.where(rng.random(features) > 0.5, -1.0, 1.0).astype(np.float16)

    ref = had_r_128(
        x.astype(np.float32),
        pre_scale=pre,
        post_scale=post,
        r_scale=0.5,
    ).astype(np.float16)
    got = had_r_128_mlx(
        mlx.array(x),
        pre_scale=mlx.array(pre),
        post_scale=mlx.array(post),
        r_scale=0.5,
    )
    assert_allclose_mlx(got, ref, atol=1e-2, label="had_r_128 scaled")


@pytest.mark.parametrize("shape", [(128, 64), (256, 128), (512, 256)])
def test_preapply_had_left_mlx_matches_ref(shape):
    rng = np.random.default_rng(sum(shape))
    x = rng.standard_normal(shape).astype(np.float16)
    ref = preapply_had_left(x.astype(np.float32)).astype(np.float16)
    got = preapply_had_left_mlx(mlx.array(x))
    assert_allclose_mlx(got, ref, atol=1e-2, label="preapply_had_left")


@pytest.mark.parametrize("shape", [(64, 128), (128, 256), (256, 512)])
def test_preapply_had_right_mlx_matches_ref(shape):
    rng = np.random.default_rng(sum(shape))
    x = rng.standard_normal(shape).astype(np.float16)
    ref = preapply_had_right(x.astype(np.float32)).astype(np.float16)
    got = preapply_had_right_mlx(mlx.array(x))
    assert_allclose_mlx(got, ref, atol=1e-2, label="preapply_had_right")
