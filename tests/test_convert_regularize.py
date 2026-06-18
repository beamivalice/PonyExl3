"""Post-M4 regularization helpers."""

from __future__ import annotations

import numpy as np

from ponyexl3.convert.hessian import public_matrix_to_inner
from ponyexl3.convert.regularize import (
    block_rms_np,
    g_scale_gss,
    regularize_public_weight,
    sample_tile_matrix,
)


def test_block_rms_np_matches_direct_mean_square():
    rng = np.random.default_rng(201)
    x = rng.standard_normal((65, 17)).astype(np.float32)

    rows = block_rms_np(x, axis=1, keepdims=True, blocksize=7)
    cols = block_rms_np(x, axis=0, keepdims=False, blocksize=11)

    np.testing.assert_allclose(rows, np.sqrt(np.mean(x * x, axis=1, keepdims=True)), rtol=1e-6)
    np.testing.assert_allclose(cols, np.sqrt(np.mean(x * x, axis=0)), rtol=1e-6)


def test_regularize_public_weight_matches_public_to_inner_inverse():
    rng = np.random.default_rng(202)
    public = (rng.standard_normal((128, 256)) * 0.05).astype(np.float32)

    regularized = regularize_public_weight(public, seed=7)
    inverted = public_matrix_to_inner(public, suh=regularized.suh, svh=regularized.svh)

    np.testing.assert_allclose(inverted, regularized.inner, rtol=1e-5, atol=2e-5)
    assert regularized.suh.shape == (128,)
    assert regularized.svh.shape == (256,)


def test_sample_tile_matrix_and_gss():
    rng = np.random.default_rng(203)
    weight = rng.standard_normal((32, 48)).astype(np.float32)
    sampled = sample_tile_matrix(weight, width=2)

    assert sampled.shape == (16 * 6, 16)

    result = g_scale_gss(lambda scale: (scale - 1.27) ** 2)
    assert abs(result.scale - 1.27) < 0.02
    assert result.evaluations >= 2
