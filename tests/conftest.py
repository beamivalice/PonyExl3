"""Shared fixtures for EXL3 correctness tests."""

from __future__ import annotations

import os

import pytest

from ponyexl3.mlx._parity import assert_allclose_mlx, assert_allclose_np, mlx_available
from ponyexl3.ref.synthetic import make_exl3_layer
from ponyexl3.testing import require_finite

MODEL_2B = os.environ.get("PONYEXL3_MODEL_2B", "")
MODEL_27B = os.environ.get("PONYEXL3_MODEL_27B", "")

__all__ = ["MODEL_2B", "MODEL_27B", "make_exl3_layer", "require_finite"]


@pytest.fixture
def assert_close():
    return assert_allclose_np


@pytest.fixture
def assert_close_mlx():
    return assert_allclose_mlx


@pytest.fixture
def exl3_layer_128():
    return make_exl3_layer(k=4, in_features=128, out_features=128, seed=11)


@pytest.fixture
def skip_without_mlx():
    if not mlx_available():
        pytest.skip("mlx not installed")


@pytest.fixture(autouse=True)
def _clear_exl3_layer_caches():
    from ponyexl3.mlx.layer_state import clear_layer_caches

    clear_layer_caches()
    yield
    clear_layer_caches()
