"""Hadamard helpers for EXL3 — matches exllamav3 had_k/had_n = 128 transforms."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import DTypeLike

HAD_DIM = 128
HAD_SCALE = 1.0 / math.sqrt(HAD_DIM)


def sylvester_hadamard(n: int, dtype: DTypeLike = np.float32) -> np.ndarray:
    """Build Sylvester Hadamard matrix of order n (n must be a power of two)."""
    if n == 1:
        return np.ones((1, 1), dtype=dtype)
    if n % 2 != 0:
        raise ValueError("n must be a power of two")
    h_half = sylvester_hadamard(n // 2, dtype=dtype)
    top = np.concatenate([h_half, h_half], axis=1)
    bottom = np.concatenate([h_half, -h_half], axis=1)
    return np.concatenate([top, bottom], axis=0)


def hadamard_128(dtype: DTypeLike = np.float32) -> np.ndarray:
    return sylvester_hadamard(HAD_DIM, dtype=dtype) * HAD_SCALE


def preapply_had_left(x: np.ndarray, had_dim: int = HAD_DIM) -> np.ndarray:
    """Apply Hadamard on the left for each row block — preapply_had_l."""
    k, _n = x.shape
    had = hadamard_128(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    for i in range(0, k, had_dim):
        out[i : i + had_dim] = had @ x[i : i + had_dim].astype(np.float32)
    return out.astype(x.dtype, copy=False)


def preapply_had_right(x: np.ndarray, had_dim: int = HAD_DIM) -> np.ndarray:
    """Apply Hadamard on the right for each column block — preapply_had_r."""
    _k, n = x.shape
    had = hadamard_128(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    for j in range(0, n, had_dim):
        out[:, j : j + had_dim] = x[:, j : j + had_dim].astype(np.float32) @ had
    return out.astype(x.dtype, copy=False)


def had_r_128(
    x: np.ndarray,
    *,
    pre_scale: np.ndarray | None = None,
    post_scale: np.ndarray | None = None,
    r_scale: float = 1.0,
) -> np.ndarray:
    """
  Right-side 128-point Hadamard along the last dimension.

  Mirrors ext.had_r_128 usage in LinearEXL3.reconstruct_hgemm:
    - pre_scale (suh): multiply input blocks before transform
    - post_scale (svh): multiply output blocks after transform
  """
    if x.ndim != 2:
        raise ValueError("had_r_128 expects 2D input (rows, features)")
    _rows, features = x.shape
    if features % HAD_DIM != 0:
        raise ValueError(f"feature dim {features} not divisible by {HAD_DIM}")

    had = hadamard_128(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    for j in range(0, features, HAD_DIM):
        block = x[:, j : j + HAD_DIM].astype(np.float32)
        if pre_scale is not None:
            block *= pre_scale[j : j + HAD_DIM].astype(np.float32)
        block = block @ had
        if post_scale is not None:
            block *= post_scale[j : j + HAD_DIM].astype(np.float32)
        out[:, j : j + HAD_DIM] = block * r_scale
    return out.astype(x.dtype, copy=False)
