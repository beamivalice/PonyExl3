"""MLX Hadamard helpers for EXL3 — parity with ``exl3.ref.hadamard``."""

from __future__ import annotations

import math

import mlx.core as mx

HAD_DIM = 128
HAD_SCALE = 1.0 / math.sqrt(HAD_DIM)


def _scale_blocks(x: mx.array, scale: mx.array) -> mx.array:
    """Multiply each 128-wide block by ``scale`` (flat length = n_blocks * 128)."""
    n_blocks = x.shape[-2]
    s = scale.astype(mx.float32).reshape(1, n_blocks, HAD_DIM)
    return x * s


def had_r_128_mlx(
    x: mx.array,
    *,
    pre_scale: mx.array | None = None,
    post_scale: mx.array | None = None,
    r_scale: float = 1.0,
) -> mx.array:
    """Right-side 128-point Hadamard along the last dimension (activation path)."""
    if x.ndim != 2:
        raise ValueError("had_r_128_mlx expects 2D input (rows, features)")
    rows, features = x.shape
    if features % HAD_DIM != 0:
        raise ValueError(f"feature dim {features} not divisible by {HAD_DIM}")

    n_blocks = features // HAD_DIM
    out_dtype = x.dtype
    x32 = x.astype(mx.float32).reshape(rows, n_blocks, HAD_DIM)

    if pre_scale is not None:
        x32 = _scale_blocks(x32, pre_scale)

    x32 = mx.hadamard_transform(x32, scale=HAD_SCALE)

    if post_scale is not None:
        x32 = _scale_blocks(x32, post_scale)

    if r_scale != 1.0:
        x32 = x32 * r_scale

    return x32.reshape(rows, features).astype(out_dtype)


def preapply_had_left_mlx(x: mx.array, had_dim: int = HAD_DIM) -> mx.array:
    """Apply Hadamard on the left for each row block — ``preapply_had_l``."""
    k, n = x.shape
    if k % had_dim != 0:
        raise ValueError(f"row dim {k} not divisible by {had_dim}")

    out_dtype = x.dtype
    n_blocks = k // had_dim
    # ``had @ block`` on row blocks — transform along the 128 axis, not columns.
    x32 = x.astype(mx.float32).reshape(n_blocks, had_dim, n)
    x32 = mx.transpose(x32, (0, 2, 1))
    x32 = mx.hadamard_transform(x32, scale=HAD_SCALE)
    x32 = mx.transpose(x32, (0, 2, 1))
    return x32.reshape(k, n).astype(out_dtype)


def preapply_had_right_mlx(x: mx.array, had_dim: int = HAD_DIM) -> mx.array:
    """Apply Hadamard on the right for each column block — ``preapply_had_r``."""
    k, n = x.shape
    if n % had_dim != 0:
        raise ValueError(f"column dim {n} not divisible by {had_dim}")

    out_dtype = x.dtype
    n_blocks = n // had_dim
    x32 = x.astype(mx.float32).reshape(k, n_blocks, had_dim)
    x32 = mx.hadamard_transform(x32, scale=HAD_SCALE)
    return x32.reshape(k, n).astype(out_dtype)
