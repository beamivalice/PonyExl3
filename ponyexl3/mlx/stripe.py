"""Striped EXL3 forward — cached stripe decode + MLX ``matmul`` per chunk."""

from __future__ import annotations

from typing import Any

import numpy as np
import mlx.core as mx

from ponyexl3.mlx.layer_state import layer_runtime_mlx, stripe_weight_mlx
from ponyexl3.ref.layer import EXL3Layer

DEFAULT_STRIPE_COLS = 512


def linear_forward_stripe_mlx(
    layer: EXL3Layer,
    x: Any,
    *,
    stripe_cols: int = DEFAULT_STRIPE_COLS,
    validate: bool = True,
    use_stripe_cache: bool = True,
) -> mx.array:
    """Forward via cached chunked trellis decode + MLX ``matmul``."""
    if validate:
        layer.validate()
    if stripe_cols % 128 != 0:
        raise ValueError("stripe_cols must be a multiple of 128")

    rt = layer_runtime_mlx(layer)
    x_np = np.asarray(x, dtype=np.float16)
    orig_shape = x_np.shape
    rows = int(np.prod(orig_shape[:-1]))
    x2d = mx.array(x_np.reshape(rows, layer.in_features))

    xh = rt.prepare_xh(x2d)
    xh_f = xh.astype(mx.float32)

    y = mx.zeros((rows, layer.out_features), dtype=mx.float16)
    for n0 in range(0, layer.out_features, stripe_cols):
        n1 = min(n0 + stripe_cols, layer.out_features)
        n_count = n1 - n0
        if n_count % 16 != 0:
            raise ValueError("stripe boundary misaligned with 16-wide tiles")
        w = stripe_weight_mlx(layer, n0, n_count, use_cache=use_stripe_cache)
        y[:, n0:n1] = (xh_f @ w.astype(mx.float32)).astype(mx.float16)

    y = rt.finish_y(y)
    if rt.bias is not None:
        y = y + rt.bias
    return y.reshape(orig_shape[:-1] + (layer.out_features,))
