"""MLX prefill forward — decode-once inner weight + native ``matmul``."""

from __future__ import annotations

from typing import Any

import numpy as np
import mlx.core as mx

from ponyexl3.mlx.layer_state import inner_weight_mlx, layer_runtime_mlx
from ponyexl3.mlx.ops import prefill_matmul_mlx
from ponyexl3.ref.layer import EXL3Layer


def linear_forward_matmul_mlx(
    layer: EXL3Layer,
    x: Any,
    *,
    validate: bool = True,
    use_weight_cache: bool = True,
    use_compile: bool = True,
) -> mx.array:
    """Prefill path: cache decoded inner ``W``, then compiled MLX ``matmul`` block."""
    if validate:
        layer.validate()

    rt = layer_runtime_mlx(layer)
    x_np = np.asarray(x, dtype=np.float16)
    orig_shape = x_np.shape
    rows = int(np.prod(orig_shape[:-1]))
    x2d = mx.array(x_np.reshape(rows, layer.in_features))

    w = inner_weight_mlx(layer, use_cache=use_weight_cache)
    y = prefill_matmul_mlx(x2d, w, rt.suh, rt.svh, use_compile=use_compile)

    if rt.bias is not None:
        y = y + rt.bias
    return y.reshape(orig_shape[:-1] + (layer.out_features,))
