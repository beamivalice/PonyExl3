"""EXL3 fused GEMV/GEMM forward — Phase 2b/3b."""

from __future__ import annotations

from typing import Any

import numpy as np
import mlx.core as mx

from ponyexl3.mlx.exl3_qmm import PREFILL_ROW_LIMIT, linear_forward_gemm_mlx
from ponyexl3.mlx.gemv_metal import inner_gemv_mlx
from ponyexl3.mlx.layer_state import layer_runtime_mlx
from ponyexl3.ref.layer import EXL3Layer


def linear_forward_fused_mlx(
    layer: EXL3Layer,
    x: Any,
    *,
    validate: bool = True,
) -> mx.array:
    """Fused Metal forward for M=1 (GEMV) or M≤144 (GEMM)."""
    x_np = np.asarray(x, dtype=np.float16)
    rows = int(np.prod(x_np.shape[:-1]))
    if rows == 1:
        return linear_forward_gemv_mlx(layer, x, validate=validate)
    if rows <= PREFILL_ROW_LIMIT:
        return linear_forward_gemm_mlx(layer, x, validate=validate)
    raise ValueError(f"batch {rows} exceeds fused prefill limit {PREFILL_ROW_LIMIT}")


def linear_forward_gemv_mlx(
    layer: EXL3Layer,
    x: Any,
    *,
    validate: bool = True,
) -> mx.array:
    """M=1 forward via fused Metal GEMV."""
    if validate:
        layer.validate()

    rt = layer_runtime_mlx(layer)
    x_np = np.asarray(x, dtype=np.float16)
    orig_shape = x_np.shape
    rows = int(np.prod(orig_shape[:-1]))
    if rows != 1:
        raise ValueError(f"gemv path requires batch=1, got shape {orig_shape}")

    x1d = mx.array(x_np.reshape(layer.in_features))
    xh = rt.prepare_xh(x1d.reshape(1, layer.in_features)).reshape(-1)
    y = inner_gemv_mlx(xh, rt.trellis, rt.k, rt.cb).reshape(1, layer.out_features).astype(
        mx.float16
    )
    y = rt.finish_y(y)
    if rt.bias is not None:
        y = y + rt.bias
    return y.reshape(orig_shape[:-1] + (layer.out_features,))
