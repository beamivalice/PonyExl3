"""EXL3 fused batched GEMM forward — Phase 3b (grid-parallel batch)."""

from __future__ import annotations

from typing import Any

import numpy as np
import mlx.core as mx

from ponyexl3.mlx.gemv_metal import inner_gemm_mlx
from ponyexl3.mlx.layer_state import layer_runtime_mlx
from ponyexl3.ref.layer import EXL3Layer

PREFILL_ROW_LIMIT = 144


def linear_forward_gemm_mlx(
    layer: EXL3Layer,
    x: Any,
    *,
    validate: bool = True,
) -> mx.array:
    """Batched forward via fused Metal GEMM (v5 grid-parallel batch, no materialize)."""
    if validate:
        layer.validate()

    rt = layer_runtime_mlx(layer)
    x_np = np.asarray(x, dtype=np.float16)
    orig_shape = x_np.shape
    rows = int(np.prod(orig_shape[:-1]))
    if rows < 1:
        raise ValueError("empty batch")
    if rows > PREFILL_ROW_LIMIT:
        raise ValueError(f"prefill rows {rows} > {PREFILL_ROW_LIMIT}")

    x2d = mx.array(x_np.reshape(rows, layer.in_features))
    xh = rt.prepare_xh(x2d)
    y = inner_gemm_mlx(xh, rt.trellis, rt.k, rt.cb).astype(mx.float16)
    y = rt.finish_y(y)
    if rt.bias is not None:
        y = y + rt.bias
    return y.reshape(orig_shape[:-1] + (layer.out_features,))
