"""Route EXL3 forward to the best MLX path for batch size and layer size."""

from __future__ import annotations

from typing import Any

import numpy as np

from ponyexl3.mlx.exl3_qmm import PREFILL_ROW_LIMIT, linear_forward_gemm_mlx
from ponyexl3.mlx.exl3_qmv import linear_forward_gemv_mlx
from ponyexl3.mlx.prefill import linear_forward_matmul_mlx
from ponyexl3.mlx.stripe import DEFAULT_STRIPE_COLS, linear_forward_stripe_mlx
from ponyexl3.ref.layer import EXL3Layer

STRIPE_WEIGHT_BYTES = 64 * 1024 * 1024


def _use_stripe(layer: EXL3Layer) -> bool:
    nbytes = layer.in_features * layer.out_features * 2
    return nbytes > STRIPE_WEIGHT_BYTES


def linear_forward_fast_mlx(
    layer: EXL3Layer,
    x: Any,
    *,
    stripe_cols: int = DEFAULT_STRIPE_COLS,
    validate: bool = True,
):
    """
    Fast MLX serving forward.

    - M=1: fused Metal GEMV (no weight materialize)
    - M>1, layer fits in memory: decode-once cache + compiled MLX ``matmul``
    - M>1, huge layer, M≤144: fused Metal GEMM (v5 grid-parallel batch)
    - M>1, huge layer, M>144: striped decode + MLX ``matmul`` per chunk
    """
    from ponyexl3.mlx.forward import linear_forward_reconstruct_mlx

    x_np = np.asarray(x, dtype=np.float16)
    rows = int(np.prod(x_np.shape[:-1]))

    if rows == 1:
        return linear_forward_gemv_mlx(layer, x, validate=validate)
    if _use_stripe(layer):
        if rows <= PREFILL_ROW_LIMIT:
            return linear_forward_gemm_mlx(layer, x, validate=validate)
        return linear_forward_stripe_mlx(layer, x, stripe_cols=stripe_cols, validate=validate)
    if rows <= PREFILL_ROW_LIMIT:
        return linear_forward_matmul_mlx(layer, x, validate=validate)
    return linear_forward_reconstruct_mlx(layer, x, validate=validate)


def linear_forward_prefill_mlx(
    layer: EXL3Layer,
    x: Any,
    *,
    stripe_cols: int = DEFAULT_STRIPE_COLS,
    validate: bool = True,
):
    """Prefill forward — MLX native paths (matmul cache or fused GEMM)."""
    return linear_forward_fast_mlx(layer, x, stripe_cols=stripe_cols, validate=validate)
