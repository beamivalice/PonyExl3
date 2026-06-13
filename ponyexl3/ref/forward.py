"""Naive EXL3 linear forward via reconstruct — Phase-1 inference path."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from .hadamard import had_r_128
from .layer import EXL3Layer
from .reconstruct import reconstruct_inner
from .signs import unpack_signs_or_pass


def linear_forward_reconstruct(
    layer: EXL3Layer,
    x: np.ndarray,
    *,
    dtype: DTypeLike = np.float16,
) -> np.ndarray:
    """
  Reference forward matching LinearEXL3.reconstruct_hgemm.

  Slow but correct — golden path for MLX kernel validation.
  """
    layer.validate()
    x = np.asarray(x, dtype=dtype)
    orig_shape = x.shape
    rows = int(np.prod(orig_shape[:-1]))
    x2d: np.ndarray = np.reshape(x, (rows, layer.in_features))

    suh = unpack_signs_or_pass(layer.suh)
    svh = unpack_signs_or_pass(layer.svh)

    xh = had_r_128(x2d.astype(np.float32), pre_scale=suh, r_scale=1.0).astype(dtype)
    w = reconstruct_inner(
        layer.trellis,
        layer.k,
        mcg=layer.mcg,
        mul1=layer.mul1,
    )
    y = (xh.astype(np.float32) @ w.astype(np.float32)).astype(dtype)
    y = had_r_128(y.astype(np.float32), post_scale=svh, r_scale=1.0).astype(dtype)

    if layer.bias is not None:
        y = y + layer.bias.astype(dtype)

    out_shape = orig_shape[:-1] + (layer.out_features,)
    out: np.ndarray = np.reshape(y, out_shape)
    return out
