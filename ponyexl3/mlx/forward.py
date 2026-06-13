"""MLX EXL3 linear forward — correctness + fast striped serving path."""

from __future__ import annotations

from typing import Any

import numpy as np

from ponyexl3.mlx.dispatch import linear_forward_fast_mlx
from ponyexl3.mlx.hadamard import had_r_128_mlx
from ponyexl3.mlx.reconstruct import reconstruct_inner_mlx, reconstruct_public_mlx
from ponyexl3.mlx.signs import unpack_signs_or_pass_mlx
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.signs import unpack_signs_or_pass


def linear_forward_mlx(layer: EXL3Layer, x: Any, *, validate: bool = True, fast: bool = True):
    """MLX forward. ``fast=True`` (default) uses striped decode+matmul for serving."""
    if fast:
        return linear_forward_fast_mlx(layer, x, validate=validate)
    return linear_forward_reconstruct_mlx(layer, x, validate=validate)


def linear_forward_reconstruct_mlx(layer: EXL3Layer, x: Any, *, validate: bool = True):
    """Slow path: full trellis decode then matmul (golden / debug)."""
    import mlx.core as mx

    dtype = mx.float16

    if validate:
        layer.validate()

    x_np = np.asarray(x, dtype=np.float16)
    orig_shape = x_np.shape
    rows = int(np.prod(orig_shape[:-1]))
    x2d = mx.array(x_np.reshape(rows, layer.in_features))

    suh = unpack_signs_or_pass(layer.suh)
    svh = unpack_signs_or_pass(layer.svh)
    suh_mx = None if suh is None else unpack_signs_or_pass_mlx(mx.array(suh))
    svh_mx = None if svh is None else unpack_signs_or_pass_mlx(mx.array(svh))

    xh = had_r_128_mlx(x2d.astype(mx.float32), pre_scale=suh_mx, r_scale=1.0).astype(dtype)
    w = reconstruct_inner_mlx(
        layer.trellis,
        layer.k,
        mcg=layer.mcg,
        mul1=layer.mul1,
    )
    y = (xh.astype(mx.float32) @ w.astype(mx.float32)).astype(dtype)
    y = had_r_128_mlx(y.astype(mx.float32), post_scale=svh_mx, r_scale=1.0).astype(dtype)

    if layer.bias is not None:
        y = y + mx.array(layer.bias.astype(np.float16))

    return y.reshape(orig_shape[:-1] + (layer.out_features,))


def linear_forward_public_mlx(layer: EXL3Layer, x: Any, *, validate: bool = True):
    """Forward using public reconstructed weights (no runtime Hadamard on activations)."""
    import mlx.core as mx

    dtype = mx.float16

    if validate:
        layer.validate()

    x_np = np.asarray(x, dtype=np.float16)
    orig_shape = x_np.shape
    rows = int(np.prod(orig_shape[:-1]))
    x2d = mx.array(x_np.reshape(rows, layer.in_features))

    w = reconstruct_public_mlx(layer)
    y = (x2d.astype(mx.float32) @ w.astype(mx.float32)).astype(dtype)

    if layer.bias is not None:
        y = y + mx.array(layer.bias.astype(np.float16))

    return y.reshape(orig_shape[:-1] + (layer.out_features,))
