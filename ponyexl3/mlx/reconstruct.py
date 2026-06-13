"""MLX weight reconstruction — Metal decode path."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from ponyexl3.mlx.decode import decode_packed_trellis_mlx
from ponyexl3.mlx.hadamard import preapply_had_left_mlx, preapply_had_right_mlx
from ponyexl3.mlx.signs import unpack_signs_or_pass_mlx
from ponyexl3.ref.codebook import codebook_mode_from_flags
from ponyexl3.ref.layer import EXL3Layer


def mlx_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("mlx.core") is not None
    except ImportError:
        return False


def reconstruct_inner_mlx(
    trellis: Any,
    k: int,
    *,
    mcg: bool = False,
    mul1: bool = False,
) -> mx.array:
    cb = codebook_mode_from_flags(mcg=mcg, mul1=mul1)
    t = mx.array(trellis.astype("uint16")) if not isinstance(trellis, mx.array) else trellis
    return decode_packed_trellis_mlx(t, k, cb)


def reconstruct_public_mlx(layer: EXL3Layer) -> mx.array:
    w = reconstruct_inner_mlx(
        layer.trellis,
        layer.k,
        mcg=layer.mcg,
        mul1=layer.mul1,
    ).astype(mx.float32)
    suh = unpack_signs_or_pass_mlx(
        None if layer.suh is None else mx.array(layer.suh)
    )
    svh = unpack_signs_or_pass_mlx(
        None if layer.svh is None else mx.array(layer.svh)
    )
    if suh is not None:
        w = preapply_had_left_mlx(w)
        w = w * suh.reshape(-1, 1).astype(mx.float32)
    if svh is not None:
        w = preapply_had_right_mlx(w)
        w = w * svh.reshape(1, -1).astype(mx.float32)
    return w.astype(mx.float16)
