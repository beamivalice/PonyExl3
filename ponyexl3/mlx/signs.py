"""MLX sign unpacking for EXL3 suh/svh tensors."""

from __future__ import annotations

import mlx.core as mx


def unpack_sign_bitfield_mlx(bitfield: mx.array) -> mx.array:
    """Expand packed int16 sign words to per-element ±1 float16 scales.

    bitfield: (groups,) int16 — one word per 16 channels
    returns: (groups * 16,) float16
    """
    bf = bitfield.astype(mx.uint16).reshape(-1, 1)
    masks = mx.array([1 << i for i in range(16)], dtype=mx.uint32).reshape(1, 16)
    bits = (bf & masks) != 0
    return mx.where(bits, mx.array(-1.0, dtype=mx.float16), mx.array(1.0, dtype=mx.float16)).reshape(
        -1
    )


def unpack_signs_or_pass_mlx(values: mx.array | None) -> mx.array | None:
    if values is None:
        return None
    if values.dtype in (mx.float16, mx.bfloat16, mx.float32):
        return values.astype(mx.float16)
    return unpack_sign_bitfield_mlx(values)
