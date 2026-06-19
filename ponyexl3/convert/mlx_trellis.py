"""MLX trellis packing helpers for converter production paths."""

from __future__ import annotations

from typing import Any


def _swap_word_pairs(flat: Any) -> Any:
    import mlx.core as mx

    pairs = flat.reshape(*flat.shape[:-1], flat.shape[-1] // 2, 2)
    return mx.stack([pairs[..., 1], pairs[..., 0]], axis=-1).reshape(*flat.shape)


def _pack_trellis_div16_mlx(encoded: Any, k: int) -> Any:
    import mlx.core as mx

    vals_per_word = 16 // k
    words_per_span = k
    mask = (1 << k) - 1
    vals = (encoded.astype(mx.uint32) & mask).reshape(
        *encoded.shape[:-1],
        16,
        words_per_span,
        vals_per_word,
    )
    shifts = mx.array(
        [i * k for i in range(vals_per_word - 1, -1, -1)],
        dtype=mx.uint32,
    ).reshape(*((1,) * (vals.ndim - 1)), vals_per_word)
    words = mx.sum(vals << shifts, axis=-1).astype(mx.uint16)
    flat = words.reshape(*encoded.shape[:-1], 256 * k // 16)
    return _swap_word_pairs(flat)


def _pack_trellis_general_mlx(encoded: Any, k: int) -> Any:
    import mlx.core as mx

    prefix = encoded.shape[:-1]
    vals = (encoded.astype(mx.uint32) & ((1 << k) - 1)).reshape(*prefix, 16, 16)
    word_rows = []
    for word_idx in range(k):
        word_val = mx.zeros((*prefix, 16), dtype=mx.uint32)
        for i in range(16):
            bit_start = i * k
            word = bit_start // 16
            offset = bit_start % 16
            v = vals[..., i]
            if offset + k <= 16:
                if word == word_idx:
                    word_val = word_val | (v << (16 - offset - k))
            else:
                high_bits = 16 - offset
                low_bits = k - high_bits
                if word == word_idx:
                    word_val = word_val | (v >> low_bits)
                if word + 1 == word_idx:
                    word_val = word_val | (
                        (v & ((1 << low_bits) - 1)) << (16 - low_bits)
                    )
        word_rows.append(word_val)
    words = mx.stack(word_rows, axis=-1).astype(mx.uint16)
    flat = words.reshape(*prefix, 256 * k // 16)
    return _swap_word_pairs(flat)


def pack_trellis_mlx(encoded: Any, k: int) -> Any:
    """Pack low-K trellis codewords on MLX.

    ``encoded`` must have shape ``(..., 256)`` and contain the fresh low K bits
    for each tile position, matching ``ponyexl3.ref.trellis.pack_trellis``.
    """

    if not 1 <= k <= 8:
        raise ValueError(f"K must be in [1, 8], got {k}")

    import mlx.core as mx

    arr = mx.array(encoded, dtype=mx.uint16)
    if arr.ndim < 1 or arr.shape[-1] != 256:
        raise ValueError(f"expected encoded last dim 256, got {arr.shape}")
    if k in (1, 2, 4, 8):
        return _pack_trellis_div16_mlx(arr, k)
    return _pack_trellis_general_mlx(arr, k)
