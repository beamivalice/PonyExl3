"""MLX + Metal tile decode — no Python loops over codewords."""

from __future__ import annotations

import mlx.core as mx

from ponyexl3.mlx.metal_kernels import decode_codewords_mlx, unpack_trellis_tiles_mlx  # Metal
from ponyexl3.mlx.perm import kernel_order_to_row_major_mlx
from ponyexl3.ref.codebook import CodebookMode, codebook_mode_from_flags
from ponyexl3.ref.layer import EXL3Layer


def decode_packed_tile_mlx(
    packed_tile: mx.array,
    k: int,
    cb: CodebookMode | int,
) -> mx.array:
    """Decode one packed trellis tile to row-major 16×16 fp16."""
    codewords = unpack_trellis_tiles_mlx(packed_tile.reshape(1, -1), k).reshape(256)
    kernel_vals = decode_codewords_mlx(codewords, cb)
    return kernel_order_to_row_major_mlx(kernel_vals)


def decode_packed_trellis_mlx(
    trellis: mx.array,
    k: int,
    cb: CodebookMode | int,
) -> mx.array:
    """Reconstruct inner weight matrix from packed trellis on MLX."""
    if trellis.ndim != 3:
        raise ValueError("trellis must be 3D")
    in_tiles, out_tiles, packed_size = trellis.shape
    expected = 256 * k // 16
    if packed_size != expected:
        raise ValueError(f"packed size {packed_size} != expected {expected} for K={k}")

    flat = trellis.reshape(in_tiles * out_tiles, packed_size)
    codewords = unpack_trellis_tiles_mlx(flat, k)
    kernel_vals = decode_codewords_mlx(codewords.reshape(-1), cb).reshape(
        in_tiles * out_tiles, 256
    )
    tiles = kernel_order_to_row_major_mlx(kernel_vals).reshape(in_tiles, out_tiles, 16, 16)
    return tiles.transpose(0, 2, 1, 3).reshape(in_tiles * 16, out_tiles * 16)


def decode_inner_cols_mlx(
    trellis: mx.array,
    k: int,
    cb: CodebookMode | int,
    n_offset: int,
    n_count: int,
) -> mx.array:
    """Decode inner weights for ``W[:, n_offset:n_offset+n_count]`` without full matrix."""
    if n_offset % 16 != 0:
        raise ValueError("n_offset must be divisible by 16")
    if n_count % 16 != 0:
        raise ValueError("n_count must be divisible by 16")
    _, out_tiles, _ = trellis.shape
    out_features = out_tiles * 16
    if n_offset + n_count > out_features:
        raise ValueError("column slice exceeds out_features")

    tn0 = n_offset // 16
    tn1 = (n_offset + n_count) // 16
    w = decode_packed_trellis_mlx(trellis[:, tn0:tn1, :], k, cb)
    col0 = n_offset - tn0 * 16
    return w[:, col0 : col0 + n_count]


def decode_packed_trellis_mlx_layer(layer: EXL3Layer) -> mx.array:
    cb = codebook_mode_from_flags(mcg=layer.mcg, mul1=layer.mul1)
    return decode_packed_trellis_mlx(mx.array(layer.trellis.astype("uint16")), layer.k, cb)
