"""Decode packed EXL3 trellis tiles to fp16 weight blocks."""

from __future__ import annotations

import numpy as np

from .codebook import CodebookMode, decode_3inst
from .perm import kernel_order_to_row_major
from .trellis import unpack_trellis_tile


def decode_codewords(codewords: np.ndarray, cb: CodebookMode | int) -> np.ndarray:
    """Decode (256,) uint16 codewords to kernel-order fp16 samples."""
    out = np.empty(256, dtype=np.float16)
    for i, w in enumerate(codewords):
        out[i] = decode_3inst(int(w) & 0xFFFF, cb)
    return out


def decode_packed_tile(packed_tile: np.ndarray, k: int, cb: CodebookMode | int) -> np.ndarray:
    """
  Decode one packed trellis tile to a row-major 16x16 fp16 block.

  This is the fundamental building block for reconstruct and for validating
  future fused Metal GEMV kernels.
  """
    codewords = unpack_trellis_tile(packed_tile, k)
    kernel_vals = decode_codewords(codewords, cb)
    return kernel_order_to_row_major(kernel_vals)


def decode_packed_trellis(
    trellis: np.ndarray,
    k: int,
    cb: CodebookMode | int,
    *,
    n_offset: int = 0,
    n_count: int | None = None,
) -> np.ndarray:
    """
  Reconstruct inner weight matrix from packed trellis.

  trellis: (in_tiles, out_tiles, packed_size) uint16
  returns: (in_features, out_features) float16 inner weights (before outer Hadamard)
  """
    if trellis.ndim != 3:
        raise ValueError("trellis must be 3D")
    in_tiles, out_tiles, packed_size = trellis.shape
    expected = 256 * k // 16
    if packed_size != expected:
        raise ValueError(f"packed size {packed_size} != expected {expected} for K={k}")

    in_features = in_tiles * 16
    out_features = out_tiles * 16
    if n_offset % 128 != 0:
        raise ValueError("n_offset must be divisible by 128")
    if n_count is None:
        n_count_eff = out_features
    else:
        n_count_eff = n_count
    if n_offset + n_count_eff > out_features:
        raise ValueError("slice exceeds output features")

    w: np.ndarray = np.empty((in_features, n_count_eff), dtype=np.float16)
    tile_n0 = n_offset // 16
    tile_n1 = (n_offset + n_count_eff + 15) // 16

    col = 0
    for tn in range(tile_n0, tile_n1):
        n_start = max(0, n_offset - tn * 16)
        n_end = min(16, n_offset + n_count_eff - tn * 16)
        width = n_end - n_start
        if width <= 0:
            continue
        for tk in range(in_tiles):
            tile = decode_packed_tile(trellis[tk, tn], k, cb)
            r0 = tk * 16
            w[r0 : r0 + 16, col : col + width] = tile[:, n_start:n_end]
        col += width

    return w
