"""Unpack packed sign bitfields (suh/svh) — mirrors LinearEXL3.unpack_bf."""

from __future__ import annotations

import numpy as np


def unpack_sign_bitfield(bitfield: np.ndarray) -> np.ndarray:
    """
  Expand packed int16 sign words to per-element +/-1 float16 scales.

  bitfield: (groups,) int16 — one word per 16 channels
  returns: (groups * 16,) float16
  """
    bf = bitfield.astype(np.uint16).reshape(-1, 1)
    masks = (1 << np.arange(16, dtype=np.uint16)).reshape(1, 16)
    bits = (bf & masks) != 0
    signs = np.where(bits, -1.0, 1.0).astype(np.float16)
    return signs.reshape(-1)


def unpack_signs_or_pass(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    if values.dtype == np.float16 or values.dtype == np.float32:
        return values.astype(np.float16, copy=False)
    return unpack_sign_bitfield(values)
