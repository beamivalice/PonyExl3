"""Tensor-core tile element permutation from exllamav3/modules/quant/exl3_lib/quantize.py."""

from __future__ import annotations

import numpy as np


def tensor_core_perm() -> np.ndarray:
    perm = np.zeros(256, dtype=np.int32)
    for t in range(32):
        r0 = (t % 4) * 2
        r1 = r0 + 1
        r2 = r0 + 8
        r3 = r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        base = t * 8
        perm[base + 0] = r0 * 16 + c0
        perm[base + 1] = r1 * 16 + c0
        perm[base + 2] = r2 * 16 + c0
        perm[base + 3] = r3 * 16 + c0
        perm[base + 4] = r0 * 16 + c1
        perm[base + 5] = r1 * 16 + c1
        perm[base + 6] = r2 * 16 + c1
        perm[base + 7] = r3 * 16 + c1
    return perm


def tensor_core_perm_inverse() -> np.ndarray:
    perm = tensor_core_perm()
    inv = np.empty(256, dtype=np.int32)
    inv[perm] = np.arange(256, dtype=np.int32)
    return inv


def kernel_order_to_row_major(values: np.ndarray) -> np.ndarray:
    """Reorder 256-element tile from CUDA kernel layout to row-major 16x16."""
    inv = tensor_core_perm_inverse()
    out: np.ndarray = values[inv]
    return np.reshape(out, (16, 16))
