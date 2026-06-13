"""Custom Metal kernels for EXL3 trellis unpack and codebook decode."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import mlx.core as mx

from ponyexl3.ref.codebook import CodebookMode

_HALF2_FROM_U32 = r"""
    // half add, matching CUDA __hadd (codebook.cuh)
    half2 h2 = as_type<half2>(r);
    out[idx] = h2.x + h2.y;
"""

_LOP3_LOOP = r"""
    // lop3.b32 imm 0x6A == c ^ (a & b): forces exponent bits to 011xx, never NaN/inf
    uint r = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
"""

_DECODE_DEFAULT_SOURCE = (
    r"""
    uint idx = thread_position_in_grid.x;
    uint x = inp[idx] & 0xFFFFu;
    x = x * 89226354u + 64248484u;
"""
    + _LOP3_LOOP
    + _HALF2_FROM_U32
)

_DECODE_MCG_SOURCE = (
    r"""
    uint idx = thread_position_in_grid.x;
    uint x = inp[idx] & 0xFFFFu;
    x = x * 0xCBAC1FEDu;
"""
    + _LOP3_LOOP
    + _HALF2_FROM_U32
)

_DECODE_MUL1_SOURCE = r"""
    uint idx = thread_position_in_grid.x;
    uint x = inp[idx] & 0xFFFFu;
    x = x * 0x83DCD12Du;
    uint sum = 0x6400u;
    for (uint lane = 0; lane < 4u; lane++) {
        uint ai = (x >> (8u * lane)) & 0xFFu;
        sum += ai;
    }
    const T k_inv = as_type<T>(ushort(0x1EEEu));
    const T k_bias = as_type<T>(ushort(0xC931u));
    out[idx] = as_type<T>(ushort(sum & 0xFFFFu)) * k_inv + k_bias;
"""


def _unpack_source(k: int) -> str:
    packed_u32 = k * 256 // 32
    return f"""
#define PACKED_U32 {packed_u32}

    uint tile = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    const device uint* words = packed + tile * PACKED_U32;

    for (uint t = lane; t < 128u; t += threads_per_threadgroup.x) {{
        int b0 = int(t) * 2 * {k} + {k} - 16 + 256 * {k};
        int b1 = b0 + {k};
        int b2 = b1 + 16;
        int i0 = b0 / 32;
        int i1 = (b2 - 1) / 32;
        int s1 = (i1 + 1) * 32 - b2;

        uint a = words[i0 % PACKED_U32];
        uint b = words[i1 % PACKED_U32];
        ulong merged = ((ulong)a << 32) | ulong(b);
        uint w1 = uint(merged >> uint(s1));
        uint w0 = (w1 >> {k}u) & 0xFFFFu;
        w1 &= 0xFFFFu;
        device ushort* out_tile = out + tile * 256u;
        // Full 16-bit sliding-window codewords (exl3_dq.cuh), not the low K bits.
        out_tile[t * 2u] = ushort(w0);
        out_tile[t * 2u + 1u] = ushort(w1);
    }}
"""


_DECODE_THREADS = 256
_UNPACK_THREADS = 128

_decode_kernels: dict[int, Callable[..., Any]] = {}
_unpack_kernels: dict[int, Callable[..., Any]] = {}


def _decode_kernel(cb: CodebookMode) -> Callable[..., Any]:
    if cb not in _decode_kernels:
        sources = {
            CodebookMode.DEFAULT: _DECODE_DEFAULT_SOURCE,
            CodebookMode.MCG: _DECODE_MCG_SOURCE,
            CodebookMode.MUL1: _DECODE_MUL1_SOURCE,
        }
        _decode_kernels[cb] = mx.fast.metal_kernel(
            name=f"exl3_decode_{cb.name.lower()}",
            input_names=["inp"],
            output_names=["out"],
            source=sources[cb],
        )
    return _decode_kernels[cb]


def _unpack_kernel(k: int) -> Callable[..., Any]:
    if k not in _unpack_kernels:
        _unpack_kernels[k] = mx.fast.metal_kernel(
            name=f"exl3_unpack_trellis_k{k}_v2",
            input_names=["packed"],
            output_names=["out"],
            source=_unpack_source(k),
        )
    return _unpack_kernels[k]


def _packed_to_u32(packed_u16: np.ndarray, k: int) -> mx.array:
    """Match CPU ``uint16`` trellis view as ``uint32`` words (CUDA layout)."""
    u32_words = k * 256 // 32
    flat = np.ascontiguousarray(packed_u16.astype(np.uint16)).reshape(-1)
    return mx.array(flat.view(np.uint32).reshape(-1, u32_words))


def _run_1d(kernel: Callable[..., Any], inp: mx.array, dtype: mx.Dtype) -> mx.array:
    n = int(inp.size)
    out = kernel(
        inputs=[inp],
        template=[("T", dtype)],
        grid=(n, 1, 1),
        threadgroup=(min(_DECODE_THREADS, max(1, n)), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[dtype],
    )[0]
    mx.eval(out)
    return out


def decode_codewords_mlx(codewords: mx.array, cb: CodebookMode | int) -> mx.array:
    """Batch ``decode_3inst`` via Metal (all codebook modes)."""
    cb = CodebookMode(cb)
    flat = codewords.astype(mx.uint32).reshape(-1)
    return _run_1d(_decode_kernel(cb), flat, mx.float16).reshape(codewords.shape)


def unpack_trellis_tiles_mlx(packed: mx.array, k: int) -> mx.array:
    """Unpack trellis tiles ``(..., packed_size)`` → ``(..., 256)`` uint16."""
    if not 1 <= k <= 8:
        raise ValueError(f"K must be in [1, 8], got {k}")
    packed_size = 256 * k // 16
    if packed.shape[-1] != packed_size:
        raise ValueError(f"expected last dim {packed_size}, got {packed.shape[-1]}")

    batch_shape = packed.shape[:-1]
    n_tiles = 1
    for d in batch_shape:
        n_tiles *= int(d)
    flat_u16 = np.ascontiguousarray(np.array(packed).astype(np.uint16).reshape(n_tiles, packed_size))
    packed_u32 = _packed_to_u32(flat_u16, k)

    kernel = _unpack_kernel(k)
    out = kernel(
        inputs=[packed_u32],
        template=[("T", mx.float16)],
        grid=(n_tiles * _UNPACK_THREADS, 1, 1),
        threadgroup=(_UNPACK_THREADS, 1, 1),
        output_shapes=[(n_tiles, 256)],
        output_dtypes=[mx.uint16],
    )[0]
    mx.eval(out)
    return out.reshape(*batch_shape, 256)
