"""Metal trellis search for converter bring-up.

This is the M2 path for small batches of 16x16 tiles.  It mirrors the
compressed-state CUDA search shape: one tile per threadgroup, K-bit transition
labels, device backpointers, and tail-biting via a warmup pass followed by one
pinned pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ponyexl3.convert import timing
from ponyexl3.ref.codebook import CodebookMode


_THREADS = 256
_DEFAULT_MAX_SCRATCH_BYTES = 256 * 1024 * 1024
_KERNELS: dict[tuple[int, int], Callable[..., Any]] = {}


def _threads_for_k(k: int) -> int:
    return min(_THREADS, 1 << (16 - k))


def _decode_expr(cb: CodebookMode, *, cw_in: str = "cw", out: str = "dq_val") -> str:
    if cb == CodebookMode.DEFAULT:
        return f"""
                uint dq_cw = {cw_in} * 89226354u + 64248484u;
                uint dq_r = (dq_cw & 0x8FFF8FFFu) ^ 0x3B603B60u;
                half2 dq_h2 = as_type<half2>(dq_r);
                float {out} = float(dq_h2.x + dq_h2.y);
"""
    if cb == CodebookMode.MCG:
        return f"""
                uint dq_cw = {cw_in} * 0xCBAC1FEDu;
                uint dq_r = (dq_cw & 0x8FFF8FFFu) ^ 0x3B603B60u;
                half2 dq_h2 = as_type<half2>(dq_r);
                float {out} = float(dq_h2.x + dq_h2.y);
"""
    return f"""
                uint dq_cw = {cw_in} * 0x83DCD12Du;
                uint dq_sum = 0x6400u;
                for (uint dq_lane = 0u; dq_lane < 4u; dq_lane++) {{
                    dq_sum += (dq_cw >> (8u * dq_lane)) & 0xFFu;
                }}
                half dq_h = as_type<half>(ushort(dq_sum & 0xFFFFu));
                half dq_k_inv = as_type<half>(ushort(0x1EEEu));
                half dq_k_bias = as_type<half>(ushort(0xC931u));
                float {out} = float(dq_h * dq_k_inv + dq_k_bias);
"""


def _source(k: int, cb: CodebookMode) -> str:
    edges = 1 << (16 - k)
    kk = 1 << k
    kr = 16 - k
    threads = _threads_for_k(k)
    global_costs = k < 4
    decode_cost = _decode_expr(cb, cw_in="cw", out="dq_val")
    decode_write = _decode_expr(cb, cw_in="state", out="dq_out")
    costs_decl = (
        "    device half* costs = temp_costs + ulong(tile) * 2u * EDGES;\n"
        if global_costs
        else "    threadgroup half costs[2][EDGES];\n"
    )
    barrier = (
        "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"
        if global_costs
        else "threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    cost_store = "costs[ulong(curr) * EDGES + e]" if global_costs else "costs[curr][e]"
    prev_cost_load = (
        "costs[ulong(prev_buf) * EDGES + pred]" if global_costs else "costs[prev_buf][pred]"
    )
    argmin_cost_load = (
        "costs[ulong(curr) * EDGES + e]" if global_costs else "costs[curr][e]"
    )
    return f"""
#define K_BITS {k}u
#define KR_BITS {kr}u
#define EDGES {edges}u
#define KK {kk}u
#define H_INF_F 65504.0f

    uint tile = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;

    const device float* in_tile = tiles + tile * 256u;
    device float* out_tile = q_tiles + tile * 256u;
    device ushort* out_idx = indices + tile * 256u;
    device ushort* edge_ptr = temp_edges + ulong(tile) * 256u * EDGES;

{costs_decl}
    threadgroup uint sh_pin;

    uint pin = 0u;
    bool pinned = false;

    for (uint pass = 0u; pass < 2u; pass++) {{
        uint curr = 0u;
        uint roll = pass == 0u ? 128u : 0u;

        for (uint e = tid; e < EDGES; e += {threads}u) {{
            float target = in_tile[roll];
            float best = H_INF_F;
            uint best_pred = 0u;
            for (uint fresh = 0u; fresh < KK; fresh++) {{
                uint cw = (fresh << KR_BITS) | e;
                uint pred = cw >> K_BITS;
                float err;
                if (pinned && pred != pin) {{
                    err = H_INF_F;
                }} else {{
{decode_cost}
                    float dh = dq_val - target;
                    err = dh * dh;
                    if (err > H_INF_F) err = H_INF_F;
                }}
                if (err < best) {{
                    best = err;
                    best_pred = pred;
                }}
            }}
            {cost_store} = half(best);
            if (pass == 1u || roll < 128u) {{
                edge_ptr[roll * EDGES + e] = ushort(best_pred);
            }}
        }}
        {barrier}

        for (uint step = 1u; step < 256u; step++) {{
            uint ri = (step + roll) & 255u;
            uint prev_buf = curr;
            curr = 1u - curr;
            for (uint e = tid; e < EDGES; e += {threads}u) {{
                float target = in_tile[ri];
                float best = H_INF_F;
                uint best_pred = 0u;
                for (uint fresh = 0u; fresh < KK; fresh++) {{
                    uint cw = (fresh << KR_BITS) | e;
                    uint pred = cw >> K_BITS;
{decode_cost}
                    float dh = dq_val - target;
                    float err = fma(dh, dh, float({prev_cost_load}));
                    if (err > H_INF_F) err = H_INF_F;
                    if (err < best) {{
                        best = err;
                        best_pred = pred;
                    }}
                }}
                {cost_store} = half(best);
                if (pass == 1u || ri < 128u) {{
                    edge_ptr[ulong(ri) * EDGES + e] = ushort(best_pred);
                }}
            }}
            {barrier}
        }}

        if (tid == 0u) {{
            uint edge = pin;
            if (pass == 0u) {{
                float best = H_INF_F;
                edge = 0u;
                for (uint e = 0u; e < EDGES; e++) {{
                    float v = float({argmin_cost_load});
                    if (v < best) {{
                        best = v;
                        edge = e;
                    }}
                }}
            }}

            for (int step_i = 255; step_i >= 0; step_i--) {{
                uint ri = (uint(step_i) + roll) & 255u;
                uint prev = uint(edge_ptr[ulong(ri) * EDGES + edge]);
                uint state = (prev << K_BITS) | edge;
                edge = prev;
                if (pass == 1u) {{
                    out_idx[ri] = ushort(state);
{decode_write}
                    out_tile[ri] = dq_out;
                }} else if (ri == 0u) {{
                    break;
                }}
            }}
            sh_pin = edge;
        }}
        {barrier}
        pin = sh_pin;
        pinned = true;
    }}
"""


def _kernel(k: int, cb: CodebookMode) -> Callable[..., Any]:
    import mlx.core as mx

    key = (k, int(cb))
    if key not in _KERNELS:
        _KERNELS[key] = mx.fast.metal_kernel(
            name=f"exl3_quantize_tiles_k{k}_cb{int(cb)}_v8",
            input_names=["tiles"],
            output_names=["q_tiles", "indices", "temp_edges", "temp_costs"],
            source=_source(k, cb),
        )
    return _KERNELS[key]


def _scratch_bytes_per_tile(k: int) -> int:
    edges = 1 << (16 - k)
    return (256 * edges * np.dtype(np.uint16).itemsize) + (
        2 * edges * np.dtype(np.float16).itemsize
    )


def _quantize_tiles_mlx_launch(
    arr: Any,
    k: int,
    cb: CodebookMode,
) -> tuple[Any, Any]:
    import mlx.core as mx

    n_tiles = int(arr.shape[0])
    edges = 1 << (16 - k)
    threads = _threads_for_k(k)
    kernel = _kernel(k, cb)
    q_tiles, indices, _temp_edges, _temp_costs = kernel(
        inputs=[arr],
        template=[("T", mx.float32)],
        grid=(n_tiles * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (n_tiles, 256),
            (n_tiles, 256),
            (n_tiles, 256 * edges),
            (n_tiles, 2, edges),
        ],
        output_dtypes=[mx.float32, mx.uint16, mx.uint16, mx.float16],
    )
    return q_tiles, indices


def quantize_tiles_mlx(
    tiles: Any,
    k: int,
    cb: CodebookMode | int = CodebookMode.DEFAULT,
    *,
    max_scratch_bytes: int = _DEFAULT_MAX_SCRATCH_BYTES,
) -> tuple[Any, Any]:
    """Quantize kernel-order tiles with the Metal trellis search.

    Returns ``(quantized_tiles, indices)`` as MLX arrays, both shaped like the
    input ``(num_tiles, 256)``.  ``indices`` are full 16-bit codewords; callers
    should pass ``indices & ((1 << k) - 1)`` to the existing trellis packer.

    ``max_scratch_bytes`` bounds per-launch temporary backpointer/cost scratch;
    larger batches are split and concatenated.

    The implementation supports K>=2, which is the practical range needed by
    upstream EXL3 conversion. K=1 needs a separate low-K scratch strategy.
    """

    if not 2 <= k <= 8:
        raise ValueError("Metal trellis search currently supports K in [2, 8]")
    cb = CodebookMode(cb)

    import mlx.core as mx

    arr = mx.array(tiles, dtype=mx.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 256)
    if arr.ndim != 2 or arr.shape[1] != 256:
        raise ValueError(f"expected tiles shape (N, 256), got {arr.shape}")
    n_tiles = int(arr.shape[0])
    if n_tiles == 0:
        raise ValueError("expected at least one tile")
    if max_scratch_bytes <= 0:
        raise ValueError("max_scratch_bytes must be positive")

    launch_tiles = max(1, max_scratch_bytes // _scratch_bytes_per_tile(k))
    with timing.gpu("search"):
        if n_tiles <= launch_tiles:
            q_tiles, indices = _quantize_tiles_mlx_launch(arr, k, cb)
        else:
            q_parts = []
            idx_parts = []
            for start in range(0, n_tiles, launch_tiles):
                q_part, idx_part = _quantize_tiles_mlx_launch(arr[start : start + launch_tiles], k, cb)
                q_parts.append(q_part)
                idx_parts.append(idx_part)
            q_tiles = mx.concatenate(q_parts, axis=0)
            indices = mx.concatenate(idx_parts, axis=0)
        mx.eval(q_tiles, indices)
    return q_tiles, indices


def quantize_tiles_mlx_np(
    tiles: np.ndarray,
    k: int,
    cb: CodebookMode | int = CodebookMode.DEFAULT,
    *,
    max_scratch_bytes: int = _DEFAULT_MAX_SCRATCH_BYTES,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy wrapper around :func:`quantize_tiles_mlx` for tests/CLI glue."""

    q_tiles, indices = quantize_tiles_mlx(tiles, k, cb, max_scratch_bytes=max_scratch_bytes)
    return np.array(q_tiles), np.array(indices).astype(np.uint16, copy=False)
