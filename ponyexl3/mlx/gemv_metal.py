"""Fused Metal GEMV/GEMM: on-the-fly trellis decode + dot (Phase 2b/3b)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import mlx.core as mx

from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.perm import tensor_core_perm, tensor_core_perm_inverse

_GEM_THREADS = 128
_fwd_perm_mx: mx.array | None = None
_inv_perm_mx: mx.array | None = None
_gem_kernels: dict[tuple[int, int, int, bool], Any] = {}


def _fwd_perm_u32() -> mx.array:
    """Row-major tile position for each kernel-order weight index."""
    global _fwd_perm_mx
    if _fwd_perm_mx is None:
        _fwd_perm_mx = mx.array(tensor_core_perm().astype(np.uint32))
    return _fwd_perm_mx


def _inv_perm_u32() -> mx.array:
    """Kernel-order weight index for each row-major tile position."""
    global _inv_perm_mx
    if _inv_perm_mx is None:
        _inv_perm_mx = mx.array(tensor_core_perm_inverse().astype(np.uint32))
    return _inv_perm_mx


def _decode_expr(cb: CodebookMode, *, cw_in: str = "cw") -> str:
    """Inline ``decode_3inst``; reads ``cw_in``, writes ``dq_val``."""
    if cb == CodebookMode.DEFAULT:
        return f"""
            uint dq_cw = {cw_in} * 89226354u + 64248484u;
            // lop3.b32 imm 0x6A == c ^ (a & b): exponent forced to 011xx, never NaN/inf
            uint dq_r = (dq_cw & 0x8FFF8FFFu) ^ 0x3B603B60u;
            // half add, matching CUDA __hadd (one rounding, fewer converts)
            half2 dq_h2 = as_type<half2>(dq_r);
            float dq_val = float(dq_h2.x + dq_h2.y);
"""
    if cb == CodebookMode.MCG:
        return f"""
            uint dq_cw = {cw_in} * 0xCBAC1FEDu;
            // lop3.b32 imm 0x6A == c ^ (a & b): exponent forced to 011xx, never NaN/inf
            uint dq_r = (dq_cw & 0x8FFF8FFFu) ^ 0x3B603B60u;
            // half add, matching CUDA __hadd (one rounding, fewer converts)
            half2 dq_h2 = as_type<half2>(dq_r);
            float dq_val = float(dq_h2.x + dq_h2.y);
"""
    return f"""
            uint dq_cw = {cw_in} * 0x83DCD12Du;
            uint dq_sum = 0x6400u;
            for (uint dq_lane = 0u; dq_lane < 4u; dq_lane++) {{
                uint ai = (dq_cw >> (8u * dq_lane)) & 0xFFu;
                dq_sum += ai;
            }}
            half dq_h = as_type<half>(ushort(dq_sum & 0xFFFFu));
            half dq_k_inv = as_type<half>(ushort(0x1EEEu));
            half dq_k_bias = as_type<half>(ushort(0xC931u));
            float dq_val = float(dq_h * dq_k_inv + dq_k_bias);
"""


_HAD_SCALE_LIT = "0.08838834764831845f"  # 1/sqrt(128)


def _had_prologue(nb: int) -> str:
    """Cooperative in-place 128-point Hadamard over this split's x range.

    Replaces the separate prep dispatch (and its serial GPU gap): loads raw x,
    multiplies by suh and the Hadamard scale, then runs the 7-stage butterfly
    with pair ownership (thread owns (i, i|bit) — disjoint writes, one barrier
    per stage). The GPU is idle between dependent dispatches anyway, so the
    extra ALU is free.
    """
    return f"""
    threadgroup float tg_xh[{nb}];
    uint nx = (tk_end - tk_begin) * 16u;
    for (uint idx = tid; idx < nx; idx += 128u) {{
        uint gidx = tk_begin * 16u + idx;
        tg_xh[idx] = float(xh[gidx]) * float(suh[sub * in_features + gidx])
                   * {_HAD_SCALE_LIT};
    }}
    for (uint s_ = 0u; s_ < 7u; s_++) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint bit = 1u << s_;
        for (uint p = tid; p < nx / 2u; p += 128u) {{
            uint blk = p >> 6u;
            uint w = p & 63u;
            uint i = blk * 128u + (((w & ~(bit - 1u)) << 1u) | (w & (bit - 1u)));
            uint jj = i + bit;
            float a = tg_xh[i];
            float b = tg_xh[jj];
            tg_xh[i] = a + b;
            tg_xh[jj] = a - b;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""


def _gemv_simd_source(k: int, cb: CodebookMode, mt: int = 1, had_nb: int = 0) -> str:
    """v12: simdgroup-cooperative GEMV (M=1), ds4/llama.cpp mat-vec pattern.

    One out-tile (16 cols) per threadgroup; NSG simdgroups stride the in-tile
    range independently — ZERO barriers in the hot loop. Within a simdgroup,
    lane ``l`` owns 1-2 trellis words and the x value for row ``l & 15``; every
    lane decodes 4 codeword pairs, fetching words and x via ``simd_shuffle``
    (register speed, no threadgroup memory). The tile perm is tile-invariant,
    so each lane's 8 weights land at fixed (row, col): 8 register accumulators
    across the whole loop, one threadgroup reduction at the very end.
    """
    packed_u32 = k * 256 // 32
    decode = _decode_expr(cb, cw_in="cw")
    # Words are read directly from device (the ds4 pattern): a tile's
    # 128-256 B is L1-hot after the first touch, and direct reads beat the
    # two-word shuffle ladder needed for k >= 5 (measured).
    _word_load = ""
    _word_get = "words[i]"
    if had_nb:
        had_prologue = _had_prologue(had_nb)
        x_load = "tg_xh[(tk - tk_begin) * 16u + (lane & 15u)]"
    else:
        had_prologue = ""
        x_load = "float(xh[sub * in_features + tk * 16u + (lane & 15u)])"
    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}
#define NSG 4u
#define MT {mt}u

    uint tn = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;     // 0..127
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint in_tiles = dims[0];
    uint out_tiles = dims[1];
    uint batch = dims[2];
    uint n_splits = dims[3];
    uint in_features = in_tiles * 16u;
    uint out_features = out_tiles * 16u;

    uint n_sub = dims[4];
    uint sub = (n_sub > 1u) ? tile_sub[tn] : 0u;
    // MoE expert indirection: tn maps into a wider source trellis
    uint src_stride = (dims[5] > 0u) ? dims[5] : out_tiles;
    uint tn_src = (dims[5] > 0u) ? tile_map[tn] : tn;

    uint split = threadgroup_position_in_grid.z;
    uint tiles_per_split = (in_tiles + n_splits - 1u) / n_splits;
    uint tk_begin = split * tiles_per_split;
    uint tk_end = min(tk_begin + tiles_per_split, in_tiles);

    // Lane handles codeword pairs t = 4*lane .. 4*lane+3 (8 consecutive
    // codewords). All funnel geometry and output positions are tile-invariant.
    uint pos_[8];
    uint row_[8];
    for (uint j = 0u; j < 4u; j++) {{
        uint t = lane * 4u + j;
        pos_[j * 2u] = perm[t * 2u];
        pos_[j * 2u + 1u] = perm[t * 2u + 1u];
        row_[j * 2u] = pos_[j * 2u] >> 4u;
        row_[j * 2u + 1u] = pos_[j * 2u + 1u] >> 4u;
    }}

    float acc[8][MT];
    for (uint j = 0u; j < 8u; j++) {{
        for (uint mm = 0u; mm < MT; mm++) {{
            acc[j][mm] = 0.0f;
        }}
    }}

#if MT == 1
    // dq4 (exl3_dq.cuh): a 4-aligned group of 4 consecutive codewords spans
    // 3K+16 bits, which for K in {{1..6, 8}} always fits ONE 64-bit window of
    // exactly two words — half the loads and merges of per-pair funnels.
    uint ig0_[2];
    uint ig1_[2];
    uint s3_[2];
    for (uint g = 0u; g < 2u; g++) {{
        uint c0 = lane * 8u + g * 4u;                 // first codeword of group
        int e_last = int(c0 + 4u) * K_BITS + 256 * K_BITS;  // end of last window
        int i_end = (e_last - 1) / 32;
        ig1_[g] = uint(i_end) % PACKED_U32;
        ig0_[g] = uint(i_end - 1) % PACKED_U32;
        s3_[g] = uint((i_end + 1) * 32 - e_last);
    }}

{had_prologue}

    for (uint tk = tk_begin + sg; tk < tk_end; tk += NSG) {{
        const device uint* words = trellis + (tk * src_stride + tn_src) * PACKED_U32;
        float x_lane = {x_load};

        for (uint g = 0u; g < 2u; g++) {{
            ulong merged = ((ulong)words[ig0_[g]] << 32) | (ulong)words[ig1_[g]];
            uint s = s3_[g];
            // codewords c0..c0+3, newest (c0+3) at shift s
            uint cw3 = uint(merged >> s) & 0xFFFFu;
            uint cw2 = uint(merged >> (s + K_BITS)) & 0xFFFFu;
            uint cw1 = uint(merged >> (s + 2u * K_BITS)) & 0xFFFFu;
            uint cw0 = uint(merged >> (s + 3u * K_BITS)) & 0xFFFFu;
            uint jw = g * 4u;   // first of this group's 4 weight slots
            {{
                uint cw = cw0;
{decode}
                acc[jw][0] = fma(simd_shuffle(x_lane, ushort(row_[jw])), dq_val, acc[jw][0]);
            }}
            {{
                uint cw = cw1;
{decode}
                acc[jw + 1u][0] = fma(simd_shuffle(x_lane, ushort(row_[jw + 1u])), dq_val, acc[jw + 1u][0]);
            }}
            {{
                uint cw = cw2;
{decode}
                acc[jw + 2u][0] = fma(simd_shuffle(x_lane, ushort(row_[jw + 2u])), dq_val, acc[jw + 2u][0]);
            }}
            {{
                uint cw = cw3;
{decode}
                acc[jw + 3u][0] = fma(simd_shuffle(x_lane, ushort(row_[jw + 3u])), dq_val, acc[jw + 3u][0]);
            }}
        }}
    }}
#else
    // MT > 1 (small-batch verify): x staged in ping-pong threadgroup buffers
    // (one barrier per TBX tiles), TRANSPOSED so a lane reads its row's MT
    // values as float4 vectors; accumulation is vector fma — the per-row cost
    // collapses to ~1 op per 4 rows and decode dominates again. Words are
    // read directly from device (L1-hot), exactly like the MT == 1 path.
    #define TBX 4u
    #define MT4 (MT / 4u)
    threadgroup float tg_x[2][TBX][16u * MT];
    float4 acc4[8][MT4];
    for (uint j = 0u; j < 8u; j++) {{
        for (uint q = 0u; q < MT4; q++) {{
            acc4[j][q] = float4(0.0f);
        }}
    }}
    for (uint tk0 = tk_begin; tk0 < tk_end; tk0 += TBX) {{
        uint buf = (tk0 / TBX) & 1u;
        for (uint idx = tid; idx < TBX * MT * 16u; idx += 128u) {{
            uint t = idx / (MT * 16u);
            uint rem = idx % (MT * 16u);
            uint ri = rem / MT;
            uint mm = rem % MT;
            float v = 0.0f;
            if (tk0 + t < tk_end && mm < batch) {{
                v = float(xh[(mm * n_sub + sub) * in_features + (tk0 + t) * 16u + ri]);
            }}
            tg_x[buf][t][ri * MT + mm] = v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = sg; t < TBX && tk0 + t < tk_end; t += NSG) {{
            uint tk = tk0 + t;
            const device uint* words = trellis + (tk * src_stride + tn_src) * PACKED_U32;
            for (uint j = 0u; j < 4u; j++) {{
                ulong merged = ((ulong)words[i0_[j]] << 32) | (ulong)words[i1_[j]];
                uint w1 = uint(merged >> s1_[j]);
                uint w0 = (w1 >> K_BITS) & 0xFFFFu;
                w1 &= 0xFFFFu;
                {{
                    uint cw = w0;
{decode}
                    threadgroup const float4* xv = (threadgroup const float4*)
                        &tg_x[buf][t][row_[j * 2u] * MT];
                    for (uint q = 0u; q < MT4; q++) {{
                        acc4[j * 2u][q] = fma(xv[q], float4(dq_val), acc4[j * 2u][q]);
                    }}
                }}
                {{
                    uint cw = w1;
{decode}
                    threadgroup const float4* xv = (threadgroup const float4*)
                        &tg_x[buf][t][row_[j * 2u + 1u] * MT];
                    for (uint q = 0u; q < MT4; q++) {{
                        acc4[j * 2u + 1u][q] = fma(xv[q], float4(dq_val), acc4[j * 2u + 1u][q]);
                    }}
                }}
            }}
        }}
    }}
    for (uint j = 0u; j < 8u; j++) {{
        for (uint mm = 0u; mm < MT; mm++) {{
            acc[j][mm] = acc4[j][mm >> 2u][mm & 3u];
        }}
    }}
#endif

    // Cross-simdgroup reduction: each position is owned by exactly one lane
    // within a simdgroup, so per-sg writes are conflict-free.
    threadgroup float tg_w[NSG][256];
    for (uint mm = 0u; mm < MT; mm++) {{
        if (mm > 0u) {{
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        for (uint j = 0u; j < 8u; j++) {{
            tg_w[sg][pos_[j]] = acc[j][mm];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 16u && mm < batch) {{
            float s = 0.0f;
            for (uint r = 0u; r < 16u; r++) {{
                uint p = r * 16u + tid;
                for (uint g = 0u; g < NSG; g++) {{
                    s += tg_w[g][p];
                }}
            }}
            out[(mm * n_splits + split) * out_features + tn * 16u + tid] = s;
        }}
    }}
"""


def _gemv_post_source(k: int, cb: CodebookMode) -> str:
    """v18: M=1 GEMV with the post-Hadamard + ``svh`` fused as an epilogue.

    One threadgroup owns a 128-col output block (8 sequential out-tiles, same
    v12 inner loop each). The block's col sums collect in 512 B of threadgroup
    memory, get the 7-stage butterfly + ``svh`` at the end, and the finish
    dispatch (and its serial GPU gap) disappears. Col sums are cast through
    fp16 before the butterfly to match the unfused pipeline's cast order.
    With split-K the epilogue runs per split — the Hadamard is linear, so
    summing transformed partials on the host is exact.
    """
    packed_u32 = k * 256 // 32
    decode = _decode_expr(cb, cw_in="cw")
    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}
#define NSG 4u

    uint blk = threadgroup_position_in_grid.x;      // 128-col output block
    uint tn0 = blk * 8u;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint in_tiles = dims[0];
    uint out_tiles = dims[1];
    uint n_splits = dims[3];
    uint n_sub = dims[4];
    uint in_features = in_tiles * 16u;
    uint out_features = out_tiles * 16u;
    uint sub = (n_sub > 1u) ? tile_sub[tn0] : 0u;

    uint split = threadgroup_position_in_grid.z;
    uint tiles_per_split = (in_tiles + n_splits - 1u) / n_splits;
    uint tk_begin = split * tiles_per_split;
    uint tk_end = min(tk_begin + tiles_per_split, in_tiles);

    uint pos_[8];
    uint row_[8];
    for (uint j = 0u; j < 4u; j++) {{
        uint t = lane * 4u + j;
        pos_[j * 2u] = perm[t * 2u];
        pos_[j * 2u + 1u] = perm[t * 2u + 1u];
        row_[j * 2u] = pos_[j * 2u] >> 4u;
        row_[j * 2u + 1u] = pos_[j * 2u + 1u] >> 4u;
    }}
    uint ig0_[2];
    uint ig1_[2];
    uint s3_[2];
    for (uint g = 0u; g < 2u; g++) {{
        uint c0 = lane * 8u + g * 4u;
        int e_last = int(c0 + 4u) * K_BITS + 256 * K_BITS;
        int i_end = (e_last - 1) / 32;
        ig1_[g] = uint(i_end) % PACKED_U32;
        ig0_[g] = uint(i_end - 1) % PACKED_U32;
        s3_[g] = uint((i_end + 1) * 32 - e_last);
    }}

    threadgroup float tg_w[NSG][256];
    threadgroup float tg_y[128];

    for (uint ot = 0u; ot < 8u; ot++) {{
        uint tn = tn0 + ot;
        float acc[8] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
        for (uint tk = tk_begin + sg; tk < tk_end; tk += NSG) {{
            const device uint* words = trellis + (tk * out_tiles + tn) * PACKED_U32;
            float x_lane = float(xh[sub * in_features + tk * 16u + (lane & 15u)]);
            for (uint g = 0u; g < 2u; g++) {{
                ulong merged = ((ulong)words[ig0_[g]] << 32) | (ulong)words[ig1_[g]];
                uint s = s3_[g];
                uint cw3 = uint(merged >> s) & 0xFFFFu;
                uint cw2 = uint(merged >> (s + K_BITS)) & 0xFFFFu;
                uint cw1 = uint(merged >> (s + 2u * K_BITS)) & 0xFFFFu;
                uint cw0 = uint(merged >> (s + 3u * K_BITS)) & 0xFFFFu;
                uint jw = g * 4u;
                {{
                    uint cw = cw0;
{decode}
                    acc[jw] = fma(simd_shuffle(x_lane, ushort(row_[jw])), dq_val, acc[jw]);
                }}
                {{
                    uint cw = cw1;
{decode}
                    acc[jw + 1u] = fma(simd_shuffle(x_lane, ushort(row_[jw + 1u])), dq_val, acc[jw + 1u]);
                }}
                {{
                    uint cw = cw2;
{decode}
                    acc[jw + 2u] = fma(simd_shuffle(x_lane, ushort(row_[jw + 2u])), dq_val, acc[jw + 2u]);
                }}
                {{
                    uint cw = cw3;
{decode}
                    acc[jw + 3u] = fma(simd_shuffle(x_lane, ushort(row_[jw + 3u])), dq_val, acc[jw + 3u]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < 8u; j++) {{
            tg_w[sg][pos_[j]] = acc[j];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16u) {{
            float s = 0.0f;
            for (uint r = 0u; r < 16u; r++) {{
                uint p = r * 16u + tid;
                for (uint g = 0u; g < NSG; g++) {{
                    s += tg_w[g][p];
                }}
            }}
            tg_y[ot * 16u + tid] = float(half(s));  // fp16 cast before post-had
        }}
    }}

    // 128-point butterfly + svh epilogue (replaces the finish dispatch)
    for (uint st = 0u; st < 7u; st++) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint bit = 1u << st;
        if (tid < 64u) {{
            uint i = ((tid & ~(bit - 1u)) << 1u) | (tid & (bit - 1u));
            uint jj = i + bit;
            float a = tg_y[i];
            float b = tg_y[jj];
            tg_y[i] = a + b;
            tg_y[jj] = a - b;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128u) {{
        uint col = blk * 128u + tid;
        out[split * out_features + col] =
            tg_y[tid] * {_HAD_SCALE_LIT} * float(svh[col]);
    }}
"""


def _gemm_simd_source(k: int, cb: CodebookMode, mt: int) -> str:
    """Small-batch (2..8 rows) trellis GEMM, v16 register-budget layout.

    Like the M=1 simd kernel but each lane owns only TWO codeword pairs (4
    weights) so the per-row float4 accumulators fit the register file: two
    simdgroups cover one tile (half = sg & 1), the other two stride the next
    tile. x is staged transposed in ping-pong threadgroup buffers so a lane
    reads its row's MT values as float4 vectors (decode stays the dominant
    cost, not the per-row fma).
    """
    assert mt % 2 == 0
    packed_u32 = k * 256 // 32
    decode = _decode_expr(cb, cw_in="cw")
    vec = 4 if mt % 4 == 0 else 2
    # Deeper serial barrier blocks amortize the staging loop (swept, Phase 24:
    # mt=4: 338->359 Gw/s @ TBX=16; mt=8: 189->201 @ TBX=8; bit-exact — the
    # per-lane fma order is TBX-invariant). mt=2 unswept, keeps 4.
    tbx = {2: 4, 4: 16, 8: 8}.get(mt, 4)
    # A lane's two pairs are codewords 4*lane..4*lane+3 within its half —
    # 4-aligned, so the dq4 merged window (one 64-bit load pair per 4
    # codewords) applies for K in {1..6, 8}; k=7 keeps per-pair funnels
    # (3K+16 can straddle 3 words). Bit-exact: same cw bits, same fma order
    # (verified in tools/gemm_mt_bench.py; ~+2-3% in-stream).
    if k != 7:
        funnel_setup = """
    uint c0_ = half_ * 128u + lane * 4u;
    int e_last_ = int(c0_ + 4u) * K_BITS + 256 * K_BITS;
    int i_end_ = (e_last_ - 1) / 32;
    uint ig1_ = uint(i_end_) % PACKED_U32;
    uint ig0_ = uint(i_end_ - 1) % PACKED_U32;
    uint s3_ = uint((i_end_ + 1) * 32 - e_last_);
"""
        extract = """
                ulong merged = ((ulong)words[ig0_] << 32) | (ulong)words[ig1_];
                uint cws[4];
                cws[3] = uint(merged >> s3_) & 0xFFFFu;
                cws[2] = uint(merged >> (s3_ + K_BITS)) & 0xFFFFu;
                cws[1] = uint(merged >> (s3_ + 2u * K_BITS)) & 0xFFFFu;
                cws[0] = uint(merged >> (s3_ + 3u * K_BITS)) & 0xFFFFu;
"""
    else:
        funnel_setup = """
    uint i0_[2];
    uint i1_[2];
    uint s1_[2];
    for (uint j = 0u; j < 2u; j++) {
        uint t = half_ * 64u + lane * 2u + j;
        int b0 = int(t) * 2 * K_BITS + K_BITS - 16 + 256 * K_BITS;
        int b2 = b0 + K_BITS + 16;
        i0_[j] = uint(b0 / 32) % PACKED_U32;
        i1_[j] = uint((b2 - 1) / 32) % PACKED_U32;
        s1_[j] = uint(((b2 - 1) / 32 + 1) * 32 - b2);
    }
"""
        extract = """
                uint cws[4];
                for (uint jj = 0u; jj < 2u; jj++) {
                    ulong merged = ((ulong)words[i0_[jj]] << 32) | (ulong)words[i1_[jj]];
                    uint w1 = uint(merged >> s1_[jj]);
                    cws[jj * 2u] = (w1 >> K_BITS) & 0xFFFFu;
                    cws[jj * 2u + 1u] = w1 & 0xFFFFu;
                }
"""
    slot_body = "".join(
        f"""
            {{
                uint cw = cws[{i}];
{decode}
                threadgroup const float4_t* xv = (threadgroup const float4_t*)
                    &tg_x[buf][t][row_[{i}] * MT];
                for (uint q = 0u; q < MT4; q++) {{
                    acc4[{i}][q] = fma(xv[q], float4_t(dq_val), acc4[{i}][q]);
                }}
            }}
"""
        for i in range(4)
    )
    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}
#define MT {mt}u
#define VEC {vec}u
#define MT4 ({mt // vec}u)
#define float4_t {'float4' if vec == 4 else 'float2'}
#define TBX {tbx}u

    uint tn = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;     // 0..127 (4 simdgroups)
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint sgp = sg >> 1u;     // tile-stride group (0..1)
    uint half_ = sg & 1u;    // which 64 codewords of the tile
    uint in_tiles = dims[0];
    uint out_tiles = dims[1];
    uint batch = dims[2];
    uint n_splits = dims[3];
    uint n_sub = dims[4];
    uint in_features = in_tiles * 16u;
    uint out_features = out_tiles * 16u;
    uint sub = (n_sub > 1u) ? tile_sub[tn] : 0u;

    uint split = threadgroup_position_in_grid.z;
    uint tiles_per_split = (in_tiles + n_splits - 1u) / n_splits;
    uint tk_begin = split * tiles_per_split;
    uint tk_end = min(tk_begin + tiles_per_split, in_tiles);

    // Lane owns codeword pairs t = half*64 + lane*2 + {{0,1}} (4 weights).
    uint pos_[4];
    uint row_[4];
    for (uint j = 0u; j < 2u; j++) {{
        uint t = half_ * 64u + lane * 2u + j;
        pos_[j * 2u] = perm[t * 2u];
        pos_[j * 2u + 1u] = perm[t * 2u + 1u];
        row_[j * 2u] = pos_[j * 2u] >> 4u;
        row_[j * 2u + 1u] = pos_[j * 2u + 1u] >> 4u;
    }}
{funnel_setup}

    threadgroup float tg_x[2][TBX][16u * MT];
    float4_t acc4[4][MT4];
    for (uint j = 0u; j < 4u; j++) {{
        for (uint q = 0u; q < MT4; q++) {{
            acc4[j][q] = float4_t(0.0f);
        }}
    }}

    for (uint tk0 = tk_begin; tk0 < tk_end; tk0 += TBX) {{
        uint buf = (tk0 / TBX) & 1u;
        for (uint idx = tid; idx < TBX * MT * 16u; idx += 128u) {{
            uint t = idx / (MT * 16u);
            uint rem = idx % (MT * 16u);
            uint ri = rem / MT;
            uint mm = rem % MT;
            float v = 0.0f;
            if (tk0 + t < tk_end && mm < batch) {{
                v = float(xh[(mm * n_sub + sub) * in_features + (tk0 + t) * 16u + ri]);
            }}
            tg_x[buf][t][ri * MT + mm] = v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = sgp; t < TBX && tk0 + t < tk_end; t += 2u) {{
            const device uint* words = trellis + ((tk0 + t) * out_tiles + tn) * PACKED_U32;
{extract}
{slot_body}
        }}
    }}

    // Reduction: the two halves of an sgp cover disjoint positions; sum the
    // two sgp tile-stride groups, then fold rows per column, one row at a time.
    threadgroup float tg_w[2][256];
    for (uint mm = 0u; mm < MT; mm++) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < 4u; j++) {{
            tg_w[sgp][pos_[j]] = acc4[j][mm / VEC][mm % VEC];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 16u && mm < batch) {{
            float s = 0.0f;
            for (uint r = 0u; r < 16u; r++) {{
                uint p = r * 16u + tid;
                s += tg_w[0][p] + tg_w[1][p];
            }}
            out[(mm * n_splits + split) * out_features + tn * 16u + tid] = s;
        }}
    }}
"""


def _decode_full_source(k: int, cb: CodebookMode) -> str:
    """v13: decode a whole trellis to fp16 ``W (in, out)`` in ONE dispatch.

    Built for the prefill path: decode once per chunk, then run the GEMM on
    MLX's native matmul (tensor units). No barriers, no threadgroup memory —
    each lane owns 4 codeword pairs per tile and streams TBD tiles.
    """
    packed_u32 = k * 256 // 32
    decode = _decode_expr(cb, cw_in="cw")
    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}
// Deep serial tile loop: the per-threadgroup setup (perm reads + funnel
// geometry) amortizes over many tiles (swept 1..160 -> 64 optimal on M5 Max).
#define TBD 64u

    uint tn_local = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;     // 0..127
    uint in_tiles = dims[0];
    uint out_tiles = dims[1];     // stride of the source trellis
    uint tn = dims[2] + tn_local; // out-tile in the source trellis
    uint tn_count = dims[3];
    uint tpe = dims[4];           // >0: expert-grouped (E, in, out_e) layout
    uint out_features = tn_count * 16u;  // width of the DECODED slab
    uint tk0 = threadgroup_position_in_grid.y * TBD;

    // Thread owns ADJACENT row-major positions (2*tid, 2*tid+1): same row,
    // consecutive cols -> one coalesced half2 store. Funnel geometry of each
    // position's kernel-order codeword is tile-invariant.
    uint row = (tid * 2u) >> 4u;
    uint i0_[2];
    uint i1_[2];
    uint sh_[2];
    for (uint j = 0u; j < 2u; j++) {{
        uint src = inv_perm[tid * 2u + j];   // kernel-order weight index
        uint t = src >> 1u;                  // codeword pair
        int b0 = int(t) * 2 * K_BITS + K_BITS - 16 + 256 * K_BITS;
        int b2 = b0 + K_BITS + 16;
        i0_[j] = uint(b0 / 32) % PACKED_U32;
        i1_[j] = uint((b2 - 1) / 32) % PACKED_U32;
        // shift for w1 (odd index); even index needs K_BITS more
        sh_[j] = uint(((b2 - 1) / 32 + 1) * 32 - b2) + ((src & 1u) ? 0u : K_BITS);
    }}

    for (uint t = 0u; t < TBD; t++) {{
        uint tk = tk0 + t;
        if (tk >= in_tiles) {{
            break;
        }}
        const device uint* words = trellis + (tk * out_tiles + tn) * PACKED_U32;
        half2 val;
        for (uint j = 0u; j < 2u; j++) {{
            ulong merged = ((ulong)words[i0_[j]] << 32) | (ulong)words[i1_[j]];
            uint cw = uint(merged >> sh_[j]) & 0xFFFFu;
{decode}
            val[j] = half(dq_val);
        }}
        ulong base2;
        if (tpe > 0u) {{
            // expert-grouped (E, in, out_e): contiguous gather_mm rhs
            uint e = tn_local / tpe;
            uint tl = tn_local - e * tpe;
            base2 = (((ulong)e * (in_tiles * 16u) + tk * 16u + row) * (tpe * 16u)
                     + tl * 16u + ((tid * 2u) & 15u)) >> 1u;
        }} else {{
            base2 = ((ulong)(tk * 16u + row) * out_features + tn_local * 16u + ((tid * 2u) & 15u)) >> 1u;
        }}
        ((device half2*) wout)[base2] = val;
    }}
"""


_decode_full_kernels: dict[tuple[int, int], Callable[..., Any]] = {}


def _decode_full_kernel(k: int, cb: CodebookMode) -> Callable[..., Any]:
    key = (k, int(cb))
    if key not in _decode_full_kernels:
        _decode_full_kernels[key] = mx.fast.metal_kernel(
            name=f"exl3_decode_full_k{k}_cb{int(cb)}_v13",
            input_names=["trellis", "inv_perm", "dims"],
            output_names=["wout"],
            source=_decode_full_source(k, cb),
        )
    return _decode_full_kernels[key]


def decode_full_mlx(
    trellis_u16: mx.array,
    k: int,
    cb: CodebookMode | int,
    *,
    tn_offset: int = 0,
    tn_count: int | None = None,
) -> mx.array:
    """Decode the packed trellis (or an out-tile range) to fp16 ``W`` in one
    dispatch. ``tn_offset/tn_count`` select out-tile columns — used by fused
    groups to decode one member from the concatenated trellis contiguously."""
    cb = CodebookMode(cb)
    in_tiles, out_tiles, _ = trellis_u16.shape
    if tn_count is None:
        tn_count = out_tiles - tn_offset
    trellis_u32 = trellis_u16.reshape(-1).view(mx.uint32)
    dims = mx.array([in_tiles, out_tiles, tn_offset, tn_count, 0], dtype=mx.uint32)
    kernel = _decode_full_kernel(k, cb)
    out = kernel(
        inputs=[trellis_u32, _inv_perm_u32(), dims],
        template=[("T", mx.float16)],
        grid=(tn_count * _GEM_THREADS, (in_tiles + 63) // 64, 1),
        threadgroup=(_GEM_THREADS, 1, 1),
        output_shapes=[(in_tiles * 16 * tn_count * 16,)],
        output_dtypes=[mx.float16],
    )[0]
    return out.reshape(in_tiles * 16, tn_count * 16)


def decode_full_eg_mlx(
    trellis_u16: mx.array,
    k: int,
    cb: CodebookMode | int,
    *,
    tiles_per_e: int,
) -> mx.array:
    """Decode a stacked-expert trellis to CONTIGUOUS ``(E, in, out_e)`` fp16 —
    the exact gather_mm rhs layout (the swapaxes view off decode_full_t costs
    ~13% in the GEMM; this kernel writes expert-grouped directly)."""
    cb = CodebookMode(cb)
    in_tiles, out_tiles, _ = trellis_u16.shape
    if out_tiles % tiles_per_e != 0:
        raise ValueError("out_tiles not divisible by tiles_per_e")
    groups = out_tiles // tiles_per_e
    trellis_u32 = trellis_u16.reshape(-1).view(mx.uint32)
    dims = mx.array(
        [in_tiles, out_tiles, 0, out_tiles, tiles_per_e], dtype=mx.uint32
    )
    kernel = _decode_full_kernel(k, cb)
    out = kernel(
        inputs=[trellis_u32, _inv_perm_u32(), dims],
        template=[("T", mx.float16)],
        grid=(out_tiles * _GEM_THREADS, (in_tiles + 63) // 64, 1),
        threadgroup=(_GEM_THREADS, 1, 1),
        output_shapes=[(groups * in_tiles * 16 * tiles_per_e * 16,)],
        output_dtypes=[mx.float16],
    )[0]
    return out.reshape(groups, in_tiles * 16, tiles_per_e * 16)


def _decode_full_t_source(k: int, cb: CodebookMode) -> str:
    """v13t: decode a trellis out-tile range DIRECTLY to fp16 ``W^T`` laid out
    ``(out, in)`` — the gather_mm rhs layout — killing the per-chunk
    transpose+contiguous copies of the MoE prefill. Threads own two
    ADJACENT-ROW positions of one column (coalesced half2 along in)."""
    packed_u32 = k * 256 // 32
    decode = _decode_expr(cb, cw_in="cw")
    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}
#define TBD 64u

    uint tn_local = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint in_tiles = dims[0];
    uint out_tiles = dims[1];
    uint tn = dims[2] + tn_local;
    uint in_features = in_tiles * 16u;
    uint tk0 = threadgroup_position_in_grid.y * TBD;

    uint col = tid >> 3u;
    uint rp = tid & 7u;
    uint i0_[2];
    uint i1_[2];
    uint sh_[2];
    for (uint j = 0u; j < 2u; j++) {{
        uint src = inv_perm[(2u * rp + j) * 16u + col];
        uint t = src >> 1u;
        int b0 = int(t) * 2 * K_BITS + K_BITS - 16 + 256 * K_BITS;
        int b2 = b0 + K_BITS + 16;
        i0_[j] = uint(b0 / 32) % PACKED_U32;
        i1_[j] = uint((b2 - 1) / 32) % PACKED_U32;
        sh_[j] = uint(((b2 - 1) / 32 + 1) * 32 - b2) + ((src & 1u) ? 0u : K_BITS);
    }}

    for (uint t = 0u; t < TBD; t++) {{
        uint tk = tk0 + t;
        if (tk >= in_tiles) {{
            break;
        }}
        const device uint* words = trellis + (tk * out_tiles + tn) * PACKED_U32;
        half2 val;
        for (uint j = 0u; j < 2u; j++) {{
            ulong merged = ((ulong)words[i0_[j]] << 32) | (ulong)words[i1_[j]];
            uint cw = uint(merged >> sh_[j]) & 0xFFFFu;
{decode}
            val[j] = half(dq_val);
        }}
        ulong base2 = ((ulong)(tn_local * 16u + col) * in_features
                       + tk * 16u + 2u * rp) >> 1u;
        ((device half2*) wout)[base2] = val;
    }}
"""


_decode_full_t_kernels: dict[tuple[int, int], Any] = {}


def decode_full_t_mlx(
    trellis_u16: mx.array,
    k: int,
    cb: CodebookMode | int,
    *,
    tn_offset: int = 0,
    tn_count: int | None = None,
) -> mx.array:
    """Decode an out-tile range to fp16 ``(out, in)`` (gather_mm layout)."""
    cb = CodebookMode(cb)
    in_tiles, out_tiles, _ = trellis_u16.shape
    if tn_count is None:
        tn_count = out_tiles - tn_offset
    key = (k, int(cb))
    kernel = _decode_full_t_kernels.get(key)
    if kernel is None:
        kernel = _decode_full_t_kernels[key] = mx.fast.metal_kernel(
            name=f"exl3_decode_full_t_k{k}_cb{int(cb)}_v13t",
            input_names=["trellis", "inv_perm", "dims"],
            output_names=["wout"],
            source=_decode_full_t_source(k, cb),
        )
    dims = mx.array([in_tiles, out_tiles, tn_offset, 0, 0], dtype=mx.uint32)
    out = kernel(
        inputs=[trellis_u16.reshape(-1).view(mx.uint32), _inv_perm_u32(), dims],
        template=[("T", mx.float16)],
        grid=(tn_count * _GEM_THREADS, (in_tiles + 63) // 64, 1),
        threadgroup=(_GEM_THREADS, 1, 1),
        output_shapes=[(tn_count * 16 * in_tiles * 16,)],
        output_dtypes=[mx.float16],
    )[0]
    return out.reshape(tn_count * 16, in_tiles * 16)


def _gem_source(k: int, cb: CodebookMode, mt: int, use_lut: bool) -> str:
    packed_u32 = k * 256 // 32
    if use_lut:
        # decode_3inst is a pure function of the 16-bit codeword; the 128 KB
        # fp16 table (built by the same Metal decode kernel — bit-identical)
        # stays hot in L2 and replaces ~8 ALU ops per weight with one load.
        decode_pair = """
            float dq0 = float(lut[w0]);
            float dq1 = float(lut[w1]);
"""
    else:
        decode = _decode_expr(cb, cw_in="cw")
        decode_pair = f"""
            float dq0;
            float dq1;
            {{
                uint cw = w0;
{decode}
                dq0 = dq_val;
            }}
            {{
                uint cw = w1;
{decode}
                dq1 = dq_val;
            }}
"""
    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}
#define MT {mt}u

    // v10: cooperative tile decode, register accumulation, M-tiling, fused
    // sub-layers. Each threadgroup owns one out-tile (16 cols) and MT batch
    // rows. Every thread decodes one codeword PAIR per in-tile (one funnel
    // shift for 2 weights, CUDA dq2 pattern) in kernel order; the tile perm is
    // tile-invariant so a thread's weights always land at fixed (row, col) —
    // multiply by x[row] per batch row and accumulate in registers. Trellis
    // bytes are read once per MT rows. ``tile_sub`` selects which stacked xh
    // an out-tile belongs to (several same-K linears concatenated at load).
    uint tn = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint in_tiles = dims[0];
    uint out_tiles = dims[1];
    uint batch = dims[2];
    uint n_splits = dims[3];
    uint n_sub = dims[4];
    uint in_features = in_tiles * 16u;
    uint out_features = out_tiles * 16u;
    uint m0 = threadgroup_position_in_grid.y * MT;
    uint sub = (n_sub > 1u) ? tile_sub[tn] : 0u;

    // Tall layers (in_tiles >> out_tiles) underfill the GPU with one
    // threadgroup per out-tile; split the in-tile range across grid.z and
    // reduce the (n_splits, out) partials on the host graph.
    uint split = threadgroup_position_in_grid.z;
    uint tiles_per_split = (in_tiles + n_splits - 1u) / n_splits;
    uint tk_begin = split * tiles_per_split;
    uint tk_end = min(tk_begin + tiles_per_split, in_tiles);

    // TB in-tiles per barrier iteration: TB independent loads + decode chains
    // per thread keep ~TB x more bytes in flight and amortize the barrier.
    #define TB 4u
    threadgroup uint tg_words[2][TB][PACKED_U32];
    threadgroup float tg_x[2][TB][MT * 16u];
    threadgroup float tg_w[256];

    uint pos0 = perm[tid * 2u];
    uint pos1 = perm[tid * 2u + 1u];
    uint row0 = pos0 >> 4u;
    uint row1 = pos1 >> 4u;

    // Funnel geometry for this thread's codeword pair is loop-invariant.
    int b0 = int(tid) * 2 * K_BITS + K_BITS - 16 + 256 * K_BITS;
    int b2 = b0 + K_BITS + 16;
    uint i0 = uint(b0 / 32) % PACKED_U32;
    uint i1 = uint((b2 - 1) / 32) % PACKED_U32;
    uint s1 = uint(((b2 - 1) / 32 + 1) * 32 - b2);

    float acc0[MT];
    float acc1[MT];
    for (uint mm = 0u; mm < MT; mm++) {{
        acc0[mm] = 0.0f;
        acc1[mm] = 0.0f;
    }}

    for (uint tk = tk_begin; tk < tk_end; tk += TB) {{
        uint buf = (tk / TB) & 1u;
        if (tid < PACKED_U32) {{
            for (uint t = 0u; t < TB; t++) {{
                if (tk + t < tk_end) {{
                    tg_words[buf][t][tid] =
                        trellis[((tk + t) * out_tiles + tn) * PACKED_U32 + tid];
                }}
            }}
        }}
        // xh is logically (batch, n_sub, in_features); stage MT rows x TB tiles.
        for (uint idx = tid; idx < TB * MT * 16u; idx += 128u) {{
            uint t = idx / (MT * 16u);
            uint rem = idx % (MT * 16u);
            uint mm = rem >> 4u;
            uint ri = rem & 15u;
            float v = 0.0f;
            if (tk + t < tk_end && m0 + mm < batch) {{
                v = float(xh[((m0 + mm) * n_sub + sub) * in_features + (tk + t) * 16u + ri]);
            }}
            tg_x[buf][t][mm * 16u + ri] = v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0u; t < TB; t++) {{
            if (tk + t >= tk_end) {{
                break;
            }}
            ulong merged = ((ulong)tg_words[buf][t][i0] << 32) | (ulong)tg_words[buf][t][i1];
            uint w1 = uint(merged >> s1);
            uint w0 = (w1 >> K_BITS) & 0xFFFFu;
            w1 &= 0xFFFFu;
            // Full 16-bit sliding-window codewords (exl3_dq.cuh), not the low K bits.
{decode_pair}
            for (uint mm = 0u; mm < MT; mm++) {{
                acc0[mm] = fma(tg_x[buf][t][mm * 16u + row0], dq0, acc0[mm]);
                acc1[mm] = fma(tg_x[buf][t][mm * 16u + row1], dq1, acc1[mm]);
            }}
        }}
    }}

    // End reduction, one batch row at a time: park the per-position partial
    // dots, then 16 threads fold their column.
    for (uint mm = 0u; mm < MT; mm++) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_w[pos0] = acc0[mm];
        tg_w[pos1] = acc1[mm];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 16u && m0 + mm < batch) {{
            float s = 0.0f;
            for (uint r = 0u; r < 16u; r++) {{
                s += tg_w[r * 16u + tid];
            }}
            out[((m0 + mm) * n_splits + split) * out_features + tn * 16u + tid] = s;
        }}
    }}
"""


_M_TILE = 8
# Computed 3INST decode beats the 128 KB LUT end-to-end on M5 Max (random
# gathers are latency-bound: 27B 13.9 vs 11.7 tok/s). EXL3_GEMV_LUT=1 opts in.
_USE_LUT = os.environ.get("EXL3_GEMV_LUT", "0") == "1"
# simdgroup-cooperative GEMV (v12, ds4 pattern) for M=1; EXL3_GEMV_SIMD=0
# falls back to the staged v10 kernel.
_USE_SIMD_GEMV = os.environ.get("EXL3_GEMV_SIMD", "1") != "0"
_gemv_simd_kernels: dict[tuple[int, int, int, int], Any] = {}


def _gemv_simd_kernel(k: int, cb: CodebookMode, mt: int = 1, had_nb: int = 0) -> Any:
    key = (k, int(cb), mt, had_nb)
    if key not in _gemv_simd_kernels:
        if mt == 1:
            src = _gemv_simd_source(k, cb, 1, had_nb)
        else:
            src = _gemm_simd_source(k, cb, mt)
        names = (
            ["xh", "suh", "trellis", "perm", "tile_sub", "tile_map", "dims"]
            if had_nb
            else ["xh", "trellis", "perm", "tile_sub", "tile_map", "dims"]
        )
        _gemv_simd_kernels[key] = mx.fast.metal_kernel(
            name=f"exl3_gemv_simd_k{k}_cb{int(cb)}_mt{mt}_nb{had_nb}_v19",
            input_names=names,
            output_names=["out"],
            source=src,
        )
    return _gemv_simd_kernels[key]


_gemv_post_kernels: dict[tuple[int, int], Callable[..., Any]] = {}
# Fused post-Hadamard epilogue GEMV (v18). Measured 4-8% SLOWER than the
# separate finish dispatch on M5 Max (8 sequential out-tiles per threadgroup
# cut occupancy more than the saved dispatch gap) — opt-in for other hardware.
_USE_POST_FUSE = os.environ.get("EXL3_FUSE_POST", "0") == "1"


def _gemv_post_kernel(k: int, cb: CodebookMode) -> Callable[..., Any]:
    key = (k, int(cb))
    if key not in _gemv_post_kernels:
        _gemv_post_kernels[key] = mx.fast.metal_kernel(
            name=f"exl3_gemv_post_k{k}_cb{int(cb)}_v18",
            input_names=["xh", "svh", "trellis", "perm", "tile_sub", "dims"],
            output_names=["out"],
            source=_gemv_post_source(k, cb),
        )
    return _gemv_post_kernels[key]


def inner_gemv_post_mlx(
    xh: mx.array,
    svh: mx.array,
    trellis_u16: mx.array,
    k: int,
    cb: CodebookMode | int,
    tile_sub: mx.array | None = None,
    n_sub: int = 1,
) -> mx.array | None:
    """M=1 GEMV with post-Hadamard + svh fused (one dispatch fewer per linear).

    ``xh`` is the PREPARED activation (n_sub, in); returns the finished output
    (still fp32; caller casts). None if the shape can't use this path.
    """
    cb = CodebookMode(cb)
    if not _USE_POST_FUSE or k == 7:
        return None
    in_tiles, out_tiles, _ = trellis_u16.shape
    if out_tiles % 8 != 0:
        return None
    out_features = out_tiles * 16
    blocks = out_tiles // 8

    n_splits = 1
    while blocks * n_splits < 4096 and in_tiles // (n_splits * 2) >= 32:
        n_splits *= 2
    dims = mx.array([in_tiles, out_tiles, 1, n_splits, n_sub], dtype=mx.uint32)
    kernel = _gemv_post_kernel(k, cb)
    out = kernel(
        inputs=[
            xh.reshape(-1).astype(mx.float16),
            svh.reshape(-1).astype(mx.float16),
            trellis_u16.reshape(-1).view(mx.uint32),
            _fwd_perm_u32(),
            tile_sub if tile_sub is not None else _dummy_sub(),
            dims,
        ],
        template=[("T", mx.float32)],
        grid=(blocks * _GEM_THREADS, 1, n_splits),
        threadgroup=(_GEM_THREADS, 1, 1),
        output_shapes=[(n_splits * out_features,)],
        output_dtypes=[mx.float32],
    )[0]
    out = out.reshape(1, n_splits, out_features)
    if n_splits > 1:
        out = out.sum(axis=1)
    return out.reshape(1, out_features)


_HAD_NB_BUCKETS = (1280, 2560, 5120)


def inner_gemv_had_mlx(
    x_raw: mx.array,
    suh: mx.array,
    trellis_u16: mx.array,
    k: int,
    cb: CodebookMode | int,
    tile_sub: mx.array | None = None,
    n_sub: int = 1,
) -> mx.array | None:
    """Fused pre-Hadamard + GEMV (M=1): one dispatch instead of prep + GEMV.

    ``x_raw`` is the UNROTATED activation (in,); ``suh`` is (n_sub * in,).
    Returns None when the shape can't use the fused path (caller falls back).
    """
    cb = CodebookMode(cb)
    if k == 7:
        return None
    in_tiles, out_tiles, _ = trellis_u16.shape

    n_splits = 1
    while out_tiles * n_splits < 8192 and in_tiles // (n_splits * 2) >= 128:
        n_splits *= 2
    # split ranges must align to whole 128-wide Hadamard blocks (8 tiles)
    while n_splits > 1 and ((in_tiles + n_splits - 1) // n_splits) % 8 != 0:
        n_splits //= 2
    tiles_per_split = (in_tiles + n_splits - 1) // n_splits
    need = tiles_per_split * 16
    nb = next((b for b in _HAD_NB_BUCKETS if need <= b), None)
    if nb is None:
        return None

    trellis_u32 = trellis_u16.reshape(-1).view(mx.uint32)
    out_features = out_tiles * 16
    dims = mx.array([in_tiles, out_tiles, 1, n_splits, n_sub, 0], dtype=mx.uint32)
    kernel = _gemv_simd_kernel(k, cb, 1, nb)
    out = kernel(
        inputs=[
            x_raw.reshape(-1).astype(mx.float16),
            suh.reshape(-1).astype(mx.float16),
            trellis_u32,
            _fwd_perm_u32(),
            tile_sub if tile_sub is not None else _dummy_sub(),
            _dummy_sub(),
            dims,
        ],
        template=[("T", mx.float32)],
        grid=(out_tiles * _GEM_THREADS, 1, n_splits),
        threadgroup=(_GEM_THREADS, 1, 1),
        output_shapes=[(n_splits * out_features,)],
        output_dtypes=[mx.float32],
    )[0]
    out = out.reshape(1, n_splits, out_features)
    if n_splits > 1:
        out = out.sum(axis=1)
    return out.reshape(1, out_features)
_dummy_sub_arr: mx.array | None = None
_DECODE_LUTS: dict[int, mx.array] = {}


def _dummy_sub() -> mx.array:
    global _dummy_sub_arr
    if _dummy_sub_arr is None:
        _dummy_sub_arr = mx.zeros((1,), dtype=mx.uint32)
    return _dummy_sub_arr


def _decode_lut(cb: CodebookMode) -> mx.array:
    """All 65536 codeword values, decoded by the same Metal decode kernel."""
    key = int(cb)
    lut = _DECODE_LUTS.get(key)
    if lut is None:
        from ponyexl3.mlx.metal_kernels import decode_codewords_mlx

        lut = decode_codewords_mlx(mx.arange(65536, dtype=mx.uint32), cb)
        mx.eval(lut)
        _DECODE_LUTS[key] = lut
    return lut


def _gemm_devx_source(k: int, cb: CodebookMode, mt: int) -> str:
    """v20: small-batch GEMM, x device-direct from a TRANSPOSED (n_sub, in,
    MT) half buffer — one contiguous load per weight serves all MT rows.

    The v16 staged design is threadgroup-BANDWIDTH bound at mt=8 (32 B of x
    per weight through tg memory; measured 201 Gw/s vs the ~308 slot model,
    and halving acc registers changed nothing). Device-direct transposed
    loads ride L1 instead: mt=8 309-318 Gw/s (+58%), mt=4 456-486 (+33%,
    i.e. AT the mt=1 GEMV's rate — the mt-tax is gone). Same lane/fma
    order and dq4 bits as v16, so outputs are bit-identical; zero hot-loop
    barriers. k=7 is excluded upstream (use_simd) like v16."""
    packed_u32 = k * 256 // 32
    decode = _decode_expr(cb, cw_in="cw")
    vec = 4 if mt % 4 == 0 else 2
    nq = mt // vec
    half_t = "half4" if vec == 4 else "half2"
    float_t = "float4" if vec == 4 else "float2"
    slot_body = "".join(
        f"""
        {{
            uint cw = cws[{i}];
{decode}
            const device {half_t}* xp =
                (const device {half_t}*)(xt + (ulong)(tk * 16u + row_[{i}]) * mt_total);
"""
        + "".join(
            f"""            acc4[{i}][{q}] = fma({float_t}(xp[{q}]), {float_t}(dq_val), acc4[{i}][{q}]);
"""
            for q in range(nq)
        )
        + """        }
"""
        for i in range(4)
    )
    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}
#define MT {mt}u

    uint tn = threadgroup_position_in_grid.x;
    uint mg = threadgroup_position_in_grid.y;        // row group (8 rows each)
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint sgp = sg >> 1u;
    uint half_ = sg & 1u;
    uint in_tiles = dims[0];
    uint out_tiles = dims[1];
    uint batch = dims[2];
    uint n_splits = dims[3];
    uint n_sub = dims[4];
    uint mt_total = dims[5];                         // padded row count in xt
    uint mm_base = mg * MT;
    uint in_features = in_tiles * 16u;
    uint out_features = out_tiles * 16u;
    uint sub = (n_sub > 1u) ? tile_sub[tn] : 0u;

    uint split = threadgroup_position_in_grid.z;
    uint tiles_per_split = (in_tiles + n_splits - 1u) / n_splits;
    uint tk_begin = split * tiles_per_split;
    uint tk_end = min(tk_begin + tiles_per_split, in_tiles);

    uint pos_[4];
    uint row_[4];
    for (uint j = 0u; j < 2u; j++) {{
        uint t = half_ * 64u + lane * 2u + j;
        pos_[j * 2u] = perm[t * 2u];
        pos_[j * 2u + 1u] = perm[t * 2u + 1u];
        row_[j * 2u] = pos_[j * 2u] >> 4u;
        row_[j * 2u + 1u] = pos_[j * 2u + 1u] >> 4u;
    }}
    uint c0_ = half_ * 128u + lane * 4u;
    int e_last_ = int(c0_ + 4u) * K_BITS + 256 * K_BITS;
    int i_end_ = (e_last_ - 1) / 32;
    uint ig1_ = uint(i_end_) % PACKED_U32;
    uint ig0_ = uint(i_end_ - 1) % PACKED_U32;
    uint s3_ = uint((i_end_ + 1) * 32 - e_last_);

    {float_t} acc4[4][{nq}];
    for (uint j = 0u; j < 4u; j++) {{
        for (uint q = 0u; q < {nq}u; q++) {{
            acc4[j][q] = {float_t}(0.0f);
        }}
    }}

    const device half* xt = xh + (ulong)sub * in_features * mt_total + mm_base;
    for (uint tk = tk_begin + sgp; tk < tk_end; tk += 2u) {{
        const device uint* words = trellis + ((ulong)tk * out_tiles + tn) * PACKED_U32;
        ulong merged = ((ulong)words[ig0_] << 32) | (ulong)words[ig1_];
        uint cws[4];
        cws[3] = uint(merged >> s3_) & 0xFFFFu;
        cws[2] = uint(merged >> (s3_ + K_BITS)) & 0xFFFFu;
        cws[1] = uint(merged >> (s3_ + 2u * K_BITS)) & 0xFFFFu;
        cws[0] = uint(merged >> (s3_ + 3u * K_BITS)) & 0xFFFFu;
{slot_body}
    }}

    threadgroup float tg_w[2][256];
    for (uint mm = 0u; mm < MT; mm++) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < 4u; j++) {{
            tg_w[sgp][pos_[j]] = acc4[j][mm / {vec}u][mm % {vec}u];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 16u && mm_base + mm < batch) {{
            float s = 0.0f;
            for (uint r = 0u; r < 16u; r++) {{
                uint p = r * 16u + tid;
                s += tg_w[0][p] + tg_w[1][p];
            }}
            out[((mm_base + mm) * n_splits + split) * out_features + tn * 16u + tid] = s;
        }}
    }}
"""


_gemm_devx_kernels: dict[tuple[int, int, int], Any] = {}
_USE_DEVX = os.environ.get("EXL3_GEMM_DEVX", "1") == "1"


def _gemm_devx_kernel(k: int, cb: CodebookMode, mt: int) -> Callable[..., Any]:
    key = (k, int(cb), mt)
    if key not in _gemm_devx_kernels:
        _gemm_devx_kernels[key] = mx.fast.metal_kernel(
            name=f"exl3_gemm_devx_k{k}_cb{int(cb)}_mt{mt}_v20",
            input_names=["xh", "trellis", "perm", "tile_sub", "tile_map", "dims"],
            output_names=["out"],
            source=_gemm_devx_source(k, cb, mt),
        )
    return _gemm_devx_kernels[key]


def _gem_kernel(k: int, cb: CodebookMode, mt: int, use_lut: bool) -> Callable[..., Any]:
    key = (k, int(cb), mt, use_lut)
    if key not in _gem_kernels:
        _gem_kernels[key] = mx.fast.metal_kernel(
            name=f"exl3_gem_k{k}_cb{int(cb)}_mt{mt}_lut{int(use_lut)}_v11",
            input_names=["xh", "trellis", "perm", "tile_sub", "lut", "dims"],
            output_names=["out"],
            source=_gem_source(k, cb, mt, use_lut),
        )
    return _gem_kernels[key]


def _run_inner_gem(
    xh: mx.array,
    trellis_u16: mx.array,
    k: int,
    cb: CodebookMode | int,
    *,
    tile_sub: mx.array | None = None,
) -> mx.array:
    """Inner GEMM on the packed trellis.

    ``xh`` is (batch, in), or (batch, n_sub, in) for fused sub-layer groups
    with ``tile_sub`` mapping each out-tile to its sub-layer's xh row.
    """
    cb = CodebookMode(cb)
    if not 1 <= k <= 8:
        raise ValueError(f"K must be in [1, 8], got {k}")
    if trellis_u16.ndim != 3:
        raise ValueError("trellis must be 3D")

    if xh.ndim == 3:
        batch, n_sub, in_features = xh.shape
        if tile_sub is None:
            raise ValueError("stacked xh requires tile_sub")
    else:
        xh = xh.reshape(-1, xh.shape[-1])
        batch, in_features = xh.shape
        n_sub = 1
    # fp16 activations halve x traffic; the kernel stages them as fp32 in
    # threadgroup memory and accumulates in fp32 regardless.
    xh_flat = xh.astype(mx.float16).reshape(-1)
    in_tiles, out_tiles, _ = trellis_u16.shape
    if in_features != in_tiles * 16:
        raise ValueError("xh in_features mismatch")

    # Reinterpret the device-resident uint16 trellis as uint32 words (same
    # little-endian byte layout the CUDA kernels read) — no host roundtrip.
    trellis_u32 = trellis_u16.reshape(-1).view(mx.uint32)
    out_features = out_tiles * 16

    # Decode each tile once per MT batch rows.
    mt = 1 if batch == 1 else _M_TILE
    m_groups = (batch + mt - 1) // mt

    # The barrier-free simd kernel covers small batches with one row-group
    # (register accumulators per row); larger batches use the staged kernel.
    # k=7 breaks the dq4 two-word invariant (3K+16 = 37 bits can straddle 3
    # words); it falls back to the staged kernel.
    # The v20 devx kernel additionally covers rows 9-16 with TWO row groups
    # along grid.y (each group re-decodes the trellis — the inherent
    # per-8-rows amortization limit; still far ahead of the staged path).
    devx_ok = _USE_DEVX and 1 < batch <= 16 and k != 7
    use_simd = _USE_SIMD_GEMV and (batch <= 8 or devx_ok) and k != 7
    simd_mt = 1 if batch == 1 else (2 if batch == 2 else (4 if batch <= 4 else 8))
    devx_groups = (batch + simd_mt - 1) // simd_mt if devx_ok else 1

    # Enough threadgroups to fill the GPU (measured optimum ~8k on M5 Max),
    # but never shrink a split below 32 in-tiles — short loops drown in
    # partial-sum overhead (measured: 2B layers regressed 2x at fine splits).
    # The simd kernel strides 4 simdgroups over each split: keep >=32 tiles
    # per simdgroup so the per-lane setup amortizes. The mt>1 layout has half
    # the per-lane setup, so it tolerates finer splits (swept).
    # Exception (Phase 24, the MoE Phase 22 lesson at dense scale): tiny-out
    # layers (in_proj_b/a: out_tiles=2) at mt>1 otherwise launch ~4
    # threadgroups on the serial critical path of every verify — latency
    # beats amortization there (swept 32.5 -> 21 µs at n=16..20).
    min_split_tiles = (128 if simd_mt == 1 else 64) if use_simd else 32
    if use_simd and simd_mt > 1 and out_tiles <= 8:
        min_split_tiles = 8
    n_splits = 1
    while (
        out_tiles * m_groups * n_splits < 8192
        and in_tiles // (n_splits * 2) >= min_split_tiles
    ):
        n_splits *= 2
    dims = mx.array([in_tiles, out_tiles, batch, n_splits, n_sub, 0], dtype=mx.uint32)

    if use_simd:
        grid_y = 1
        if devx_ok and simd_mt > 1:
            # v20 device-direct path: transpose to (n_sub, in, B) and pad
            # the row dim to MT*groups so each weight's rows ride one
            # vector load; rows 9-16 take a second row group along grid.y.
            mt_total = simd_mt * devx_groups
            x3 = xh.astype(mx.float16)
            if x3.ndim == 2:
                x3 = x3[:, None, :]
            xt = x3.transpose(1, 2, 0)
            if batch < mt_total:
                xt = mx.pad(xt, [(0, 0), (0, 0), (0, mt_total - batch)])
            xin = mx.contiguous(xt).reshape(-1)
            kernel = _gemm_devx_kernel(k, cb, simd_mt)
            dims = mx.array(
                [in_tiles, out_tiles, batch, n_splits, n_sub, mt_total],
                dtype=mx.uint32,
            )
            grid_y = devx_groups
        else:
            xin = xh_flat
            kernel = _gemv_simd_kernel(k, cb, simd_mt)
        out = kernel(
            inputs=[
                xin,
                trellis_u32,
                _fwd_perm_u32(),
                tile_sub if tile_sub is not None else _dummy_sub(),
                _dummy_sub(),
                dims,
            ],
            template=[("T", mx.float32)],
            grid=(out_tiles * _GEM_THREADS, grid_y, n_splits),
            threadgroup=(_GEM_THREADS, 1, 1),
            output_shapes=[(batch * n_splits * out_features,)],
            output_dtypes=[mx.float32],
        )[0]
        out = out.reshape(batch, n_splits, out_features)
        if n_splits > 1:
            out = out.sum(axis=1)
        return out.reshape(batch, out_features)

    kernel = _gem_kernel(k, cb, mt, _USE_LUT)
    out = kernel(
        inputs=[
            xh_flat,
            trellis_u32,
            _fwd_perm_u32(),
            tile_sub if tile_sub is not None else _dummy_sub(),
            _decode_lut(cb) if _USE_LUT else _dummy_sub(),
            dims,
        ],
        template=[("T", mx.float32)],
        grid=(out_tiles * _GEM_THREADS, m_groups, n_splits),
        threadgroup=(_GEM_THREADS, 1, 1),
        output_shapes=[(batch * n_splits * out_features,)],
        output_dtypes=[mx.float32],
    )[0]
    out = out.reshape(batch, n_splits, out_features)
    if n_splits > 1:
        out = out.sum(axis=1)
    return out.reshape(batch, out_features)


def inner_gem_fused_mlx(
    xh_stack: mx.array,
    trellis_u16: mx.array,
    k: int,
    cb: CodebookMode | int,
    tile_sub: mx.array,
) -> mx.array:
    """Fused-group GEMM: ``xh_stack`` is (batch, n_sub, in) — one launch for
    several same-K linears whose trellises are concatenated along out_tiles."""
    if xh_stack.ndim == 2:
        xh_stack = xh_stack[None]
    return _run_inner_gem(xh_stack, trellis_u16, k, cb, tile_sub=tile_sub)


def inner_gemv_mlx(
    xh: mx.array,
    trellis_u16: mx.array,
    k: int,
    cb: CodebookMode | int,
) -> mx.array:
    """Inner GEMV ``y = xh @ W`` for a single Hadamard-transformed row."""
    flat = xh.reshape(-1).astype(mx.float32)
    if flat.size != flat.shape[0]:
        raise ValueError("invalid xh")
    return _run_inner_gem(flat.reshape(1, -1), trellis_u16, k, cb).reshape(-1)


def inner_gemm_mlx(
    xh: mx.array,
    trellis_u16: mx.array,
    k: int,
    cb: CodebookMode | int,
) -> mx.array:
    """Inner batched GEMM ``Y = XH @ W`` without materializing ``W`` (Phase 3b)."""
    return _run_inner_gem(xh, trellis_u16, k, cb)


# ---------------------------------------------------------------------------
# v19c: segmented trellis GEMM on simdgroup matrices (MoE prefill, M > 8)
# ---------------------------------------------------------------------------
# llama.cpp mul_mm pattern with trellis decode as the dequant stage. Each
# threadgroup owns a (64-row, 64-col) tile of one expert SEGMENT of the
# pre-sorted (token, slot) rows; 4 simdgroups split it 2x2 into 32x32
# sub-tiles (16 fp32 accumulator fragments each — the register budget).
# Per 32-deep k-stage it decodes 2 in-tiles x 4 out-tiles ONCE into
# threadgroup memory and runs simdgroup_half8x8 mma with A fragments loaded
# DEVICE-DIRECT from the sorted x rows (callers pad x by one block so tail
# loads stay in bounds; out-of-segment rows compute garbage that the store
# discards). No fp16 W ever touches device memory.
#
# Measured apportioning (S=4096, 35B-A3B dims): decode ~1%, x staging ~8%,
# mma loop ~91% — so v19b's dq4 funnels and double-buffering were noise and
# were dropped; this version spends its complexity budget on the mma loop
# (0.5 loads/mma vs 0.625, one barrier per 32-k vs two per 16-k).
# EXL3_MM_NODECODE=1 keeps the decode-ablation diagnostic.

_MM_BM = 64  # token rows per block
_MM_BN = 64  # out cols per threadgroup
_MM_WS = 72  # wblk row stride in halfs (32 rows x 64 cols, padded)

_mm_seg_kernels: dict[tuple[int, int], Callable[..., Any]] = {}


def _mm_seg_source(k: int, cb: CodebookMode) -> str:
    packed_u32 = k * 256 // 32
    decode = _decode_expr(cb, cw_in="cw")
    if os.environ.get("EXL3_MM_NODECODE", "0") == "1":
        decode = "float dq_val = float(cw & 1u);"
    decode_lines = decode_bs(decode)
    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}

    uint b = threadgroup_position_in_grid.y;
    if (b >= nbr[0]) {{
        return;
    }}
    uint colb = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;     // 128 threads
    uint lane = tid & 31u;
    uint sgid = tid >> 5u;
    uint sgr = sgid >> 1u;                            // 2x2 simdgroup grid
    uint sgc = sgid & 1u;

    uint in_tiles = dims[0];
    uint tiles_per_e = dims[1];
    uint src_tiles = dims[2];
    uint tn_base = dims[3];
    uint out_e = dims[4];
    uint nb_max = dims[5];
    uint in_features = in_tiles * 16u;

    uint e = blk_tab[b];
    uint row0 = blk_tab[nb_max + b];
    uint blk_len = blk_tab[2u * nb_max + b];
    uint tn0 = tn_base + e * tiles_per_e + colb * 4u;

    // decode-position precompute (fixed per thread; decode_full_t mapping)
    uint dcol = tid >> 3u;        // out col 0..15 within tile
    uint rp = tid & 7u;           // in row-pair 0..7
    uint i0_[2];
    uint i1_[2];
    uint sh_[2];
    for (uint j = 0u; j < 2u; j++) {{
        uint src = inv_perm[(2u * rp + j) * 16u + dcol];
        uint t = src >> 1u;
        int b0 = int(t) * 2 * K_BITS + K_BITS - 16 + 256 * K_BITS;
        int b2 = b0 + K_BITS + 16;
        i0_[j] = uint(b0 / 32) % PACKED_U32;
        i1_[j] = uint((b2 - 1) / 32) % PACKED_U32;
        sh_[j] = uint(((b2 - 1) / 32 + 1) * 32 - b2) + ((src & 1u) ? 0u : K_BITS);
    }}

    threadgroup half wblk[32u * {_MM_WS}u];   // 2 in-tiles x 4 out-tiles
    threadgroup float csc[4u * 64u];

    simdgroup_float8x8 C[4][4];
    for (uint r = 0u; r < 4u; r++) {{
        for (uint c = 0u; c < 4u; c++) {{
            C[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }}
    }}

    const device half* xrow = xh + (ulong)(row0 + sgr * 32u) * in_features;
    uint stages = in_tiles >> 1u;             // in_tiles is even (>=32)
    for (uint ks = 0u; ks < stages; ks++) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint it2 = 0u; it2 < 2u; it2++) {{
            uint tk = ks * 2u + it2;
            for (uint lt = 0u; lt < 4u; lt++) {{
                const device uint* words =
                    trellis + ((ulong)tk * src_tiles + tn0 + lt) * PACKED_U32;
                for (uint j = 0u; j < 2u; j++) {{
                    ulong merged =
                        ((ulong)words[i0_[j]] << 32) | (ulong)words[i1_[j]];
                    uint cw = uint(merged >> sh_[j]) & 0xFFFFu;
{decode_lines}
                    wblk[(it2 * 16u + 2u * rp + j) * {_MM_WS}u + lt * 16u + dcol] =
                        half(dq_val);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k2 = 0u; k2 < 4u; k2++) {{
            simdgroup_half8x8 bf0;
            simdgroup_half8x8 bf1;
            simdgroup_half8x8 bf2;
            simdgroup_half8x8 bf3;
            const threadgroup half* wrow =
                &wblk[k2 * 8u * {_MM_WS}u + sgc * 32u];
            simdgroup_load(bf0, wrow, {_MM_WS}u);
            simdgroup_load(bf1, wrow + 8u, {_MM_WS}u);
            simdgroup_load(bf2, wrow + 16u, {_MM_WS}u);
            simdgroup_load(bf3, wrow + 24u, {_MM_WS}u);
            for (uint r = 0u; r < 4u; r++) {{
                simdgroup_half8x8 a;
                simdgroup_load(
                    a, xrow + (ulong)(r * 8u) * in_features + ks * 32u + k2 * 8u,
                    in_features);
                simdgroup_multiply_accumulate(C[r][0], a, bf0, C[r][0]);
                simdgroup_multiply_accumulate(C[r][1], a, bf1, C[r][1]);
                simdgroup_multiply_accumulate(C[r][2], a, bf2, C[r][2]);
                simdgroup_multiply_accumulate(C[r][3], a, bf3, C[r][3]);
            }}
        }}
    }}

    // store: fp32 frags -> tg scratch -> half device rows within the segment
    uint cb0 = colb * {_MM_BN}u + sgc * 32u;
    uint rbase = sgr * 32u;
    for (uint r = 0u; r < 4u; r++) {{
        if (rbase + r * 8u >= blk_len) {{
            break;
        }}
        for (uint cf = 0u; cf < 4u; cf++) {{
            simdgroup_store(C[r][cf], &csc[sgid * 64u], 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 64u; i += 32u) {{
                uint rr = i >> 3u;
                uint cc = i & 7u;
                uint gr = rbase + r * 8u + rr;
                if (gr < blk_len) {{
                    out[(ulong)(row0 + gr) * out_e + cb0 + cf * 8u + cc] =
                        half(csc[sgid * 64u + i]);
                }}
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}
"""


def decode_bs(decode: str) -> str:
    """Strip ``//`` comments (line splicing runs BEFORE comment removal in
    the preprocessor) and drop blank lines; used for code inlined into
    deeply-indented kernel loops."""
    lines = []
    for line in decode.rstrip().splitlines():
        code = line.split("//", 1)[0].rstrip()
        if code.strip():
            lines.append(code)
    return "\n".join(lines)


def inner_mm_seg_mlx(
    xh_sorted: mx.array,  # (>= n_rows + 64, in) fp16, rows sorted by expert
    trellis_u16: mx.array,  # stacked (in_tiles, src_tiles, P)
    k: int,
    cb: CodebookMode | int,
    blk_tab: mx.array,  # (3, nb_max) u32: [expert, row0, len] per block
    nb_real: mx.array,  # (1,) u32 device scalar: live block count
    *,
    n_rows: int,
    tn_base: int,
    tiles_per_e: int,
    out_e: int,
) -> mx.array:
    """Segmented trellis GEMM: one launch over all experts' sorted row blocks.

    ``xh_sorted`` must be padded to at least ``n_rows + 64`` rows (the kernel
    loads A fragments device-direct; the pad keeps ragged tail blocks in
    bounds — pad values are multiplied into discarded output rows).
    """
    cb = CodebookMode(cb)
    in_tiles, src_tiles, _ = trellis_u16.shape
    nb_max = int(blk_tab.shape[1])
    if out_e % _MM_BN != 0:
        raise ValueError(f"out_e {out_e} not a multiple of {_MM_BN}")
    if in_tiles % 2 != 0:
        raise ValueError("in_tiles must be even")
    if int(xh_sorted.shape[0]) < n_rows + _MM_BM:
        raise ValueError("xh_sorted must be padded by one block")
    key = (k, int(cb))
    kernel = _mm_seg_kernels.get(key)
    if kernel is None:
        kernel = _mm_seg_kernels[key] = mx.fast.metal_kernel(
            name=f"exl3_mm_seg_k{k}_cb{int(cb)}_v19c",
            input_names=["xh", "trellis", "inv_perm", "blk_tab", "nbr", "dims"],
            output_names=["out"],
            source=_mm_seg_source(k, cb),
            header="#include <metal_simdgroup_matrix>\n#include <metal_stdlib>\nusing namespace metal;\n",
        )
    dims = mx.array(
        [in_tiles, tiles_per_e, src_tiles, tn_base, out_e, nb_max], dtype=mx.uint32
    )
    out = kernel(
        inputs=[
            xh_sorted.reshape(-1),
            trellis_u16.reshape(-1).view(mx.uint32),
            _inv_perm_u32(),
            blk_tab.reshape(-1),
            nb_real,
            dims,
        ],
        template=[("T", mx.float16)],
        grid=((out_e // _MM_BN) * _GEM_THREADS, nb_max, 1),
        threadgroup=(_GEM_THREADS, 1, 1),
        output_shapes=[(n_rows, out_e)],
        output_dtypes=[mx.float16],
    )[0]
    return out
