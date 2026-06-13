#!/usr/bin/env python3
"""mt=8 GEMM cliff: v16 (2 pairs/lane, 32 acc regs) vs 1-pair/lane variant.

The slot model says mt=8 should run ~308 Gw/s; it measures 201 — the gap
profile matches register pressure (acc4[4][2] = 32 fp32 accumulators).
Variant: each lane owns ONE codeword pair (acc4[2][2] = 16 regs, the mt=4
footprint); a tile is covered by all four simdgroups in quarters and the
final reduction is a single disjoint tg_w[256] pass.

Also benches the pre-simd staged kernel (EXL3_GEMV_SIMD=0 path) for
routing reference, and a half-staged-x variant of the winner.
"""

from __future__ import annotations

import time


import numpy as np
import mlx.core as mx

from ponyexl3.mlx.gemv_metal import (
    _GEM_THREADS,
    _decode_expr,
    _fwd_perm_u32,
    _gemv_simd_kernel,
)
from ponyexl3.ref.codebook import CodebookMode

_cache: dict = {}


def _onepair_source(k: int, cb: CodebookMode, *, tbx: int = 8, half_x: bool = False) -> str:
    packed_u32 = k * 256 // 32
    decode = _decode_expr(cb, cw_in="cw")
    xt = "half" if half_x else "float"
    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}
#define MT 8u
#define MT4 2u
#define TBX {tbx}u

    uint tn = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;     // 128 threads, 4 sgs
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
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

    // Lane owns ONE codeword pair t (2 weights): 128 lanes cover the tile.
    uint t = sg * 32u + lane;
    int b0 = int(t) * 2 * K_BITS + K_BITS - 16 + 256 * K_BITS;
    int b2 = b0 + K_BITS + 16;
    uint i0_ = uint(b0 / 32) % PACKED_U32;
    uint i1_ = uint((b2 - 1) / 32) % PACKED_U32;
    uint s1_ = uint(((b2 - 1) / 32 + 1) * 32 - b2);
    uint pos_[2];
    uint row_[2];
    pos_[0] = perm[t * 2u];
    pos_[1] = perm[t * 2u + 1u];
    row_[0] = pos_[0] >> 4u;
    row_[1] = pos_[1] >> 4u;

    threadgroup {xt} tg_x[2][TBX][16u * MT];
    float4 acc4[2][MT4];
    for (uint j = 0u; j < 2u; j++) {{
        for (uint q = 0u; q < MT4; q++) {{
            acc4[j][q] = float4(0.0f);
        }}
    }}

    for (uint tk0 = tk_begin; tk0 < tk_end; tk0 += TBX) {{
        uint buf = (tk0 / TBX) & 1u;
        for (uint idx = tid; idx < TBX * MT * 16u; idx += 128u) {{
            uint tt = idx / (MT * 16u);
            uint rem = idx % (MT * 16u);
            uint ri = rem / MT;
            uint mm = rem % MT;
            float v = 0.0f;
            if (tk0 + tt < tk_end && mm < batch) {{
                v = float(xh[(mm * n_sub + sub) * in_features + (tk0 + tt) * 16u + ri]);
            }}
            tg_x[buf][tt][ri * MT + mm] = ({xt})v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tt = 0u; tt < TBX && tk0 + tt < tk_end; tt++) {{
            const device uint* words = trellis + ((tk0 + tt) * out_tiles + tn) * PACKED_U32;
            ulong merged = ((ulong)words[i0_] << 32) | (ulong)words[i1_];
            uint w1 = uint(merged >> s1_);
            uint w0 = (w1 >> K_BITS) & 0xFFFFu;
            w1 &= 0xFFFFu;
            {{
                uint cw = w0;
{decode}
                threadgroup const {xt}* xp = &tg_x[buf][tt][row_[0] * MT];
                for (uint q = 0u; q < MT4; q++) {{
                    float4 xv = float4(xp[q*4u], xp[q*4u+1u], xp[q*4u+2u], xp[q*4u+3u]);
                    acc4[0][q] = fma(xv, float4(dq_val), acc4[0][q]);
                }}
            }}
            {{
                uint cw = w1;
{decode}
                threadgroup const {xt}* xp = &tg_x[buf][tt][row_[1] * MT];
                for (uint q = 0u; q < MT4; q++) {{
                    float4 xv = float4(xp[q*4u], xp[q*4u+1u], xp[q*4u+2u], xp[q*4u+3u]);
                    acc4[1][q] = fma(xv, float4(dq_val), acc4[1][q]);
                }}
            }}
        }}
    }}

    // Every position is owned by exactly one lane across the whole tg:
    // one disjoint write pass per mm, then a 16-row column sum.
    threadgroup float tg_w[256];
    for (uint mm = 0u; mm < MT; mm++) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_w[pos_[0]] = acc4[0][mm >> 2u][mm & 3u];
        tg_w[pos_[1]] = acc4[1][mm >> 2u][mm & 3u];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 16u && mm < batch) {{
            float s = 0.0f;
            for (uint r = 0u; r < 16u; r++) {{
                s += tg_w[r * 16u + tid];
            }}
            out[(mm * n_splits + split) * out_features + tn * 16u + tid] = s;
        }}
    }}
"""


def _devx_source(k: int, cb: CodebookMode, mt: int = 8) -> str:
    """v16 2-pair ownership + dq4 funnels, x read device-direct from a
    TRANSPOSED (in, MT) half buffer — one contiguous load per weight
    serves all MT rows; no tg staging, no hot-loop barriers."""
    packed_u32 = k * 256 // 32
    decode = _decode_expr(cb, cw_in="cw")
    nq = mt // 4
    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}
#define MT {mt}u

    uint tn = threadgroup_position_in_grid.x;
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

    float4 acc4[4][{nq}];
    for (uint j = 0u; j < 4u; j++) {{
        for (uint q = 0u; q < {nq}u; q++) {{
            acc4[j][q] = float4(0.0f);
        }}
    }}

    // xh is TRANSPOSED+padded: (n_sub, in_features, MT) half
    const device half* xt = xh + (ulong)sub * in_features * MT;
    for (uint tk = tk_begin + sgp; tk < tk_end; tk += 2u) {{
        const device uint* words = trellis + ((ulong)tk * out_tiles + tn) * PACKED_U32;
        ulong merged = ((ulong)words[ig0_] << 32) | (ulong)words[ig1_];
        uint cws[4];
        cws[3] = uint(merged >> s3_) & 0xFFFFu;
        cws[2] = uint(merged >> (s3_ + K_BITS)) & 0xFFFFu;
        cws[1] = uint(merged >> (s3_ + 2u * K_BITS)) & 0xFFFFu;
        cws[0] = uint(merged >> (s3_ + 3u * K_BITS)) & 0xFFFFu;
""" + "".join(
        f"""
        {{
            uint cw = cws[{i}];
{decode}
            const device half4* xp = (const device half4*)(xt + (tk * 16u + row_[{i}]) * MT);
"""
        + "".join(
            f"""            half4 x{q}_{i} = xp[{q}];
            acc4[{i}][{q}] = fma(float4(x{q}_{i}), float4(dq_val), acc4[{i}][{q}]);
"""
            for q in range(nq)
        )
        + """        }
"""
        for i in range(4)
    ) + """
    }
""" + f"""
    threadgroup float tg_w[2][256];
    for (uint mm = 0u; mm < MT; mm++) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < 4u; j++) {{
            tg_w[sgp][pos_[j]] = acc4[j][mm >> 2u][mm & 3u];
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


def variant(k, cb, *, tbx, half_x):
    key = (k, int(cb), tbx, half_x)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"exl3_gemm8_1p_tbx{tbx}_hx{int(half_x)}_k{k}",
            input_names=["xh", "trellis", "perm", "tile_sub", "tile_map", "dims"],
            output_names=["out"],
            source=_onepair_source(k, cb, tbx=tbx, half_x=half_x),
        )
    return _cache[key]


def n_splits_for(in_tiles, out_tiles):
    n = 1
    while out_tiles * n < 8192 and in_tiles // (n * 2) >= 64:
        n *= 2
    return n


def main() -> int:
    k = 4
    cb = CodebookMode.MCG
    batch = 8
    rng = np.random.default_rng(13)
    dummy = mx.zeros((1,), dtype=mx.uint32)

    for label, (in_tiles, out_tiles) in {
        "gate+up (320x2176)": (320, 2176),
        "down    (1088x320)": (1088, 320),
        "qkv+z   (320x832)": (320, 832),
    }.items():
        trellis = mx.array(
            rng.integers(0, 65536, (in_tiles, out_tiles, k * 16), dtype=np.uint16)
        )
        t32 = trellis.reshape(-1).view(mx.uint32)
        in_features, out_features = in_tiles * 16, out_tiles * 16
        weights = in_features * out_features
        n = n_splits_for(in_tiles, out_tiles)
        xh = mx.array(
            rng.standard_normal((batch, in_features)).astype(np.float16)
        ).reshape(-1)
        dims = mx.array([in_tiles, out_tiles, batch, n, 1, 0], dtype=mx.uint32)
        mx.eval(t32, xh)

        def run(kern):
            return kern(
                inputs=[xh, t32, _fwd_perm_u32(), dummy, dummy, dims],
                template=[("T", mx.float32)],
                grid=(out_tiles * _GEM_THREADS, 1, n),
                threadgroup=(_GEM_THREADS, 1, 1),
                output_shapes=[(batch * n * out_features,)],
                output_dtypes=[mx.float32],
            )[0]

        # transposed + MT-padded x for the device-direct variant
        xt = mx.zeros((in_features, 8), dtype=mx.float16)
        xt[:, :batch] = xh.reshape(batch, in_features).T
        xt_flat = mx.contiguous(xt).reshape(-1)
        mx.eval(xt_flat)

        def run_devx(kern):
            return kern(
                inputs=[xt_flat, t32, _fwd_perm_u32(), dummy, dummy, dims],
                template=[("T", mx.float32)],
                grid=(out_tiles * _GEM_THREADS, 1, n),
                threadgroup=(_GEM_THREADS, 1, 1),
                output_shapes=[(batch * n * out_features,)],
                output_dtypes=[mx.float32],
            )[0]

        def devx_kernel(mt):
            key = ("devx", k, mt)
            if key not in _cache:
                _cache[key] = mx.fast.metal_kernel(
                    name=f"exl3_gemm_devx_mt{mt}_k{k}",
                    input_names=["xh", "trellis", "perm", "tile_sub", "tile_map", "dims"],
                    output_names=["out"],
                    source=_devx_source(k, cb, mt),
                )
            return _cache[key]
        devx = devx_kernel(8)

        prod = _gemv_simd_kernel(k, cb, 8)
        ref = run(prod).reshape(batch, n, out_features).sum(axis=1)
        mx.eval(ref)

        kerns = {"v16-2pair(prod)": (prod, run)}
        for tbx, hx, name in ((8, False, "1pair-tbx8"), (8, True, "1pair-halfx")):
            kv = variant(k, cb, tbx=tbx, half_x=hx)
            got = run(kv).reshape(batch, n, out_features).sum(axis=1)
            mx.eval(got)
            rel = float(mx.max(mx.abs(got - ref)) / (mx.max(mx.abs(ref)) + 1e-6))
            kerns[name + ("" if rel < 1e-4 else f" ✗rel={rel:.0e}")] = (kv, run)

        got = run_devx(devx).reshape(batch, n, out_features).sum(axis=1)
        mx.eval(got)
        rel = float(mx.max(mx.abs(got - ref)) / (mx.max(mx.abs(ref)) + 1e-6))
        kerns["devx-T" + ("" if rel < 1e-4 else f" ✗rel={rel:.0e}")] = (devx, run_devx)

        # mt=4 devx: 4-row batch on the same design vs production mt=4
        b4 = 4
        xh4 = mx.array(rng.standard_normal((b4, in_features)).astype(np.float16))
        xt4 = mx.zeros((in_features, 4), dtype=mx.float16)
        xt4[:, :b4] = xh4.T
        xt4_flat = mx.contiguous(xt4).reshape(-1)
        dims4 = mx.array([in_tiles, out_tiles, b4, n, 1, 0], dtype=mx.uint32)
        mx.eval(xt4_flat, xh4)

        def run4(kern):
            return kern(
                inputs=[xh4.reshape(-1), t32, _fwd_perm_u32(), dummy, dummy, dims4],
                template=[("T", mx.float32)], grid=(out_tiles * _GEM_THREADS, 1, n),
                threadgroup=(_GEM_THREADS, 1, 1),
                output_shapes=[(b4 * n * out_features,)], output_dtypes=[mx.float32])[0]

        def run4_devx(kern):
            return kern(
                inputs=[xt4_flat, t32, _fwd_perm_u32(), dummy, dummy, dims4],
                template=[("T", mx.float32)], grid=(out_tiles * _GEM_THREADS, 1, n),
                threadgroup=(_GEM_THREADS, 1, 1),
                output_shapes=[(b4 * n * out_features,)], output_dtypes=[mx.float32])[0]

        prod4 = _gemv_simd_kernel(k, cb, 4)
        devx4 = devx_kernel(4)
        ref4 = run4(prod4).reshape(b4, n, out_features).sum(axis=1)
        got4 = run4_devx(devx4).reshape(b4, n, out_features).sum(axis=1)
        mx.eval(ref4, got4)
        rel4 = float(mx.max(mx.abs(got4 - ref4)) / (mx.max(mx.abs(ref4)) + 1e-6))
        kerns["mt4-prod"] = (prod4, run4)
        kerns["mt4-devx" + ("" if rel4 < 1e-4 else f" XREL={rel4:.0e}")] = (devx4, run4_devx)

        line = f"{label}:"
        for name, (kern, runner) in kerns.items():
            outs = [runner(kern) for _ in range(12)]
            mx.eval(*outs)
            mx.synchronize()
            tic = time.perf_counter()
            for _ in range(6):
                outs = [runner(kern) for _ in range(12)]
                mx.eval(*outs)
            mx.synchronize()
            dt = (time.perf_counter() - tic) / (6 * 12)
            line += f"  {name} {weights/dt/1e9:4.0f}"
        print(line, "Gw/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
