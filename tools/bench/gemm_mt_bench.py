#!/usr/bin/env python3
"""Prototype bench: v16 mt=4 GEMM vs shuffle-x / dq4-funnel variants.

The mt=1 GEMV reads x via simd_shuffle with zero hot-loop barriers; the v16
mt>1 GEMM stages x in threadgroup memory (barrier per TBX tiles) and still
uses per-pair funnels. A lane's 4 codewords are 4-aligned, so the dq4 merged
window applies, and x can ride float4 simd_shuffle. Variants must be
bit-exact vs v16 before timing counts.
"""

from __future__ import annotations

import argparse
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


def _variant_source(k: int, cb: CodebookMode, mt: int, *, shuffle_x: bool, dq4: bool) -> str:
    assert mt == 4
    packed_u32 = k * 256 // 32
    decode = _decode_expr(cb, cw_in="cw")

    if dq4:
        funnel_setup = """
    uint c0 = half_ * 128u + lane * 4u;
    int e_last = int(c0 + 4u) * K_BITS + 256 * K_BITS;
    int i_end = (e_last - 1) / 32;
    uint ig1_ = uint(i_end) % PACKED_U32;
    uint ig0_ = uint(i_end - 1) % PACKED_U32;
    uint s3_ = uint((i_end + 1) * 32 - e_last);
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
            for (uint j = 0u; j < 2u; j++) {
                ulong merged = ((ulong)words[i0_[j]] << 32) | (ulong)words[i1_[j]];
                uint w1 = uint(merged >> s1_[j]);
                cws[j * 2u] = (w1 >> K_BITS) & 0xFFFFu;
                cws[j * 2u + 1u] = w1 & 0xFFFFu;
            }
"""

    if shuffle_x:
        x_pre = ""
        x_tile = """
        float4 x_lane = float4(0.0f);
        uint xb = tk * 16u + (lane & 15u);
        x_lane.x = float(xh[sub * in_features + xb]);
        if (batch > 1u) x_lane.y = float(xh[(1u * n_sub + sub) * in_features + xb]);
        if (batch > 2u) x_lane.z = float(xh[(2u * n_sub + sub) * in_features + xb]);
        if (batch > 3u) x_lane.w = float(xh[(3u * n_sub + sub) * in_features + xb]);
"""
        x_get = "simd_shuffle(x_lane, ushort(row_[{i}]))"
        loop_open = """
    for (uint tk = tk_begin + sgp; tk < tk_end; tk += 2u) {
        const device uint* words = trellis + (tk * out_tiles + tn) * PACKED_U32;
"""
        loop_close = "    }\n"
    else:
        x_pre = """
    threadgroup float tg_x[2][TBX][16u * MT];
"""
        x_get = "*((threadgroup const float4*)&tg_x[buf][t][row_[{i}] * MT])"
        loop_open = """
    for (uint tk0 = tk_begin; tk0 < tk_end; tk0 += TBX) {
        uint buf = (tk0 / TBX) & 1u;
        for (uint idx = tid; idx < TBX * MT * 16u; idx += 128u) {
            uint t = idx / (MT * 16u);
            uint rem = idx % (MT * 16u);
            uint ri = rem / MT;
            uint mm = rem % MT;
            float v = 0.0f;
            if (tk0 + t < tk_end && mm < batch) {
                v = float(xh[(mm * n_sub + sub) * in_features + (tk0 + t) * 16u + ri]);
            }
            tg_x[buf][t][ri * MT + mm] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = sgp; t < TBX && tk0 + t < tk_end; t += 2u) {
            uint tk = tk0 + t;
            const device uint* words = trellis + (tk * out_tiles + tn) * PACKED_U32;
"""
        loop_close = "        }\n    }\n"
        x_tile = ""

    body = ""
    for i in range(4):
        body += f"""
        {{
            uint cw = cws[{i}];
{decode}
            acc4[{i}] = fma({x_get.format(i=i)}, float4(dq_val), acc4[{i}]);
        }}
"""

    return f"""
#define PACKED_U32 {packed_u32}
#define K_BITS {k}
#define MT {mt}u
#define TBX 4u

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
{funnel_setup}
{x_pre}
    float4 acc4[4];
    for (uint j = 0u; j < 4u; j++) {{
        acc4[j] = float4(0.0f);
    }}
{loop_open}
{x_tile}
{extract}
{body}
{loop_close}
    threadgroup float tg_w[2][256];
    for (uint mm = 0u; mm < MT; mm++) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < 4u; j++) {{
            tg_w[sgp][pos_[j]] = acc4[j][mm];
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


_variant_cache: dict = {}


def _variant_kernel(k: int, cb: CodebookMode, *, shuffle_x: bool, dq4: bool):
    key = (k, int(cb), shuffle_x, dq4)
    if key not in _variant_cache:
        _variant_cache[key] = mx.fast.metal_kernel(
            name=f"exl3_gemm_proto_k{k}_sx{int(shuffle_x)}_dq{int(dq4)}",
            input_names=["xh", "trellis", "perm", "tile_sub", "tile_map", "dims"],
            output_names=["out"],
            source=_variant_source(k, cb, 4, shuffle_x=shuffle_x, dq4=dq4),
        )
    return _variant_cache[key]


def _n_splits(in_tiles: int, out_tiles: int) -> int:
    n = 1
    while out_tiles * n < 8192 and in_tiles // (n * 2) >= 64:
        n *= 2
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()
    k = args.k
    cb = CodebookMode.MCG
    batch = args.batch

    shapes = {
        "gate+up (320x2176)": (320, 2176),
        "down (1088x320)": (1088, 320),
        "qkv+z (320x832)": (320, 832),
    }

    dummy = mx.zeros((1,), dtype=mx.uint32)

    for label, (in_tiles, out_tiles) in shapes.items():
        rng = np.random.default_rng(7)
        trellis = mx.array(
            rng.integers(0, 65536, (in_tiles, out_tiles, k * 16), dtype=np.uint16)
        )
        trellis_u32 = trellis.reshape(-1).view(mx.uint32)
        in_features = in_tiles * 16
        out_features = out_tiles * 16
        xh = mx.array(rng.standard_normal((batch, in_features)).astype(np.float16))
        xh_flat = xh.reshape(-1)
        mx.eval(trellis_u32, xh_flat)

        n_splits = _n_splits(in_tiles, out_tiles)
        dims = mx.array([in_tiles, out_tiles, batch, n_splits, 1, 0], dtype=mx.uint32)
        grid = (out_tiles * _GEM_THREADS, 1, n_splits)
        tg = (_GEM_THREADS, 1, 1)
        oshape = [(batch * n_splits * out_features,)]

        def run(kern):
            return kern(
                inputs=[xh_flat, trellis_u32, _fwd_perm_u32(), dummy, dummy, dims],
                template=[("T", mx.float32)],
                grid=grid,
                threadgroup=tg,
                output_shapes=oshape,
                output_dtypes=[mx.float32],
            )[0]

        base_kern = _gemv_simd_kernel(k, cb, 4)
        ref = run(base_kern)
        mx.eval(ref)
        ref_np = np.array(ref)

        kerns = {"v16 (base)": base_kern}
        for sx, dq, name in (
            (False, True, "dq4 only"),
            (True, False, "shuffle-x only"),
            (True, True, "shuffle-x + dq4"),
        ):
            kern = _variant_kernel(k, cb, shuffle_x=sx, dq4=dq)
            got = run(kern)
            mx.eval(got)
            exact = np.array_equal(np.array(got), ref_np)
            kerns[name + ("" if exact else "  ✗NOT-EXACT")] = kern

        weights = in_features * out_features
        print(f"--- {label} k={k} batch={batch} splits={n_splits}")
        for name, kern in kerns.items():
            n, reps = 16, 8
            outs = [run(kern) for _ in range(n)]
            mx.eval(*outs)
            mx.synchronize()
            tic = time.perf_counter()
            for _ in range(reps):
                outs = [run(kern) for _ in range(n)]
                mx.eval(*outs)
            mx.synchronize()
            dt = (time.perf_counter() - tic) / (reps * n)
            print(f"  {name:28s} {dt*1000:7.3f} ms   {weights/dt/1e9:6.0f} Gw/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
