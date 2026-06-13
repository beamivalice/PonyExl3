#!/usr/bin/env python3
"""Sweep the v16 small-batch GEMM's TBX (tiles per barrier block) at mt=4/8.

TBX=4 shipped unswept; deeper serial blocks halve barrier count (the
decode_full TBD sweep won 8->64 the same way). Bit-exactness: TBX does not
change per-lane accumulation order (same tile->sgp assignment, same fma
sequence), so outputs must match the production kernel exactly.
"""

from __future__ import annotations

import time


import numpy as np
import mlx.core as mx

from ponyexl3.mlx.gemv_metal import (
    _GEM_THREADS,
    _fwd_perm_u32,
    _gemm_simd_source,
    _gemv_simd_kernel,
)
from ponyexl3.ref.codebook import CodebookMode

_cache: dict = {}


def variant(k: int, cb: CodebookMode, mt: int, tbx: int):
    key = (k, int(cb), mt, tbx)
    if key not in _cache:
        src = _gemm_simd_source(k, cb, mt).replace(
            "#define TBX 4u", f"#define TBX {tbx}u"
        )
        _cache[key] = mx.fast.metal_kernel(
            name=f"exl3_gemm_tbx{tbx}_k{k}_mt{mt}",
            input_names=["xh", "trellis", "perm", "tile_sub", "tile_map", "dims"],
            output_names=["out"],
            source=src,
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
    rng = np.random.default_rng(9)
    dummy = mx.zeros((1,), dtype=mx.uint32)

    for label, (in_tiles, out_tiles) in {
        "gate+up (320x2176)": (320, 2176),
        "down    (1088x320)": (1088, 320),
    }.items():
        trellis = mx.array(
            rng.integers(0, 65536, (in_tiles, out_tiles, k * 16), dtype=np.uint16)
        )
        trellis_u32 = trellis.reshape(-1).view(mx.uint32)
        in_features, out_features = in_tiles * 16, out_tiles * 16
        weights = in_features * out_features
        n = n_splits_for(in_tiles, out_tiles)

        for mt, batch in ((4, 4), (8, 8)):
            xh = mx.array(
                rng.standard_normal((batch, in_features)).astype(np.float16)
            ).reshape(-1)
            dims = mx.array([in_tiles, out_tiles, batch, n, 1, 0], dtype=mx.uint32)
            mx.eval(trellis_u32, xh)

            def run(kern):
                return kern(
                    inputs=[xh, trellis_u32, _fwd_perm_u32(), dummy, dummy, dims],
                    template=[("T", mx.float32)],
                    grid=(out_tiles * _GEM_THREADS, 1, n),
                    threadgroup=(_GEM_THREADS, 1, 1),
                    output_shapes=[(batch * n * out_features,)],
                    output_dtypes=[mx.float32],
                )[0]

            prod = _gemv_simd_kernel(k, cb, mt)
            ref = run(prod)
            mx.eval(ref)
            ref_np = np.array(ref)
            kerns = {"TBX=4 (prod)": prod}
            for tbx in (8, 16):
                kv = variant(k, cb, mt, tbx)
                exact = np.array_equal(np.array(run(kv)), ref_np)
                kerns[f"TBX={tbx}" + ("" if exact else " ✗NOT-EXACT")] = kv

            line = f"{label} mt={mt} splits={n}:"
            for name, kern in kerns.items():
                outs = [run(kern) for _ in range(12)]
                mx.eval(*outs)
                mx.synchronize()
                tic = time.perf_counter()
                for _ in range(6):
                    outs = [run(kern) for _ in range(12)]
                    mx.eval(*outs)
                mx.synchronize()
                dt = (time.perf_counter() - tic) / (6 * 12)
                line += f"  {name} {weights/dt/1e9:4.0f} Gw/s"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
