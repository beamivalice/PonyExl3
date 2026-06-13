#!/usr/bin/env python3
"""Split-K re-sweep for the mt=1 simd GEMV on 27B decode shapes (post-dq4).

The production split chooser targets 8192 threadgroups with a >=128
in-tiles-per-split floor (tuned pre-dq4). This sweeps n_splits directly on
the production kernel per shape; * marks the auto choice.
"""

from __future__ import annotations

import time


import numpy as np
import mlx.core as mx

from ponyexl3.mlx.gemv_metal import _GEM_THREADS, _fwd_perm_u32, _gemv_simd_kernel
from ponyexl3.ref.codebook import CodebookMode


def auto_splits(in_tiles: int, out_tiles: int) -> int:
    n = 1
    while out_tiles * n < 8192 and in_tiles // (n * 2) >= 128:
        n *= 2
    return n


def main() -> int:
    cb = CodebookMode.MCG
    shapes = {
        "gate+up  (320x2176) k=4": (320, 2176, 4),
        "down     (1088x320) k=4": (1088, 320, 4),
        "qkv+z    (320x832)  k=4": (320, 832, 4),
        "out_proj (256x320)  k=4": (256, 320, 4),
        "lm_head  (320x9472) k=6": (320, 9472, 6),
    }
    dummy = mx.zeros((1,), dtype=mx.uint32)
    rng = np.random.default_rng(11)

    for label, (in_tiles, out_tiles, k) in shapes.items():
        trellis = mx.array(
            rng.integers(0, 65536, (in_tiles, out_tiles, k * 16), dtype=np.uint16)
        )
        trellis_u32 = trellis.reshape(-1).view(mx.uint32)
        in_features = in_tiles * 16
        out_features = out_tiles * 16
        xh = mx.array(rng.standard_normal((in_features,)).astype(np.float16))
        mx.eval(trellis_u32, xh)
        kern = _gemv_simd_kernel(k, cb, 1)
        auto = auto_splits(in_tiles, out_tiles)
        weights = in_features * out_features

        def run(n_splits):
            dims = mx.array(
                [in_tiles, out_tiles, 1, n_splits, 1, 0], dtype=mx.uint32
            )
            return kern(
                inputs=[xh, trellis_u32, _fwd_perm_u32(), dummy, dummy, dims],
                template=[("T", mx.float32)],
                grid=(out_tiles * _GEM_THREADS, 1, n_splits),
                threadgroup=(_GEM_THREADS, 1, 1),
                output_shapes=[(n_splits * out_features,)],
                output_dtypes=[mx.float32],
            )[0]

        splits = [n for n in (1, 2, 4, 8, 16, 32) if in_tiles // n >= 8]
        # warm all variants first (compile + clocks), then interleave
        for n in splits:
            mx.eval(run(n))
        mx.synchronize()
        times = {n: 0.0 for n in splits}
        reps, batch = 6, 12
        for _ in range(reps):
            for n in splits:
                outs = [run(n) for _ in range(batch)]
                tic = time.perf_counter()
                mx.eval(*outs)
                mx.synchronize()
                times[n] += time.perf_counter() - tic
        print(f"--- {label}  (auto n={auto})")
        for n in splits:
            dt = times[n] / (reps * batch)
            mark = " *" if n == auto else ""
            print(f"  splits={n:2d}: {dt*1000:7.3f} ms  {weights/dt/1e9:6.0f} Gw/s{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
