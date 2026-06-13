#!/usr/bin/env python3
"""Dense small-R GEMM: v16 register kernel vs v19c mma kernel (one block).

The MoE Phase 23 lesson transferred: verify rows 2-8 should cost ~one plain
step. The mma kernel decodes each tile once into threadgroup memory and the
mma over a 64-row padded block is ~free, so it should run near the mt=1
GEMV's decode-bound rate (~480-490 Gw/s) where the v16 GEMM pays 330-350.
Caveats probed here: skinny shapes underfill (no split-K), fp16-store parity.
"""

from __future__ import annotations

import argparse
import time


import numpy as np
import mlx.core as mx

from ponyexl3.mlx.gemv_metal import inner_gemm_mlx, inner_mm_seg_mlx
from ponyexl3.ref.codebook import CodebookMode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()
    k = args.k
    cb = CodebookMode.MCG

    shapes = {
        "gate+up (320x2176)": (320, 2176),
        "qkv+z   (320x832)": (320, 832),
        "down    (1088x320)": (1088, 320),
        "out_proj(256x320)": (256, 320),
    }
    rng = np.random.default_rng(5)

    for label, (in_tiles, out_tiles) in shapes.items():
        trellis = mx.array(
            rng.integers(0, 65536, (in_tiles, out_tiles, k * 16), dtype=np.uint16)
        )
        in_features = in_tiles * 16
        out_e = out_tiles * 16
        weights = in_features * out_e
        mx.eval(trellis)

        for R in (4, 8):
            xh = mx.array(rng.standard_normal((R, in_features)).astype(np.float16))
            xh_pad = mx.concatenate(
                [xh, mx.zeros((64, in_features), dtype=mx.float16)], axis=0
            )
            blk_tab = mx.array([[0], [0], [R]], dtype=mx.uint32)
            nb_real = mx.array([1], dtype=mx.uint32)
            mx.eval(xh, xh_pad, blk_tab, nb_real)

            def run_v16():
                return inner_gemm_mlx(xh, trellis, k, cb)

            def run_mm():
                return inner_mm_seg_mlx(
                    xh_pad, trellis, k, cb, blk_tab, nb_real,
                    n_rows=R, tn_base=0, tiles_per_e=0, out_e=out_e,
                )

            ref = run_v16().astype(mx.float32)
            got = run_mm().astype(mx.float32)
            mx.eval(ref, got)
            rel = float(mx.max(mx.abs(got - ref)) / (mx.max(mx.abs(ref)) + 1e-6))

            res = {}
            for name, fn in (("v16", run_v16), ("mma", run_mm)):
                outs = [fn() for _ in range(12)]
                mx.eval(*outs)
                mx.synchronize()
                tic = time.perf_counter()
                for _ in range(6):
                    outs = [fn() for _ in range(12)]
                    mx.eval(*outs)
                mx.synchronize()
                res[name] = (time.perf_counter() - tic) / (6 * 12)
            v16_ms, mm_ms = res["v16"] * 1000, res["mma"] * 1000
            print(
                f"{label} R={R}: v16 {v16_ms:7.3f} ms ({weights/res['v16']/1e9:4.0f} Gw/s)"
                f"  mma {mm_ms:7.3f} ms ({weights/res['mma']/1e9:4.0f} Gw/s)"
                f"  x{v16_ms/mm_ms:4.2f}  rel {rel:.1e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
