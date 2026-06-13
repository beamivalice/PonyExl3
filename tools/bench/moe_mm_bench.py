"""A/B the MoE prefill paths at 35B-A3B dims: sorted gather_mm vs v19 GEMM.

Synthetic trellis at real shapes (E=257, D=2048, H=512, k=4, 9 pairs/token).
Respects the EXL3_MM_* ablation knobs (BN / DQ4 / DB) — set them in the env
before running to ablate kernel variants.

Run:  uv run python pony/tools/moe_mm_bench.py
"""

from __future__ import annotations

import time

import numpy as np
import mlx.core as mx

import ponyexl3.mlx.exl3_moe as moe
from ponyexl3.ref.codebook import CodebookMode

E, D, H, K = 257, 2048, 512, 4
LAYERS = 40


def bench(fn, reps=8, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3


def main():
    import os

    print(f"v19c  NODECODE={os.environ.get('EXL3_MM_NODECODE', '0')}")
    rng = np.random.default_rng(0)
    P = 16 * K
    sign = lambda *s: mx.array(rng.choice([-1.0, 1.0], s).astype(np.float16))
    gu_t = mx.array(
        rng.integers(0, 1 << 16, (D // 16, E * 2 * (H // 16), P)).astype(np.uint16)
    )
    dn_t = mx.array(
        rng.integers(0, 1 << 16, (H // 16, E * (D // 16), P)).astype(np.uint16)
    )
    sw = moe.EXL3SwitchGLU(
        gu_trellis=gu_t, gu_suh=sign(E, 2, D), gu_svh=sign(E, 2 * H),
        dn_trellis=dn_t, dn_suh=sign(E, H), dn_svh=sign(E, D),
        k=K, cb=CodebookMode.DEFAULT,
    )
    for S in (512, 2048, 4096, 8192):
        x = mx.array((rng.standard_normal((1, S, D)) * 0.02).astype(np.float16))
        routed = np.argpartition(rng.random((S, E - 1)), -8, axis=-1)[:, -8:]
        inds = mx.array(
            np.concatenate([routed, np.full((S, 1), E - 1)], 1)
            .astype(np.int32)
            .reshape(1, S, 9)
        )
        mx.eval(x, inds)
        moe._MOE_MM = False
        t_g = bench(lambda: sw._prefill(x, inds))
        moe._MOE_MM = True
        t_m = bench(lambda: sw._prefill(x, inds))
        print(
            f"S={S:5d}: gather {t_g:7.2f} ms/layer   mm {t_m:7.2f} ms/layer "
            f"({t_g / t_m:4.2f}x)   FFN ceiling {S / (t_g * LAYERS) * 1e3:5.0f} "
            f"-> {S / (t_m * LAYERS) * 1e3:5.0f} tok/s"
        )


if __name__ == "__main__":
    main()
