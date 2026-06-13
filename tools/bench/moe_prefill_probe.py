"""Probe: where does MoE prefill time go at Qwen3.6-35B-A3B dims?

Synthetic trellis (random uint16 — decode timing is value-independent, 3INST
never NaNs), real shapes: E=257 (256 routed + shared), D=2048, H=512, k=4,
top-9 pairs/token, S=512 chunk. Apportions the measured ~840 ms chunk into
decode-all / gather_mm / rotations, and A/Bs the two cheap fixes:

  1. sorted gather_mm (mlx_lm SwitchGLU's gather_sort pattern)
  2. contiguous rhs vs swapaxes view

Run:  uv run python pony/tools/moe_prefill_probe.py
"""

from __future__ import annotations

import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU, _rows_finish, _rows_prep
from ponyexl3.mlx.gemv_metal import decode_full_t_mlx
from ponyexl3.ref.codebook import CodebookMode

E, D, H, K, TOPK, S = 257, 2048, 512, 4, 9, 512
N = S * TOPK
P = 16 * K
GU_TILES = H // 16
DN_TILES = D // 16
CB = CodebookMode.DEFAULT
LAYERS = 40


def bench(fn, reps=10, warm=3):
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
    rng = np.random.default_rng(0)
    gu_trellis = mx.array(
        rng.integers(0, 1 << 16, (D // 16, E * 2 * GU_TILES, P)).astype(np.uint16)
    )
    dn_trellis = mx.array(
        rng.integers(0, 1 << 16, (H // 16, E * DN_TILES, P)).astype(np.uint16)
    )
    sign = lambda *s: mx.array(rng.choice([-1.0, 1.0], s).astype(np.float16))
    sw = EXL3SwitchGLU(
        gu_trellis=gu_trellis,
        gu_suh=sign(E, 2, D),
        gu_svh=sign(E, 2 * H),
        dn_trellis=dn_trellis,
        dn_suh=sign(E, H),
        dn_svh=sign(E, D),
        k=K,
        cb=CB,
    )
    x = mx.array((rng.standard_normal((1, S, D)) * 0.02).astype(np.float16))
    # distinct top-8 per token + shared expert appended (mirrors the router)
    routed = np.argpartition(rng.random((S, E - 1)), -8, axis=-1)[:, -8:]
    indices = mx.array(
        np.concatenate([routed, np.full((S, 1), E - 1)], axis=1)
        .astype(np.int32)
        .reshape(1, S, TOPK)
    )
    mx.eval(sw.parameters(), x, indices)

    flat = indices.reshape(-1)
    idx = flat.reshape(N, 1).astype(mx.uint32)

    # ---- component: decode-all (3 calls, as _prefill does) ---------------
    def decode_all():
        w_g = decode_full_t_mlx(gu_trellis, K, CB, tn_offset=0, tn_count=E * GU_TILES)
        w_u = decode_full_t_mlx(
            gu_trellis, K, CB, tn_offset=E * GU_TILES, tn_count=E * GU_TILES
        )
        wd = decode_full_t_mlx(dn_trellis, K, CB)
        return w_g, w_u, wd

    t_dec = bench(decode_all)
    w_g, w_u, wd = decode_all()
    w_g = w_g.reshape(E, H, D)
    w_u = w_u.reshape(E, H, D)
    wd = wd.reshape(E, D, H)
    mx.eval(w_g, w_u, wd)

    # ---- component: rotations for N pairs --------------------------------
    x_pairs = mx.broadcast_to(x.reshape(S, 1, D), (S, TOPK, D)).reshape(N, D)
    t_rot = bench(lambda: _rows_prep()(x_pairs, sw._gu_suh[flat, 0]))

    # ---- component: gather_mm gate, unsorted (current) -------------------
    xg = _rows_prep()(x_pairs, sw._gu_suh[flat, 0])
    mx.eval(xg)
    t_gmm_uns = bench(
        lambda: mx.gather_mm(
            xg.reshape(N, 1, 1, D), w_g.swapaxes(-1, -2), rhs_indices=idx
        )
    )

    # ---- component: gather_mm gate, sorted --------------------------------
    order = mx.argsort(flat)
    inv = mx.argsort(order)
    sidx = flat[order].reshape(N, 1).astype(mx.uint32)
    xs = xg[order]
    mx.eval(order, inv, sidx, xs)
    t_gmm_srt = bench(
        lambda: mx.gather_mm(
            xs.reshape(N, 1, 1, D),
            w_g.swapaxes(-1, -2),
            rhs_indices=sidx,
            sorted_indices=True,
        )
    )

    # ---- component: gather_mm gate, sorted + contiguous rhs ---------------
    w_g_c = mx.contiguous(w_g.swapaxes(-1, -2))
    mx.eval(w_g_c)
    t_gmm_srt_c = bench(
        lambda: mx.gather_mm(
            xs.reshape(N, 1, 1, D), w_g_c, rhs_indices=sidx, sorted_indices=True
        )
    )

    # ---- reference: dense matmul at same FLOPs ----------------------------
    w_dense = w_g[0].swapaxes(0, 1)
    t_dense = bench(lambda: xg @ w_dense)

    # ---- end-to-end: one layer, current _prefill --------------------------
    t_layer_now = bench(lambda: sw._prefill(x, indices))

    # ---- end-to-end: one layer, sorted variant ----------------------------
    def prefill_sorted():
        flat_s = flat[order]
        tok = (mx.arange(N, dtype=mx.uint32) // TOPK)[order]
        xp = x.reshape(S, D)[tok]
        xg_ = _rows_prep()(xp, sw._gu_suh[flat_s, 0])
        xu_ = _rows_prep()(xp, sw._gu_suh[flat_s, 1])
        wg_ = decode_full_t_mlx(
            gu_trellis, K, CB, tn_offset=0, tn_count=E * GU_TILES
        ).reshape(E, H, D)
        wu_ = decode_full_t_mlx(
            gu_trellis, K, CB, tn_offset=E * GU_TILES, tn_count=E * GU_TILES
        ).reshape(E, H, D)
        g = mx.gather_mm(
            xg_.reshape(N, 1, 1, D), wg_.swapaxes(-1, -2),
            rhs_indices=sidx, sorted_indices=True,
        )
        u = mx.gather_mm(
            xu_.reshape(N, 1, 1, D), wu_.swapaxes(-1, -2),
            rhs_indices=sidx, sorted_indices=True,
        )
        g = _rows_finish()(g.reshape(N, H).astype(mx.float16), sw._gu_svh[flat_s, :H])
        u = _rows_finish()(u.reshape(N, H).astype(mx.float16), sw._gu_svh[flat_s, H:])
        h = nn.silu(g) * u
        xd_ = _rows_prep()(h, sw._dn_suh[flat_s])
        wd_ = decode_full_t_mlx(dn_trellis, K, CB).reshape(E, D, H)
        y = mx.gather_mm(
            xd_.reshape(N, 1, 1, H), wd_.swapaxes(-1, -2),
            rhs_indices=sidx, sorted_indices=True,
        )
        y = _rows_finish()(y.reshape(N, D).astype(mx.float16), sw._dn_svh[flat_s])
        return y[inv].reshape(1, S, TOPK, D)

    # parity first (same weights, real values)
    ref = sw._prefill(x, indices)
    got = prefill_sorted()
    mx.eval(ref, got)
    err = float(mx.abs(ref - got).max()) / max(float(mx.abs(ref).max()), 1e-9)
    t_layer_sorted = bench(prefill_sorted)

    gw = 3 * E * H * D / 1e9
    print(f"dims: E={E} D={D} H={H} k={K}  S={S} pairs N={N}  layers={LAYERS}")
    print(f"expert weights/layer: {gw:.2f} Gw  ({gw*2:.2f} GB fp16, {gw/2:.2f} GB trellis)")
    print()
    print(f"decode-all (3x decode_full_t)        {t_dec:8.2f} ms/layer   ({gw/t_dec*1e3:6.0f} Gw/s)")
    print(f"rotations (one _rows_prep, N pairs)  {t_rot:8.2f} ms")
    print(f"gather_mm gate UNSORTED (current)    {t_gmm_uns:8.2f} ms")
    print(f"gather_mm gate sorted                {t_gmm_srt:8.2f} ms")
    print(f"gather_mm gate sorted+contig rhs     {t_gmm_srt_c:8.2f} ms")
    print(f"dense matmul same flops (ref)        {t_dense:8.2f} ms")
    print()
    print(f"one MoE layer _prefill (current)     {t_layer_now:8.2f} ms  -> x{LAYERS} = {t_layer_now*LAYERS:6.0f} ms/chunk")
    print(f"one MoE layer sorted variant         {t_layer_sorted:8.2f} ms  -> x{LAYERS} = {t_layer_sorted*LAYERS:6.0f} ms/chunk")
    print(f"sorted parity vs current: rel err {err:.2e}")
    print()
    print(f"chunk tok/s ceiling (MoE FFN only): now {S/(t_layer_now*LAYERS)*1e3:5.0f}, sorted {S/(t_layer_sorted*LAYERS)*1e3:5.0f}")


if __name__ == "__main__":
    main()
