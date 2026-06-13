#!/usr/bin/env python3
"""Time individual ops of one GatedDeltaNet layer at S=1 vs S=4.

Identifies which op pays for the verify forward's S-scaling.
"""

from __future__ import annotations

import argparse
import time


import mlx.core as mx
import mlx.nn as nn


def bench(make_input, fn, n=24, reps=8, warm=2):
    """Launch n independent op instances per eval — amortizes the eval/sync
    floor and defeats single-slot input-identity caches."""
    xs = [make_input() for _ in range(n)]
    mx.eval(*xs)
    for _ in range(warm):
        mx.eval(*[fn(x) for x in xs])
    mx.synchronize()
    tic = time.perf_counter()
    for _ in range(reps):
        mx.eval(*[fn(x) for x in xs])
    mx.synchronize()
    return (time.perf_counter() - tic) / (reps * n) * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()

    from ponyexl3.mlx.model import load_model

    model, config = load_model(args.model, engine="exl3", verbose=False)
    lm = model.language_model

    # find the first linear (DeltaNet) layer and the first full-attention layer
    layers = lm.model.layers
    dn = next(l.linear_attn for l in layers if getattr(l, "is_linear", False))
    mlp = layers[0].mlp

    H = dn.hidden_size
    from mlx_lm.models.gated_delta import gated_delta_update

    for S in (1, args.rows):
        def xin():
            return mx.random.normal((1, S, H)).astype(mx.float16)

        r = {}
        r["in_proj_qkv(z fused)"] = bench(xin, lambda x: dn.in_proj_qkv(x))
        r["in_proj_b"] = bench(xin, lambda x: dn.in_proj_b(x))
        r["in_proj_a"] = bench(xin, lambda x: dn.in_proj_a(x))

        r["out_proj"] = bench(
            lambda: mx.random.normal((1, S, dn.value_dim)).astype(mx.float16),
            lambda y: dn.out_proj(y),
        )
        r["mlp.gate(up fused)"] = bench(xin, lambda x: mlp.gate_proj(x))
        r["mlp.down"] = bench(
            lambda: mx.random.normal((1, S, mlp.down_proj.in_features)).astype(mx.float16),
            lambda d: mlp.down_proj(d),
        )

        st = mx.zeros((1, dn.num_v_heads, dn.head_v_dim, dn.head_k_dim), dtype=mx.float32)
        ab = mx.random.normal((1, S, dn.num_v_heads)).astype(mx.float16)
        mx.eval(st, ab)

        def qkv_in():
            return (
                mx.random.normal((1, S, dn.num_k_heads, dn.head_k_dim)).astype(mx.float16),
                mx.random.normal((1, S, dn.num_k_heads, dn.head_k_dim)).astype(mx.float16),
                mx.random.normal((1, S, dn.num_v_heads, dn.head_v_dim)).astype(mx.float16),
            )

        def scan(t):
            q, k, v = t
            mx.eval(q, k, v)
            return gated_delta_update(q, k, v, ab, ab, dn.A_log, dn.dt_bias, st, None)[1]

        r["gated_delta_update"] = bench(qkv_in, scan)

        r["conv1d+silu"] = bench(
            lambda: mx.random.normal((1, S + dn.conv_kernel_size - 1, dn.conv_dim)).astype(mx.float16),
            lambda c: nn.silu(mx.conv1d(c, dn.conv1d.weight, groups=dn.conv_dim)),
        )

        print(f"--- S={S} (ms, in-stream x50)")
        for name, ms in r.items():
            print(f"  {name:24s} {ms:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
