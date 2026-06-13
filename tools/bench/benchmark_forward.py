#!/usr/bin/env python3
"""Benchmark EXL3 MLX forward paths: reconstruct vs fast dispatch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", type=Path, nargs="?")
    p.add_argument("module_key", nargs="?")
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warm-cache", action="store_true", help="warm layer caches before timing")
    p.add_argument("--synthetic", action="store_true", help="use in-memory synthetic layer")
    p.add_argument("--in-features", type=int, default=512)
    p.add_argument("--out-features", type=int, default=512)
    p.add_argument("-k", type=int, default=4)
    args = p.parse_args()

    import mlx.core as mx

    from ponyexl3.mlx.exl3_qmm import linear_forward_gemm_mlx
    from ponyexl3.mlx.exl3_qmv import linear_forward_gemv_mlx
    from ponyexl3.mlx.forward import linear_forward_mlx, linear_forward_reconstruct_mlx
    from ponyexl3.mlx.layer_state import clear_layer_caches, warm_layer_mlx
    from ponyexl3.mlx.stripe import linear_forward_stripe_mlx
    from ponyexl3.ref.forward import linear_forward_reconstruct
    from ponyexl3.ref.loader import layer_meta_from_config, load_exl3_layer
    from ponyexl3.ref.synthetic import make_exl3_layer

    if args.synthetic or not args.model_dir:
        layer = make_exl3_layer(
            k=args.k,
            in_features=args.in_features,
            out_features=args.out_features,
            seed=0,
            mcg=True,
        )
        label = f"synthetic {args.in_features}x{args.out_features} k={args.k}"
    else:
        if not args.module_key:
            raise SystemExit("module_key required for checkpoint benchmark")
        layer_meta_from_config(str(args.model_dir), args.module_key)
        layer = load_exl3_layer(str(args.model_dir), args.module_key)
        label = args.module_key

    rng = np.random.default_rng(0)
    x = rng.standard_normal((args.rows, layer.in_features)).astype(np.float16)

    nbytes = layer.in_features * layer.out_features * 2
    huge = nbytes > 64 * 1024 * 1024

    print(f"module: {label}")
    print(f"  shape: ({args.rows}, {layer.in_features}) -> ({args.rows}, {layer.out_features})  K={layer.k}")
    print(f"  decoded_fp16: {nbytes / 1e6:.1f} MB  huge={huge}")

    if args.synthetic or (not huge and layer.in_features <= 2048):
        t_ref = _bench(lambda: linear_forward_reconstruct(layer, x), 1, max(1, args.iters))
        print(f"  ref reconstruct: {t_ref*1e3:.1f} ms")
    else:
        t_ref = None
        print("  ref reconstruct: skipped (layer too large)")

    mx.eval(mx.array(0))
    clear_layer_caches()

    if args.warm_cache:
        warm_layer_mlx(layer, inner=True, stripes=huge)
        print("  cache: warmed")

    t_cold = _bench(
        lambda: mx.eval(linear_forward_mlx(layer, x, fast=True)),
        0,
        1,
    )
    if args.warm_cache:
        warm_layer_mlx(layer, inner=True, stripes=huge)
    t_fast = _bench(
        lambda: mx.eval(linear_forward_mlx(layer, x, fast=True)),
        1,
        max(1, args.iters),
    )
    t_slow = _bench(
        lambda: mx.eval(linear_forward_reconstruct_mlx(layer, x)),
        1,
        max(1, args.iters),
    )
    t_gemv = None
    if args.rows == 1:
        if args.warm_cache:
            warm_layer_mlx(layer, inner=False)
        t_gemv = _bench(
            lambda: mx.eval(linear_forward_gemv_mlx(layer, x)),
            1,
            max(1, args.iters),
        )
    t_fused_gemm = None
    t_stripe = None
    if args.rows > 1 and huge:
        if args.warm_cache:
            warm_layer_mlx(layer, inner=False)
        t_fused_gemm = _bench(
            lambda: mx.eval(linear_forward_gemm_mlx(layer, x)),
            1,
            max(1, args.iters),
        )
        if args.warm_cache:
            warm_layer_mlx(layer, inner=True, stripes=True)
        t_stripe = _bench(
            lambda: mx.eval(linear_forward_stripe_mlx(layer, x)),
            1,
            max(1, args.iters),
        )

    y_slow = np.array(linear_forward_reconstruct_mlx(layer, x))
    y_fast = np.array(linear_forward_mlx(layer, x, fast=True))
    diff = np.abs(y_slow.astype(np.float32) - y_fast.astype(np.float32))
    both_nan = np.isnan(y_slow) & np.isnan(y_fast)
    diff[both_nan] = 0.0

    report = {
        "module": label,
        "rows": args.rows,
        "k": layer.k,
        "decoded_mb": nbytes / 1e6,
        "huge": huge,
        "ref_ms": None if t_ref is None else t_ref * 1e3,
        "mlx_reconstruct_ms": t_slow * 1e3,
        "mlx_fast_cold_ms": t_cold * 1e3,
        "mlx_fast_warm_ms": t_fast * 1e3,
        "mlx_gemv_ms": None if t_gemv is None else t_gemv * 1e3,
        "mlx_fused_gemm_ms": None if t_fused_gemm is None else t_fused_gemm * 1e3,
        "mlx_stripe_ms": None if t_stripe is None else t_stripe * 1e3,
        "speedup_vs_reconstruct": t_slow / t_fast if t_fast > 0 else None,
        "max_abs_fast_vs_slow": float(np.nanmax(diff)),
    }
    print(f"  mlx reconstruct: {t_slow*1e3:.1f} ms")
    print(f"  mlx fast (cold): {t_cold*1e3:.1f} ms")
    print(f"  mlx fast (warm): {t_fast*1e3:.1f} ms  ({report['speedup_vs_reconstruct']:.1f}x vs reconstruct)")
    if t_gemv is not None:
        print(f"  mlx gemv:        {t_gemv*1e3:.1f} ms")
    if t_fused_gemm is not None:
        print(f"  mlx fused gemm:  {t_fused_gemm*1e3:.1f} ms")
    if t_stripe is not None:
        print(f"  mlx stripe:      {t_stripe*1e3:.1f} ms")
    print(f"  fast vs reconstruct max_abs: {report['max_abs_fast_vs_slow']:.2e}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
