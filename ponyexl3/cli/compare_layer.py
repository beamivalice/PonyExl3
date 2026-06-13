#!/usr/bin/env python3
"""
Compare EXL3 layer correctness across backends.

On Apple Silicon the GPU reference is **MLX (Metal)**, not CUDA or PyTorch MPS.
ExLlamaV3's ``exllamav3_ext`` reconstruct kernels are CUDA-only; when a CUDA GPU
is available this tool also compares against ExLlamaV3.

**Start with ``--probe``** on large models — full ``--mode forward`` materializes the
entire weight matrix on CPU and can take tens of minutes per layer on 27B-class models.

Usage:
  # Instant cost estimate (no weight load)
  python tools/compare_layer.py MODEL MODULE --probe

  # Fast correctness check: one 16x16 tile decode (~seconds)
  python tools/compare_layer.py MODEL MODULE --mode tile

  # Partial forward on first N output columns (multiple of 128)
  python tools/compare_layer.py MODEL MODULE --mode slice --out-cols 128

  # Full layer forward (slow on large layers)
  python tools/compare_layer.py MODEL MODULE --mode forward --rows 2

  # List EXL3 module keys
  python tools/compare_layer.py MODEL --list
"""

from __future__ import annotations

from typing import Any

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.types import LayerMeta


def _load_layer(model_dir: Path, module_key: str):
    from ponyexl3.ref.loader import load_exl3_layer

    return load_exl3_layer(str(model_dir), module_key)


def _layer_meta(model_dir: Path, module_key: str) -> LayerMeta:
    from ponyexl3.ref.loader import layer_meta_from_config

    return layer_meta_from_config(str(model_dir), module_key)


def _forward_ref(layer: EXL3Layer, x: np.ndarray, *, out_cols: int | None = None) -> np.ndarray:
    from ponyexl3.ref.forward import linear_forward_reconstruct
    from ponyexl3.ref.hadamard import had_r_128
    from ponyexl3.ref.reconstruct import reconstruct_inner
    from ponyexl3.ref.signs import unpack_signs_or_pass

    if out_cols is None:
        return linear_forward_reconstruct(layer, x)

    layer.validate()
    if out_cols % 128 != 0:
        raise ValueError("--out-cols must be a multiple of 128")
    if out_cols > layer.out_features:
        raise ValueError("--out-cols exceeds layer.out_features")

    x = np.asarray(x, dtype=np.float16)
    orig_shape = x.shape
    rows = int(np.prod(orig_shape[:-1]))
    x2d = x.reshape(rows, layer.in_features)

    suh = unpack_signs_or_pass(layer.suh)
    svh = unpack_signs_or_pass(layer.svh)
    xh = had_r_128(x2d.astype(np.float32), pre_scale=suh, r_scale=1.0).astype(np.float16)
    w = reconstruct_inner(
        layer.trellis,
        layer.k,
        mcg=layer.mcg,
        mul1=layer.mul1,
        n_offset=0,
        n_count=out_cols,
    )
    y = (xh.astype(np.float32) @ w.astype(np.float32)).astype(np.float16)
    y = had_r_128(
        y.astype(np.float32),
        post_scale=None if svh is None else svh[:out_cols],
        r_scale=1.0,
    ).astype(np.float16)
    if layer.bias is not None:
        y = y + layer.bias[:out_cols].astype(np.float16)
    return y.reshape(orig_shape[:-1] + (out_cols,))


def _forward_mlx(layer: EXL3Layer, x: np.ndarray, *, out_cols: int | None = None) -> np.ndarray:
    import mlx.core as mx

    from ponyexl3.mlx.decode import decode_packed_trellis_mlx
    from ponyexl3.mlx.forward import linear_forward_mlx
    from ponyexl3.mlx.hadamard import had_r_128_mlx
    from ponyexl3.mlx.signs import unpack_signs_or_pass_mlx
    from ponyexl3.ref.codebook import codebook_mode_from_flags
    from ponyexl3.ref.signs import unpack_signs_or_pass

    if out_cols is None:
        return np.array(linear_forward_mlx(layer, x))

    layer.validate()
    if out_cols % 128 != 0:
        raise ValueError("--out-cols must be a multiple of 128")

    x_np = np.asarray(x, dtype=np.float16)
    orig_shape = x_np.shape
    rows = int(np.prod(orig_shape[:-1]))
    x2d = mx.array(x_np.reshape(rows, layer.in_features))

    suh = unpack_signs_or_pass(layer.suh)
    svh = unpack_signs_or_pass(layer.svh)
    suh_mx = None if suh is None else unpack_signs_or_pass_mlx(mx.array(suh))
    svh_mx = (
        None
        if svh is None
        else unpack_signs_or_pass_mlx(mx.array(svh[:out_cols]))
    )

    xh = had_r_128_mlx(x2d.astype(mx.float32), pre_scale=suh_mx, r_scale=1.0).astype(
        mx.float16
    )
    cb = codebook_mode_from_flags(mcg=layer.mcg, mul1=layer.mul1)
    w_full = decode_packed_trellis_mlx(mx.array(layer.trellis.astype(np.uint16)), layer.k, cb)
    w = w_full[:, :out_cols]
    y = (xh.astype(mx.float32) @ w.astype(mx.float32)).astype(mx.float16)
    y = had_r_128_mlx(y.astype(mx.float32), post_scale=svh_mx, r_scale=1.0).astype(mx.float16)
    if layer.bias is not None:
        y = y + mx.array(layer.bias[:out_cols].astype(np.float16))
    return np.array(y.reshape(orig_shape[:-1] + (out_cols,)))


def _forward_cuda(
    model_dir: Path, module_key: str, x: np.ndarray
) -> np.ndarray | None:
    try:
        import torch
        from exllamav3.model import Config, Model
        from exllamav3.modules.quant.exl3 import LinearEXL3
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    config = Config.from_directory(str(model_dir))
    model = Model.from_config(config)
    linear = None
    for m in model:
        if m.key == module_key:
            linear = m
            break
    if linear is None or not isinstance(linear, LinearEXL3):
        raise SystemExit(f"module {module_key!r} not found or not EXL3")

    device = torch.device("cuda")
    xt = torch.tensor(x, dtype=torch.half, device=device)
    with torch.no_grad():
        y = linear.forward(xt, {"reconstruct": True})
    return y.cpu().numpy()


def _compare_tile(layer: EXL3Layer, tile_index: int | None, seed: int) -> dict[str, Any]:
    from ponyexl3.mlx.decode import decode_packed_tile_mlx
    from ponyexl3.ref.codebook import codebook_mode_from_flags
    from ponyexl3.ref.decode import decode_packed_tile

    import mlx.core as mx

    in_tiles, out_tiles, _ = layer.trellis.shape
    n_tiles = in_tiles * out_tiles
    if tile_index is None:
        rng = np.random.default_rng(seed)
        tile_index = int(rng.integers(0, n_tiles))
    if not 0 <= tile_index < n_tiles:
        raise ValueError(f"--tile-index must be in [0, {n_tiles})")

    tk = tile_index // out_tiles
    tn = tile_index % out_tiles
    packed = layer.trellis[tk, tn]
    cb = codebook_mode_from_flags(mcg=layer.mcg, mul1=layer.mul1)

    t0 = time.perf_counter()
    ref = decode_packed_tile(packed, layer.k, cb)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    got = np.array(decode_packed_tile_mlx(mx.array(packed), layer.k, cb))
    t_mlx = time.perf_counter() - t0

    st = _stats(got, ref)
    return {
        "tile_index": tile_index,
        "tile_coords": [int(tk), int(tn)],
        "ref_seconds": t_ref,
        "mlx_seconds": t_mlx,
        "stats": st,
    }


def _stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    both_nan = np.isnan(af) & np.isnan(bf)
    diff = np.abs(af - bf)
    diff[both_nan] = 0.0
    denom = np.abs(bf)
    mask = (denom > 1e-6) & np.isfinite(denom)
    return {
        "max_abs": float(np.nanmax(diff)),
        "mean_abs": float(np.nanmean(diff)),
        "max_rel": float((diff[mask] / denom[mask]).max()) if np.any(mask) else 0.0,
        "finite": bool(np.isfinite(af).all() and np.isfinite(bf).all()),
        "nonfinite_ref": int((~np.isfinite(af)).sum()),
        "nonfinite_mlx": int((~np.isfinite(bf)).sum()),
    }


def _format_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def _probe(model_dir: Path, module_key: str, *, rows: int, seed: int, benchmark: bool) -> dict[str, Any]:
    meta = _layer_meta(model_dir, module_key)
    n_tiles = meta["n_tiles"]
    weight_mb = meta["weight_fp16_bytes"] / (1024 * 1024)
    trellis_mb = meta["trellis_bytes"] / (1024 * 1024)
    _ = trellis_mb  # reserved for future probe output

    report: dict[str, Any] = {
        "module": module_key,
        "k": meta["k"],
        "bits_per_weight": meta["bits_per_weight"],
        "in_features": meta["in_features"],
        "out_features": meta["out_features"],
        "tiles": n_tiles,
        "trellis": _format_bytes(meta["trellis_bytes"]),
        "weight_fp16": _format_bytes(meta["weight_fp16_bytes"]),
        "mcg": meta["mcg"],
        "mul1": meta["mul1"],
    }

    print(f"module: {module_key}")
    print(
        f"  K={meta['k']}  bpw={meta['bits_per_weight']}  "
        f"in={meta['in_features']}  out={meta['out_features']}"
    )
    print(f"  tiles={n_tiles:,}  trellis={report['trellis']}  weight_fp16={report['weight_fp16']}")

    if weight_mb > 64:
        print(
            "  warning: full forward materializes the entire weight matrix on CPU — "
            "expect minutes to tens of minutes per layer"
        )
    print("  recommended:")
    print("    --mode tile          one 16x16 decode (~seconds)")
    print("    --mode slice --out-cols 128   partial forward on 128 output cols")
    if weight_mb < 8:
        print("    --mode forward       feasible at this size")

    if not benchmark:
        return report

    print("\nbenchmark (loads trellis from disk once):")
    t0 = time.perf_counter()
    layer = _load_layer(model_dir, module_key)
    t_load = time.perf_counter() - t0
    report["load_seconds"] = t_load
    print(f"  load trellis + scales: {t_load:.2f}s")

    tile_report = _compare_tile(layer, tile_index=0, seed=seed)
    report["tile_benchmark"] = tile_report
    t_tile_ref = tile_report["ref_seconds"]
    est_reconstruct = t_tile_ref * n_tiles
    est_matmul = 2 * rows * meta["in_features"] * meta["out_features"] / 5e9
    print(
        f"  tile[0] ref={t_tile_ref*1e3:.2f}ms  mlx={tile_report['mlx_seconds']*1e3:.2f}ms  "
        f"max_abs={tile_report['stats']['max_abs']:.2e}"
    )
    print(f"  estimated CPU reconstruct: {est_reconstruct:.0f}s (~{est_reconstruct/60:.1f} min)")
    print(f"  estimated matmul (rows={rows}): {est_matmul:.2f}s")
    print(f"  estimated full forward (ref): {est_reconstruct + est_matmul:.0f}s")
    return report


def _list_layers(model_dir: Path, limit: int) -> None:
    from ponyexl3.ref.loader import list_exl3_layers

    layers = list_exl3_layers(str(model_dir))
    print(f"{len(layers)} EXL3 layers in {model_dir}")
    for info in layers[:limit]:
        key = info["key"]
        trellis = info["stored_tensors"].get(f"{key}.trellis", {}).get("shape")
        print(f"  {key}  bpw={info.get('bits_per_weight')}  trellis={trellis}")
    if len(layers) > limit:
        print(f"  ... and {len(layers) - limit} more (use --list-limit)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path)
    p.add_argument("module_key", nargs="?")
    p.add_argument("--list", action="store_true", help="list EXL3 module keys and exit")
    p.add_argument("--list-limit", type=int, default=20)
    p.add_argument(
        "--mode",
        choices=("forward", "tile", "slice"),
        default="forward",
        help="forward=full layer; tile=one 16x16 block; slice=partial output cols",
    )
    p.add_argument(
        "--probe",
        action="store_true",
        help="print size estimates; with --benchmark time load + one tile",
    )
    p.add_argument("--benchmark", action="store_true", help="with --probe, run a quick tile timing")
    p.add_argument("--rows", type=int, default=4, help="batch rows for forward/slice modes")
    p.add_argument("--out-cols", type=int, default=128, help="output columns for slice mode")
    p.add_argument("--tile-index", type=int, default=None, help="flat tile index for tile mode")
    p.add_argument(
        "--backend",
        action="append",
        dest="backends",
        choices=("ref", "mlx", "cuda"),
        help="default: ref + mlx, and cuda when available",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.list:
        _list_layers(args.model_dir, args.list_limit)
        return

    if not args.module_key:
        raise SystemExit("module_key required unless --list")

    if args.probe:
        report = _probe(
            args.model_dir,
            args.module_key,
            rows=args.rows,
            seed=args.seed,
            benchmark=args.benchmark,
        )
        print(json.dumps(report, indent=2))
        return

    try:
        import mlx.core as mx
    except ImportError:
        mx = None

    if args.mode == "tile":
        t0 = time.perf_counter()
        layer = _load_layer(args.model_dir, args.module_key)
        t_load = time.perf_counter() - t0
        report = _compare_tile(layer, args.tile_index, args.seed)
        print(f"module: {args.module_key}  K={layer.k}  load={t_load:.2f}s")
        print(
            f"  tile {report['tile_index']} {report['tile_coords']}: "
            f"max_abs={report['stats']['max_abs']:.2e}  "
            f"ref={report['ref_seconds']*1e3:.2f}ms  mlx={report['mlx_seconds']*1e3:.2f}ms"
        )
        print(json.dumps(report, indent=2))
        return

    layer = _load_layer(args.model_dir, args.module_key)
    layer.validate()

    rng = np.random.default_rng(args.seed)
    out_cols = args.out_cols if args.mode == "slice" else None
    out_dim = out_cols if out_cols is not None else layer.out_features
    _ = out_dim
    x = rng.standard_normal((args.rows, layer.in_features)).astype(np.float16)

    backends = list(args.backends or ["ref", "mlx"])
    if "cuda" not in backends:
        try:
            import torch

            if torch.cuda.is_available():
                backends.append("cuda")
        except ImportError:
            pass

    results: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    if "ref" in backends:
        t0 = time.perf_counter()
        results["ref"] = _forward_ref(layer, x, out_cols=out_cols)
        timings["ref"] = time.perf_counter() - t0
    if "mlx" in backends:
        if mx is None:
            print("mlx not installed — skipping mlx backend", file=sys.stderr)
        elif not mx.metal.is_available():
            print("Metal not available — skipping mlx backend", file=sys.stderr)
        else:
            t0 = time.perf_counter()
            results["mlx"] = _forward_mlx(layer, x, out_cols=out_cols)
            timings["mlx"] = time.perf_counter() - t0
    if "cuda" in backends and args.mode == "forward":
        t0 = time.perf_counter()
        y_cuda = _forward_cuda(args.model_dir, args.module_key, x)
        if y_cuda is not None:
            results["cuda"] = y_cuda
            timings["cuda"] = time.perf_counter() - t0
        else:
            print("CUDA / exllamav3 not available — skipping cuda backend", file=sys.stderr)

    if "ref" not in results:
        raise SystemExit("ref backend required as comparison baseline")

    y_ref = results["ref"]
    report: dict[str, Any] = {
        "module": args.module_key,
        "mode": args.mode,
        "shape": list(y_ref.shape),
        "k": layer.k,
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "timings_seconds": timings,
        "backends": {},
    }

    print(f"module: {args.module_key}  mode={args.mode}")
    print(f"shape: {tuple(y_ref.shape)}  K={layer.k}")
    for name, sec in timings.items():
        print(f"  {name}: {sec:.2f}s")
    for name, y in results.items():
        if name == "ref":
            continue
        st = _stats(y, y_ref)
        report["backends"][name] = st
        print(
            f"  {name} vs ref: max_abs={st['max_abs']:.6f} "
            f"mean_abs={st['mean_abs']:.6f} max_rel={st['max_rel']:.6f}"
        )

    if mx is not None and mx.metal.is_available():
        print("note: on Apple Silicon, mlx (Metal) is the GPU reference — not PyTorch MPS")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
