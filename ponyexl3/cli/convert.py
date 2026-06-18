"""HF -> EXL3 converter bring-up CLI.

The default mode is the fast oracle-comparable tile pilot. `--direct-window`
quantizes one 128x128 Hadamard block, `--direct-layer` quantizes a whole
linear module without error feedback, and `--ldlq-layer` adds Hessian/LDLQ
error feedback. Layer modes can emit a minimal loadable EXL3 bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ponyexl3.convert.fixtures import run_tile_pilot, tile_pilot_summary


DEFAULT_PILOT_MODULE = "model.language_model.layers.0.linear_attn.in_proj_qkv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True, type=Path, help="source HF checkpoint")
    parser.add_argument("--out-dir", type=Path, help="reserved for full conversion output")
    parser.add_argument("--work-dir", type=Path, help="reserved for resumable conversion state")
    parser.add_argument("--oracle-dir", required=True, type=Path, help="EXL3 oracle checkpoint")
    parser.add_argument("--bits", type=float, default=4.0, help="target decoder bpw")
    parser.add_argument("--head-bits", type=int, default=6, help="target head bpw")
    parser.add_argument(
        "--codebook",
        choices=("mcg", "mul1", "3inst"),
        default="mcg",
        help="target codebook; tile pilot uses the oracle layer's stored mode",
    )
    parser.add_argument("--only-layer", type=int, help="reserved for layer-scoped conversion")
    parser.add_argument(
        "--only-module",
        default=DEFAULT_PILOT_MODULE,
        help="module key to pilot",
    )
    parser.add_argument("--tile-k", type=int, default=0, help="input tile index")
    parser.add_argument("--tile-n", type=int, default=0, help="output tile index")
    parser.add_argument("--block-k", type=int, default=0, help="128-row block index for direct mode")
    parser.add_argument("--block-n", type=int, default=0, help="128-column block index for direct mode")
    parser.add_argument(
        "--direct-window",
        action="store_true",
        help="directly quantize one 128x128 block instead of one 16x16 tile",
    )
    parser.add_argument(
        "--direct-layer",
        action="store_true",
        help="directly quantize the whole selected linear module",
    )
    parser.add_argument(
        "--ldlq-layer",
        action="store_true",
        help="quantize the whole selected linear module with Hessian/LDLQ",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("oracle", "oracle_safe", "identity"),
        default="oracle_safe",
        help="scale source for layer modes; oracle_safe replaces zero oracle scales with 1",
    )
    parser.add_argument(
        "--search-backend",
        choices=("cpu", "metal"),
        default="cpu",
        help="trellis search backend for the tile pilot",
    )
    parser.add_argument("--sigma-reg", type=float, default=0.025, help="Hessian diagonal damping")
    parser.add_argument(
        "--buf-size-rows",
        type=int,
        default=128,
        help="LDLQ row buffer size; must be a 16-multiple",
    )
    parser.add_argument("--resume", action="store_true", help="reserved for full conversion")
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    try:
        layer_modes = int(args.direct_window) + int(args.direct_layer) + int(args.ldlq_layer)
        if layer_modes > 1:
            raise ValueError("--direct-window, --direct-layer, and --ldlq-layer are mutually exclusive")
        if args.direct_layer or args.ldlq_layer:
            from ponyexl3.convert.direct import (
                direct_layer_summary,
                direct_quantize_layer,
                write_direct_layer_bundle,
            )

            if args.ldlq_layer:
                from ponyexl3.convert.hessian import ldlq_layer_summary, ldlq_quantize_layer

                result = ldlq_quantize_layer(
                    args.in_dir,
                    args.oracle_dir,
                    args.only_module,
                    search_backend=args.search_backend,
                    scale_mode=args.scale_mode,
                    sigma_reg=args.sigma_reg,
                    buf_size_rows=args.buf_size_rows,
                )
                summary = ldlq_layer_summary(result)
            else:
                result = direct_quantize_layer(
                    args.in_dir,
                    args.oracle_dir,
                    args.only_module,
                    search_backend=args.search_backend,
                    scale_mode=args.scale_mode,
                )
                summary = direct_layer_summary(result)
            summary["requested"] = {
                "bits": args.bits,
                "head_bits": args.head_bits,
                "codebook": args.codebook,
                "out_dir": None if args.out_dir is None else str(args.out_dir),
                "work_dir": None if args.work_dir is None else str(args.work_dir),
                "only_layer": args.only_layer,
                "search_backend": args.search_backend,
                "scale_mode": args.scale_mode,
                "sigma_reg": args.sigma_reg,
                "buf_size_rows": args.buf_size_rows,
                "resume": bool(args.resume),
            }
            if args.out_dir is not None:
                loaded = write_direct_layer_bundle(result, args.out_dir)
                summary["emitted"] = {
                    "out_dir": str(args.out_dir),
                    "loaded_shape": [loaded.in_features, loaded.out_features],
                    "trellis_shape": [int(x) for x in loaded.trellis.shape],
                }
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 0
            stats = summary["stats"]
            print(f"module: {summary['module']}")
            print(
                f"layer: shape={summary['shape']}  K={summary['k']}  "
                f"codebook={summary['codebook']}  backend={summary['search_backend']}  "
                f"scale_mode={summary['scale_mode']}  quantizer={summary.get('quantizer', 'direct')}"
            )
            print(
                "public rel RMS: "
                f"{stats['public_rel_rms']:.6f}  output rel RMS: {stats['output_rel_rms']:.6f}"
            )
            print(
                "MSE: "
                f"inner={stats['inner_mse']:.6e}  "
                f"public={stats['public_mse']:.6e}  "
                f"output={stats['output_mse']:.6e}"
            )
            print(
                "scale replacements: "
                f"suh={stats.get('suh_zero_replacements', 0):.0f}  "
                f"svh={stats.get('svh_zero_replacements', 0):.0f}"
            )
            if "hessian_proxy_rel_rms" in stats:
                print(
                    "Hessian proxy: "
                    f"rel RMS={stats['hessian_proxy_rel_rms']:.6f}  "
                    f"diag_mean={stats['hessian_diag_mean']:.6e}  "
                    f"ldl_retries={stats['ldl_retries']:.0f}"
                )
            print(f"pack roundtrip: {stats['pack_roundtrip']}")
            if "emitted" in summary:
                print(f"emitted: {summary['emitted']['out_dir']}")
            return 0

        if args.direct_window:
            from ponyexl3.convert.direct import (
                direct_quantize_window,
                direct_window_summary,
                write_direct_window_bundle,
            )

            result = direct_quantize_window(
                args.in_dir,
                args.oracle_dir,
                args.only_module,
                in_start=args.block_k * 128,
                out_start=args.block_n * 128,
                search_backend=args.search_backend,
            )
            summary = direct_window_summary(result)
            summary["requested"] = {
                "bits": args.bits,
                "head_bits": args.head_bits,
                "codebook": args.codebook,
                "out_dir": None if args.out_dir is None else str(args.out_dir),
                "work_dir": None if args.work_dir is None else str(args.work_dir),
                "only_layer": args.only_layer,
                "search_backend": args.search_backend,
                "resume": bool(args.resume),
            }
            if args.out_dir is not None:
                loaded = write_direct_window_bundle(result, args.out_dir)
                summary["emitted"] = {
                    "out_dir": str(args.out_dir),
                    "loaded_shape": [loaded.in_features, loaded.out_features],
                    "trellis_shape": [int(x) for x in loaded.trellis.shape],
                }
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 0
            stats = summary["stats"]
            print(f"module: {summary['module']}")
            print(
                f"block: [{args.block_k}, {args.block_n}]  "
                f"shape={summary['shape']}  K={summary['k']}  "
                f"codebook={summary['codebook']}  backend={summary['search_backend']}"
            )
            print(
                "public rel RMS: "
                f"{stats['public_rel_rms']:.6f}  output rel RMS: {stats['output_rel_rms']:.6f}"
            )
            print(
                "MSE: "
                f"inner={stats['inner_mse']:.6e}  "
                f"public={stats['public_mse']:.6e}  "
                f"output={stats['output_mse']:.6e}"
            )
            print(f"pack roundtrip: {stats['pack_roundtrip']}")
            if "emitted" in summary:
                print(f"emitted: {summary['emitted']['out_dir']}")
            return 0

        result = run_tile_pilot(
            args.in_dir,
            args.oracle_dir,
            args.only_module,
            tile_k=args.tile_k,
            tile_n=args.tile_n,
            search_backend=args.search_backend,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    summary = tile_pilot_summary(result)
    summary["requested"] = {
        "bits": args.bits,
        "head_bits": args.head_bits,
        "codebook": args.codebook,
        "out_dir": None if args.out_dir is None else str(args.out_dir),
        "work_dir": None if args.work_dir is None else str(args.work_dir),
        "only_layer": args.only_layer,
        "search_backend": args.search_backend,
        "resume": bool(args.resume),
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    stats = summary["stats"]
    print(f"module: {summary['module']}")
    print(
        f"tile: {summary['tile']}  K={summary['k']}  "
        f"codebook={summary['codebook']}  backend={summary['search_backend']}"
    )
    print(
        "target MSE: "
        f"converted={stats['converted_target_mse']:.6e}  "
        f"oracle={stats['oracle_target_mse']:.6e}"
    )
    print(
        "rel RMS: "
        f"converted={stats['converted_target_rel_rms']:.6f}  "
        f"oracle={stats['oracle_target_rel_rms']:.6f}  "
        f"converted-vs-oracle={stats['converted_vs_oracle_rel_rms']:.6f}"
    )
    print(
        "pack roundtrip: "
        f"converted={stats['converted_pack_roundtrip']}  "
        f"oracle={stats['oracle_pack_roundtrip']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
