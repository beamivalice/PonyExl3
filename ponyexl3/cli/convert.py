"""HF -> EXL3 converter bring-up CLI.

The default mode is the fast oracle-comparable tile pilot. `--direct-window`
quantizes one 128x128 Hadamard block, `--direct-layer` quantizes a whole
linear module without error feedback, and `--ldlq-layer` adds Hessian/LDLQ
error feedback. `--layer-modules` promotes layer modes to a bounded module-set
driver and emits one multi-layer EXL3 bundle plus a manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from ponyexl3.convert.calibration import (
    load_calibration_activations,
    load_calibration_activations_map,
)
from ponyexl3.convert.fixtures import run_tile_pilot, tile_pilot_summary


DEFAULT_PILOT_MODULE = "model.language_model.layers.0.linear_attn.in_proj_qkv"


def _as_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _format_seconds(value: object) -> str:
    seconds = _as_float(value)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{rem:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m{rem:04.1f}s"


def _progress_value(value: object) -> str:
    return "nan" if value is None else f"{_as_float(value):.6f}"


def _parse_layer_bit_overrides(
    specs: list[str],
    module_keys: list[str],
) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for spec in specs:
        pattern, sep, bits_s = spec.rpartition(":")
        if not sep or not pattern:
            raise ValueError(f"--layer-bits must be REGEX:K, got {spec!r}")
        bits = int(bits_s)
        if bits < 1 or bits > 8:
            raise ValueError(f"--layer-bits K must be in [1, 8], got {bits}")
        rx = re.compile(pattern)
        matched = [key for key in module_keys if rx.search(key)]
        if not matched:
            raise ValueError(f"--layer-bits pattern matched no selected modules: {pattern!r}")
        for key in matched:
            overrides[key] = bits
    return overrides


def _module_set_progress(scope: str):
    def emit(event: str, data: dict[str, object]) -> None:
        prefix = f"[convert:{scope}]"
        if event == "start":
            alloc = " allocation=on" if data.get("bit_plan_enabled") else ""
            print(
                f"{prefix} start modules={data['total']} quantizer={data['quantizer']} "
                f"backend={data['search_backend']} scale={data['scale_mode']}{alloc}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "module_start":
            planned = data.get("planned_k")
            planned_s = "" if planned is None else f" K={planned}"
            print(
                f"{prefix} {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
                f"start {data['module']}{planned_s} "
                f"elapsed={_format_seconds(data['elapsed_s'])}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "module_done":
            shape = data.get("shape")
            shape_s = "?"
            if isinstance(shape, list | tuple) and len(shape) == 2:
                shape_s = f"{shape[0]}x{shape[1]}"
            proxy = data.get("hessian_proxy_rel_rms")
            proxy_s = "" if proxy is None else f" proxy={_progress_value(proxy)}"
            print(
                f"{prefix} {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
                f"done {data['module']} shape={shape_s} k={data.get('k')} "
                f"output={_progress_value(data.get('output_rel_rms'))} "
                f"public={_progress_value(data.get('public_rel_rms'))}{proxy_s} "
                f"module={_format_seconds(data['module_s'])} "
                f"elapsed={_format_seconds(data['elapsed_s'])}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "module_resumed":
            print(
                f"{prefix} {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
                f"resumed {data['module']} elapsed={_format_seconds(data['elapsed_s'])}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "module_skipped":
            print(
                f"{prefix} {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
                f"skipped {data['module']}: {data['reason']}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "plain_start":
            if data.get("include_plain_tensors"):
                print(
                    f"{prefix} reading plain tensors elapsed={_format_seconds(data['elapsed_s'])}",
                    file=sys.stderr,
                    flush=True,
                )
        elif event == "write_start":
            print(
                f"{prefix} writing layers={data['layers']} plain={data['plain_tensors']} "
                f"out={data['out_dir']} elapsed={_format_seconds(data['elapsed_s'])}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "write_done":
            print(
                f"{prefix} wrote loaded_layers={data['loaded_layers']} out={data['out_dir']} "
                f"elapsed={_format_seconds(data['elapsed_s'])}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "done":
            print(
                f"{prefix} done completed={data['completed']} skipped={data['skipped']} "
                f"elapsed={_format_seconds(data['elapsed_s'])}",
                file=sys.stderr,
                flush=True,
            )

    return emit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True, type=Path, help="source HF checkpoint")
    parser.add_argument("--out-dir", type=Path, help="write converted EXL3 output bundle")
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
    parser.add_argument("--only-layer", type=int, help="layer index for --layer-modules")
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
        "--layer-modules",
        action="store_true",
        help="convert supported EXL3 modules in --only-layer instead of --only-module",
    )
    parser.add_argument(
        "--model-modules",
        action="store_true",
        help="convert all supported EXL3 modules in the oracle checkpoint",
    )
    parser.add_argument(
        "--include-routed-experts",
        action="store_true",
        help="include routed MoE experts in --layer-modules",
    )
    parser.add_argument(
        "--module-limit",
        type=int,
        help="limit selected modules for bounded layer-driver smoke runs",
    )
    parser.add_argument(
        "--use-bit-allocation",
        action="store_true",
        help="apply the M5a weighted priority K plan from --bits/--head-bits",
    )
    parser.add_argument(
        "--allocation-dry-run",
        action="store_true",
        help="print the selected module K allocation plan and exit",
    )
    parser.add_argument(
        "--layer-bits",
        action="append",
        default=[],
        metavar="REGEX:K",
        help="force selected module bits; may be repeated and later matches win",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("oracle", "oracle_safe", "identity", "computed"),
        default="oracle_safe",
        help="scale source for layer modes; computed derives fresh suh/svh from source weights",
    )
    parser.add_argument(
        "--calibration-activations",
        type=Path,
        help="pre-captured 2D activations (.npy/.npz/.safetensors) for layer modes",
    )
    parser.add_argument(
        "--calibration-activations-map",
        type=Path,
        help=(
            "pre-captured per-module activations (.npz/.safetensors or directory of .npy files); "
            "entries are keyed by module name and override --calibration-activations"
        ),
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
    parser.add_argument(
        "--ldlq-feedback-rows",
        type=int,
        default=16,
        help=(
            "LDLQ feedback granularity; 16 is exact, larger 16-multiples batch "
            "more rows per Metal launch for faster approximate LDLQ"
        ),
    )
    parser.add_argument(
        "--skip-oracle-metrics",
        action="store_true",
        help="skip expensive oracle dequantization metrics in LDLQ conversion",
    )
    parser.add_argument("--resume", action="store_true", help="reserved for full conversion")
    parser.add_argument(
        "--skip-g-scale",
        action="store_true",
        help="skip computed-scale global scale search for faster smoke runs",
    )
    parser.add_argument(
        "--regularization-seed",
        type=int,
        default=0,
        help="seed for computed-scale sign flips",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    try:
        calibration_activations = (
            None
            if args.calibration_activations is None
            else load_calibration_activations(args.calibration_activations)
        )
        calibration_activations_by_module = (
            None
            if args.calibration_activations_map is None
            else load_calibration_activations_map(args.calibration_activations_map)
        )
        layer_modes = int(args.direct_window) + int(args.direct_layer) + int(args.ldlq_layer)
        if layer_modes > 1:
            raise ValueError("--direct-window, --direct-layer, and --ldlq-layer are mutually exclusive")
        allocation_requested = bool(
            args.use_bit_allocation or args.allocation_dry_run or args.layer_bits
        )
        if allocation_requested and not (args.layer_modules or args.model_modules):
            raise ValueError(
                "--use-bit-allocation, --allocation-dry-run, and --layer-bits "
                "require --layer-modules or --model-modules"
            )
        if args.direct_layer or args.ldlq_layer:
            if args.layer_modules or args.model_modules:
                if args.layer_modules and args.model_modules:
                    raise ValueError("--layer-modules and --model-modules are mutually exclusive")
                if args.layer_modules and args.only_layer is None:
                    raise ValueError("--layer-modules requires --only-layer")
                from ponyexl3.convert.driver import (
                    bit_allocation_summary,
                    bit_plan_from_allocations,
                    convert_module_set,
                    model_module_keys,
                    module_set_summary,
                    priority_bit_allocations,
                    supported_model_module_keys,
                    supported_module_keys,
                )

                if args.model_modules:
                    module_keys, pre_skipped = supported_model_module_keys(
                        args.in_dir,
                        args.oracle_dir,
                        include_routed_experts=args.include_routed_experts,
                        module_limit=args.module_limit,
                    )
                    selected_scope = "model"
                    selected_total = len(model_module_keys(args.oracle_dir))
                else:
                    module_keys, pre_skipped = supported_module_keys(
                        args.in_dir,
                        args.oracle_dir,
                        args.only_layer,
                        include_routed_experts=args.include_routed_experts,
                        module_limit=args.module_limit,
                    )
                    selected_scope = f"layer={args.only_layer}"
                    selected_total = len(module_keys) + len(pre_skipped)
                bit_overrides = _parse_layer_bit_overrides(args.layer_bits, module_keys)
                allocation = None
                allocation_info = None
                bit_plan = bit_overrides or None
                if args.use_bit_allocation or args.allocation_dry_run:
                    allocation = priority_bit_allocations(
                        args.oracle_dir,
                        module_keys,
                        target_bpw=args.bits,
                        head_bits=args.head_bits,
                        bit_overrides=bit_overrides,
                    )
                    bit_plan = bit_plan_from_allocations(allocation)
                    allocation_info = bit_allocation_summary(
                        allocation,
                        target_bpw=args.bits,
                    )
                elif bit_overrides:
                    allocation_info = {
                        "target_bpw": args.bits,
                        "manual_overrides": bit_overrides,
                    }
                if args.allocation_dry_run:
                    dry = {
                        "scope": selected_scope,
                        "selected_total": selected_total,
                        "supported_modules": len(module_keys),
                        "pre_skipped": pre_skipped,
                        "allocation": allocation_info,
                        "bit_plan": bit_plan or {},
                    }
                    if args.json:
                        print(json.dumps(dry, indent=2, sort_keys=True))
                    else:
                        plan_for_print = bit_plan or {}
                        print(f"allocation dry-run: {selected_scope}")
                        print(
                            f"modules={len(module_keys)} selected_total={selected_total} "
                            f"target_bpw={args.bits:.3f}"
                        )
                        if allocation_info is not None:
                            print(
                                f"weighted average bits="
                                f"{allocation_info['average_bits']:.6f} "
                                f"range={allocation_info['min_bits']}-"
                                f"{allocation_info['max_bits']}"
                            )
                        for key in module_keys:
                            print(f"  K={plan_for_print.get(key, 'oracle')} {key}")
                    return 0
                result = convert_module_set(
                    args.in_dir,
                    args.oracle_dir,
                    module_keys,
                    quantizer="ldlq" if args.ldlq_layer else "direct",
                    out_dir=args.out_dir,
                    search_backend=args.search_backend,
                    scale_mode=args.scale_mode,
                    sigma_reg=args.sigma_reg,
                    buf_size_rows=args.buf_size_rows,
                    feedback_rows=args.ldlq_feedback_rows,
                    compare_oracle=not args.skip_oracle_metrics,
                    resume=bool(args.resume),
                    calibration_activations=calibration_activations,
                    calibration_activations_by_module=calibration_activations_by_module,
                    skip_g_scale=bool(args.skip_g_scale),
                    regularization_seed=args.regularization_seed,
                    include_plain_tensors=bool(args.model_modules),
                    bit_plan=bit_plan,
                    progress=_module_set_progress(selected_scope),
                )
                summary = module_set_summary(result)
                summary["allocation"] = allocation_info
                summary["pre_skipped"] = pre_skipped
                summary["requested"] = {
                    "bits": args.bits,
                    "head_bits": args.head_bits,
                    "codebook": args.codebook,
                    "out_dir": None if args.out_dir is None else str(args.out_dir),
                    "work_dir": None if args.work_dir is None else str(args.work_dir),
                    "only_layer": args.only_layer,
                    "model_modules": bool(args.model_modules),
                    "include_routed_experts": bool(args.include_routed_experts),
                    "module_limit": args.module_limit,
                    "selected_scope": selected_scope,
                    "selected_total": selected_total,
                    "search_backend": args.search_backend,
                    "scale_mode": args.scale_mode,
                    "sigma_reg": args.sigma_reg,
                    "buf_size_rows": args.buf_size_rows,
                    "ldlq_feedback_rows": args.ldlq_feedback_rows,
                    "oracle_metrics": not args.skip_oracle_metrics,
                    "resume": bool(args.resume),
                    "calibration_activations": None
                    if args.calibration_activations is None
                    else str(args.calibration_activations),
                    "calibration_activations_map": None
                    if args.calibration_activations_map is None
                    else str(args.calibration_activations_map),
                    "calibration_rows": 0
                    if calibration_activations is None
                    else int(calibration_activations.shape[0]),
                    "calibration_module_count": 0
                    if calibration_activations_by_module is None
                    else len(calibration_activations_by_module),
                    "skip_g_scale": bool(args.skip_g_scale),
                    "regularization_seed": args.regularization_seed,
                }
                if args.json:
                    print(json.dumps(summary, indent=2, sort_keys=True))
                    return 0
                print(f"module set: {selected_scope} quantizer={summary['quantizer']}")
                print(
                    f"completed={len(summary['completed'])} "
                    f"skipped={len(summary['skipped']) + len(pre_skipped)} "
                    f"out_dir={summary['out_dir']}"
                )
                for item in summary["completed"]:
                    if item.get("resumed"):
                        print(f"  {item['module']}: resumed")
                        continue
                    stats = item["stats"]
                    print(
                        f"  {item['module']}: output={stats['output_rel_rms']:.6f} "
                        f"proxy={stats.get('hessian_proxy_rel_rms', float('nan')):.6f}"
                    )
                return 0

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
                    feedback_rows=args.ldlq_feedback_rows,
                    compare_oracle=not args.skip_oracle_metrics,
                    calibration_activations=calibration_activations,
                    skip_g_scale=bool(args.skip_g_scale),
                    regularization_seed=args.regularization_seed,
                )
                summary = ldlq_layer_summary(result)
            else:
                result = direct_quantize_layer(
                    args.in_dir,
                    args.oracle_dir,
                    args.only_module,
                    search_backend=args.search_backend,
                    scale_mode=args.scale_mode,
                    calibration_activations=calibration_activations,
                    skip_g_scale=bool(args.skip_g_scale),
                    regularization_seed=args.regularization_seed,
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
                "ldlq_feedback_rows": args.ldlq_feedback_rows,
                "oracle_metrics": not args.skip_oracle_metrics,
                "resume": bool(args.resume),
                "calibration_activations": None
                if args.calibration_activations is None
                else str(args.calibration_activations),
                "calibration_rows": 0
                if calibration_activations is None
                else int(calibration_activations.shape[0]),
                "skip_g_scale": bool(args.skip_g_scale),
                "regularization_seed": args.regularization_seed,
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
                    f"oracle={stats.get('oracle_hessian_proxy_rel_rms', float('nan')):.6f}  "
                    f"ratio={stats.get('hessian_proxy_rel_rms_over_oracle', float('nan')):.6f}  "
                    f"diag_mean={stats['hessian_diag_mean']:.6e}  "
                    f"ldl_retries={stats['ldl_retries']:.0f}"
                )
            if "oracle_output_rel_rms" in stats:
                print(
                    "Oracle output: "
                    f"rel RMS={stats['oracle_output_rel_rms']:.6f}  "
                    f"converted/oracle={stats['output_rel_rms_over_oracle']:.6f}"
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
