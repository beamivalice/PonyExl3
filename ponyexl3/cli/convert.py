"""HF -> EXL3 converter bring-up CLI.

The default mode is the fast oracle-comparable tile pilot. `--direct-window`
quantizes one 128x128 Hadamard block, `--direct-layer` quantizes a whole
linear module without error feedback, and `--ldlq-layer` adds Hessian/LDLQ
error feedback. `--layer-modules` promotes layer modes to a bounded module-set
driver and emits one multi-layer EXL3 bundle plus a manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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


def _load_measurement_plan(path: Path) -> tuple[dict[str, int], float | None, dict[str, object]]:
    data_obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data_obj, dict):
        raise ValueError("--measurement-plan JSON root must be an object")
    plan_obj = data_obj.get("bit_plan")
    if not isinstance(plan_obj, dict) or not plan_obj:
        raise ValueError("--measurement-plan must contain a non-empty bit_plan object")
    bit_plan: dict[str, int] = {}
    for key_obj, bits_obj in plan_obj.items():
        key = str(key_obj)
        if not key:
            raise ValueError("--measurement-plan contains an empty module key")
        if isinstance(bits_obj, bool):
            raise ValueError(f"--measurement-plan bit for {key!r} must be an integer")
        if isinstance(bits_obj, int):
            bits = bits_obj
        elif isinstance(bits_obj, float):
            if not bits_obj.is_integer():
                raise ValueError(f"--measurement-plan bit for {key!r} must be an integer")
            bits = int(bits_obj)
        elif isinstance(bits_obj, str):
            bits_s = bits_obj.strip()
            if not re.fullmatch(r"\d+", bits_s):
                raise ValueError(f"--measurement-plan bit for {key!r} must be an integer")
            bits = int(bits_s)
        else:
            raise ValueError(f"--measurement-plan bit for {key!r} must be an integer")
        if bits < 1 or bits > 8:
            raise ValueError(f"--measurement-plan bit for {key!r} must be in [1, 8], got {bits}")
        bit_plan[key] = bits

    shrinkage_obj = data_obj.get("hessian_shrinkage")
    if shrinkage_obj is None:
        shrinkage = None
    else:
        if isinstance(shrinkage_obj, bool) or not isinstance(shrinkage_obj, int | float | str):
            raise ValueError("--measurement-plan hessian_shrinkage must be a number or null")
        try:
            shrinkage = float(shrinkage_obj)
        except ValueError as exc:
            raise ValueError("--measurement-plan hessian_shrinkage must be a number or null") from exc
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("--measurement-plan hessian_shrinkage must be in [0, 1]")
    return bit_plan, shrinkage, data_obj


def _parse_csv_ints(value: str | None, *, flag: str, min_value: int, max_value: int) -> list[int] | None:
    if value is None:
        return None
    out: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise ValueError(f"{flag} must be a comma-separated integer list") from exc
        if parsed < min_value or parsed > max_value:
            raise ValueError(f"{flag} entries must be in [{min_value}, {max_value}], got {parsed}")
        out.append(parsed)
    if not out:
        raise ValueError(f"{flag} must contain at least one value")
    return out


def _parse_csv_floats(value: str | None, *, flag: str, min_value: float, max_value: float) -> list[float] | None:
    if value is None:
        return None
    out: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            parsed = float(item)
        except ValueError as exc:
            raise ValueError(f"{flag} must be a comma-separated number list") from exc
        if parsed < min_value or parsed > max_value:
            raise ValueError(f"{flag} entries must be in [{min_value}, {max_value}], got {parsed}")
        out.append(parsed)
    if not out:
        raise ValueError(f"{flag} must contain at least one value")
    return out


def _measurement_progress(scope: str):
    def emit(event: str, data: dict[str, object]) -> None:
        prefix = f"[measure:{scope}]"
        if event == "measure_start":
            bits = data.get("candidate_bits")
            bits_s = "oracle" if bits is None else str(bits)
            print(
                f"{prefix} {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
                f"start {data['module']} K={bits_s} "
                f"shrink={_as_float(data.get('hessian_shrinkage')):.3f} "
                f"elapsed={_format_seconds(data['elapsed_s'])}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "measure_done":
            print(
                f"{prefix} {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
                f"done {data['module']} K={data.get('k')} "
                f"shrink={_as_float(data.get('hessian_shrinkage')):.3f} "
                f"{data.get('score_metric')}={_progress_value(data.get('score'))} "
                f"candidate={_format_seconds(data['candidate_s'])} "
                f"elapsed={_format_seconds(data['elapsed_s'])}",
                file=sys.stderr,
                flush=True,
            )

    return emit


def _print_measurement_summary(summary: dict[str, object]) -> None:
    print(
        f"measurement: modules={_as_int(summary.get('module_count'))} "
        f"candidates={_as_int(summary.get('candidate_count'))} "
        f"score={summary.get('score_metric')} "
        f"elapsed={_format_seconds(summary.get('elapsed_s'))}"
    )
    best = summary.get("best_by_module")
    if isinstance(best, list):
        for item in best:
            if not isinstance(item, dict):
                continue
            requested = item.get("candidate_bits")
            requested_s = "oracle" if requested is None else str(requested)
            print(
                f"  {item.get('module')}: best K={item.get('k')} "
                f"requested={requested_s} "
                f"shrink={_as_float(item.get('hessian_shrinkage')):.3f} "
                f"{item.get('score_metric')}={_progress_value(item.get('score'))}"
            )


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
        elif event == "module_group_start":
            modules = data.get("modules")
            group_size = len(modules) if isinstance(modules, list | tuple) else 0
            print(
                f"{prefix} {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
                f"batch-start size={group_size} first={data['module']} "
                f"elapsed={_format_seconds(data['elapsed_s'])}",
                file=sys.stderr,
                flush=True,
            )
        elif event == "module_group_fallback":
            modules = data.get("modules")
            group_size = len(modules) if isinstance(modules, list | tuple) else 0
            print(
                f"{prefix} {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
                f"batch-fallback size={group_size} first={data['module']}: {data['reason']}",
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
            shrinkage = _as_float(data.get("hessian_shrinkage"))
            shrinkage_s = (
                ""
                if shrinkage == 0.0
                else (
                    f" shrink={shrinkage:.3f} "
                    f"offdiag={_progress_value(data.get('hessian_offdiag_rel'))}"
                )
            )
            print(
                f"{prefix} {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
                f"done {data['module']} shape={shape_s} k={data.get('k')} "
                f"output={_progress_value(data.get('output_rel_rms'))} "
                f"public={_progress_value(data.get('public_rel_rms'))}{proxy_s}{shrinkage_s} "
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


def _calibration_capture_progress(event: str, data: dict[str, object]) -> None:
    if event != "calibration_seq":
        return
    print(
        "calibration "
        f"seqs={_as_int(data.get('seqs_run'))} "
        f"captured={_as_int(data.get('captured_modules'))}/"
        f"{_as_int(data.get('module_count'))}",
        file=sys.stderr,
        flush=True,
    )


def _source_plan_dir(args: argparse.Namespace) -> Path:
    if args.work_dir is not None:
        return args.work_dir / "source_quant_plan"
    if args.out_dir is not None:
        return args.out_dir.with_name(f".{args.out_dir.name}.source_quant_plan")
    if args.capture_calibration_map is not None:
        capture_path = Path(args.capture_calibration_map)
        return capture_path.with_name(f".{capture_path.stem}.source_quant_plan")
    raise ValueError("source-only conversion requires --work-dir or --out-dir")


def _write_source_quant_plan(args: argparse.Namespace, plan_dir: Path) -> dict[str, object]:
    from ponyexl3.convert.discovery import (
        discover_exl3_module_keys,
        write_quantization_plan,
    )

    bit_overrides: dict[str, int] = {}
    if args.layer_bits:
        module_keys = discover_exl3_module_keys(
            args.in_dir,
            include_routed_experts=args.include_routed_experts,
        )
        bit_overrides = _parse_layer_bit_overrides(args.layer_bits, module_keys)
    return write_quantization_plan(
        args.in_dir,
        plan_dir,
        bits=args.bits,
        head_bits=args.head_bits,
        codebook=args.codebook,
        include_routed_experts=args.include_routed_experts,
        bit_overrides=bit_overrides or None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True, type=Path, help="source HF checkpoint")
    parser.add_argument("--out-dir", type=Path, help="write converted EXL3 output bundle")
    parser.add_argument("--work-dir", type=Path, help="reserved for resumable conversion state")
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        help="EXL3 oracle checkpoint, or a plan-only dir from --init-quant-config",
    )
    parser.add_argument(
        "--init-quant-config",
        action="store_true",
        help="write quantization_config.json (+ HF assets) from BF16 source only",
    )
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
        "--capture-calibration-map",
        type=Path,
        help=(
            "capture real BF16 source activations for selected modules and exit; "
            "output may be .safetensors or .npz and is consumable by "
            "--calibration-activations-map"
        ),
    )
    parser.add_argument(
        "--calibration-text",
        type=Path,
        help="text file used by --capture-calibration-map",
    )
    parser.add_argument(
        "--calibration-rows",
        type=int,
        default=250,
        help="activation rows per module captured by --capture-calibration-map",
    )
    parser.add_argument(
        "--calibration-seq-len",
        type=int,
        default=2048,
        help="token sequence length per forward for --capture-calibration-map",
    )
    parser.add_argument(
        "--calibration-max-seqs",
        type=int,
        help="maximum forwards for --capture-calibration-map; default is enough to fill rows",
    )
    parser.add_argument(
        "--calibration-capture-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="stored activation dtype for --capture-calibration-map",
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
        "--measurement-plan",
        type=Path,
        help=(
            "optimized JSON plan from ponyexl3-optimize-measurements; applies "
            "its bit_plan and, unless overridden, its global hessian_shrinkage"
        ),
    )
    parser.add_argument(
        "--measure-candidates",
        action="store_true",
        help="run bounded LDLQ candidate measurements for selected modules and exit",
    )
    parser.add_argument(
        "--candidate-bits",
        help=(
            "comma-separated K candidates for --measure-candidates; omitted means "
            "the selected bit plan or oracle/plan K"
        ),
    )
    parser.add_argument(
        "--candidate-hessian-shrinkages",
        help=(
            "comma-separated Hessian shrinkage candidates for --measure-candidates; "
            "omitted means the single --hessian-shrinkage value"
        ),
    )
    parser.add_argument(
        "--measure-score",
        choices=(
            "output_rel_rms",
            "hessian_proxy_rel_rms",
            "public_rel_rms",
            "hessian_proxy_rel_rms_over_oracle",
            "output_rel_rms_over_oracle",
        ),
        default="output_rel_rms",
        help="stats key used to rank --measure-candidates results",
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
        choices=("auto", "cpu", "metal"),
        default="auto",
        help="trellis search backend: 'auto' (default) uses the Metal GPU search on "
        "Apple Silicon and falls back to the CPU reference search only when Metal is "
        "unavailable; the CPU search is ~10000x slower per tile and is a reference path",
    )
    parser.add_argument("--sigma-reg", type=float, default=0.025, help="Hessian diagonal damping")
    parser.add_argument(
        "--hessian-shrinkage",
        type=float,
        default=None,
        help=(
            "shrink Hessian off-diagonal covariance toward a diagonal estimate; "
            "0.0 preserves the empirical Hessian, 1.0 is diagonal-only; "
            "default uses --measurement-plan when present, otherwise 0.0"
        ),
    )
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
        "--oracle-metrics",
        dest="oracle_metrics",
        action="store_true",
        default=False,
        help="compute expensive oracle dequantization metrics in LDLQ conversion",
    )
    parser.add_argument(
        "--skip-oracle-metrics",
        dest="oracle_metrics",
        action="store_false",
        help="skip expensive oracle dequantization metrics in LDLQ conversion (default)",
    )
    parser.add_argument(
        "--fast-layer-metrics",
        dest="fast_layer_metrics",
        action="store_true",
        default=True,
        help=(
            "skip full reconstructed-public/output/proxy diagnostics in LDLQ conversion; "
            "default production speed path"
        ),
    )
    parser.add_argument(
        "--full-layer-metrics",
        dest="fast_layer_metrics",
        action="store_false",
        help=(
            "compute full reconstructed-public/output/proxy diagnostics in LDLQ conversion; "
            "required with --oracle-metrics"
        ),
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

    if args.init_quant_config:
        if args.out_dir is None:
            raise SystemExit("--init-quant-config requires --out-dir")
        try:
            summary = _write_source_quant_plan(args, args.out_dir)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(
                f"quant plan: out={summary['out_dir']} "
                f"exl3={summary['exl3_modules']} plain={summary['plain_tensors']} "
                f"bits={_as_float(summary['bits']):.2f} "
                f"head_bits={_as_int(summary['head_bits'])} "
                f"codebook={summary['codebook']}"
            )
            copied_assets = summary.get("copied_assets")
            if isinstance(copied_assets, list) and copied_assets:
                print(f"assets: {', '.join(str(item) for item in copied_assets)}")
            print(
                "next: ponyexl3-convert --in-dir ... --oracle-dir "
                f"{summary['out_dir']} --out-dir <weights> "
                "--direct-layer --model-modules --scale-mode computed"
            )
        return 0

    source_plan_generated = False
    if args.oracle_dir is None:
        try:
            plan_dir = _source_plan_dir(args)
            summary = _write_source_quant_plan(args, plan_dir)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        args.oracle_dir = plan_dir
        source_plan_generated = True
        if not args.json:
            print(
                f"source quant plan: out={summary['out_dir']} "
                f"exl3={summary['exl3_modules']} plain={summary['plain_tensors']} "
                f"bits={_as_float(summary['bits']):.2f} "
                f"head_bits={_as_int(summary['head_bits'])} "
                f"codebook={summary['codebook']}",
                file=sys.stderr,
                flush=True,
            )

    from ponyexl3.convert.discovery import is_plan_only_checkpoint

    plan_only = is_plan_only_checkpoint(args.oracle_dir)
    if plan_only and args.oracle_metrics:
        raise SystemExit(
            f"{args.oracle_dir} is plan-only (no trellis weights); "
            "--oracle-metrics requires a real EXL3 oracle"
        )
    if plan_only and args.scale_mode in ("oracle", "oracle_safe"):
        if source_plan_generated:
            args.scale_mode = "computed"
            if not args.json:
                print(
                    "source quant plan is plan-only; using --scale-mode computed",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            raise SystemExit(
                f"{args.oracle_dir} is plan-only (no trellis weights); "
                "use --scale-mode computed or identity"
            )

    if args.search_backend == "auto":
        try:
            import mlx.core as mx

            metal_ok = bool(mx.metal.is_available())
        except Exception:
            metal_ok = False
        args.search_backend = "metal" if metal_ok else "cpu"
        if not args.json:
            detail = "Metal GPU" if metal_ok else "CPU reference (Metal unavailable)"
            print(f"search-backend=auto -> {args.search_backend} ({detail})", file=sys.stderr)

    try:
        if args.capture_calibration_map is not None:
            if args.calibration_text is None:
                raise ValueError("--capture-calibration-map requires --calibration-text")
            from ponyexl3.convert.capture import capture_calibration_activations
            from ponyexl3.convert.driver import (
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
            elif args.layer_modules:
                if args.only_layer is None:
                    raise ValueError("--layer-modules requires --only-layer")
                module_keys, pre_skipped = supported_module_keys(
                    args.in_dir,
                    args.oracle_dir,
                    args.only_layer,
                    include_routed_experts=args.include_routed_experts,
                    module_limit=args.module_limit,
                )
            else:
                module_keys = [args.only_module]
                pre_skipped = []

            if not module_keys:
                raise ValueError("no supported modules selected for calibration capture")
            if pre_skipped and not args.json:
                print(
                    f"calibration capture: skipped {len(pre_skipped)} unsupported modules",
                    file=sys.stderr,
                    flush=True,
                )
            summary = capture_calibration_activations(
                args.in_dir,
                module_keys,
                args.capture_calibration_map,
                text_path=args.calibration_text,
                rows=args.calibration_rows,
                seq_len=args.calibration_seq_len,
                max_seqs=args.calibration_max_seqs,
                dtype=args.calibration_capture_dtype,
                progress=None if args.json else _calibration_capture_progress,
            )
            out = asdict(summary)
            out["pre_skipped"] = pre_skipped
            if args.json:
                print(json.dumps(out, indent=2, sort_keys=True))
            else:
                print(
                    f"calibration map: output={summary.output} "
                    f"captured={summary.captured_count}/{summary.module_count} "
                    f"rows={summary.rows} seqs={summary.seqs_run}"
                )
                if summary.missing:
                    print(f"missing={summary.missing[:10]}")
            return 0

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
        measurement_plan_bits: dict[str, int] | None = None
        measurement_plan_shrinkage: float | None = None
        measurement_plan_summary: dict[str, object] | None = None
        if args.measurement_plan is not None:
            if args.measure_candidates:
                raise ValueError("--measurement-plan consumes optimized output; do not combine with --measure-candidates")
            if args.use_bit_allocation or args.allocation_dry_run:
                raise ValueError("--measurement-plan cannot be combined with --use-bit-allocation or --allocation-dry-run")
            if not (args.layer_modules or args.model_modules):
                raise ValueError("--measurement-plan requires --layer-modules or --model-modules")
            if not (args.direct_layer or args.ldlq_layer):
                raise ValueError("--measurement-plan requires --direct-layer or --ldlq-layer")
            measurement_plan_bits, measurement_plan_shrinkage, measurement_plan_summary = _load_measurement_plan(
                args.measurement_plan,
            )
        hessian_shrinkage_source = "default"
        if args.hessian_shrinkage is not None:
            effective_hessian_shrinkage = float(args.hessian_shrinkage)
            hessian_shrinkage_source = "cli"
        elif measurement_plan_shrinkage is not None:
            effective_hessian_shrinkage = measurement_plan_shrinkage
            hessian_shrinkage_source = "measurement_plan"
        else:
            effective_hessian_shrinkage = 0.0
        layer_modes = int(args.direct_window) + int(args.direct_layer) + int(args.ldlq_layer)
        if layer_modes > 1:
            raise ValueError("--direct-window, --direct-layer, and --ldlq-layer are mutually exclusive")
        if args.measure_candidates and not args.ldlq_layer:
            raise ValueError("--measure-candidates requires --ldlq-layer")
        if args.measure_candidates and args.out_dir is not None:
            raise ValueError("--measure-candidates does not write --out-dir")
        if not args.measure_candidates and (
            args.candidate_bits is not None or args.candidate_hessian_shrinkages is not None
        ):
            raise ValueError("--candidate-bits and --candidate-hessian-shrinkages require --measure-candidates")
        ldlq_fast_metrics = bool(args.ldlq_layer and args.fast_layer_metrics and not args.measure_candidates)
        if args.oracle_metrics and ldlq_fast_metrics:
            raise ValueError("--oracle-metrics requires --full-layer-metrics")
        if not 0.0 <= effective_hessian_shrinkage <= 1.0:
            raise ValueError("--hessian-shrinkage must be in [0, 1]")
        candidate_bits = _parse_csv_ints(
            args.candidate_bits,
            flag="--candidate-bits",
            min_value=1,
            max_value=8,
        )
        candidate_shrinkages = _parse_csv_floats(
            args.candidate_hessian_shrinkages,
            flag="--candidate-hessian-shrinkages",
            min_value=0.0,
            max_value=1.0,
        ) or [effective_hessian_shrinkage]
        allocation_requested = bool(
            args.use_bit_allocation
            or args.allocation_dry_run
            or args.layer_bits
            or args.measurement_plan is not None
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
                bit_plan = dict(measurement_plan_bits) if measurement_plan_bits is not None else None
                if bit_plan is not None:
                    missing_plan = [key for key in module_keys if key not in bit_plan]
                    if missing_plan:
                        preview = ", ".join(missing_plan[:5])
                        suffix = "" if len(missing_plan) <= 5 else f", ... +{len(missing_plan) - 5}"
                        raise ValueError(f"--measurement-plan missing selected modules: {preview}{suffix}")
                    if bit_overrides:
                        bit_plan.update(bit_overrides)
                    allocation_info = {
                        "target_bpw": measurement_plan_summary.get("target_bpw")
                        if measurement_plan_summary is not None
                        else args.bits,
                        "average_bits": measurement_plan_summary.get("average_bits")
                        if measurement_plan_summary is not None
                        else None,
                        "objective": measurement_plan_summary.get("objective")
                        if measurement_plan_summary is not None
                        else None,
                        "measurement_plan": None
                        if args.measurement_plan is None
                        else str(args.measurement_plan),
                        "measurement_plan_modules": len(bit_plan),
                        "hessian_shrinkage": measurement_plan_shrinkage,
                        "manual_overrides": bit_overrides,
                    }
                elif bit_overrides:
                    bit_plan = bit_overrides
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
                if args.measure_candidates:
                    from ponyexl3.convert.measure import measure_ldlq_candidates

                    measurement = measure_ldlq_candidates(
                        args.in_dir,
                        args.oracle_dir,
                        module_keys,
                        candidate_bits=candidate_bits,
                        hessian_shrinkages=candidate_shrinkages,
                        bit_plan=bit_plan,
                        search_backend=args.search_backend,
                        scale_mode=args.scale_mode,
                        sigma_reg=args.sigma_reg,
                        buf_size_rows=args.buf_size_rows,
                        feedback_rows=args.ldlq_feedback_rows,
                        calibration_activations=calibration_activations,
                        calibration_activations_by_module=calibration_activations_by_module,
                        skip_g_scale=bool(args.skip_g_scale),
                        regularization_seed=args.regularization_seed,
                        compare_oracle=bool(args.oracle_metrics),
                        score_metric=args.measure_score,
                        progress=None if args.json else _measurement_progress(selected_scope),
                    )
                    measurement["scope"] = selected_scope
                    measurement["selected_total"] = selected_total
                    measurement["pre_skipped"] = pre_skipped
                    measurement["allocation"] = allocation_info
                    measurement["bit_plan"] = bit_plan or {}
                    measurement["requested"] = {
                        "bits": args.bits,
                        "head_bits": args.head_bits,
                        "candidate_bits": candidate_bits,
                        "candidate_hessian_shrinkages": candidate_shrinkages,
                        "measure_score": args.measure_score,
                        "search_backend": args.search_backend,
                        "scale_mode": args.scale_mode,
                        "sigma_reg": args.sigma_reg,
                        "hessian_shrinkage": effective_hessian_shrinkage,
                        "hessian_shrinkage_source": hessian_shrinkage_source,
                        "buf_size_rows": args.buf_size_rows,
                        "ldlq_feedback_rows": args.ldlq_feedback_rows,
                        "oracle_metrics": bool(args.oracle_metrics),
                        "calibration_activations": None
                        if args.calibration_activations is None
                        else str(args.calibration_activations),
                        "calibration_activations_map": None
                        if args.calibration_activations_map is None
                        else str(args.calibration_activations_map),
                        "skip_g_scale": bool(args.skip_g_scale),
                        "regularization_seed": args.regularization_seed,
                    }
                    if args.json:
                        print(json.dumps(measurement, indent=2, sort_keys=True))
                    else:
                        _print_measurement_summary(measurement)
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
                    hessian_shrinkage=effective_hessian_shrinkage,
                    buf_size_rows=args.buf_size_rows,
                    feedback_rows=args.ldlq_feedback_rows,
                    compare_oracle=bool(args.oracle_metrics),
                    fast_metrics=ldlq_fast_metrics,
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
                    "hessian_shrinkage": effective_hessian_shrinkage,
                    "hessian_shrinkage_source": hessian_shrinkage_source,
                    "measurement_plan": None
                    if args.measurement_plan is None
                    else str(args.measurement_plan),
                    "buf_size_rows": args.buf_size_rows,
                    "ldlq_feedback_rows": args.ldlq_feedback_rows,
                    "oracle_metrics": bool(args.oracle_metrics),
                    "fast_layer_metrics": ldlq_fast_metrics,
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
                        f"  {item['module']}: output={stats.get('output_rel_rms', float('nan')):.6f} "
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

                if args.measure_candidates:
                    from ponyexl3.convert.measure import measure_ldlq_candidates

                    measurement = measure_ldlq_candidates(
                        args.in_dir,
                        args.oracle_dir,
                        [args.only_module],
                        candidate_bits=candidate_bits,
                        hessian_shrinkages=candidate_shrinkages,
                        search_backend=args.search_backend,
                        scale_mode=args.scale_mode,
                        sigma_reg=args.sigma_reg,
                        buf_size_rows=args.buf_size_rows,
                        feedback_rows=args.ldlq_feedback_rows,
                        calibration_activations=calibration_activations,
                        skip_g_scale=bool(args.skip_g_scale),
                        regularization_seed=args.regularization_seed,
                        compare_oracle=bool(args.oracle_metrics),
                        score_metric=args.measure_score,
                        progress=None if args.json else _measurement_progress("module"),
                    )
                    measurement["scope"] = "module"
                    measurement["selected_total"] = 1
                    measurement["pre_skipped"] = []
                    measurement["bit_plan"] = {}
                    measurement["requested"] = {
                        "bits": args.bits,
                        "head_bits": args.head_bits,
                        "candidate_bits": candidate_bits,
                        "candidate_hessian_shrinkages": candidate_shrinkages,
                        "measure_score": args.measure_score,
                        "only_module": args.only_module,
                        "search_backend": args.search_backend,
                        "scale_mode": args.scale_mode,
                        "sigma_reg": args.sigma_reg,
                        "hessian_shrinkage": effective_hessian_shrinkage,
                        "hessian_shrinkage_source": hessian_shrinkage_source,
                        "buf_size_rows": args.buf_size_rows,
                        "ldlq_feedback_rows": args.ldlq_feedback_rows,
                        "oracle_metrics": bool(args.oracle_metrics),
                        "calibration_activations": None
                        if args.calibration_activations is None
                        else str(args.calibration_activations),
                        "skip_g_scale": bool(args.skip_g_scale),
                        "regularization_seed": args.regularization_seed,
                    }
                    if args.json:
                        print(json.dumps(measurement, indent=2, sort_keys=True))
                    else:
                        _print_measurement_summary(measurement)
                    return 0

                result = ldlq_quantize_layer(
                    args.in_dir,
                    args.oracle_dir,
                    args.only_module,
                    search_backend=args.search_backend,
                    scale_mode=args.scale_mode,
                    sigma_reg=args.sigma_reg,
                    hessian_shrinkage=effective_hessian_shrinkage,
                    buf_size_rows=args.buf_size_rows,
                    feedback_rows=args.ldlq_feedback_rows,
                    compare_oracle=bool(args.oracle_metrics),
                    fast_metrics=ldlq_fast_metrics,
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
                "hessian_shrinkage": effective_hessian_shrinkage,
                "hessian_shrinkage_source": hessian_shrinkage_source,
                "buf_size_rows": args.buf_size_rows,
                "ldlq_feedback_rows": args.ldlq_feedback_rows,
                "oracle_metrics": bool(args.oracle_metrics),
                "fast_layer_metrics": ldlq_fast_metrics,
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
                    f"shrinkage={stats.get('hessian_shrinkage', 0.0):.3f}  "
                    f"offdiag={stats.get('hessian_offdiag_rel', float('nan')):.6f}  "
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
