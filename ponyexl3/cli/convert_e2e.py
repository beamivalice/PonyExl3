"""End-to-end BF16 -> EXL3 conversion pipeline with resumable stages."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from ponyexl3.convert import reuse
from ponyexl3.convert.calibration import load_calibration_activations_map
from ponyexl3.convert.capture import capture_calibration_activations
from ponyexl3.convert.discovery import write_quantization_plan
from ponyexl3.convert.driver import (
    convert_module_set,
    module_set_summary,
    supported_model_module_keys,
)
from ponyexl3.convert.measure import measure_ldlq_candidates, optimize_measurement_plan


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


def _progress_value(value: object) -> str:
    return "nan" if value is None else f"{_as_float(value):.6f}"


def _write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} JSON root must be an object")
    return data


def _default_work_dir(out_dir: Path) -> Path:
    return out_dir.with_name(f".{out_dir.name}.ponyexl3-work")


def _default_candidate_bits(bits: float, head_bits: int) -> list[int]:
    _ = head_bits
    base = max(1, min(8, int(math.floor(bits))))
    out = {base}
    if not math.isclose(float(bits), float(base), rel_tol=0.0, abs_tol=1e-9):
        out.add(max(1, min(8, int(math.ceil(bits)))))
    return sorted(bits for bits in out if 1 <= bits <= 8)


def _parse_csv_ints(value: str | None, *, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    out: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        bits = int(item)
        if bits < 1 or bits > 8:
            raise ValueError(f"candidate bits must be in [1, 8], got {bits}")
        out.append(bits)
    if not out:
        raise ValueError("candidate bits must contain at least one value")
    return sorted(set(out))


def _parse_csv_floats(value: str | None) -> list[float]:
    if value is None:
        return [0.0]
    out: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        shrinkage = float(item)
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"hessian shrinkage candidates must be in [0, 1], got {shrinkage}")
        out.append(shrinkage)
    if not out:
        raise ValueError("hessian shrinkage candidates must contain at least one value")
    return sorted(set(out))


def _resolve_search_backend(value: str) -> str:
    if value != "auto":
        return value
    try:
        import mlx.core as mx

        return "metal" if bool(mx.metal.is_available()) else "cpu"
    except Exception:
        return "cpu"


def _stage(state_path: Path, name: str, data: dict[str, Any]) -> None:
    payload = {
        "stage": name,
        "updated_at": time.time(),
        **data,
    }
    _write_json_atomic(state_path, payload)


def _request_mismatches(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    return [
        key
        for key, value in current.items()
        if previous.get(key) != value
    ]


def _calibration_progress(event: str, data: dict[str, object]) -> None:
    if event != "calibration_seq":
        return
    print(
        "[e2e:calib] "
        f"seqs={_as_int(data.get('seqs_run'))} "
        f"captured={_as_int(data.get('captured_modules'))}/"
        f"{_as_int(data.get('module_count'))}",
        file=sys.stderr,
        flush=True,
    )


def _measurement_progress(event: str, data: dict[str, object]) -> None:
    if event == "measure_start":
        bits = data.get("candidate_bits")
        bits_s = "oracle" if bits is None else str(bits)
        print(
            f"[e2e:measure] {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
            f"start {data['module']} K={bits_s} "
            f"shrink={_as_float(data.get('hessian_shrinkage')):.3f}",
            file=sys.stderr,
            flush=True,
        )
    elif event == "measure_done":
        print(
            f"[e2e:measure] {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
            f"done {data['module']} K={data.get('k')} "
            f"{data.get('score_metric')}={_progress_value(data.get('score'))}",
            file=sys.stderr,
            flush=True,
        )
    elif event == "measure_resumed":
        bits = data.get("candidate_bits")
        bits_s = "oracle" if bits is None else str(bits)
        print(
            f"[e2e:measure] {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
            f"resumed {data['module']} K={bits_s} "
            f"shrink={_as_float(data.get('hessian_shrinkage')):.3f}",
            file=sys.stderr,
            flush=True,
        )


def _convert_progress(event: str, data: dict[str, object]) -> None:
    if event == "module_start":
        planned = data.get("planned_k")
        planned_s = "" if planned is None else f" K={planned}"
        print(
            f"[e2e:convert] {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
            f"start {data['module']}{planned_s}",
            file=sys.stderr,
            flush=True,
        )
    elif event == "module_done":
        print(
            f"[e2e:convert] {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
            f"done {data['module']} k={data.get('k')}",
            file=sys.stderr,
            flush=True,
        )
    elif event == "module_resumed":
        print(
            f"[e2e:convert] {_as_int(data['index']):03d}/{_as_int(data['total']):03d} "
            f"resumed {data['module']}",
            file=sys.stderr,
            flush=True,
        )
    elif event == "done":
        print(
            f"[e2e:convert] done completed={data.get('completed')} skipped={data.get('skipped')}",
            file=sys.stderr,
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True, type=Path, help="source BF16 HF checkpoint")
    parser.add_argument("--out-dir", required=True, type=Path, help="final EXL3 output bundle")
    parser.add_argument("--work-dir", type=Path, help="resumable stage artifacts")
    parser.add_argument("--bits", required=True, type=float, help="target weighted EXL3 bpw")
    parser.add_argument("--head-bits", type=int, default=6, help="forced lm_head K")
    parser.add_argument(
        "--codebook",
        choices=("mcg", "mul1", "3inst"),
        default="mcg",
        help="target EXL3 codebook",
    )
    parser.add_argument(
        "--calibration-text",
        required=True,
        type=Path,
        help="text corpus used to capture BF16 calibration activations",
    )
    parser.add_argument("--calibration-rows", type=int, default=250)
    parser.add_argument("--calibration-seq-len", type=int, default=2048)
    parser.add_argument("--calibration-max-seqs", type=int)
    parser.add_argument(
        "--calibration-capture-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--candidate-bits",
        help=(
            "comma-separated global non-head K candidates; default is floor(bits) "
            "and ceil(bits), while lm_head is measured only at --head-bits"
        ),
    )
    parser.add_argument(
        "--candidate-hessian-shrinkages",
        help="comma-separated global shrinkage candidates; default: 0.0",
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
    )
    parser.add_argument(
        "--search-backend",
        choices=("auto", "cpu", "metal"),
        default="auto",
    )
    parser.add_argument("--sigma-reg", type=float, default=0.025)
    parser.add_argument("--buf-size-rows", type=int, default=128)
    parser.add_argument("--ldlq-feedback-rows", type=int, default=16)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help=(
            "measurement threads (default 2). >1 overlaps CPU-bound prep with the "
            "GPU search to fill GPU idle time, but uses up to ~N x the per-module "
            "transient memory (Hessian/scratch — the model itself stays shared); "
            "automatically throttled toward sequential when free memory is tight. "
            "Use 1 for the plain sequential path."
        ),
    )
    parser.add_argument("--module-limit", type=int, help="limit selected modules for smoke runs")
    parser.add_argument("--include-routed-experts", action="store_true")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="resume completed artifacts in --work-dir/--out-dir (default)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="recompute stages even when artifacts exist",
    )
    parser.add_argument("--json", action="store_true", help="print final JSON summary")
    args = parser.parse_args()

    try:
        if args.bits <= 0.0:
            raise ValueError("--bits must be positive")
        if not 1 <= args.head_bits <= 8:
            raise ValueError("--head-bits must be in [1, 8]")
        work_dir = args.work_dir or _default_work_dir(args.out_dir)
        if not args.resume:
            if args.out_dir.exists() and any(args.out_dir.iterdir()):
                raise ValueError("--no-resume requires an empty --out-dir; remove it or choose a new output path")
            if work_dir.exists() and any(work_dir.iterdir()):
                raise ValueError("--no-resume requires an empty --work-dir; remove it or choose a new work path")
        work_dir.mkdir(parents=True, exist_ok=True)
        state_path = work_dir / "pipeline_state.json"
        request_path = work_dir / "pipeline_request.json"
        plan_dir = work_dir / "source_quant_plan"
        calibration_path = work_dir / "calibration.safetensors"
        measurement_path = work_dir / "measurements.json"
        measurement_plan_path = work_dir / "measurement.plan.json"
        search_backend = _resolve_search_backend(args.search_backend)
        candidate_bits = _parse_csv_ints(
            args.candidate_bits,
            default=_default_candidate_bits(args.bits, args.head_bits),
        )
        shrinkages = _parse_csv_floats(args.candidate_hessian_shrinkages)
        candidate_bits_by_module = {"lm_head": [args.head_bits]}
        request = {
            "in_dir": str(args.in_dir),
            "out_dir": str(args.out_dir),
            "bits": float(args.bits),
            "head_bits": int(args.head_bits),
            "codebook": args.codebook,
            "calibration_text": str(args.calibration_text),
            "calibration_rows": int(args.calibration_rows),
            "calibration_seq_len": int(args.calibration_seq_len),
            "calibration_max_seqs": args.calibration_max_seqs,
            "calibration_capture_dtype": args.calibration_capture_dtype,
            "candidate_bits": candidate_bits,
            "candidate_bits_by_module": candidate_bits_by_module,
            "candidate_hessian_shrinkages": shrinkages,
            "measure_score": args.measure_score,
            "search_backend": search_backend,
            "sigma_reg": float(args.sigma_reg),
            "buf_size_rows": int(args.buf_size_rows),
            "ldlq_feedback_rows": int(args.ldlq_feedback_rows),
            "module_limit": args.module_limit,
            "include_routed_experts": bool(args.include_routed_experts),
        }
        if args.resume and request_path.is_file():
            previous_request = _load_json_object(request_path)
            mismatches = _request_mismatches(previous_request, request)
            if mismatches:
                preview = ", ".join(mismatches[:8])
                suffix = "" if len(mismatches) <= 8 else f", ... +{len(mismatches) - 8}"
                raise ValueError(
                    f"work-dir request mismatch on resume: {preview}{suffix}; "
                    "use the original command or choose a new --work-dir/--out-dir"
                )
        else:
            _write_json_atomic(request_path, request)

        if not args.json:
            print(
                f"[e2e] work_dir={work_dir} bits={args.bits:.3f} "
                f"candidate_bits={candidate_bits} shrinkages={shrinkages} backend={search_backend}",
                file=sys.stderr,
                flush=True,
            )

        if args.resume and (plan_dir / "quantization_config.json").is_file():
            if not args.json:
                print(f"[e2e] resume quant plan: {plan_dir}", file=sys.stderr, flush=True)
            plan_summary: dict[str, Any] = {"out_dir": str(plan_dir), "resumed": True}
        else:
            plan_summary = write_quantization_plan(
                args.in_dir,
                plan_dir,
                bits=args.bits,
                head_bits=args.head_bits,
                codebook=args.codebook,
                include_routed_experts=bool(args.include_routed_experts),
            )
        _stage(state_path, "quant_plan", {"plan_dir": str(plan_dir), "summary": plan_summary})

        module_keys, pre_skipped = supported_model_module_keys(
            args.in_dir,
            plan_dir,
            include_routed_experts=bool(args.include_routed_experts),
            module_limit=args.module_limit,
        )
        if not module_keys:
            raise ValueError("no supported modules selected for end-to-end conversion")

        if args.resume and calibration_path.is_file():
            if not args.json:
                print(f"[e2e] resume calibration: {calibration_path}", file=sys.stderr, flush=True)
            calibration_summary: dict[str, Any] = {"output": str(calibration_path), "resumed": True}
        else:
            capture_summary = capture_calibration_activations(
                args.in_dir,
                module_keys,
                calibration_path,
                text_path=args.calibration_text,
                rows=args.calibration_rows,
                seq_len=args.calibration_seq_len,
                max_seqs=args.calibration_max_seqs,
                dtype=args.calibration_capture_dtype,
                progress=None if args.json else _calibration_progress,
            )
            calibration_summary = capture_summary.__dict__
        _stage(
            state_path,
            "calibration",
            {
                "calibration_path": str(calibration_path),
                "summary": calibration_summary,
                "pre_skipped": pre_skipped,
            },
        )

        # Reuse the measurement's quantized layers in the convert/emit pass: the
        # winning (K, shrinkage) candidate is bit-identical to what convert would
        # re-quantize, so this skips the redundant second pass (~2x on the
        # measure+convert span). Opt out with PONYEXL3_E2E_NO_REUSE=1.
        if os.environ.get("PONYEXL3_E2E_NO_REUSE", "") in ("", "0", "false", "no"):
            reuse.enable()
        activations = load_calibration_activations_map(calibration_path)
        measurement = measure_ldlq_candidates(
            args.in_dir,
            plan_dir,
            module_keys,
            candidate_bits=candidate_bits,
            candidate_bits_by_module=candidate_bits_by_module,
            hessian_shrinkages=shrinkages,
            search_backend=search_backend,  # type: ignore[arg-type]
            scale_mode="computed",
            sigma_reg=args.sigma_reg,
            buf_size_rows=args.buf_size_rows,
            feedback_rows=args.ldlq_feedback_rows,
            calibration_activations_by_module=activations,
            compare_oracle=False,
            score_metric=args.measure_score,
            checkpoint_path=measurement_path,
            resume=bool(args.resume),
            max_workers=args.max_workers,
            progress=None if args.json else _measurement_progress,
        )
        measurement["scope"] = "e2e"
        measurement["selected_total"] = len(module_keys) + len(pre_skipped)
        measurement["pre_skipped"] = pre_skipped
        measurement["requested"] = {
            "bits": args.bits,
            "head_bits": args.head_bits,
            "candidate_bits": candidate_bits,
            "candidate_bits_by_module": candidate_bits_by_module,
            "candidate_hessian_shrinkages": shrinkages,
            "measure_score": args.measure_score,
            "search_backend": search_backend,
            "scale_mode": "computed",
            "sigma_reg": args.sigma_reg,
            "buf_size_rows": args.buf_size_rows,
            "ldlq_feedback_rows": args.ldlq_feedback_rows,
            "calibration_activations_map": str(calibration_path),
        }
        _write_json_atomic(measurement_path, measurement)
        _stage(state_path, "measurement", {"measurement_path": str(measurement_path)})

        measurement_plan = optimize_measurement_plan(
            measurement,
            target_bpw=args.bits,
            score_metric=args.measure_score,
            fixed_bits={"lm_head": args.head_bits},
        )
        _write_json_atomic(measurement_plan_path, measurement_plan)
        _stage(
            state_path,
            "measurement_plan",
            {
                "measurement_plan_path": str(measurement_plan_path),
                "average_bits": measurement_plan.get("average_bits"),
                "objective": measurement_plan.get("objective"),
            },
        )
        hessian_shrinkage = measurement_plan.get("hessian_shrinkage")
        effective_shrinkage = 0.0 if hessian_shrinkage is None else float(hessian_shrinkage)

        result = convert_module_set(
            args.in_dir,
            plan_dir,
            module_keys,
            quantizer="ldlq",
            out_dir=args.out_dir,
            search_backend=search_backend,  # type: ignore[arg-type]
            scale_mode="computed",
            sigma_reg=args.sigma_reg,
            hessian_shrinkage=effective_shrinkage,
            buf_size_rows=args.buf_size_rows,
            feedback_rows=args.ldlq_feedback_rows,
            compare_oracle=False,
            fast_metrics=True,
            resume=bool(args.resume),
            calibration_activations_by_module=activations,
            include_plain_tensors=True,
            bit_plan={str(key): int(bits) for key, bits in measurement_plan["bit_plan"].items()},
            incremental_output=True,
            progress=None if args.json else _convert_progress,
        )
        reuse_stats = reuse.disable()
        conversion_summary = module_set_summary(result)
        conversion_summary["layer_reuse"] = reuse_stats
        conversion_summary["pre_skipped"] = pre_skipped
        conversion_summary["measurement_plan"] = str(measurement_plan_path)
        conversion_summary["requested"] = {
            "bits": args.bits,
            "head_bits": args.head_bits,
            "codebook": args.codebook,
            "work_dir": str(work_dir),
            "out_dir": str(args.out_dir),
            "search_backend": search_backend,
            "scale_mode": "computed",
            "hessian_shrinkage": effective_shrinkage,
            "resume": bool(args.resume),
            "incremental_output": True,
        }
        final = {
            "pipeline": "ponyexl3_e2e",
            "work_dir": str(work_dir),
            "out_dir": str(args.out_dir),
            "quant_plan": plan_summary,
            "calibration": calibration_summary,
            "measurement": {
                "path": str(measurement_path),
                "candidate_count": measurement.get("candidate_count"),
            },
            "measurement_plan": measurement_plan,
            "conversion": conversion_summary,
        }
        _write_json_atomic(work_dir / "pipeline_summary.json", final)
        _stage(state_path, "done", {"summary": str(work_dir / "pipeline_summary.json")})
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        # Always release the reuse cache, including on KeyboardInterrupt or any
        # uncaught error, so it can't hold layers wired after the run.
        reuse.disable()

    if args.json:
        print(json.dumps(final, indent=2, sort_keys=True))
    else:
        reuse_hits = reuse_stats["hits"] if reuse_stats else 0
        reuse_total = (reuse_stats["hits"] + reuse_stats["misses"]) if reuse_stats else 0
        print(
            "e2e conversion complete: "
            f"out={args.out_dir} modules={len(conversion_summary['completed'])} "
            f"plan_avg_bits={float(measurement_plan['average_bits']):.6f} "
            f"layer_reuse={reuse_hits}/{reuse_total}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
