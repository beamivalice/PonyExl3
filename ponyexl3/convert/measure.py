"""Bounded candidate measurement for LDLQ conversion quality knobs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import math
import re
import time
from typing import Any

import numpy as np

from ponyexl3.convert.calibration import activation_for_module
from ponyexl3.convert.direct import ScaleMode
from ponyexl3.convert.fixtures import SearchBackend
from ponyexl3.convert.hessian import ldlq_layer_summary, ldlq_quantize_layer

ProgressCallback = Callable[[str, dict[str, object]], None]

_DEFAULT_SCORE_METRIC = "output_rel_rms"


@dataclass(frozen=True)
class _MeasuredCandidate:
    module: str
    k: int
    hessian_shrinkage: float
    score: float
    weight: int
    record: Mapping[str, Any]


def _as_score(value: object) -> float:
    if isinstance(value, int | float):
        score = float(value)
        return score if math.isfinite(score) else float("inf")
    return float("inf")


def candidate_score(stats: Mapping[str, object], metric: str = _DEFAULT_SCORE_METRIC) -> float:
    """Return a finite-sortable score for one measured candidate."""

    return _as_score(stats.get(metric))


def _as_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"measurement {name} must be an object")
    return value


def _required_int(value: object, *, name: str) -> int:
    if not isinstance(value, int | float | str):
        raise ValueError(f"measurement {name} must be an integer")
    return int(value)


def _required_float(value: object, *, name: str) -> float:
    if not isinstance(value, int | float | str):
        raise ValueError(f"measurement {name} must be a number")
    return float(value)


def _record_shape_weight(record: Mapping[str, Any]) -> int:
    summary = _as_mapping(record.get("summary"), name="record.summary")
    shape = summary.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, str) or len(shape) != 2:
        raise ValueError(f"measurement record for {record.get('module')!r} is missing shape")
    return max(1, int(shape[0]) * int(shape[1]))


def _record_stats(record: Mapping[str, Any]) -> Mapping[str, object]:
    summary = _as_mapping(record.get("summary"), name="record.summary")
    return _as_mapping(summary.get("stats"), name="record.summary.stats")


def _candidate_from_record(record: Mapping[str, Any], *, score_metric: str) -> _MeasuredCandidate:
    summary = _as_mapping(record.get("summary"), name="record.summary")
    stats = _record_stats(record)
    module = str(record.get("module") or summary.get("module") or "")
    if not module:
        raise ValueError("measurement record is missing module")
    k = _required_int(record.get("k") or summary.get("k"), name=f"{module}.k")
    if k < 1 or k > 8:
        raise ValueError(f"measurement record for {module} has invalid K={k}")
    shrinkage = _required_float(
        record.get("hessian_shrinkage", stats.get("hessian_shrinkage", 0.0)),
        name=f"{module}.hessian_shrinkage",
    )
    score = candidate_score(stats, score_metric)
    if math.isinf(score):
        score = _as_score(record.get("score"))
    return _MeasuredCandidate(
        module=module,
        k=k,
        hessian_shrinkage=shrinkage,
        score=score,
        weight=_record_shape_weight(record),
        record=record,
    )


def _measurement_candidates(
    measurement: Mapping[str, Any],
    *,
    score_metric: str,
) -> list[_MeasuredCandidate]:
    records_obj = measurement.get("records")
    if not isinstance(records_obj, Sequence) or isinstance(records_obj, str):
        raise ValueError("measurement records must be a list")
    candidates = [
        _candidate_from_record(_as_mapping(record, name="record"), score_metric=score_metric)
        for record in records_obj
    ]
    if not candidates:
        raise ValueError("measurement records are empty")
    return candidates


def _objective(candidates: Sequence[_MeasuredCandidate]) -> float:
    total_weight = sum(candidate.weight for candidate in candidates)
    if total_weight <= 0:
        return float("inf")
    return float(
        sum(candidate.score * candidate.weight for candidate in candidates)
        / total_weight
    )


def _layer_bits_specs(bit_plan: Mapping[str, int]) -> list[str]:
    return [f"^{re.escape(module)}$:{bits}" for module, bits in bit_plan.items()]


def _plan_from_candidates(
    candidates: Sequence[_MeasuredCandidate],
    *,
    target_bpw: float,
    score_metric: str,
    hessian_shrinkage: float | None,
    fixed_bits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if target_bpw <= 0.0:
        raise ValueError(f"target_bpw must be positive, got {target_bpw}")
    by_module: dict[str, dict[int, _MeasuredCandidate]] = {}
    module_order: list[str] = []
    for candidate in candidates:
        if candidate.module not in by_module:
            by_module[candidate.module] = {}
            module_order.append(candidate.module)
        by_k = by_module[candidate.module]
        current = by_k.get(candidate.k)
        if current is None or (
            candidate.score,
            candidate.hessian_shrinkage,
        ) < (
            current.score,
            current.hessian_shrinkage,
        ):
            by_k[candidate.k] = candidate

    selected: dict[str, _MeasuredCandidate] = {}
    fixed = {module: int(bits) for module, bits in (fixed_bits or {}).items()}
    for module in module_order:
        by_k = by_module[module]
        if not by_k:
            raise ValueError(f"measurement has no candidates for {module}")
        if module in fixed:
            fixed_k = fixed[module]
            if fixed_k not in by_k:
                raise ValueError(f"measurement has no fixed K={fixed_k} candidate for {module}")
            selected[module] = by_k[fixed_k]
        else:
            min_k = min(by_k)
            selected[module] = by_k[min_k]

    baseline = dict(selected)
    total_weight = sum(candidate.weight for candidate in selected.values())
    target_weighted_bits = int(round(target_bpw * total_weight))
    spent_weighted_bits = sum(candidate.k * candidate.weight for candidate in selected.values())
    feasible = spent_weighted_bits <= target_weighted_bits
    upgrades: list[dict[str, Any]] = []

    while spent_weighted_bits < target_weighted_bits:
        best_upgrade: tuple[tuple[float, float, int, int, str], _MeasuredCandidate] | None = None
        for module in module_order:
            if module in fixed:
                continue
            current = selected[module]
            for candidate in by_module[module].values():
                if candidate.k <= current.k:
                    continue
                cost = (candidate.k - current.k) * candidate.weight
                if cost <= 0 or spent_weighted_bits + cost > target_weighted_bits:
                    continue
                improvement = (current.score - candidate.score) * candidate.weight
                if improvement <= 0.0:
                    continue
                ratio = improvement / cost
                key = (
                    ratio,
                    improvement,
                    -cost,
                    -candidate.k,
                    module,
                )
                if best_upgrade is None or key > best_upgrade[0]:
                    best_upgrade = (key, candidate)
        if best_upgrade is None:
            break
        (_ratio, improvement, cost_neg, _k_neg, module), candidate = best_upgrade
        previous = selected[module]
        cost = -cost_neg
        selected[module] = candidate
        spent_weighted_bits += cost
        upgrades.append(
            {
                "module": module,
                "from_k": previous.k,
                "to_k": candidate.k,
                "from_score": previous.score,
                "to_score": candidate.score,
                "score_improvement": improvement / candidate.weight,
                "weighted_score_improvement": improvement,
                "weighted_bit_cost": cost,
                "hessian_shrinkage": candidate.hessian_shrinkage,
            }
        )

    selected_list = [selected[module] for module in module_order]
    baseline_list = [baseline[module] for module in module_order]
    bit_plan = {module: int(selected[module].k) for module in module_order}
    hessian_plan = {
        module: float(selected[module].hessian_shrinkage)
        for module in module_order
    }
    layer_bits_specs = _layer_bits_specs(bit_plan)
    average_bits = spent_weighted_bits / total_weight if total_weight else 0.0
    baseline_weighted_bits = sum(candidate.k * candidate.weight for candidate in baseline_list)
    return {
        "target_bpw": float(target_bpw),
        "score_metric": score_metric,
        "hessian_shrinkage": hessian_shrinkage,
        "fixed_bits": {module: bits for module, bits in fixed.items() if module in by_module},
        "module_count": len(module_order),
        "candidate_count": len(candidates),
        "total_weight": int(total_weight),
        "target_weighted_bits": int(target_weighted_bits),
        "spent_weighted_bits": int(spent_weighted_bits),
        "baseline_weighted_bits": int(baseline_weighted_bits),
        "average_bits": float(average_bits),
        "average_bits_delta": float(average_bits - target_bpw),
        "feasible": bool(feasible),
        "objective": _objective(selected_list),
        "baseline_objective": _objective(baseline_list),
        "bit_plan": bit_plan,
        "hessian_shrinkage_plan": hessian_plan,
        "layer_bits": layer_bits_specs,
        "layer_bits_args": [
            item
            for spec in layer_bits_specs
            for item in ("--layer-bits", spec)
        ],
        "selected": [
            {
                "module": candidate.module,
                "k": int(candidate.k),
                "hessian_shrinkage": float(candidate.hessian_shrinkage),
                "score": float(candidate.score),
                "weight": int(candidate.weight),
            }
            for candidate in selected_list
        ],
        "upgrades": upgrades,
    }


def optimize_measurement_plan(
    measurement: Mapping[str, Any],
    *,
    target_bpw: float,
    score_metric: str | None = None,
    hessian_shrinkage: float | None = None,
    per_module_shrinkage: bool = False,
    fixed_bits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build a budgeted K plan from candidate measurement records."""

    metric = str(score_metric or measurement.get("score_metric") or _DEFAULT_SCORE_METRIC)
    candidates = _measurement_candidates(measurement, score_metric=metric)
    if per_module_shrinkage:
        plan = _plan_from_candidates(
            candidates,
            target_bpw=target_bpw,
            score_metric=metric,
            hessian_shrinkage=None,
            fixed_bits=fixed_bits,
        )
        plan["mode"] = "per_module_shrinkage"
        return plan

    shrinkages = sorted({candidate.hessian_shrinkage for candidate in candidates})
    selected_shrinkages = [float(hessian_shrinkage)] if hessian_shrinkage is not None else shrinkages
    global_plans: list[dict[str, Any]] = []
    module_count = len({candidate.module for candidate in candidates})
    for shrinkage in selected_shrinkages:
        filtered = [
            candidate
            for candidate in candidates
            if math.isclose(candidate.hessian_shrinkage, shrinkage, rel_tol=0.0, abs_tol=1e-9)
        ]
        if len({candidate.module for candidate in filtered}) != module_count:
            continue
        global_plans.append(
            _plan_from_candidates(
                filtered,
                target_bpw=target_bpw,
                score_metric=metric,
                hessian_shrinkage=shrinkage,
                fixed_bits=fixed_bits,
            )
        )
    if not global_plans:
        raise ValueError("no complete global hessian-shrinkage candidate plan found")
    best = min(
        global_plans,
        key=lambda plan: (
            float(plan["objective"]),
            abs(float(plan["average_bits_delta"])),
            float(plan["hessian_shrinkage"]),
        ),
    )
    out = dict(best)
    out["mode"] = "global_hessian_shrinkage"
    out["global_plans"] = global_plans
    return out


def _candidate_bits_for_module(
    module_key: str,
    *,
    candidate_bits: Sequence[int] | None,
    bit_plan: Mapping[str, int] | None,
) -> list[int | None]:
    if candidate_bits:
        return [int(bits) for bits in candidate_bits]
    if bit_plan is not None and module_key in bit_plan:
        return [int(bit_plan[module_key])]
    return [None]


def _candidate_identity(module_key: str, bits: int | None, shrinkage: float) -> tuple[str, int | None, str]:
    return (module_key, None if bits is None else int(bits), f"{float(shrinkage):.12g}")


def _measurement_summary(
    *,
    modules: Sequence[str],
    candidate_bits: Sequence[int] | None,
    shrinkages: Sequence[float],
    score_metric: str,
    records: Sequence[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    best_by_module: list[dict[str, Any]] = []
    for key in modules:
        candidates = [record for record in records if record["module"] == key]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda record: (
                float(record["score"]),
                int(record["k"]),
                float(record["hessian_shrinkage"]),
            ),
        )
        best_by_module.append(
            {
                "module": key,
                "candidate_bits": best["candidate_bits"],
                "k": best["k"],
                "hessian_shrinkage": best["hessian_shrinkage"],
                "score_metric": score_metric,
                "score": best["score"],
            }
        )

    return {
        "measure": "ldlq_candidates",
        "module_count": len(modules),
        "candidate_count": len(records),
        "candidate_bits": None if candidate_bits is None else [int(bits) for bits in candidate_bits],
        "hessian_shrinkages": [float(item) for item in shrinkages],
        "score_metric": score_metric,
        "records": list(records),
        "best_by_module": best_by_module,
        "elapsed_s": elapsed_s,
    }


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _load_measurement_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data_obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data_obj, Mapping):
        raise ValueError(f"measurement checkpoint root must be an object: {path}")
    records_obj = data_obj.get("records", [])
    if not isinstance(records_obj, Sequence) or isinstance(records_obj, str):
        raise ValueError(f"measurement checkpoint records must be a list: {path}")
    out: list[dict[str, Any]] = []
    for record_obj in records_obj:
        if not isinstance(record_obj, Mapping):
            raise ValueError(f"measurement checkpoint record must be an object: {path}")
        out.append(dict(record_obj))
    return out


def measure_ldlq_candidates(
    source_dir: str | Path,
    oracle_dir: str | Path,
    module_keys: Sequence[str],
    *,
    candidate_bits: Sequence[int] | None = None,
    hessian_shrinkages: Sequence[float] = (0.0,),
    bit_plan: Mapping[str, int] | None = None,
    search_backend: SearchBackend = "metal",
    scale_mode: ScaleMode = "oracle_safe",
    sigma_reg: float = 0.025,
    buf_size_rows: int = 128,
    feedback_rows: int = 16,
    calibration_activations: np.ndarray | None = None,
    calibration_activations_by_module: Mapping[str, np.ndarray] | None = None,
    skip_g_scale: bool = False,
    regularization_seed: int = 0,
    compare_oracle: bool = False,
    score_metric: str = _DEFAULT_SCORE_METRIC,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Quantize selected modules over candidate K/shrinkage grids and rank them."""

    modules = list(module_keys)
    shrinkages = [float(item) for item in hessian_shrinkages]
    if not modules:
        raise ValueError("no modules selected for measurement")
    if not shrinkages:
        raise ValueError("at least one hessian shrinkage candidate is required")
    for shrinkage in shrinkages:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"hessian shrinkage candidates must be in [0, 1], got {shrinkage}")

    total = sum(
        len(_candidate_bits_for_module(key, candidate_bits=candidate_bits, bit_plan=bit_plan))
        * len(shrinkages)
        for key in modules
    )
    planned: set[tuple[str, int | None, str]] = set()
    for key in modules:
        for bits in _candidate_bits_for_module(key, candidate_bits=candidate_bits, bit_plan=bit_plan):
            for shrinkage in shrinkages:
                planned.add(_candidate_identity(key, bits, shrinkage))

    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    checkpoint_records = _load_measurement_checkpoint(checkpoint) if checkpoint is not None and resume else []
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, str]] = set()
    for record in checkpoint_records:
        ident = _candidate_identity(
            str(record.get("module")),
            None if record.get("candidate_bits") is None else int(record["candidate_bits"]),
            float(record.get("hessian_shrinkage", 0.0)),
        )
        if ident not in planned or ident in seen:
            continue
        records.append(record)
        seen.add(ident)
    start = time.perf_counter()
    index = 0
    for key in modules:
        module_acts = activation_for_module(
            calibration_activations,
            calibration_activations_by_module,
            key,
        )
        for bits in _candidate_bits_for_module(
            key,
            candidate_bits=candidate_bits,
            bit_plan=bit_plan,
        ):
            if bits is not None and not 1 <= int(bits) <= 8:
                raise ValueError(f"candidate bits for {key} must be in [1, 8], got {bits}")
            for shrinkage in shrinkages:
                index += 1
                candidate_start = time.perf_counter()
                ident = _candidate_identity(key, bits, shrinkage)
                if ident in seen:
                    if progress is not None:
                        progress(
                            "measure_resumed",
                            {
                                "index": index,
                                "total": total,
                                "module": key,
                                "candidate_bits": bits,
                                "hessian_shrinkage": shrinkage,
                                "elapsed_s": candidate_start - start,
                            },
                        )
                    continue
                if progress is not None:
                    progress(
                        "measure_start",
                        {
                            "index": index,
                            "total": total,
                            "module": key,
                            "candidate_bits": bits,
                            "hessian_shrinkage": shrinkage,
                            "elapsed_s": candidate_start - start,
                        },
                    )
                result = ldlq_quantize_layer(
                    source_dir,
                    oracle_dir,
                    key,
                    search_backend=search_backend,
                    scale_mode=scale_mode,
                    sigma_reg=sigma_reg,
                    hessian_shrinkage=shrinkage,
                    buf_size_rows=buf_size_rows,
                    feedback_rows=feedback_rows,
                    calibration_activations=module_acts,
                    skip_g_scale=skip_g_scale,
                    regularization_seed=regularization_seed,
                    quant_bits=bits,
                    compare_oracle=compare_oracle,
                    fast_metrics=False,
                )
                item = ldlq_layer_summary(result)
                stats = item["stats"]
                score = candidate_score(stats, score_metric)
                record = {
                    "module": key,
                    "candidate_bits": bits,
                    "k": item["k"],
                    "hessian_shrinkage": shrinkage,
                    "score_metric": score_metric,
                    "score": score,
                    "elapsed_s": time.perf_counter() - candidate_start,
                    "summary": item,
                }
                records.append(record)
                seen.add(ident)
                if checkpoint is not None:
                    _write_json_atomic(
                        checkpoint,
                        _measurement_summary(
                            modules=modules,
                            candidate_bits=candidate_bits,
                            shrinkages=shrinkages,
                            score_metric=score_metric,
                            records=records,
                            elapsed_s=time.perf_counter() - start,
                        ),
                    )
                if progress is not None:
                    progress(
                        "measure_done",
                        {
                            "index": index,
                            "total": total,
                            "module": key,
                            "candidate_bits": bits,
                            "k": item["k"],
                            "hessian_shrinkage": shrinkage,
                            "score_metric": score_metric,
                            "score": score,
                            "elapsed_s": time.perf_counter() - start,
                            "candidate_s": record["elapsed_s"],
                        },
                    )

    summary = _measurement_summary(
        modules=modules,
        candidate_bits=candidate_bits,
        shrinkages=shrinkages,
        score_metric=score_metric,
        records=records,
        elapsed_s=time.perf_counter() - start,
    )
    if checkpoint is not None:
        _write_json_atomic(checkpoint, summary)
    return summary
