"""Bounded candidate measurement for LDLQ conversion quality knobs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import math
import time
from typing import Any

import numpy as np

from ponyexl3.convert.calibration import activation_for_module
from ponyexl3.convert.direct import ScaleMode
from ponyexl3.convert.fixtures import SearchBackend
from ponyexl3.convert.hessian import ldlq_layer_summary, ldlq_quantize_layer

ProgressCallback = Callable[[str, dict[str, object]], None]

_DEFAULT_SCORE_METRIC = "output_rel_rms"


def _as_score(value: object) -> float:
    if isinstance(value, int | float):
        score = float(value)
        return score if math.isfinite(score) else float("inf")
    return float("inf")


def candidate_score(stats: Mapping[str, object], metric: str = _DEFAULT_SCORE_METRIC) -> float:
    """Return a finite-sortable score for one measured candidate."""

    return _as_score(stats.get(metric))


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
    records: list[dict[str, Any]] = []
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
        "hessian_shrinkages": shrinkages,
        "score_metric": score_metric,
        "records": records,
        "best_by_module": best_by_module,
        "elapsed_s": time.perf_counter() - start,
    }
