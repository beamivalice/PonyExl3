"""Small M4 conversion driver for module sets and layer-scoped pilots."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import json
import re
from pathlib import Path
import time
from typing import Any, Literal, Sequence

import numpy as np

from ponyexl3.convert.allocation import (
    ModuleAllocation,
    allocate_priority_bits,
    allocation_summary,
    default_module_priority,
)
from ponyexl3.convert.calibration import activation_for_module
from ponyexl3.convert.direct import (
    DirectLayerResult,
    ScaleMode,
    direct_layer_summary,
    direct_quantize_layer,
    read_source_plain_tensors,
    write_exl3_layers_bundle,
)
from ponyexl3.convert.fixtures import SearchBackend, resolve_source_linear
from ponyexl3.convert.hessian import (
    LDLQGroupIncompatible,
    ldlq_layer_summary,
    ldlq_quantize_group,
    ldlq_quantize_layer,
)
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.loader import layer_meta_from_config, list_exl3_layers, load_exl3_layer


LayerQuantizer = Literal["direct", "ldlq"]
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ModuleSetResult:
    """Summary of a module-set conversion run."""

    quantizer: LayerQuantizer
    requested_modules: list[str]
    completed: list[dict[str, Any]]
    skipped: list[dict[str, str]]
    loaded_layers: list[EXL3Layer]
    out_dir: str | None


def _natural_key(key: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", key)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def layer_module_keys(
    oracle_dir: str | Path,
    layer_index: int,
    *,
    include_routed_experts: bool = False,
    module_limit: int | None = None,
) -> list[str]:
    """List EXL3 module keys for one transformer layer."""

    prefixes = (
        f"model.language_model.layers.{layer_index}.",
        f"model.layers.{layer_index}.",
    )
    keys = [info["key"] for info in list_exl3_layers(str(oracle_dir))]
    out = []
    for key in keys:
        if not key.startswith(prefixes):
            continue
        if not include_routed_experts and ".experts." in key:
            continue
        out.append(key)
    out = sorted(out, key=_natural_key)
    if module_limit is not None:
        out = out[:module_limit]
    return out


def model_module_keys(
    oracle_dir: str | Path,
    *,
    include_routed_experts: bool = False,
    module_limit: int | None = None,
) -> list[str]:
    """List all EXL3 module keys in one oracle checkpoint."""

    keys = [info["key"] for info in list_exl3_layers(str(oracle_dir))]
    out = []
    for key in keys:
        if not include_routed_experts and ".experts." in key:
            continue
        out.append(key)
    out = sorted(out, key=_natural_key)
    if module_limit is not None:
        out = out[:module_limit]
    return out


def plain_tensor_keys(oracle_dir: str | Path) -> list[str]:
    """Return non-EXL3 tensor names from an oracle quantization config."""

    qcfg = Path(oracle_dir) / "quantization_config.json"
    with qcfg.open(encoding="utf-8") as f:
        storage = dict(json.load(f).get("tensor_storage", {}))
    out: list[str] = []
    for info in storage.values():
        if not isinstance(info, dict) or info.get("quant_format") == "exl3":
            continue
        stored = info.get("stored_tensors", {})
        if isinstance(stored, dict):
            out.extend(str(name) for name in stored)
    return sorted(out, key=_natural_key)


def module_weight_map(
    oracle_dir: str | Path,
    module_keys: Sequence[str],
) -> dict[str, int]:
    """Return parameter-count weights for EXL3 modules."""

    out: dict[str, int] = {}
    for key in module_keys:
        meta = layer_meta_from_config(str(oracle_dir), key)
        out[key] = int(meta["in_features"] * meta["out_features"])
    return out


def priority_bit_allocations(
    oracle_dir: str | Path,
    module_keys: Sequence[str],
    *,
    target_bpw: float,
    head_bits: int | None = None,
    bit_overrides: dict[str, int] | None = None,
) -> list[ModuleAllocation]:
    """Build the M5a parameter-weighted priority allocation plan."""

    keys = list(module_keys)
    weights = module_weight_map(oracle_dir, keys)
    priorities = {key: default_module_priority(key) for key in keys}
    forced: dict[str, int] = {}
    if head_bits is not None:
        forced.update({key: int(head_bits) for key in keys if key == "lm_head"})
    forced.update({key: int(bits) for key, bits in (bit_overrides or {}).items()})
    return allocate_priority_bits(
        keys,
        target_bpw=target_bpw,
        priorities=priorities,
        weights=weights,
        fixed_bits=forced,
    )


def bit_plan_from_allocations(allocations: Sequence[ModuleAllocation]) -> dict[str, int]:
    """Convert allocation records into the driver bit-plan mapping."""

    return {item.key: int(item.bits) for item in allocations}


def bit_allocation_summary(
    allocations: Sequence[ModuleAllocation],
    *,
    target_bpw: float,
) -> dict[str, Any]:
    """JSON-friendly allocation plan summary."""

    summary: dict[str, Any] = dict(allocation_summary(allocations))
    summary["target_bpw"] = float(target_bpw)
    summary["average_bits_delta"] = float(summary["average_bits"] - target_bpw)
    summary["allocations"] = [
        {
            "module": item.key,
            "bits": int(item.bits),
            "priority": float(item.priority),
            "weight": int(item.weight),
        }
        for item in allocations
    ]
    return summary


def supported_module_keys(
    source_dir: str | Path,
    oracle_dir: str | Path,
    layer_index: int,
    *,
    include_routed_experts: bool = False,
    module_limit: int | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Return layer module keys supported by the current source adapters."""

    selected = layer_module_keys(
        oracle_dir,
        layer_index,
        include_routed_experts=include_routed_experts,
        module_limit=module_limit,
    )
    supported: list[str] = []
    skipped: list[dict[str, str]] = []
    for key in selected:
        try:
            resolve_source_linear(source_dir, key)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            skipped.append({"module": key, "reason": str(exc)})
            continue
        supported.append(key)
    return supported, skipped


def supported_model_module_keys(
    source_dir: str | Path,
    oracle_dir: str | Path,
    *,
    include_routed_experts: bool = False,
    module_limit: int | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Return all model module keys supported by current source adapters."""

    selected = model_module_keys(
        oracle_dir,
        include_routed_experts=include_routed_experts,
        module_limit=module_limit,
    )
    supported: list[str] = []
    skipped: list[dict[str, str]] = []
    for key in selected:
        try:
            resolve_source_linear(source_dir, key)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            skipped.append({"module": key, "reason": str(exc)})
            continue
        supported.append(key)
    return supported, skipped


def _convert_one(
    quantizer: LayerQuantizer,
    source_dir: str | Path,
    oracle_dir: str | Path,
    module_key: str,
    *,
    search_backend: SearchBackend,
    scale_mode: ScaleMode,
    sigma_reg: float,
    hessian_shrinkage: float,
    buf_size_rows: int,
    feedback_rows: int,
    calibration_activations: np.ndarray | None,
    calibration_activations_by_module: dict[str, np.ndarray] | None,
    skip_g_scale: bool,
    regularization_seed: int,
    quant_bits: int | None,
    compare_oracle: bool,
    fast_metrics: bool,
) -> DirectLayerResult:
    module_activations = activation_for_module(
        calibration_activations,
        calibration_activations_by_module,
        module_key,
    )
    if quantizer == "direct":
        return direct_quantize_layer(
            source_dir,
            oracle_dir,
            module_key,
            search_backend=search_backend,
            scale_mode=scale_mode,
            calibration_activations=module_activations,
            skip_g_scale=skip_g_scale,
            regularization_seed=regularization_seed,
            quant_bits=quant_bits,
        )
    return ldlq_quantize_layer(
        source_dir,
        oracle_dir,
        module_key,
        search_backend=search_backend,
        scale_mode=scale_mode,
        sigma_reg=sigma_reg,
        hessian_shrinkage=hessian_shrinkage,
        buf_size_rows=buf_size_rows,
        feedback_rows=feedback_rows,
        calibration_activations=module_activations,
        skip_g_scale=skip_g_scale,
        regularization_seed=regularization_seed,
        quant_bits=quant_bits,
        compare_oracle=compare_oracle,
        fast_metrics=fast_metrics,
    )


def _summary(quantizer: LayerQuantizer, result: DirectLayerResult) -> dict[str, Any]:
    if quantizer == "direct":
        return direct_layer_summary(result)
    return ldlq_layer_summary(result)


def _sibling_group_signature(key: str) -> str | None:
    prefix, sep, suffix = key.rpartition(".")
    if not sep:
        return None
    if prefix.endswith(".self_attn") and suffix in {"q_proj", "k_proj", "v_proj"}:
        return f"{prefix}._qkv"
    if prefix.endswith(".linear_attn") and suffix in {"in_proj_qkv", "in_proj_z"}:
        return f"{prefix}._in_proj_qkvz"
    if suffix in {"gate_proj", "up_proj"} and (
        prefix.endswith(".mlp") or ".experts." in prefix
    ):
        return f"{prefix}._gate_up"
    return None


def _candidate_sibling_group(
    requested: Sequence[str],
    key: str,
    processed: set[str],
    existing: dict[str, EXL3Layer],
) -> list[str]:
    signature = _sibling_group_signature(key)
    if signature is None:
        return [key]
    group = [
        item
        for item in requested
        if item not in processed
        and item not in existing
        and _sibling_group_signature(item) == signature
    ]
    return group if len(group) >= 2 else [key]


def _grouping_enabled(
    *,
    quantizer: LayerQuantizer,
    search_backend: SearchBackend,
    compare_oracle: bool,
    fast_metrics: bool,
) -> bool:
    return (
        quantizer == "ldlq"
        and search_backend == "metal"
        and fast_metrics
        and not compare_oracle
    )


def convert_module_set(
    source_dir: str | Path,
    oracle_dir: str | Path,
    module_keys: Sequence[str],
    *,
    quantizer: LayerQuantizer = "ldlq",
    out_dir: str | Path | None = None,
    search_backend: SearchBackend = "metal",
    scale_mode: ScaleMode = "oracle_safe",
    sigma_reg: float = 0.025,
    hessian_shrinkage: float = 0.0,
    buf_size_rows: int = 128,
    feedback_rows: int = 16,
    compare_oracle: bool = True,
    fast_metrics: bool = False,
    skip_unsupported: bool = True,
    asset_dir: str | Path | None = None,
    resume: bool = False,
    calibration_activations: np.ndarray | None = None,
    calibration_activations_by_module: dict[str, np.ndarray] | None = None,
    skip_g_scale: bool = False,
    regularization_seed: int = 0,
    include_plain_tensors: bool = False,
    bit_plan: dict[str, int] | None = None,
    progress: ProgressCallback | None = None,
) -> ModuleSetResult:
    """Convert a selected module set and optionally emit one bundle."""

    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    layers: list[EXL3Layer] = []
    requested = list(module_keys)
    planned_bits = {key: int(bits) for key, bits in (bit_plan or {}).items()}
    total_start = time.perf_counter()
    if progress is not None:
        progress(
            "start",
            {
                "total": len(requested),
                "quantizer": quantizer,
                "search_backend": search_backend,
                "scale_mode": scale_mode,
                "feedback_rows": feedback_rows,
                "hessian_shrinkage": hessian_shrinkage,
                "compare_oracle": compare_oracle,
                "fast_metrics": fast_metrics,
                "bit_plan_enabled": bool(planned_bits),
            },
        )
    existing: dict[str, EXL3Layer] = {}
    if resume:
        if out_dir is None:
            raise ValueError("resume requires out_dir")
        qcfg = Path(out_dir) / "quantization_config.json"
        if qcfg.is_file():
            available = {info["key"] for info in list_exl3_layers(str(out_dir))}
            for key in requested:
                if key in available:
                    existing[key] = load_exl3_layer(str(out_dir), key)

    positions = {key: index for index, key in enumerate(requested, start=1)}
    processed: set[str] = set()
    for key in requested:
        if key in processed:
            continue
        index = positions[key]
        module_start = time.perf_counter()
        if progress is not None:
            planned_k = planned_bits.get(key)
            progress(
                "module_start",
                {
                    "index": index,
                    "total": len(requested),
                    "module": key,
                    "planned_k": planned_k,
                    "elapsed_s": module_start - total_start,
                },
            )
        if key in existing:
            processed.add(key)
            layer = existing[key]
            layers.append(layer)
            completed.append(
                {
                    "module": key,
                    "resumed": True,
                    "shape": [layer.in_features, layer.out_features],
                    "k": layer.k,
                    "stats": {},
                }
            )
            if progress is not None:
                progress(
                    "module_resumed",
                    {
                        "index": index,
                        "total": len(requested),
                        "module": key,
                        "shape": [layer.in_features, layer.out_features],
                        "k": layer.k,
                        "module_s": time.perf_counter() - module_start,
                        "elapsed_s": time.perf_counter() - total_start,
                    },
                )
            continue
        group_keys = (
            _candidate_sibling_group(requested, key, processed, existing)
            if _grouping_enabled(
                quantizer=quantizer,
                search_backend=search_backend,
                compare_oracle=compare_oracle,
                fast_metrics=fast_metrics,
            )
            else [key]
        )
        if len(group_keys) > 1:
            if progress is not None:
                progress(
                    "module_group_start",
                    {
                        "index": index,
                        "total": len(requested),
                        "module": key,
                        "modules": group_keys,
                        "elapsed_s": time.perf_counter() - total_start,
                    },
                )
            try:
                grouped_results = ldlq_quantize_group(
                    source_dir,
                    oracle_dir,
                    group_keys,
                    search_backend=search_backend,
                    scale_mode=scale_mode,
                    sigma_reg=sigma_reg,
                    hessian_shrinkage=hessian_shrinkage,
                    buf_size_rows=buf_size_rows,
                    feedback_rows=feedback_rows,
                    calibration_activations=calibration_activations,
                    calibration_activations_by_module=calibration_activations_by_module,
                    skip_g_scale=skip_g_scale,
                    regularization_seed=regularization_seed,
                    quant_bits_by_module={
                        group_key: planned_bits.get(group_key) for group_key in group_keys
                    },
                )
            except LDLQGroupIncompatible as exc:
                if progress is not None:
                    progress(
                        "module_group_fallback",
                        {
                            "index": index,
                            "total": len(requested),
                            "module": key,
                            "modules": group_keys,
                            "reason": str(exc),
                            "elapsed_s": time.perf_counter() - total_start,
                        },
                    )
                group_keys = [key]
            else:
                group_elapsed = time.perf_counter() - module_start
                for result in grouped_results:
                    processed.add(result.module_key)
                    layers.append(result.layer)
                    item = _summary(quantizer, result)
                    completed.append(item)
                    if progress is not None:
                        stats = item.get("stats", {})
                        progress(
                            "module_done",
                            {
                                "index": positions[result.module_key],
                                "total": len(requested),
                                "module": result.module_key,
                                "shape": item.get("shape"),
                                "k": item.get("k"),
                                "output_rel_rms": stats.get("output_rel_rms"),
                                "public_rel_rms": stats.get("public_rel_rms"),
                                "hessian_proxy_rel_rms": stats.get(
                                    "hessian_proxy_rel_rms"
                                ),
                                "hessian_shrinkage": stats.get("hessian_shrinkage"),
                                "hessian_offdiag_rel": stats.get("hessian_offdiag_rel"),
                                "module_s": group_elapsed,
                                "elapsed_s": time.perf_counter() - total_start,
                            },
                        )
                continue
        try:
            result = _convert_one(
                quantizer,
                source_dir,
                oracle_dir,
                key,
                search_backend=search_backend,
                scale_mode=scale_mode,
                sigma_reg=sigma_reg,
                hessian_shrinkage=hessian_shrinkage,
                buf_size_rows=buf_size_rows,
                feedback_rows=feedback_rows,
                calibration_activations=calibration_activations,
                calibration_activations_by_module=calibration_activations_by_module,
                skip_g_scale=skip_g_scale,
                regularization_seed=regularization_seed,
                quant_bits=planned_bits.get(key),
                compare_oracle=compare_oracle,
                fast_metrics=fast_metrics,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            if not skip_unsupported:
                raise
            processed.add(key)
            skipped.append({"module": key, "reason": str(exc)})
            if progress is not None:
                progress(
                    "module_skipped",
                    {
                        "index": index,
                        "total": len(requested),
                        "module": key,
                        "reason": str(exc),
                        "module_s": time.perf_counter() - module_start,
                        "elapsed_s": time.perf_counter() - total_start,
                    },
            )
            continue
        processed.add(key)
        layers.append(result.layer)
        item = _summary(quantizer, result)
        completed.append(item)
        if progress is not None:
            stats = item.get("stats", {})
            progress(
                "module_done",
                {
                    "index": index,
                    "total": len(requested),
                    "module": key,
                    "shape": item.get("shape"),
                    "k": item.get("k"),
                    "output_rel_rms": stats.get("output_rel_rms"),
                    "public_rel_rms": stats.get("public_rel_rms"),
                    "hessian_proxy_rel_rms": stats.get("hessian_proxy_rel_rms"),
                    "hessian_shrinkage": stats.get("hessian_shrinkage"),
                    "hessian_offdiag_rel": stats.get("hessian_offdiag_rel"),
                    "module_s": time.perf_counter() - module_start,
                    "elapsed_s": time.perf_counter() - total_start,
                },
            )

    loaded: list[EXL3Layer] = []
    if out_dir is not None and layers:
        if progress is not None:
            progress(
                "plain_start",
                {
                    "include_plain_tensors": include_plain_tensors,
                    "elapsed_s": time.perf_counter() - total_start,
                },
            )
        plain_tensors = (
            read_source_plain_tensors(source_dir, plain_tensor_keys(oracle_dir))
            if include_plain_tensors
            else None
        )
        if progress is not None:
            progress(
                "write_start",
                {
                    "layers": len(layers),
                    "plain_tensors": 0 if plain_tensors is None else len(plain_tensors),
                    "out_dir": str(out_dir),
                    "elapsed_s": time.perf_counter() - total_start,
                },
            )
        manifest = {
            "quantizer": quantizer,
            "search_backend": search_backend,
            "scale_mode": scale_mode,
            "sigma_reg": sigma_reg,
            "hessian_shrinkage": hessian_shrinkage,
            "buf_size_rows": buf_size_rows,
            "feedback_rows": feedback_rows,
            "compare_oracle": compare_oracle,
            "fast_metrics": fast_metrics,
            "resume": resume,
            "calibration_rows": 0
            if calibration_activations is None
            else int(calibration_activations.shape[0]),
            "calibration_module_count": 0
            if calibration_activations_by_module is None
            else len(calibration_activations_by_module),
            "skip_g_scale": skip_g_scale,
            "regularization_seed": regularization_seed,
            "include_plain_tensors": include_plain_tensors,
            "plain_tensor_count": 0 if plain_tensors is None else len(plain_tensors),
            "bit_plan": planned_bits,
            "requested_modules": requested,
            "completed": completed,
            "skipped": skipped,
        }
        loaded = write_exl3_layers_bundle(
            layers,
            out_dir,
            asset_dir=source_dir if asset_dir is None else asset_dir,
            manifest=manifest,
            plain_tensors=plain_tensors,
        )
        if progress is not None:
            progress(
                "write_done",
                {
                    "layers": len(layers),
                    "loaded_layers": len(loaded),
                    "out_dir": str(out_dir),
                    "elapsed_s": time.perf_counter() - total_start,
                },
            )

    if progress is not None:
        progress(
            "done",
            {
                "completed": len(completed),
                "skipped": len(skipped),
                "elapsed_s": time.perf_counter() - total_start,
            },
        )

    return ModuleSetResult(
        quantizer=quantizer,
        requested_modules=requested,
        completed=completed,
        skipped=skipped,
        loaded_layers=loaded,
        out_dir=None if out_dir is None else str(out_dir),
    )


def module_set_summary(result: ModuleSetResult) -> dict[str, Any]:
    """JSON-friendly module-set summary."""

    return {
        "quantizer": result.quantizer,
        "requested_modules": result.requested_modules,
        "completed": result.completed,
        "skipped": result.skipped,
        "out_dir": result.out_dir,
        "loaded_layers": [
            {
                "module": layer.key,
                "shape": [layer.in_features, layer.out_features],
                "trellis_shape": [int(x) for x in layer.trellis.shape],
                "k": layer.k,
            }
            for layer in result.loaded_layers
        ],
    }
