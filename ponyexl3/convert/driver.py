"""Small M4 conversion driver for module sets and layer-scoped pilots."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Literal, Sequence

from ponyexl3.convert.direct import (
    DirectLayerResult,
    ScaleMode,
    direct_layer_summary,
    direct_quantize_layer,
    write_exl3_layers_bundle,
)
from ponyexl3.convert.fixtures import SearchBackend, resolve_source_linear
from ponyexl3.convert.hessian import ldlq_layer_summary, ldlq_quantize_layer
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.loader import list_exl3_layers, load_exl3_layer


LayerQuantizer = Literal["direct", "ldlq"]


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

    prefix = f"model.language_model.layers.{layer_index}."
    keys = [info["key"] for info in list_exl3_layers(str(oracle_dir))]
    out = []
    for key in keys:
        if not key.startswith(prefix):
            continue
        if not include_routed_experts and ".experts." in key:
            continue
        out.append(key)
    out = sorted(out, key=_natural_key)
    if module_limit is not None:
        out = out[:module_limit]
    return out


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


def _convert_one(
    quantizer: LayerQuantizer,
    source_dir: str | Path,
    oracle_dir: str | Path,
    module_key: str,
    *,
    search_backend: SearchBackend,
    scale_mode: ScaleMode,
    sigma_reg: float,
    buf_size_rows: int,
) -> DirectLayerResult:
    if quantizer == "direct":
        return direct_quantize_layer(
            source_dir,
            oracle_dir,
            module_key,
            search_backend=search_backend,
            scale_mode=scale_mode,
        )
    return ldlq_quantize_layer(
        source_dir,
        oracle_dir,
        module_key,
        search_backend=search_backend,
        scale_mode=scale_mode,
        sigma_reg=sigma_reg,
        buf_size_rows=buf_size_rows,
    )


def _summary(quantizer: LayerQuantizer, result: DirectLayerResult) -> dict[str, Any]:
    if quantizer == "direct":
        return direct_layer_summary(result)
    return ldlq_layer_summary(result)


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
    buf_size_rows: int = 128,
    skip_unsupported: bool = True,
    asset_dir: str | Path | None = None,
    resume: bool = False,
) -> ModuleSetResult:
    """Convert a selected module set and optionally emit one bundle."""

    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    layers: list[EXL3Layer] = []
    requested = list(module_keys)
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

    for key in requested:
        if key in existing:
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
                buf_size_rows=buf_size_rows,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            if not skip_unsupported:
                raise
            skipped.append({"module": key, "reason": str(exc)})
            continue
        layers.append(result.layer)
        completed.append(_summary(quantizer, result))

    loaded: list[EXL3Layer] = []
    if out_dir is not None and layers:
        manifest = {
            "quantizer": quantizer,
            "search_backend": search_backend,
            "scale_mode": scale_mode,
            "sigma_reg": sigma_reg,
            "buf_size_rows": buf_size_rows,
            "resume": resume,
            "requested_modules": requested,
            "completed": completed,
            "skipped": skipped,
        }
        loaded = write_exl3_layers_bundle(
            layers,
            out_dir,
            asset_dir=source_dir if asset_dir is None else asset_dir,
            manifest=manifest,
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
