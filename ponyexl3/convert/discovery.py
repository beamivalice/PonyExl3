"""Discover quantizable modules and build EXL3 plans from BF16 HF checkpoints."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ponyexl3.convert.allocation import (
    allocate_priority_bits,
    default_module_priority,
)
from ponyexl3.convert.driver import bit_plan_from_allocations
from ponyexl3.convert.fixtures import SafetensorIndex, resolve_source_linear
from ponyexl3.ref.hadamard import HAD_DIM

_MODEL_ASSET_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
)

_NATURAL_SPLIT = re.compile(r"(\d+)")


def _is_exl3_linear_key(module_key: str) -> bool:
    """Heuristic: quantize attention/MLP projections and lm_head, not embeddings."""

    if module_key == "lm_head":
        return True
    if ".embed_" in module_key:
        return False
    if ".experts." in module_key:
        return True
    return ".self_attn." in module_key or ".mlp." in module_key or ".linear_attn." in module_key


def _natural_key(key: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in _NATURAL_SPLIT.split(key))


def _dtype_torch_name(dtype: str) -> str:
    mapping = {
        "BF16": "torch.bfloat16",
        "F16": "torch.float16",
        "F32": "torch.float32",
        "I32": "torch.int32",
        "U32": "torch.uint32",
        "I16": "torch.int16",
        "U16": "torch.uint16",
    }
    return mapping.get(dtype, dtype)


def _trellis_packed_size(k: int) -> int:
    return 256 * k // 16


def _trellis_shape(in_features: int, out_features: int, k: int) -> tuple[int, int, int]:
    if in_features % 16 != 0 or out_features % 16 != 0:
        raise ValueError(f"features must be multiples of 16, got {in_features}x{out_features}")
    return in_features // 16, out_features // 16, _trellis_packed_size(k)


def _meta_shape(dtype: str, shape: tuple[int, ...]) -> dict[str, Any]:
    bpe = 2 if dtype in ("BF16", "F16", "I16", "U16") else 4 if dtype in ("F32", "I32", "U32") else 1
    n_elem = 1
    for dim in shape:
        n_elem *= int(dim)
    return {
        "dtype": _dtype_torch_name(dtype),
        "shape": [int(x) for x in shape],
        "n_bytes": int(n_elem * bpe),
    }


def _is_excluded_key(tensor_key: str) -> bool:
    """Heads the EXL3 base bundle omits — converted/used separately, not part of
    the model the runtime loads. Notably Qwen3.6's multi-token-prediction (``mtp``)
    draft head, which the official oracle also excludes and which the inference
    architecture has no slot for."""
    return "mtp" in tensor_key.split(".")


def discover_exl3_module_keys(
    source_dir: str | Path,
    *,
    include_routed_experts: bool = False,
) -> list[str]:
    """List EXL3 module keys derivable from a BF16 safetensors checkpoint."""

    index = SafetensorIndex(source_dir)
    keys: set[str] = set()
    for tensor_key in index.weight_map:
        if not tensor_key.endswith(".weight"):
            continue
        if _is_excluded_key(tensor_key):
            continue
        module_key = tensor_key[: -len(".weight")]
        if not include_routed_experts and ".experts." in module_key:
            continue
        try:
            source = resolve_source_linear(source_dir, module_key)
        except (KeyError, ValueError, FileNotFoundError):
            continue
        if not _is_exl3_linear_key(module_key):
            continue
        if source.in_features % HAD_DIM != 0 or source.out_features % HAD_DIM != 0:
            continue
        keys.add(module_key)
    return sorted(keys, key=_natural_key)


def discover_plain_tensor_keys(
    source_dir: str | Path,
    exl3_module_keys: list[str],
) -> list[str]:
    """Non-EXL3 source tensors to copy verbatim.

    Everything except the weight matrices replaced by EXL3 trellises: embeddings,
    norms, biases, and non-``.weight`` parameters like the GatedDeltaNet SSM
    ``A_log`` / ``dt_bias`` (Qwen3.6 ``linear_attn``). The earlier ``.weight``-only
    filter silently dropped those, leaving the bundle unloadable.
    """

    index = SafetensorIndex(source_dir)
    exl3_weights = {f"{key}.weight" for key in exl3_module_keys}
    plain: list[str] = []
    for tensor_key in sorted(index.weight_map, key=_natural_key):
        if tensor_key in exl3_weights:
            continue  # quantized linear weight -> replaced by trellis/suh/svh
        if _is_excluded_key(tensor_key):
            continue  # separate head (e.g. mtp) not part of the base model
        plain.append(tensor_key)
    return plain


def module_weight_map_from_source(
    source_dir: str | Path,
    module_keys: list[str],
) -> dict[str, int]:
    weights: dict[str, int] = {}
    for key in module_keys:
        source = resolve_source_linear(source_dir, key)
        weights[key] = int(source.in_features * source.out_features)
    return weights


def default_bit_plan(
    source_dir: str | Path,
    module_keys: list[str],
    *,
    target_bpw: float,
    head_bits: int,
    bit_overrides: dict[str, int] | None = None,
) -> dict[str, int]:
    """M5a priority allocation from source parameter counts."""

    weights = module_weight_map_from_source(source_dir, module_keys)
    priorities = {key: default_module_priority(key) for key in module_keys}
    forced: dict[str, int] = {}
    forced.update({key: int(head_bits) for key in module_keys if key == "lm_head"})
    forced.update({key: int(bits) for key, bits in (bit_overrides or {}).items()})
    allocation = allocate_priority_bits(
        module_keys,
        target_bpw=target_bpw,
        priorities=priorities,
        weights=weights,
        fixed_bits=forced,
    )
    return bit_plan_from_allocations(allocation)


def _codebook_flags(codebook: str) -> tuple[bool, bool]:
    if codebook == "mcg":
        return True, False
    if codebook == "mul1":
        return False, True
    if codebook == "3inst":
        return False, False
    raise ValueError(f"unknown codebook {codebook!r}")


def exl3_storage_entry(
    module_key: str,
    *,
    in_features: int,
    out_features: int,
    k: int,
    mcg: bool,
    mul1: bool,
) -> dict[str, Any]:
    in_tiles, out_tiles, packed = _trellis_shape(in_features, out_features, k)
    return {
        "quant_format": "exl3",
        "bits_per_weight": float(k),
        "mcg_multiplier": bool(mcg),
        "mul1_multiplier": bool(mul1),
        "stored_tensors": {
            f"{module_key}.suh": _meta_shape("F16", (in_features,)),
            f"{module_key}.svh": _meta_shape("F16", (out_features,)),
            f"{module_key}.mcg": _meta_shape("I32", ()),
            f"{module_key}.trellis": _meta_shape("I16", (in_tiles, out_tiles, packed)),
        },
    }


def generate_quantization_config(
    source_dir: str | Path,
    *,
    bits: float = 4.0,
    head_bits: int = 6,
    codebook: str = "mcg",
    include_routed_experts: bool = False,
    bit_overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a quantization_config.json-compatible plan from BF16 weights only."""

    source_dir = Path(source_dir)
    exl3_keys = discover_exl3_module_keys(
        source_dir,
        include_routed_experts=include_routed_experts,
    )
    if not exl3_keys:
        raise ValueError(f"no quantizable EXL3 modules found under {source_dir}")

    bit_plan = default_bit_plan(
        source_dir,
        exl3_keys,
        target_bpw=bits,
        head_bits=head_bits,
        bit_overrides=bit_overrides,
    )
    mcg, mul1 = _codebook_flags(codebook)
    index = SafetensorIndex(source_dir)

    tensor_storage: dict[str, Any] = {}
    for module_key in exl3_keys:
        source = resolve_source_linear(source_dir, module_key)
        k = int(bit_plan[module_key])
        tensor_storage[module_key] = exl3_storage_entry(
            module_key,
            in_features=source.in_features,
            out_features=source.out_features,
            k=k,
            mcg=mcg,
            mul1=mul1,
        )

    for tensor_key in discover_plain_tensor_keys(source_dir, exl3_keys):
        info = index.tensor_info(tensor_key)
        storage_key = tensor_key[: -len(".weight")]
        tensor_storage[storage_key] = {
            "stored_tensors": {
                tensor_key: _meta_shape(info.dtype, info.shape),
            },
        }

    return {
        "quant_method": "exl3",
        "version": "0.0.37",
        "bits": float(bits),
        "head_bits": int(head_bits),
        "codebook": codebook,
        "out_scales": "always",
        "tensor_storage": tensor_storage,
    }


def is_plan_only_checkpoint(model_dir: str | Path) -> bool:
    """True when quantization_config exists but no EXL3 trellis weights are on disk."""

    plan_dir = Path(model_dir)
    if not (plan_dir / "quantization_config.json").is_file():
        return False
    from ponyexl3.ref.loader import has_exl3_layer_weights, list_exl3_layers

    for info in list_exl3_layers(str(plan_dir)):
        if has_exl3_layer_weights(str(plan_dir), info["key"]):
            return False
    return True


def write_quantization_plan(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    bits: float = 4.0,
    head_bits: int = 6,
    codebook: str = "mcg",
    include_routed_experts: bool = False,
    bit_overrides: dict[str, int] | None = None,
    copy_assets: bool = True,
) -> dict[str, Any]:
    """Write quantization_config.json (and optional HF assets) for a source model."""

    source_dir = Path(source_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    qcfg = generate_quantization_config(
        source_dir,
        bits=bits,
        head_bits=head_bits,
        codebook=codebook,
        include_routed_experts=include_routed_experts,
        bit_overrides=bit_overrides,
    )
    (out / "quantization_config.json").write_text(
        json.dumps(qcfg, indent=4),
        encoding="utf-8",
    )

    copied: list[str] = []
    if copy_assets:
        for name in _MODEL_ASSET_NAMES:
            src = source_dir / name
            if src.is_file():
                shutil.copy2(src, out / name)
                copied.append(name)

    exl3_count = sum(
        1 for info in qcfg["tensor_storage"].values() if info.get("quant_format") == "exl3"
    )
    return {
        "out_dir": str(out),
        "exl3_modules": exl3_count,
        "plain_tensors": len(qcfg["tensor_storage"]) - exl3_count,
        "copied_assets": copied,
        "bits": bits,
        "head_bits": head_bits,
        "codebook": codebook,
    }
