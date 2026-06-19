"""Load EXL3 layer tensors from converted model directories."""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Any, cast

import numpy as np
from safetensors import safe_open

from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.types import Exl3LayerInfo, LayerMeta

_WEIGHT_INDEX_CACHE: dict[str, dict[str, str] | None] = {}


def clear_weight_index_cache(model_dir: str) -> None:
    """Drop the cached safetensors index for a model directory."""

    _WEIGHT_INDEX_CACHE.pop(model_dir, None)


def _read_tensor(st: Any, key: str) -> np.ndarray:
    t = st.get_tensor(key)
    return np.array(t)


def _weight_index(model_dir: str) -> dict[str, str] | None:
    if model_dir not in _WEIGHT_INDEX_CACHE:
        path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                _WEIGHT_INDEX_CACHE[model_dir] = json.load(f).get("weight_map", {})
        else:
            _WEIGHT_INDEX_CACHE[model_dir] = None
    return _WEIGHT_INDEX_CACHE[model_dir]


def layer_meta_from_config(model_dir: str, module_key: str) -> LayerMeta:
    """Layer dimensions and on-disk sizes without loading weight tensors."""
    meta = {x["key"]: x for x in list_exl3_layers(model_dir)}
    if module_key not in meta:
        raise KeyError(f"{module_key!r} not in quantization_config tensor_storage")

    info = meta[module_key]
    stored = info["stored_tensors"]
    trellis_key = f"{module_key}.trellis"
    if trellis_key not in stored:
        raise KeyError(f"missing {trellis_key}")

    trellis_shape: tuple[int, int, int] = (
        int(stored[trellis_key]["shape"][0]),
        int(stored[trellis_key]["shape"][1]),
        int(stored[trellis_key]["shape"][2]),
    )
    in_tiles, out_tiles, packed_size = trellis_shape
    in_features = in_tiles * 16
    out_features = out_tiles * 16
    k = packed_size * 16 // 256
    trellis_bytes = int(stored[trellis_key].get("n_bytes", 0))

    return cast(
        LayerMeta,
        {
            "key": module_key,
            "k": k,
            "bits_per_weight": info.get("bits_per_weight"),
            "in_features": in_features,
            "out_features": out_features,
            "in_tiles": in_tiles,
            "out_tiles": out_tiles,
            "n_tiles": in_tiles * out_tiles,
            "trellis_shape": trellis_shape,
            "trellis_bytes": trellis_bytes,
            "weight_fp16_bytes": in_features * out_features * 2,
            "mcg": info["mcg"],
            "mul1": info["mul1"],
        },
    )


def list_exl3_layers(model_dir: str) -> list[Exl3LayerInfo]:
    """List EXL3 linear modules from quantization_config.json."""
    qpath = os.path.join(model_dir, "quantization_config.json")
    if not os.path.isfile(qpath):
        raise FileNotFoundError(f"missing {qpath}")
    with open(qpath, encoding="utf-8") as f:
        qcfg = json.load(f)
    storage = qcfg.get("tensor_storage", {})
    layers: list[Exl3LayerInfo] = []
    for key, info in storage.items():
        if info.get("quant_format") != "exl3":
            continue
        layers.append(
            {
                "key": key,
                "bits_per_weight": info.get("bits_per_weight"),
                "stored_tensors": info.get("stored_tensors", {}),
                "mcg": bool(info.get("mcg_multiplier")),
                "mul1": bool(info.get("mul1_multiplier")),
            }
        )
    return sorted(layers, key=lambda x: x["key"])


def _find_shard(model_dir: str, tensor_key: str) -> str:
    index = _weight_index(model_dir)
    if index is not None:
        shard = index.get(tensor_key)
        if shard is not None:
            path = os.path.join(model_dir, shard)
            if os.path.isfile(path):
                return path

    patterns = [
        os.path.join(model_dir, "*.safetensors"),
        os.path.join(model_dir, "**", "*.safetensors"),
    ]
    for pattern in patterns:
        for path in sorted(glob(pattern, recursive=True)):
            with safe_open(path, framework="np") as st:
                if tensor_key in st.keys():
                    return path
    raise FileNotFoundError(f"tensor {tensor_key!r} not found under {model_dir}")


def _load_tensor(model_dir: str, tensor_key: str) -> np.ndarray | None:
    try:
        path = _find_shard(model_dir, tensor_key)
    except FileNotFoundError:
        return None
    with safe_open(path, framework="np") as st:
        return _read_tensor(st, tensor_key)


def has_exl3_layer_weights(model_dir: str, module_key: str) -> bool:
    """True when trellis weights exist on disk (vs a plan-only quantization_config)."""

    return _load_tensor(model_dir, f"{module_key}.trellis") is not None


def load_exl3_layer(model_dir: str, module_key: str) -> EXL3Layer:
    """Load one EXL3 linear layer by module key (e.g. model.layers.0.mlp.gate_proj)."""
    if not has_exl3_layer_weights(model_dir, module_key):
        raise FileNotFoundError(f"{module_key}.trellis")
    cfg = layer_meta_from_config(model_dir, module_key)
    trellis_key = f"{module_key}.trellis"
    trellis = _load_tensor(model_dir, trellis_key)
    if trellis is None:
        raise FileNotFoundError(trellis_key)

    suh = _load_tensor(model_dir, f"{module_key}.suh")
    if suh is None:
        suh = _load_tensor(model_dir, f"{module_key}.su")
    svh = _load_tensor(model_dir, f"{module_key}.svh")
    if svh is None:
        svh = _load_tensor(model_dir, f"{module_key}.sv")
    bias = _load_tensor(model_dir, f"{module_key}.bias")

    return EXL3Layer(
        key=module_key,
        in_features=cfg["in_features"],
        out_features=cfg["out_features"],
        k=cfg["k"],
        trellis=trellis.astype(np.uint16, copy=False),
        suh=suh,
        svh=svh,
        bias=bias,
        mcg=cfg["mcg"],
        mul1=cfg["mul1"],
    )
