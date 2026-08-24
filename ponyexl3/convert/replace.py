"""Clone an EXL3 checkpoint and replace selected quantized linears.

Used to re-quantize one module (typically the output head) from a BF16 source
and drop it into an existing multi-shard EXL3 bundle without rewriting the
untouched shards. Safetensors shards are hardlinked; metadata is copied and
then patched so the source checkpoint is never mutated.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
from safetensors.numpy import save_file

from ponyexl3.convert.heads import is_output_head
from ponyexl3.ref.codebook import MCG_MULT, MUL1_MULT
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.loader import clear_weight_index_cache, load_exl3_layer


def _layer_shard_name(layer_key: str) -> str:
    digest = hashlib.sha1(layer_key.encode("utf-8")).hexdigest()[:16]
    return f"ponyexl3-layer-{digest}.safetensors"


def _write_json_atomic(path: Path, data: object) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=4), encoding="utf-8")
    tmp.replace(path)


def _torch_dtype_name(arr: np.ndarray) -> str:
    if arr.dtype == np.float16:
        return "torch.float16"
    if arr.dtype == np.float32:
        return "torch.float32"
    if arr.dtype in (np.int16, np.uint16):
        return "torch.int16"
    if arr.dtype in (np.int32, np.uint32):
        return "torch.int32"
    raise ValueError(f"unsupported splice dtype {arr.dtype}")


def _stored_tensor_meta(arr: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(x) for x in arr.shape],
        "n_bytes": int(arr.nbytes),
        "dtype": _torch_dtype_name(arr),
    }


def _layer_tensors_host(layer: EXL3Layer) -> dict[str, np.ndarray]:
    """EXL3 tensors in the exllamav3 on-disk convention (signed ints, codebook flag)."""

    tensors: dict[str, np.ndarray] = {}
    if layer.suh is not None:
        tensors[f"{layer.key}.suh"] = np.ascontiguousarray(layer.suh.astype(np.float16, copy=False))
    if layer.svh is not None:
        tensors[f"{layer.key}.svh"] = np.ascontiguousarray(layer.svh.astype(np.float16, copy=False))
    if layer.mul1:
        tensors[f"{layer.key}.mul1"] = np.array(np.uint32(int(MUL1_MULT))).view(np.int32)
    if layer.mcg:
        tensors[f"{layer.key}.mcg"] = np.array(np.uint32(int(MCG_MULT))).view(np.int32)
    tensors[f"{layer.key}.trellis"] = np.ascontiguousarray(layer.trellis.astype(np.int16, copy=False))
    if layer.bias is not None:
        tensors[f"{layer.key}.bias"] = layer.bias
    return tensors


def _host_storage_entry(layer: EXL3Layer, tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "stored_tensors": {name: _stored_tensor_meta(arr) for name, arr in tensors.items()},
        "quant_format": "exl3",
        "bits_per_weight": int(layer.k),
    }
    if layer.mul1:
        entry["mul1_multiplier"] = int(MUL1_MULT)
    if layer.mcg:
        entry["mcg_multiplier"] = int(MCG_MULT)
    return entry


def _key_belongs_to_module(tensor_key: str, module_key: str) -> bool:
    return tensor_key == module_key or tensor_key.startswith(f"{module_key}.")


def _drop_module_tensors_from_shard(path: Path, module_keys: Sequence[str]) -> list[str]:
    """Rewrite ``path`` without tensors for ``module_keys``. Breaks a hardlink.

    Copies raw safetensors payloads so BF16/FP8 tensors that NumPy cannot
    decode still survive the rewrite.
    """

    import struct

    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header_obj = json.loads(handle.read(header_len))
        if not isinstance(header_obj, dict):
            raise ValueError(f"{path} safetensors header must be an object")
        data_base = 8 + header_len
        dropped: list[str] = []
        kept_items: list[tuple[str, dict[str, Any], bytes]] = []
        for key, meta in header_obj.items():
            if key == "__metadata__":
                continue
            if not isinstance(meta, dict):
                raise ValueError(f"{path} tensor {key!r} metadata must be an object")
            if any(_key_belongs_to_module(key, module) for module in module_keys):
                dropped.append(key)
                continue
            start, end = meta["data_offsets"]
            handle.seek(data_base + int(start))
            payload = handle.read(int(end) - int(start))
            kept_items.append((key, dict(meta), payload))

    if not dropped:
        return []

    new_header: dict[str, Any] = {}
    offset = 0
    blobs: list[bytes] = []
    for key, meta, payload in kept_items:
        meta["data_offsets"] = [offset, offset + len(payload)]
        new_header[key] = meta
        blobs.append(payload)
        offset += len(payload)
    metadata = header_obj.get("__metadata__")
    if metadata is not None:
        new_header["__metadata__"] = metadata

    header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for blob in blobs:
            handle.write(blob)
    tmp.replace(path)
    return dropped


def clone_exl3_checkpoint(src: str | Path, dst: str | Path) -> list[str]:
    """Copy an EXL3 directory, hardlinking weight shards so the body is not duplicated."""

    src_dir = Path(src)
    dst_dir = Path(dst)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"--replace-into checkpoint not found: {src_dir}")
    if dst_dir.exists():
        if any(dst_dir.iterdir()):
            raise ValueError(
                f"--out-dir {dst_dir} already exists and is not empty; "
                "choose a new path or remove it"
            )
    else:
        dst_dir.mkdir(parents=True)

    copied: list[str] = []
    for item in sorted(src_dir.iterdir()):
        if item.name.startswith("."):
            continue
        if not item.is_file():
            continue
        dest = dst_dir / item.name
        if item.suffix == ".safetensors":
            os.link(item, dest)
        else:
            shutil.copy2(item, dest)
        copied.append(item.name)
    if "quantization_config.json" not in copied:
        raise FileNotFoundError(f"{src_dir} is missing quantization_config.json")
    if "model.safetensors.index.json" not in copied:
        raise FileNotFoundError(f"{src_dir} is missing model.safetensors.index.json")
    return copied


def replace_layers_into_checkpoint(
    replace_into: str | Path,
    out_dir: str | Path,
    layers: Sequence[EXL3Layer],
    *,
    bits: float | None = None,
    head_bits: int | None = None,
) -> dict[str, Any]:
    """Clone ``replace_into`` to ``out_dir`` and overwrite the given EXL3 modules."""

    if not layers:
        raise ValueError("replace_layers_into_checkpoint requires at least one layer")

    src = Path(replace_into)
    out = Path(out_dir)
    cloned = clone_exl3_checkpoint(src, out)

    index_path = out / "model.safetensors.index.json"
    qcfg_path = out / "quantization_config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    qcfg = json.loads(qcfg_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not isinstance(qcfg, dict):
        raise ValueError("cloned checkpoint JSON roots must be objects")
    weight_map_obj = index.setdefault("weight_map", {})
    storage_obj = qcfg.setdefault("tensor_storage", {})
    if not isinstance(weight_map_obj, dict) or not isinstance(storage_obj, dict):
        raise ValueError("cloned checkpoint index/config maps must be objects")
    weight_map: dict[str, str] = {str(key): str(value) for key, value in weight_map_obj.items()}
    tensor_storage: dict[str, Any] = dict(storage_obj)

    replaced: list[str] = []
    stripped: dict[str, list[str]] = {}
    dirty_shards: set[str] = set()
    for layer in layers:
        for name, shard_name in weight_map.items():
            if _key_belongs_to_module(name, layer.key):
                dirty_shards.add(shard_name)
    for shard_name in sorted(dirty_shards):
        shard_path = out / shard_name
        if shard_path.is_file():
            stripped[shard_name] = _drop_module_tensors_from_shard(
                shard_path,
                [layer.key for layer in layers],
            )

    for layer in layers:
        layer.validate()
        tensors = _layer_tensors_host(layer)
        shard = _layer_shard_name(layer.key)
        tmp = out / f".{shard}.tmp"
        save_file(tensors, str(tmp))
        tmp.replace(out / shard)
        for name in list(weight_map):
            if _key_belongs_to_module(name, layer.key):
                del weight_map[name]
        for name in tensors:
            weight_map[name] = shard
        tensor_storage[layer.key] = _host_storage_entry(layer, tensors)
        replaced.append(layer.key)

    if bits is not None:
        qcfg["bits"] = float(bits)
    if head_bits is not None:
        qcfg["head_bits"] = int(head_bits)
        if any(is_output_head(key) for key in replaced):
            qcfg["head_bits"] = int(head_bits)

    index["weight_map"] = weight_map
    metadata = dict(index.get("metadata") or {})
    metadata["total_size"] = int(sum((out / name).stat().st_size for name in sorted(set(weight_map.values()))))
    index["metadata"] = metadata
    qcfg["tensor_storage"] = tensor_storage
    qcfg["quant_method"] = "exl3"

    _write_json_atomic(index_path, index)
    _write_json_atomic(qcfg_path, qcfg)

    config_path = out / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config, dict):
            nested = config.get("quantization_config")
            if isinstance(nested, dict):
                if bits is not None:
                    nested["bits"] = float(bits)
                if head_bits is not None:
                    nested["head_bits"] = int(head_bits)
                config["quantization_config"] = nested
                _write_json_atomic(config_path, config)

    clear_weight_index_cache(str(out))
    loaded = [load_exl3_layer(str(out), layer.key) for layer in layers]
    for item in loaded:
        item.validate()

    return {
        "out_dir": str(out),
        "replace_into": str(src),
        "cloned_files": cloned,
        "replaced": replaced,
        "stripped_from_shards": stripped,
        "shards": sorted({weight_map[f"{key}.trellis"] for key in replaced}),
        "bits": qcfg.get("bits"),
        "head_bits": qcfg.get("head_bits"),
        "loaded_k": [int(item.k) for item in loaded],
    }
