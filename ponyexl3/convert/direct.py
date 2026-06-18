"""Direct no-LDL conversion pilots for one EXL3 Hadamard block."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from ponyexl3.convert.fixtures import (
    SearchBackend,
    build_layer_fixture,
    public_block_to_inner_block,
    read_source_public_block,
)
from ponyexl3.convert.reference_search import quantize_tile_reference
from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.hadamard import HAD_DIM
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.loader import load_exl3_layer
from ponyexl3.ref.perm import kernel_order_to_row_major, tensor_core_perm
from ponyexl3.ref.reconstruct import reconstruct_public_weights
from ponyexl3.ref.signs import unpack_signs_or_pass
from ponyexl3.ref.trellis import pack_trellis, unpack_trellis


@dataclass(frozen=True)
class DirectWindowResult:
    """One directly quantized 128x128 EXL3 window."""

    module_key: str
    search_backend: SearchBackend
    in_start: int
    out_start: int
    layer: EXL3Layer
    source_public: np.ndarray
    target_inner: np.ndarray
    reconstructed_inner: np.ndarray
    reconstructed_public: np.ndarray
    activations: np.ndarray
    stats: dict[str, float | bool]


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(d * d))


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    denom = float(np.sqrt(np.mean(b.astype(np.float32) ** 2))) + 1e-20
    return float(np.sqrt(np.mean(d * d)) / denom)


def _scale_window(scale: np.ndarray | None, start: int, size: int) -> np.ndarray | None:
    unpacked = unpack_signs_or_pass(scale)
    if unpacked is None:
        return None
    return unpacked[start : start + size].astype(np.float16, copy=True)


def _inner_to_kernel_tiles(inner: np.ndarray) -> np.ndarray:
    if inner.ndim != 2 or inner.shape[0] % 16 != 0 or inner.shape[1] % 16 != 0:
        raise ValueError(f"expected 2D matrix with 16-multiple dims, got {inner.shape}")
    in_tiles = inner.shape[0] // 16
    out_tiles = inner.shape[1] // 16
    perm = tensor_core_perm()
    tiles = np.empty((in_tiles * out_tiles, 256), dtype=np.float32)
    i = 0
    for tk in range(in_tiles):
        r0 = tk * 16
        for tn in range(out_tiles):
            c0 = tn * 16
            tiles[i] = inner[r0 : r0 + 16, c0 : c0 + 16].reshape(256)[perm]
            i += 1
    return tiles


def _kernel_tiles_to_inner(tiles: np.ndarray, rows: int, cols: int) -> np.ndarray:
    in_tiles = rows // 16
    out_tiles = cols // 16
    if tiles.shape != (in_tiles * out_tiles, 256):
        raise ValueError(f"decoded tile shape {tiles.shape} does not match {(rows, cols)}")
    out = np.empty((rows, cols), dtype=np.float32)
    i = 0
    for tk in range(in_tiles):
        r0 = tk * 16
        for tn in range(out_tiles):
            c0 = tn * 16
            out[r0 : r0 + 16, c0 : c0 + 16] = kernel_order_to_row_major(tiles[i])
            i += 1
    return out


def quantize_inner_matrix_direct(
    inner: np.ndarray,
    *,
    k: int,
    cb: CodebookMode,
    search_backend: SearchBackend = "metal",
    max_pins: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize an inner-domain matrix tilewise, without LDL error feedback."""

    kernel_tiles = _inner_to_kernel_tiles(inner.astype(np.float32, copy=False))
    if search_backend == "cpu":
        state_rows: list[np.ndarray] = []
        decoded_rows: list[np.ndarray] = []
        for tile in kernel_tiles:
            states, decoded = quantize_tile_reference(tile, k=k, cb=cb, max_pins=max_pins)
            state_rows.append(states)
            decoded_rows.append(decoded)
        states_flat = np.stack(state_rows, axis=0).astype(np.uint16, copy=False)
        decoded_tiles = np.stack(decoded_rows, axis=0).astype(np.float32, copy=False)
    elif search_backend == "metal":
        from ponyexl3.convert.metal_search import quantize_tiles_mlx_np

        decoded_tiles, states_flat = quantize_tiles_mlx_np(kernel_tiles, k=k, cb=cb)
    else:
        raise ValueError(f"unknown search backend: {search_backend}")

    in_tiles = inner.shape[0] // 16
    out_tiles = inner.shape[1] // 16
    states = states_flat.reshape(in_tiles, out_tiles, 256)
    packed = pack_trellis((states & ((1 << k) - 1)).astype(np.uint16), k)
    roundtrip = unpack_trellis(packed, k)
    if not np.array_equal(roundtrip.astype(np.uint16), states):
        raise AssertionError("direct quantization produced non-round-trippable trellis")
    reconstructed_inner = _kernel_tiles_to_inner(decoded_tiles, inner.shape[0], inner.shape[1])
    return packed, states, reconstructed_inner


def direct_quantize_window(
    source_dir: str | Path,
    oracle_dir: str | Path,
    module_key: str,
    *,
    in_start: int = 0,
    out_start: int = 0,
    search_backend: SearchBackend = "metal",
    max_pins: int = 4,
) -> DirectWindowResult:
    """Directly quantize one 128x128 oracle-comparable window."""

    if in_start % HAD_DIM != 0 or out_start % HAD_DIM != 0:
        raise ValueError("direct window starts must be aligned to 128-channel Hadamard blocks")
    fixture = build_layer_fixture(source_dir, oracle_dir, module_key)
    layer = fixture.oracle.layer
    source_public = read_source_public_block(
        source_dir,
        fixture.source,
        in_start=in_start,
        out_start=out_start,
        rows=HAD_DIM,
        cols=HAD_DIM,
    )
    target_inner = public_block_to_inner_block(
        source_public,
        suh=layer.suh,
        svh=layer.svh,
        in_start=in_start,
        out_start=out_start,
    )
    packed, _states, reconstructed_inner = quantize_inner_matrix_direct(
        target_inner,
        k=layer.k,
        cb=fixture.oracle.cb,
        search_backend=search_backend,
        max_pins=max_pins,
    )
    suh = _scale_window(layer.suh, in_start, HAD_DIM)
    svh = _scale_window(layer.svh, out_start, HAD_DIM)
    out_layer = EXL3Layer(
        key=module_key,
        in_features=HAD_DIM,
        out_features=HAD_DIM,
        k=layer.k,
        trellis=packed,
        suh=suh,
        svh=svh,
        mcg=layer.mcg,
        mul1=layer.mul1,
    )
    out_layer.validate()
    reconstructed_public = reconstruct_public_weights(
        out_layer.trellis,
        out_layer.suh,
        out_layer.svh,
        out_layer.k,
        mcg=out_layer.mcg,
        mul1=out_layer.mul1,
    ).astype(np.float32)

    x = fixture.activations[:, in_start : in_start + HAD_DIM].astype(np.float32)
    source_y = x @ source_public.astype(np.float32)
    converted_y = x @ reconstructed_public.astype(np.float32)
    stats: dict[str, float | bool] = {
        "inner_mse": _mse(reconstructed_inner, target_inner),
        "inner_rel_rms": _rel_rms(reconstructed_inner, target_inner),
        "public_mse": _mse(reconstructed_public, source_public),
        "public_rel_rms": _rel_rms(reconstructed_public, source_public),
        "output_mse": _mse(converted_y, source_y),
        "output_rel_rms": _rel_rms(converted_y, source_y),
        "pack_roundtrip": bool(np.array_equal(unpack_trellis(packed, out_layer.k), _states)),
    }
    return DirectWindowResult(
        module_key=module_key,
        search_backend=search_backend,
        in_start=in_start,
        out_start=out_start,
        layer=out_layer,
        source_public=source_public.astype(np.float32),
        target_inner=target_inner.astype(np.float32),
        reconstructed_inner=reconstructed_inner.astype(np.float32),
        reconstructed_public=reconstructed_public.astype(np.float32),
        activations=x.astype(np.float32),
        stats=stats,
    )


def _stored_tensor_meta(arr: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(arr.dtype),
        "shape": [int(x) for x in arr.shape],
        "n_bytes": int(arr.nbytes),
    }


def write_direct_window_bundle(result: DirectWindowResult, out_dir: str | Path) -> EXL3Layer:
    """Write a minimal safetensors bundle and load it back through `load_exl3_layer`."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = result.module_key
    tensors: dict[str, np.ndarray] = {
        f"{key}.trellis": result.layer.trellis.astype(np.uint16, copy=False),
    }
    if result.layer.suh is not None:
        tensors[f"{key}.suh"] = result.layer.suh.astype(np.float16, copy=False)
    if result.layer.svh is not None:
        tensors[f"{key}.svh"] = result.layer.svh.astype(np.float16, copy=False)
    if result.layer.bias is not None:
        tensors[f"{key}.bias"] = result.layer.bias

    shard = "model.safetensors"
    save_file(tensors, str(out / shard))
    weight_map = {name: shard for name in tensors}
    total_size = int((out / shard).stat().st_size)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )

    qcfg = {
        "quant_method": "exl3",
        "tensor_storage": {
            key: {
                "quant_format": "exl3",
                "bits_per_weight": float(result.layer.k),
                "mcg_multiplier": bool(result.layer.mcg),
                "mul1_multiplier": bool(result.layer.mul1),
                "stored_tensors": {name: _stored_tensor_meta(arr) for name, arr in tensors.items()},
            }
        },
    }
    (out / "quantization_config.json").write_text(json.dumps(qcfg, indent=2), encoding="utf-8")
    loaded = load_exl3_layer(str(out), key)
    loaded.validate()
    return loaded


def direct_window_summary(result: DirectWindowResult) -> dict[str, Any]:
    """JSON-friendly summary for direct-window pilots."""

    return {
        "module": result.module_key,
        "search_backend": result.search_backend,
        "in_start": result.in_start,
        "out_start": result.out_start,
        "k": result.layer.k,
        "codebook": CodebookMode(result.layer.codebook_mode).name.lower(),
        "shape": [result.layer.in_features, result.layer.out_features],
        "stats": result.stats,
    }
