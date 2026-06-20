"""Direct no-LDL conversion pilots for one EXL3 Hadamard block."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Literal, Sequence

import numpy as np
from safetensors.numpy import save_file

from ponyexl3.convert.fixtures import (
    SafetensorIndex,
    SearchBackend,
    SourceLinear,
    build_layer_fixture,
    read_source_public_block,
)
from ponyexl3.convert.reference_search import quantize_tile_reference
from ponyexl3.convert.regularize import (
    apply_global_scale,
    g_scale_gss,
    regularize_public_weight,
    sample_tile_matrix,
)
from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.hadamard import HAD_DIM, preapply_had_left, preapply_had_right
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.loader import clear_weight_index_cache, load_exl3_layer
from ponyexl3.ref.perm import tensor_core_perm, tensor_core_perm_inverse
from ponyexl3.ref.signs import unpack_signs_or_pass
from ponyexl3.ref.trellis import pack_trellis, unpack_trellis


ScaleMode = Literal["oracle", "oracle_safe", "identity", "computed"]


@dataclass(frozen=True)
class DirectWindowResult:
    """One directly quantized 128x128 EXL3 window."""

    module_key: str
    search_backend: SearchBackend
    scale_mode: ScaleMode
    in_start: int
    out_start: int
    layer: EXL3Layer
    source_public: np.ndarray
    target_inner: np.ndarray
    reconstructed_inner: np.ndarray
    reconstructed_public: np.ndarray
    activations: np.ndarray
    stats: dict[str, float | bool]


@dataclass(frozen=True)
class DirectLayerResult:
    """One directly quantized EXL3 linear layer."""

    module_key: str
    search_backend: SearchBackend
    scale_mode: ScaleMode
    layer: EXL3Layer
    activations: np.ndarray
    source_output: np.ndarray
    converted_output: np.ndarray
    stats: dict[str, float | bool]


DirectResult = DirectWindowResult | DirectLayerResult


@dataclass(frozen=True)
class LayerQuantizationBasis:
    """Source public weights plus the inner-domain target and scales."""

    source_public: np.ndarray
    target_inner: np.ndarray
    suh: np.ndarray | None
    svh: np.ndarray | None
    stats: dict[str, float | bool]

_MODEL_ASSET_NAMES = (
    "config.json",
    "configuration.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
_TENSOR_CORE_PERM = tensor_core_perm()
_TENSOR_CORE_PERM_INV = tensor_core_perm_inverse()


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(d * d))


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    denom = float(np.sqrt(np.mean(b.astype(np.float32) ** 2))) + 1e-20
    return float(np.sqrt(np.mean(d * d)) / denom)


def mse_from_sse(sse: float, count: int) -> float:
    return float(sse / max(1, count))


def rel_rms_from_sse(sse: float, ref_ss: float, count: int) -> float:
    return float(np.sqrt(mse_from_sse(sse, count)) / (np.sqrt(ref_ss / max(1, count)) + 1e-20))


def _scale_window(scale: np.ndarray | None, start: int, size: int) -> np.ndarray | None:
    unpacked = unpack_signs_or_pass(scale)
    if unpacked is None:
        return None
    return unpacked[start : start + size].astype(np.float16, copy=True)


def scale_full_for_mode(
    scale: np.ndarray | None,
    size: int,
    mode: ScaleMode,
) -> tuple[np.ndarray | None, int]:
    if mode == "computed":
        raise ValueError("computed scale mode requires source-weight regularization")
    if mode == "identity":
        return None, 0
    out = _scale_window(scale, 0, size)
    if out is None:
        return None, 0
    zero_mask = np.abs(out.astype(np.float32)) < 1e-30
    zero_count = int(np.sum(zero_mask))
    if zero_count:
        if mode == "oracle":
            raise ValueError(f"oracle scale contains {zero_count} zero entries")
        out = out.copy()
        out[zero_mask] = np.float16(1.0)
    return out, zero_count


def public_block_to_inner_with_scale_slices(
    public_block: np.ndarray,
    *,
    su: np.ndarray | None,
    sv: np.ndarray | None,
) -> np.ndarray:
    if public_block.shape != (HAD_DIM, HAD_DIM):
        raise ValueError(f"expected {(HAD_DIM, HAD_DIM)} block, got {public_block.shape}")
    block = public_block.astype(np.float32)
    if sv is not None:
        block = block / sv.reshape(1, HAD_DIM).astype(np.float32)
        block = preapply_had_right(block.astype(np.float32)).astype(np.float32)
    if su is not None:
        block = block / su.reshape(HAD_DIM, 1).astype(np.float32)
        block = preapply_had_left(block.astype(np.float32)).astype(np.float32)
    return block


def inner_matrix_to_public(
    inner: np.ndarray,
    *,
    suh: np.ndarray | None,
    svh: np.ndarray | None,
) -> np.ndarray:
    """Reconstruct public weights from an already-decoded inner matrix."""

    if inner.ndim != 2:
        raise ValueError(f"expected 2D inner matrix, got {inner.shape}")
    rows, cols = inner.shape
    if rows % HAD_DIM != 0 or cols % HAD_DIM != 0:
        raise ValueError(f"inner weight shape must be 128-multiple, got {inner.shape}")
    out = inner.astype(np.float32, copy=True)
    if suh is not None:
        if suh.shape[0] != rows:
            raise ValueError(f"suh shape {suh.shape} does not match rows {rows}")
        out = preapply_had_left(out).astype(np.float32, copy=False)
        out *= suh.reshape(rows, 1).astype(np.float32)
    if svh is not None:
        if svh.shape[0] != cols:
            raise ValueError(f"svh shape {svh.shape} does not match cols {cols}")
        out = preapply_had_right(out).astype(np.float32, copy=False)
        out *= svh.reshape(1, cols).astype(np.float32)
    return out.astype(np.float16).astype(np.float32)


def read_source_public_matrix(
    source_dir: str | Path,
    source: SourceLinear,
) -> np.ndarray:
    """Read a full source linear in public EXL3 layout ``(in, out)``."""

    if source.in_features % HAD_DIM != 0 or source.out_features % HAD_DIM != 0:
        raise ValueError("full source matrix read requires 128-multiple dimensions")
    index = SafetensorIndex(source_dir)
    if source.layout == "linear_t":
        weight = index.read_tensor(source.source_tensor_key)
        return weight.T.astype(np.float32, copy=False)

    if source.expert is None:
        raise ValueError(f"{source.key}: expert index missing")
    if source.layout in ("qwen_gate", "qwen_up"):
        if source.gate_up_mid is None:
            raise ValueError(f"{source.key}: gate/up midpoint missing")
        source_out_start = 0 if source.layout == "qwen_gate" else source.gate_up_mid
        weight = index.read_slice(
            source.source_tensor_key,
            (source.expert, source_out_start, 0),
            (1, source.out_features, source.in_features),
        )[0]
        return weight.T.astype(np.float32, copy=False)

    weight = index.read_slice(
        source.source_tensor_key,
        (source.expert, 0, 0),
        (1, source.out_features, source.in_features),
    )[0]
    return weight.T.astype(np.float32, copy=False)


def _oracle_or_identity_basis(
    source_public: np.ndarray,
    ref_layer: EXL3Layer,
    *,
    scale_mode: ScaleMode,
    search_backend: SearchBackend,
) -> LayerQuantizationBasis:
    suh, suh_zero_count = scale_full_for_mode(ref_layer.suh, ref_layer.in_features, scale_mode)
    svh, svh_zero_count = scale_full_for_mode(ref_layer.svh, ref_layer.out_features, scale_mode)
    if scale_mode != "identity" and search_backend == "metal":
        try:
            target_inner = _oracle_basis_mlx_hadamard(source_public, suh=suh, svh=svh)
        except Exception:
            target_inner = None
        if target_inner is not None:
            return LayerQuantizationBasis(
                source_public=source_public,
                target_inner=target_inner,
                suh=suh,
                svh=svh,
                stats={
                    "suh_zero_replacements": float(suh_zero_count),
                    "svh_zero_replacements": float(svh_zero_count),
                    "regularize_computed_scales": False,
                    "basis_mlx_hadamard": True,
                },
            )
    target_inner = np.empty_like(source_public, dtype=np.float32)
    for in_start in range(0, ref_layer.in_features, HAD_DIM):
        su_slice = None if suh is None else suh[in_start : in_start + HAD_DIM]
        for out_start in range(0, ref_layer.out_features, HAD_DIM):
            sv_slice = None if svh is None else svh[out_start : out_start + HAD_DIM]
            target_inner[
                in_start : in_start + HAD_DIM,
                out_start : out_start + HAD_DIM,
            ] = public_block_to_inner_with_scale_slices(
                source_public[
                    in_start : in_start + HAD_DIM,
                    out_start : out_start + HAD_DIM,
                ],
                su=su_slice,
                sv=sv_slice,
            )
    return LayerQuantizationBasis(
        source_public=source_public,
        target_inner=target_inner,
        suh=suh,
        svh=svh,
        stats={
            "suh_zero_replacements": float(suh_zero_count),
            "svh_zero_replacements": float(svh_zero_count),
            "regularize_computed_scales": False,
            "basis_mlx_hadamard": False,
        },
    )


def _oracle_basis_mlx_hadamard(
    source_public: np.ndarray,
    *,
    suh: np.ndarray | None,
    svh: np.ndarray | None,
) -> np.ndarray | None:
    if suh is None and svh is None:
        return source_public.astype(np.float32, copy=False)
    try:
        import mlx.core as mx

        from ponyexl3.mlx.hadamard import preapply_had_left_mlx, preapply_had_right_mlx
    except Exception:
        return None

    out = mx.array(source_public.astype(np.float32, copy=False), dtype=mx.float32)
    rows, cols = source_public.shape
    if svh is not None:
        out = out / mx.array(svh.astype(np.float32, copy=False), dtype=mx.float32).reshape(
            1,
            cols,
        )
        out = preapply_had_right_mlx(out).astype(mx.float32)
    if suh is not None:
        out = out / mx.array(suh.astype(np.float32, copy=False), dtype=mx.float32).reshape(
            rows,
            1,
        )
        out = preapply_had_left_mlx(out).astype(mx.float32)
    mx.eval(out)
    return np.array(out).astype(np.float32, copy=False)


def prepare_layer_quantization_basis(
    source_dir: str | Path,
    source: SourceLinear,
    ref_layer: EXL3Layer,
    cb: CodebookMode,
    *,
    scale_mode: ScaleMode,
    search_backend: SearchBackend,
    max_pins: int = 4,
    skip_g_scale: bool = False,
    regularization_seed: int = 0,
    g_scale_width: int = 3,
    quant_bits: int | None = None,
) -> LayerQuantizationBasis:
    """Build the source, inner target, and scales shared by direct and LDLQ."""

    source_public = read_source_public_matrix(source_dir, source)
    if scale_mode != "computed":
        return _oracle_or_identity_basis(
            source_public,
            ref_layer,
            scale_mode=scale_mode,
            search_backend=search_backend,
        )

    k = ref_layer.k if quant_bits is None else int(quant_bits)
    regularized = regularize_public_weight(
        source_public,
        seed=regularization_seed,
    )
    stats = dict(regularized.stats)
    stats.update(
        {
            "regularize_computed_scales": True,
            "suh_zero_replacements": 0.0,
            "svh_zero_replacements": float(stats["regularize_zero_out_scales"]),
        }
    )
    if skip_g_scale:
        stats["regularize_g_scale_skipped"] = True
    else:
        sample = sample_tile_matrix(regularized.inner, width=g_scale_width)

        def score(scale: float) -> float:
            scaled = (sample * np.float32(scale)).astype(np.float32, copy=False)
            _packed, _states, reconstructed = quantize_inner_matrix_direct(
                scaled,
                k=k,
                cb=cb,
                search_backend=search_backend,
                max_pins=max_pins,
                return_states=False,
            )
            delta = reconstructed / np.float32(scale) - sample
            return float(np.mean(delta * delta, dtype=np.float64))

        gss = g_scale_gss(score)
        regularized = apply_global_scale(regularized, gss)
        stats = dict(regularized.stats)
        stats.update(
            {
                "regularize_computed_scales": True,
                "regularize_g_scale_skipped": False,
                "suh_zero_replacements": 0.0,
                "svh_zero_replacements": float(stats["regularize_zero_out_scales"]),
            }
        )

    return LayerQuantizationBasis(
        source_public=source_public,
        target_inner=regularized.inner,
        suh=regularized.suh,
        svh=regularized.svh,
        stats=stats,
    )


def read_source_plain_tensors(
    source_dir: str | Path,
    tensor_keys: Sequence[str],
) -> dict[str, np.ndarray]:
    """Read non-EXL3 source tensors for a strict-loadable emitted model."""

    index = SafetensorIndex(source_dir)
    tensors: dict[str, np.ndarray] = {}
    for key in tensor_keys:
        tensors[key] = index.read_tensor(key).astype(np.float16, copy=False)
    return tensors


def _inner_to_kernel_tiles(inner: np.ndarray) -> np.ndarray:
    if inner.ndim != 2 or inner.shape[0] % 16 != 0 or inner.shape[1] % 16 != 0:
        raise ValueError(f"expected 2D matrix with 16-multiple dims, got {inner.shape}")
    in_tiles = inner.shape[0] // 16
    out_tiles = inner.shape[1] // 16
    row_major = (
        inner.reshape(in_tiles, 16, out_tiles, 16)
        .transpose(0, 2, 1, 3)
        .reshape(in_tiles * out_tiles, 256)
    )
    return row_major[:, _TENSOR_CORE_PERM].astype(np.float32, copy=False)


def _kernel_tiles_to_inner(tiles: np.ndarray, rows: int, cols: int) -> np.ndarray:
    in_tiles = rows // 16
    out_tiles = cols // 16
    if tiles.shape != (in_tiles * out_tiles, 256):
        raise ValueError(f"decoded tile shape {tiles.shape} does not match {(rows, cols)}")
    return (
        tiles[:, _TENSOR_CORE_PERM_INV]
        .reshape(in_tiles, out_tiles, 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(rows, cols)
        .astype(np.float32, copy=False)
    )


def _inner_to_kernel_tiles_mlx(inner: Any) -> Any:
    import mlx.core as mx

    if inner.ndim != 2 or inner.shape[0] % 16 != 0 or inner.shape[1] % 16 != 0:
        raise ValueError(f"expected 2D matrix with 16-multiple dims, got {inner.shape}")
    in_tiles = inner.shape[0] // 16
    out_tiles = inner.shape[1] // 16
    row_major = (
        inner.reshape(in_tiles, 16, out_tiles, 16)
        .transpose(0, 2, 1, 3)
        .reshape(in_tiles * out_tiles, 256)
    )
    perm = mx.array(_TENSOR_CORE_PERM.astype(np.int32))
    return mx.take(row_major, perm, axis=1).astype(mx.float32)


def _kernel_tiles_to_inner_mlx(tiles: Any, rows: int, cols: int) -> Any:
    import mlx.core as mx

    in_tiles = rows // 16
    out_tiles = cols // 16
    if tiles.shape != (in_tiles * out_tiles, 256):
        raise ValueError(f"decoded tile shape {tiles.shape} does not match {(rows, cols)}")
    perm_inv = mx.array(_TENSOR_CORE_PERM_INV.astype(np.int32))
    return (
        mx.take(tiles, perm_inv, axis=1)
        .reshape(in_tiles, out_tiles, 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(rows, cols)
        .astype(mx.float32)
    )


def quantize_inner_matrix_direct_mlx(
    inner: Any,
    *,
    k: int,
    cb: CodebookMode,
) -> tuple[Any, Any]:
    """MLX-native Metal direct quantization for production LDLQ loops."""

    import mlx.core as mx

    arr = mx.array(inner, dtype=mx.float32)
    kernel_tiles = _inner_to_kernel_tiles_mlx(arr)
    from ponyexl3.convert.metal_search import quantize_tiles_mlx
    from ponyexl3.convert.mlx_trellis import pack_trellis_mlx

    decoded_mx, states_mx = quantize_tiles_mlx(kernel_tiles, k=k, cb=cb)
    in_tiles = arr.shape[0] // 16
    out_tiles = arr.shape[1] // 16
    packed_mx = pack_trellis_mlx(
        (states_mx & ((1 << k) - 1)).reshape(in_tiles, out_tiles, 256),
        k,
    )
    reconstructed_mx = _kernel_tiles_to_inner_mlx(decoded_mx, arr.shape[0], arr.shape[1])
    return packed_mx, reconstructed_mx


def quantize_inner_matrix_direct(
    inner: np.ndarray,
    *,
    k: int,
    cb: CodebookMode,
    search_backend: SearchBackend = "metal",
    max_pins: int = 4,
    verify_roundtrip: bool = False,
    return_states: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
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
        if return_states or verify_roundtrip:
            from ponyexl3.convert.metal_search import quantize_tiles_mlx_np

            decoded_tiles, states_flat = quantize_tiles_mlx_np(kernel_tiles, k=k, cb=cb)
        else:
            import mlx.core as mx

            from ponyexl3.convert.metal_search import quantize_tiles_mlx
            from ponyexl3.convert.mlx_trellis import pack_trellis_mlx

            decoded_mx, states_mx = quantize_tiles_mlx(kernel_tiles, k=k, cb=cb)
            in_tiles = inner.shape[0] // 16
            out_tiles = inner.shape[1] // 16
            packed_mx = pack_trellis_mlx(
                (states_mx & ((1 << k) - 1)).reshape(in_tiles, out_tiles, 256),
                k,
            )
            mx.eval(decoded_mx, packed_mx)
            decoded_tiles = np.array(decoded_mx).astype(np.float32, copy=False)
            packed = np.array(packed_mx).astype(np.uint16, copy=False)
            reconstructed_inner = _kernel_tiles_to_inner(
                decoded_tiles,
                inner.shape[0],
                inner.shape[1],
            )
            return packed, None, reconstructed_inner
    else:
        raise ValueError(f"unknown search backend: {search_backend}")

    in_tiles = inner.shape[0] // 16
    out_tiles = inner.shape[1] // 16
    states = states_flat.reshape(in_tiles, out_tiles, 256)
    packed = pack_trellis((states & ((1 << k) - 1)).astype(np.uint16), k)
    if verify_roundtrip:
        roundtrip = unpack_trellis(packed, k)
        if not np.array_equal(roundtrip.astype(np.uint16), states):
            raise AssertionError("direct quantization produced non-round-trippable trellis")
    reconstructed_inner = _kernel_tiles_to_inner(decoded_tiles, inner.shape[0], inner.shape[1])
    return packed, states if return_states else None, reconstructed_inner


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
    su = _scale_window(layer.suh, in_start, HAD_DIM)
    sv = _scale_window(layer.svh, out_start, HAD_DIM)
    if su is not None and np.any(np.abs(su.astype(np.float32)) < 1e-30):
        raise ValueError("direct window oracle suh contains zero scales")
    if sv is not None and np.any(np.abs(sv.astype(np.float32)) < 1e-30):
        raise ValueError("direct window oracle svh contains zero scales")
    target_inner = public_block_to_inner_with_scale_slices(
        source_public,
        su=su,
        sv=sv,
    )
    packed, _states, reconstructed_inner = quantize_inner_matrix_direct(
        target_inner,
        k=layer.k,
        cb=fixture.oracle.cb,
        search_backend=search_backend,
        max_pins=max_pins,
    )
    out_layer = EXL3Layer(
        key=module_key,
        in_features=HAD_DIM,
        out_features=HAD_DIM,
        k=layer.k,
        trellis=packed,
        suh=su,
        svh=sv,
        mcg=layer.mcg,
        mul1=layer.mul1,
    )
    out_layer.validate()
    reconstructed_public = inner_matrix_to_public(
        reconstructed_inner,
        suh=out_layer.suh,
        svh=out_layer.svh,
    )

    x = fixture.activations[:, in_start : in_start + HAD_DIM].astype(np.float32)
    source_y = x @ source_public.astype(np.float32)
    converted_y = x @ reconstructed_public.astype(np.float32)
    assert _states is not None
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
        scale_mode="oracle",
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


def direct_quantize_layer(
    source_dir: str | Path,
    oracle_dir: str | Path,
    module_key: str,
    *,
    search_backend: SearchBackend = "metal",
    scale_mode: ScaleMode = "oracle_safe",
    max_pins: int = 4,
    calibration_activations: np.ndarray | None = None,
    skip_g_scale: bool = False,
    regularization_seed: int = 0,
    quant_bits: int | None = None,
) -> DirectLayerResult:
    """Directly quantize one full linear module."""

    fixture = build_layer_fixture(
        source_dir,
        oracle_dir,
        module_key,
        activations=calibration_activations,
    )
    source = fixture.source
    oracle = fixture.oracle
    ref_layer = oracle.layer
    if ref_layer.in_features % HAD_DIM != 0 or ref_layer.out_features % HAD_DIM != 0:
        raise ValueError("direct layer conversion requires 128-multiple dimensions")
    k = ref_layer.k if quant_bits is None else int(quant_bits)
    if k < 1 or k > 8:
        raise ValueError(f"EXL3 trellis K must be in [1, 8], got {k}")

    in_blocks = ref_layer.in_features // HAD_DIM
    in_tiles = ref_layer.in_features // 16
    out_tiles = ref_layer.out_features // 16
    packed_size = 256 * k // 16
    trellis = np.empty((in_tiles, out_tiles, packed_size), dtype=np.uint16)
    basis = prepare_layer_quantization_basis(
        source_dir,
        source,
        ref_layer,
        oracle.cb,
        scale_mode=scale_mode,
        search_backend=search_backend,
        max_pins=max_pins,
        skip_g_scale=skip_g_scale,
        regularization_seed=regularization_seed,
        quant_bits=k,
    )
    source_public = basis.source_public
    target_inner = basis.target_inner
    suh = basis.suh
    svh = basis.svh

    x = fixture.activations.astype(np.float32)
    source_y = np.zeros((x.shape[0], ref_layer.out_features), dtype=np.float32)
    converted_y = np.zeros_like(source_y)

    inner_sse = inner_ref_ss = 0.0
    public_sse = public_ref_ss = 0.0
    inner_count = public_count = 0

    for ib in range(in_blocks):
        in_start = ib * HAD_DIM
        source_public_row = source_public[in_start : in_start + HAD_DIM]
        target_inner_row = target_inner[in_start : in_start + HAD_DIM]

        packed_row, _states, reconstructed_inner_row = quantize_inner_matrix_direct(
            target_inner_row,
            k=k,
            cb=oracle.cb,
            search_backend=search_backend,
            max_pins=max_pins,
            return_states=False,
        )
        tk0 = in_start // 16
        trellis[tk0 : tk0 + HAD_DIM // 16, :, :] = packed_row

        reconstructed_public_row = inner_matrix_to_public(
            reconstructed_inner_row,
            suh=None if suh is None else suh[in_start : in_start + HAD_DIM],
            svh=svh,
        )

        inner_delta = reconstructed_inner_row.astype(np.float32) - target_inner_row
        public_delta = reconstructed_public_row - source_public_row
        inner_sse += float(np.sum(inner_delta * inner_delta))
        inner_ref_ss += float(np.sum(target_inner_row * target_inner_row))
        inner_count += int(target_inner_row.size)
        public_sse += float(np.sum(public_delta * public_delta))
        public_ref_ss += float(np.sum(source_public_row * source_public_row))
        public_count += int(source_public_row.size)

        x_block = x[:, in_start : in_start + HAD_DIM]
        source_y += x_block @ source_public_row
        converted_y += x_block @ reconstructed_public_row

    out_layer = EXL3Layer(
        key=module_key,
        in_features=ref_layer.in_features,
        out_features=ref_layer.out_features,
        k=k,
        trellis=trellis,
        suh=suh,
        svh=svh,
        mcg=ref_layer.mcg,
        mul1=ref_layer.mul1,
    )
    out_layer.validate()

    output_delta = converted_y - source_y
    output_sse = float(np.sum(output_delta * output_delta))
    output_ref_ss = float(np.sum(source_y * source_y))
    output_count = int(source_y.size)
    stats: dict[str, float | bool] = {
        "inner_mse": mse_from_sse(inner_sse, inner_count),
        "inner_rel_rms": rel_rms_from_sse(inner_sse, inner_ref_ss, inner_count),
        "public_mse": mse_from_sse(public_sse, public_count),
        "public_rel_rms": rel_rms_from_sse(public_sse, public_ref_ss, public_count),
        "output_mse": mse_from_sse(output_sse, output_count),
        "output_rel_rms": rel_rms_from_sse(output_sse, output_ref_ss, output_count),
        "pack_roundtrip": True,
    }
    stats.update(basis.stats)
    return DirectLayerResult(
        module_key=module_key,
        search_backend=search_backend,
        scale_mode=scale_mode,
        layer=out_layer,
        activations=x.astype(np.float32),
        source_output=source_y.astype(np.float32),
        converted_output=converted_y.astype(np.float32),
        stats=stats,
    )


def _stored_tensor_meta(arr: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(arr.dtype),
        "shape": [int(x) for x in arr.shape],
        "n_bytes": int(arr.nbytes),
    }


def _layer_tensors(layer: EXL3Layer) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        f"{layer.key}.trellis": layer.trellis.astype(np.uint16, copy=False),
    }
    if layer.suh is not None:
        tensors[f"{layer.key}.suh"] = layer.suh.astype(np.float16, copy=False)
    if layer.svh is not None:
        tensors[f"{layer.key}.svh"] = layer.svh.astype(np.float16, copy=False)
    if layer.bias is not None:
        tensors[f"{layer.key}.bias"] = layer.bias
    return tensors


def _copy_model_assets(asset_dir: str | Path | None, out: Path) -> list[str]:
    if asset_dir is None:
        return []
    src = Path(asset_dir)
    copied: list[str] = []
    for name in _MODEL_ASSET_NAMES:
        path = src / name
        if path.is_file():
            shutil.copy2(path, out / name)
            copied.append(name)
    return copied


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _save_safetensors_atomic(tensors: dict[str, np.ndarray], path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    save_file(tensors, str(tmp))
    tmp.replace(path)


def _load_json_object(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} JSON root must be an object")
    return data


def _layer_shard_name(layer_key: str) -> str:
    digest = hashlib.sha1(layer_key.encode("utf-8")).hexdigest()[:16]
    return f"ponyexl3-layer-{digest}.safetensors"


def _refresh_total_size(out: Path, weight_map: dict[str, str]) -> int:
    total = 0
    for shard in sorted(set(weight_map.values())):
        path = out / shard
        if path.is_file():
            total += path.stat().st_size
    return int(total)


def _clear_loader_index_cache(out: Path) -> None:
    clear_weight_index_cache(str(out))


def write_exl3_incremental_bundle(
    layers: Sequence[EXL3Layer],
    out_dir: str | Path,
    *,
    asset_dir: str | Path | None = None,
    manifest: dict[str, Any] | None = None,
    plain_tensors: dict[str, np.ndarray] | None = None,
) -> list[EXL3Layer]:
    """Write converted layers as stable per-layer shards and update indexes."""

    if not layers and not plain_tensors and manifest is None:
        raise ValueError("nothing to write to incremental EXL3 bundle")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "model.safetensors.index.json"
    qcfg_path = out / "quantization_config.json"
    index = _load_json_object(index_path, {"metadata": {"total_size": 0}, "weight_map": {}})
    qcfg = _load_json_object(qcfg_path, {"quant_method": "exl3", "tensor_storage": {}})
    weight_map_obj = index.setdefault("weight_map", {})
    storage_obj = qcfg.setdefault("tensor_storage", {})
    if not isinstance(weight_map_obj, dict):
        raise ValueError(f"{index_path} weight_map must be an object")
    if not isinstance(storage_obj, dict):
        raise ValueError(f"{qcfg_path} tensor_storage must be an object")
    weight_map: dict[str, str] = {str(key): str(value) for key, value in weight_map_obj.items()}
    tensor_storage: dict[str, Any] = dict(storage_obj)

    if plain_tensors:
        shard = "ponyexl3-plain.safetensors"
        _save_safetensors_atomic(plain_tensors, out / shard)
        for name, arr in plain_tensors.items():
            weight_map[name] = shard
            storage_key = name[: -len(".weight")] if name.endswith(".weight") else name
            tensor_storage[storage_key] = {
                "stored_tensors": {name: _stored_tensor_meta(arr)},
            }

    for layer in layers:
        layer.validate()
        shard = _layer_shard_name(layer.key)
        tensors = _layer_tensors(layer)
        _save_safetensors_atomic(tensors, out / shard)
        for name in tensors:
            weight_map[name] = shard
        tensor_storage[layer.key] = {
            "quant_format": "exl3",
            "bits_per_weight": float(layer.k),
            "mcg_multiplier": bool(layer.mcg),
            "mul1_multiplier": bool(layer.mul1),
            "stored_tensors": {name: _stored_tensor_meta(arr) for name, arr in tensors.items()},
        }

    index["weight_map"] = weight_map
    index["metadata"] = {"total_size": _refresh_total_size(out, weight_map)}
    qcfg["quant_method"] = "exl3"
    qcfg["tensor_storage"] = tensor_storage
    _write_json_atomic(index_path, index)
    _write_json_atomic(qcfg_path, qcfg)

    copied_assets = _copy_model_assets(asset_dir, out)
    if manifest is not None:
        manifest_out = dict(manifest)
        existing_assets = manifest_out.get("asset_files")
        if isinstance(existing_assets, list):
            copied_assets = sorted(set(str(item) for item in existing_assets) | set(copied_assets))
        manifest_out["asset_files"] = copied_assets
        manifest_out["tensor_count"] = len(weight_map)
        manifest_out["layer_count"] = sum(
            1
            for info in tensor_storage.values()
            if isinstance(info, dict) and info.get("quant_format") == "exl3"
        )
        _write_json_atomic(out / "ponyexl3_convert_manifest.json", manifest_out)

    _clear_loader_index_cache(out)
    loaded: list[EXL3Layer] = []
    for layer in layers:
        item = load_exl3_layer(str(out), layer.key)
        item.validate()
        loaded.append(item)
    return loaded


def _safetensors_tensor_sizes(path: Path) -> dict[str, int]:
    """Per-tensor byte size from a safetensors header (no data load)."""
    import struct

    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    sizes: dict[str, int] = {}
    for key, meta in header.items():
        if key == "__metadata__" or not isinstance(meta, dict):
            continue
        start, end = meta["data_offsets"]
        sizes[key] = int(end) - int(start)
    return sizes


def finalize_bundle(
    out_dir: str | Path,
    *,
    max_shard_bytes: int = 5 * 1024**3,
    shard_prefix: str = "model",
) -> list[str]:
    """Consolidate per-layer shards into HF-standard ``model-NNNNN-of-MMMMM`` files.

    The incremental writer emits one small ``ponyexl3-layer-<hash>`` shard per
    layer (stable names so resume can overwrite) — convenient mid-run, but it
    leaves hundreds of tiny, oddly-named files. This repacks them into a few
    size-bounded shards under the conventional naming + index that HF tooling and
    users expect. Only the safetensors shards + index change; the tensor-keyed
    ``quantization_config.json`` is untouched, so loading is unaffected.
    """
    import re

    from safetensors import safe_open

    out = Path(out_dir)
    index_path = out / "model.safetensors.index.json"
    index = _load_json_object(index_path, {"metadata": {"total_size": 0}, "weight_map": {}})
    weight_map = {str(k): str(v) for k, v in dict(index.get("weight_map", {})).items()}
    if not weight_map:
        return []

    old_shards = sorted(set(weight_map.values()))
    sizes: dict[str, int] = {}
    for shard in old_shards:
        sizes.update(_safetensors_tensor_sizes(out / shard))

    def _nat(key: str) -> list[Any]:
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", key)]

    keys = sorted(weight_map, key=_nat)

    # Bin-pack tensors into size-bounded shards (an oversized single tensor gets
    # its own shard, matching HF behaviour).
    plan: list[list[str]] = [[]]
    current = 0
    for key in keys:
        size = sizes.get(key, 0)
        if plan[-1] and current + size > max_shard_bytes:
            plan.append([])
            current = 0
        plan[-1].append(key)
        current += size
    total_shards = len(plan)

    new_weight_map: dict[str, str] = {}
    new_names: list[str] = []
    for shard_index, shard_keys in enumerate(plan, start=1):
        name = f"{shard_prefix}-{shard_index:05d}-of-{total_shards:05d}.safetensors"
        new_names.append(name)
        by_old: dict[str, list[str]] = {}
        for key in shard_keys:
            by_old.setdefault(weight_map[key], []).append(key)
        tensors: dict[str, np.ndarray] = {}
        for old_shard, member_keys in by_old.items():
            with safe_open(str(out / old_shard), framework="numpy") as handle:
                for key in member_keys:
                    tensors[key] = handle.get_tensor(key)
        _save_safetensors_atomic(tensors, out / name)
        for key in shard_keys:
            new_weight_map[key] = name
        del tensors

    # Swap the index to the new shards atomically *before* removing the old
    # ones, so an interrupt mid-finalize leaves either the original bundle (index
    # still points at the intact old shards) or the new one — never a dangling
    # index referencing deleted files.
    index["weight_map"] = new_weight_map
    index["metadata"] = {"total_size": _refresh_total_size(out, new_weight_map)}
    _write_json_atomic(index_path, index)
    _clear_loader_index_cache(out)

    new_set = set(new_names)
    for old_shard in old_shards:
        if old_shard not in new_set:
            (out / old_shard).unlink(missing_ok=True)
    return new_names


def write_exl3_layers_bundle(
    layers: Sequence[EXL3Layer],
    out_dir: str | Path,
    *,
    asset_dir: str | Path | None = None,
    manifest: dict[str, Any] | None = None,
    plain_tensors: dict[str, np.ndarray] | None = None,
) -> list[EXL3Layer]:
    """Write one safetensors shard containing multiple converted EXL3 layers."""

    if not layers:
        raise ValueError("cannot write an empty EXL3 layer bundle")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, np.ndarray] = {}
    for name, arr in (plain_tensors or {}).items():
        if name in tensors:
            raise ValueError(f"duplicate tensor in bundle: {name}")
        tensors[name] = arr
    for layer in layers:
        layer.validate()
        for name, arr in _layer_tensors(layer).items():
            if name in tensors:
                raise ValueError(f"duplicate tensor in bundle: {name}")
            tensors[name] = arr

    shard = "model.safetensors"
    save_file(tensors, str(out / shard))
    weight_map = {name: shard for name in tensors}
    total_size = int((out / shard).stat().st_size)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )

    tensor_storage: dict[str, Any] = {}
    for name, arr in (plain_tensors or {}).items():
        storage_key = name[: -len(".weight")] if name.endswith(".weight") else name
        tensor_storage[storage_key] = {
            "stored_tensors": {name: _stored_tensor_meta(arr)},
        }
    for layer in layers:
        layer_tensor_names = [name for name in tensors if name.startswith(f"{layer.key}.")]
        tensor_storage[layer.key] = {
            "quant_format": "exl3",
            "bits_per_weight": float(layer.k),
            "mcg_multiplier": bool(layer.mcg),
            "mul1_multiplier": bool(layer.mul1),
            "stored_tensors": {name: _stored_tensor_meta(tensors[name]) for name in layer_tensor_names},
        }
    qcfg = {
        "quant_method": "exl3",
        "tensor_storage": tensor_storage,
    }
    (out / "quantization_config.json").write_text(json.dumps(qcfg, indent=2), encoding="utf-8")

    copied_assets = _copy_model_assets(asset_dir, out)
    if manifest is not None:
        manifest_out = dict(manifest)
        manifest_out["asset_files"] = copied_assets
        manifest_out["tensor_count"] = len(tensors)
        manifest_out["layer_count"] = len(layers)
        (out / "ponyexl3_convert_manifest.json").write_text(
            json.dumps(manifest_out, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    loaded: list[EXL3Layer] = []
    for layer in layers:
        item = load_exl3_layer(str(out), layer.key)
        item.validate()
        loaded.append(item)
    return loaded


def write_direct_bundle(result: DirectResult, out_dir: str | Path) -> EXL3Layer:
    """Write a minimal safetensors bundle and load it back through `load_exl3_layer`."""

    return write_exl3_layers_bundle([result.layer], out_dir)[0]


def write_direct_window_bundle(result: DirectWindowResult, out_dir: str | Path) -> EXL3Layer:
    """Backward-compatible wrapper for direct-window tests."""

    return write_direct_bundle(result, out_dir)


def write_direct_layer_bundle(result: DirectLayerResult, out_dir: str | Path) -> EXL3Layer:
    """Write a direct full-layer result and load it back."""

    return write_direct_bundle(result, out_dir)


def direct_result_summary(result: DirectResult) -> dict[str, Any]:
    """JSON-friendly summary for direct conversion pilots."""

    return {
        "module": result.module_key,
        "search_backend": result.search_backend,
        "scale_mode": result.scale_mode,
        "k": result.layer.k,
        "codebook": CodebookMode(result.layer.codebook_mode).name.lower(),
        "shape": [result.layer.in_features, result.layer.out_features],
        "stats": result.stats,
    }


def direct_window_summary(result: DirectWindowResult) -> dict[str, Any]:
    """JSON-friendly summary for direct-window pilots."""

    summary = direct_result_summary(result)
    summary.update({"in_start": result.in_start, "out_start": result.out_start})
    return summary


def direct_layer_summary(result: DirectLayerResult) -> dict[str, Any]:
    """JSON-friendly summary for full direct-layer pilots."""

    return direct_result_summary(result)
