"""Hessian capture and NumPy LDLQ primitives for EXL3 conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ponyexl3.convert.direct import (
    DirectLayerResult,
    ScaleMode,
    direct_layer_summary,
    inner_matrix_to_public,
    mse_from_sse,
    public_block_to_inner_with_scale_slices,
    prepare_layer_quantization_basis,
    quantize_inner_matrix_direct,
    rel_rms_from_sse,
)
from ponyexl3.convert.fixtures import SearchBackend, build_layer_fixture
from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.hadamard import HAD_DIM, had_r_128, preapply_had_left, preapply_had_right
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.reconstruct import reconstruct_public_weights
from ponyexl3.ref.signs import unpack_signs_or_pass


@dataclass(frozen=True)
class PreparedHessian:
    """Regularized input Hessian ready for block LDL factorization."""

    hessian: np.ndarray
    diag: np.ndarray
    dead: np.ndarray
    diag_mean: float
    sigma_reg: float


@dataclass(frozen=True)
class BlockLDLResult:
    """Block-LDL lower factor with identity diagonal blocks."""

    l: np.ndarray
    hessian: np.ndarray
    block_size: int
    retries: int


@dataclass(frozen=True)
class HessianProxyStats:
    """Activation-weighted error proxy for an inner-domain weight matrix."""

    proxy_sse: float
    proxy_mse: float
    proxy_rel_rms: float


@dataclass(frozen=True)
class LDLQResult:
    """One LDLQ-quantized inner-domain matrix."""

    packed: np.ndarray
    states: np.ndarray | None
    reconstructed: np.ndarray
    stats: dict[str, float | bool]


def capture_hessian(activations: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Accumulate ``X.T @ X`` from calibration activations."""

    if activations.ndim != 2:
        raise ValueError(f"expected 2D activations, got {activations.shape}")
    x = activations.astype(np.float32, copy=False)
    h = x.T @ x
    if normalize:
        h /= max(1, x.shape[0])
    h = (h + h.T) * np.float32(0.5)
    return h.astype(np.float32, copy=False)


def prepare_hessian_for_ldl(
    hessian: np.ndarray,
    *,
    sigma_reg: float = 0.025,
) -> PreparedHessian:
    """Apply upstream-style dead-channel handling and diagonal damping."""

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"expected square Hessian, got {hessian.shape}")
    h = hessian.astype(np.float32, copy=True)
    diag = np.diag(h).astype(np.float32, copy=True)
    dead = diag == 0
    if np.any(dead):
        idx = np.flatnonzero(dead)
        h[idx, idx] = np.float32(1.0)
        diag = np.diag(h).astype(np.float32, copy=True)
    diag_mean = float(np.mean(diag)) if diag.size else 0.0
    if diag_mean > 0.0:
        h[np.diag_indices_from(h)] += np.float32(sigma_reg * diag_mean)
    return PreparedHessian(
        hessian=h,
        diag=diag,
        dead=dead,
        diag_mean=diag_mean,
        sigma_reg=float(sigma_reg),
    )


def block_ldl(
    hessian: np.ndarray,
    *,
    block_size: int = 16,
    sigma_reg: float = 0.025,
    max_retries: int = 10,
) -> BlockLDLResult:
    """Port of upstream EXL3 ``block_ldl`` using NumPy/Accelerate Cholesky."""

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"expected square Hessian, got {hessian.shape}")
    n = hessian.shape[0]
    if block_size <= 0 or n % block_size != 0:
        raise ValueError(f"Hessian size {n} must be divisible by block size {block_size}")

    h = hessian.astype(np.float32, copy=True)
    diag_mean = float(np.mean(np.diag(h))) if n else 0.0
    retries = 0
    while True:
        try:
            chol = np.linalg.cholesky(h.astype(np.float64)).astype(np.float32)
            break
        except np.linalg.LinAlgError:
            retries += 1
            if retries > max_retries:
                raise
            if diag_mean <= 0.0:
                raise
            h[np.diag_indices_from(h)] += np.float32(2.0 * sigma_reg * diag_mean)

    m = n // block_size
    l_factor = chol.copy()
    for i in range(m):
        c0 = i * block_size
        c1 = c0 + block_size
        diag_block = chol[c0:c1, c0:c1]
        inv_diag_block = np.linalg.inv(diag_block.astype(np.float64)).astype(np.float32)
        l_factor[:, c0:c1] = l_factor[:, c0:c1] @ inv_diag_block

    for i in range(m):
        c0 = i * block_size
        c1 = c0 + block_size
        l_factor[c0:c1, c0:c1] = np.eye(block_size, dtype=np.float32)

    return BlockLDLResult(
        l=l_factor.astype(np.float32, copy=False),
        hessian=h.astype(np.float32, copy=False),
        block_size=block_size,
        retries=retries,
    )


def hessian_proxy_stats(
    error: np.ndarray,
    hessian: np.ndarray,
    *,
    reference: np.ndarray | None = None,
) -> HessianProxyStats:
    """Compute ``tr(E.T @ H @ E)`` and optional relative RMS versus reference."""

    if error.ndim != 2:
        raise ValueError(f"expected 2D error matrix, got {error.shape}")
    if hessian.shape != (error.shape[0], error.shape[0]):
        raise ValueError(f"Hessian shape {hessian.shape} does not match error rows {error.shape[0]}")
    e = error.astype(np.float32, copy=False)
    h = hessian.astype(np.float32, copy=False)
    proxy_sse = max(0.0, float(np.sum(e * (h @ e), dtype=np.float64)))
    proxy_mse = proxy_sse / max(1, e.size)
    if reference is None:
        proxy_rel_rms = float("nan")
    else:
        r = reference.astype(np.float32, copy=False)
        ref_sse = max(0.0, float(np.sum(r * (h @ r), dtype=np.float64)))
        proxy_rel_rms = float(np.sqrt(proxy_sse / (ref_sse + 1e-20)))
    return HessianProxyStats(
        proxy_sse=proxy_sse,
        proxy_mse=float(proxy_mse),
        proxy_rel_rms=proxy_rel_rms,
    )


def public_activations_to_inner(activations: np.ndarray, suh: np.ndarray | None) -> np.ndarray:
    """Transform public activations into the EXL3 inner row domain."""

    x = activations.astype(np.float32, copy=False)
    if suh is None:
        return x.copy()
    return had_r_128(x, pre_scale=suh).astype(np.float32, copy=False)


def public_matrix_to_inner(
    public_weight: np.ndarray,
    *,
    suh: np.ndarray | None,
    svh: np.ndarray | None,
) -> np.ndarray:
    """Transform a public weight matrix into the comparable EXL3 inner domain."""

    if public_weight.ndim != 2:
        raise ValueError(f"expected 2D public weight matrix, got {public_weight.shape}")
    rows, cols = public_weight.shape
    if rows % HAD_DIM != 0 or cols % HAD_DIM != 0:
        raise ValueError(f"public weight shape must be 128-multiple, got {public_weight.shape}")
    out = np.empty((rows, cols), dtype=np.float32)
    for in_start in range(0, rows, HAD_DIM):
        su_slice = None if suh is None else suh[in_start : in_start + HAD_DIM]
        for out_start in range(0, cols, HAD_DIM):
            sv_slice = None if svh is None else svh[out_start : out_start + HAD_DIM]
            out[in_start : in_start + HAD_DIM, out_start : out_start + HAD_DIM] = (
                public_block_to_inner_with_scale_slices(
                    public_weight[
                        in_start : in_start + HAD_DIM,
                        out_start : out_start + HAD_DIM,
                    ],
                    su=su_slice,
                    sv=sv_slice,
                )
            )
    return out


def reconstruct_oracle_public_fast(layer: EXL3Layer) -> np.ndarray:
    """Metal-decoded twin of ``ref.reconstruct.reconstruct_public_weights``.

    The reference reconstruct decodes the trellis one codeword at a time in
    Python (``lop3_b32`` runs once per weight element: ~54s for a 7M-param
    matrix, and it dominates ``--oracle-metrics``). Only the inner trellis decode
    is hot; the outer Hadamard + per-channel scales are cheap NumPy. This swaps
    the inner decode for the Metal trellis kernel — validated against the Python
    reference in ``tests/test_mlx_decode.py`` and bit-identical in practice — and
    reuses the reference outer steps verbatim, so the public weights are identical
    at ~4000x the speed. Falls back to the pure-Python reference if Metal is
    unavailable.
    """

    try:
        from ponyexl3.mlx.decode import decode_packed_trellis_mlx_layer

        inner = np.array(decode_packed_trellis_mlx_layer(layer)).astype(np.float16)
    except Exception:
        return reconstruct_public_weights(
            layer.trellis,
            layer.suh,
            layer.svh,
            layer.k,
            mcg=layer.mcg,
            mul1=layer.mul1,
        )
    w = inner.astype(np.float32)
    suh_u = unpack_signs_or_pass(layer.suh)
    svh_u = unpack_signs_or_pass(layer.svh)
    if suh_u is not None:
        w = preapply_had_left(w)
        w *= suh_u.reshape(-1, 1).astype(np.float32)
    if svh_u is not None:
        w = preapply_had_right(w)
        w *= svh_u.reshape(1, -1).astype(np.float32)
    return w.astype(np.float16)


def oracle_comparison_weights(
    layer: EXL3Layer,
    *,
    suh: np.ndarray | None,
    svh: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return oracle weights in both public and comparable inner domains."""

    oracle_public = reconstruct_oracle_public_fast(layer).astype(np.float32)
    oracle_inner = public_matrix_to_inner(oracle_public, suh=suh, svh=svh)
    return oracle_inner, oracle_public


def ldlq_inner_matrix(
    inner: np.ndarray,
    l_factor: np.ndarray,
    *,
    k: int,
    cb: CodebookMode,
    hessian: np.ndarray | None = None,
    search_backend: SearchBackend = "metal",
    buf_size_rows: int = 128,
    feedback_rows: int = 16,
    max_pins: int = 4,
    collect_states: bool = True,
    compute_proxy: bool = True,
) -> LDLQResult:
    """Reverse 16-row LDLQ over an EXL3 inner-domain matrix."""

    if inner.ndim != 2 or inner.shape[0] % 16 != 0 or inner.shape[1] % 16 != 0:
        raise ValueError(f"expected inner matrix with 16-multiple dims, got {inner.shape}")
    rows, cols = inner.shape
    if l_factor.shape != (rows, rows):
        raise ValueError(f"L factor shape {l_factor.shape} does not match rows {rows}")
    if buf_size_rows < 16 or buf_size_rows % 16 != 0:
        raise ValueError(f"buf_size_rows must be a positive 16-multiple, got {buf_size_rows}")
    if feedback_rows < 16 or feedback_rows % 16 != 0:
        raise ValueError(f"feedback_rows must be a positive 16-multiple, got {feedback_rows}")
    if feedback_rows > buf_size_rows:
        raise ValueError("feedback_rows must be <= buf_size_rows")
    if hessian is not None and hessian.shape != (rows, rows):
        raise ValueError(f"Hessian shape {hessian.shape} does not match rows {rows}")

    weight = inner.astype(np.float32, copy=True)
    l = l_factor.astype(np.float32, copy=True)
    l[np.diag_indices_from(l)] = np.float32(0.0)

    tiles_k = rows // 16
    tiles_n = cols // 16
    packed_size = 256 * k // 16
    packed = np.empty((tiles_k, tiles_n, packed_size), dtype=np.uint16)
    states = np.empty((tiles_k, tiles_n, 256), dtype=np.uint16) if collect_states else None
    reconstructed = np.zeros_like(weight, dtype=np.float32)
    prod_cache = np.zeros_like(weight, dtype=np.float32)

    j = rows
    while j > 0:
        i = max(0, j - buf_size_rows)
        if (j - i) % 16 != 0:
            raise ValueError("LDLQ row chunks must stay 16-aligned")
        b_weight = weight[i:j]
        b_reconstructed = reconstructed[i:j]
        b_prod_cache = prod_cache[i:j]
        b_l = l[i:j]
        chunk_rows = j - i

        for bj in range(chunk_rows, 0, -feedback_rows):
            bi = max(0, bj - feedback_rows)
            if (bj - bi) % 16 != 0:
                raise ValueError("LDLQ feedback groups must stay 16-aligned")
            bb_err = b_weight[bj:] - b_reconstructed[bj:]
            bb_l = b_l[bj:, i + bi : i + bj]
            compensation = b_prod_cache[bi:bj].copy()
            if bb_err.size:
                compensation += bb_l.T @ bb_err
            rows_to_quantize = b_weight[bi:bj] + compensation
            block_packed, block_states, block_reconstructed = quantize_inner_matrix_direct(
                rows_to_quantize,
                k=k,
                cb=cb,
                search_backend=search_backend,
                max_pins=max_pins,
                return_states=collect_states,
            )
            tk = (i + bi) // 16
            tk1 = tk + (bj - bi) // 16
            packed[tk:tk1] = block_packed
            if states is not None and block_states is not None:
                states[tk:tk1] = block_states
            b_reconstructed[bi:bj] = block_reconstructed

        b_err = b_weight - b_reconstructed
        if i > 0:
            prod_cache[:i] += b_l[:, :i].T @ b_err
        j = i

    delta = reconstructed - weight
    mse = float(np.mean(delta * delta))
    ref_rms = float(np.sqrt(np.mean(weight * weight))) + 1e-20
    stats: dict[str, float | bool] = {
        "inner_mse": mse,
        "inner_rel_rms": float(np.sqrt(mse) / ref_rms),
        "pack_roundtrip": True,
        "ldlq_feedback_rows": float(feedback_rows),
    }
    if hessian is not None and compute_proxy:
        proxy = hessian_proxy_stats(delta, hessian, reference=weight)
        stats.update(
            {
                "hessian_proxy_sse": proxy.proxy_sse,
                "hessian_proxy_mse": proxy.proxy_mse,
                "hessian_proxy_rel_rms": proxy.proxy_rel_rms,
            }
        )

    return LDLQResult(
        packed=packed,
        states=states,
        reconstructed=reconstructed,
        stats=stats,
    )


def ldlq_quantize_layer(
    source_dir: str | Path,
    oracle_dir: str | Path,
    module_key: str,
    *,
    search_backend: SearchBackend = "metal",
    scale_mode: ScaleMode = "oracle_safe",
    sigma_reg: float = 0.025,
    buf_size_rows: int = 128,
    feedback_rows: int = 16,
    max_pins: int = 4,
    calibration_activations: np.ndarray | None = None,
    skip_g_scale: bool = False,
    regularization_seed: int = 0,
    quant_bits: int | None = None,
    compare_oracle: bool = True,
    fast_metrics: bool = False,
) -> DirectLayerResult:
    """Quantize one full linear module with activation-aware LDLQ."""

    if fast_metrics and compare_oracle:
        raise ValueError("fast_metrics requires compare_oracle=False")

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
        raise ValueError("LDLQ layer conversion requires 128-multiple dimensions")
    k = ref_layer.k if quant_bits is None else int(quant_bits)
    if k < 1 or k > 8:
        raise ValueError(f"EXL3 trellis K must be in [1, 8], got {k}")

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
    target_inner = basis.target_inner
    source_public = basis.source_public
    suh = basis.suh
    svh = basis.svh

    inner_acts = public_activations_to_inner(fixture.activations, suh)
    prepared = prepare_hessian_for_ldl(capture_hessian(inner_acts), sigma_reg=sigma_reg)
    ldl = block_ldl(prepared.hessian, block_size=16, sigma_reg=sigma_reg)
    quantized = ldlq_inner_matrix(
        target_inner,
        ldl.l,
        k=k,
        cb=oracle.cb,
        hessian=prepared.hessian,
        search_backend=search_backend,
        buf_size_rows=buf_size_rows,
        feedback_rows=feedback_rows,
        max_pins=max_pins,
        collect_states=False,
        compute_proxy=not fast_metrics,
    )

    out_layer = EXL3Layer(
        key=module_key,
        in_features=ref_layer.in_features,
        out_features=ref_layer.out_features,
        k=k,
        trellis=quantized.packed,
        suh=suh,
        svh=svh,
        mcg=ref_layer.mcg,
        mul1=ref_layer.mul1,
    )
    out_layer.validate()

    x = fixture.activations.astype(np.float32)
    public_ref_ss = float(np.sum(source_public * source_public))

    stats = dict(quantized.stats)
    stats.update(
        {
            "hessian_diag_mean": prepared.diag_mean,
            "ldl_retries": float(ldl.retries),
            "oracle_metrics": bool(compare_oracle),
            "fast_metrics": bool(fast_metrics),
        }
    )
    if fast_metrics:
        source_y = np.empty((0, 0), dtype=np.float32)
        converted_y = np.empty((0, 0), dtype=np.float32)
        stats.update(
            {
                "public_mse": float("nan"),
                "public_rel_rms": float("nan"),
                "output_mse": float("nan"),
                "output_rel_rms": float("nan"),
            }
        )
    else:
        reconstructed_public = inner_matrix_to_public(
            quantized.reconstructed,
            suh=suh,
            svh=svh,
        )
        source_y = x @ source_public
        converted_y = x @ reconstructed_public
        public_delta = reconstructed_public - source_public
        output_delta = converted_y - source_y
        public_sse = float(np.sum(public_delta * public_delta))
        output_sse = float(np.sum(output_delta * output_delta))
        output_ref_ss = float(np.sum(source_y * source_y))
        stats.update(
            {
                "public_mse": mse_from_sse(public_sse, int(source_public.size)),
                "public_rel_rms": rel_rms_from_sse(
                    public_sse,
                    public_ref_ss,
                    int(source_public.size),
                ),
                "output_mse": mse_from_sse(output_sse, int(source_y.size)),
                "output_rel_rms": rel_rms_from_sse(
                    output_sse,
                    output_ref_ss,
                    int(source_y.size),
                ),
            }
        )
        if compare_oracle:
            oracle_inner, oracle_public = oracle_comparison_weights(ref_layer, suh=suh, svh=svh)
            oracle_y = x @ oracle_public
            oracle_public_delta = oracle_public - source_public
            oracle_output_delta = oracle_y - source_y
            oracle_public_sse = float(np.sum(oracle_public_delta * oracle_public_delta))
            oracle_output_sse = float(np.sum(oracle_output_delta * oracle_output_delta))
            oracle_proxy = hessian_proxy_stats(
                oracle_inner - target_inner,
                prepared.hessian,
                reference=target_inner,
            )
            converted_proxy_rel = float(quantized.stats.get("hessian_proxy_rel_rms", float("nan")))
            oracle_proxy_rel = oracle_proxy.proxy_rel_rms
            stats.update(
                {
                    "oracle_public_mse": mse_from_sse(oracle_public_sse, int(source_public.size)),
                    "oracle_public_rel_rms": rel_rms_from_sse(
                        oracle_public_sse,
                        public_ref_ss,
                        int(source_public.size),
                    ),
                    "oracle_output_mse": mse_from_sse(oracle_output_sse, int(source_y.size)),
                    "oracle_output_rel_rms": rel_rms_from_sse(
                        oracle_output_sse,
                        output_ref_ss,
                        int(source_y.size),
                    ),
                    "oracle_hessian_proxy_sse": oracle_proxy.proxy_sse,
                    "oracle_hessian_proxy_mse": oracle_proxy.proxy_mse,
                    "oracle_hessian_proxy_rel_rms": oracle_proxy_rel,
                    "hessian_proxy_rel_rms_over_oracle": float(
                        converted_proxy_rel / (oracle_proxy_rel + 1e-20)
                    ),
                    "output_rel_rms_over_oracle": float(
                        np.sqrt(mse_from_sse(output_sse, int(source_y.size)))
                        / (np.sqrt(mse_from_sse(oracle_output_sse, int(source_y.size))) + 1e-20)
                    ),
                }
            )
    stats.update(basis.stats)
    return DirectLayerResult(
        module_key=module_key,
        search_backend=search_backend,
        scale_mode=scale_mode,
        layer=out_layer,
        activations=x,
        source_output=source_y.astype(np.float32),
        converted_output=converted_y.astype(np.float32),
        stats=stats,
    )


def ldlq_layer_summary(result: DirectLayerResult) -> dict[str, Any]:
    """JSON-friendly summary for LDLQ layer pilots."""

    summary = direct_layer_summary(result)
    summary["quantizer"] = "ldlq"
    return summary
