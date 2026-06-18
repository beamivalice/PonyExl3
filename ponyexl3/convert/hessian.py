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
    mse_from_sse,
    public_block_to_inner_with_scale_slices,
    quantize_inner_matrix_direct,
    rel_rms_from_sse,
    scale_full_for_mode,
)
from ponyexl3.convert.fixtures import SearchBackend, build_layer_fixture, read_source_public_block
from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.hadamard import HAD_DIM, had_r_128
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.reconstruct import reconstruct_public_weights


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
    states: np.ndarray
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


def ldlq_inner_matrix(
    inner: np.ndarray,
    l_factor: np.ndarray,
    *,
    k: int,
    cb: CodebookMode,
    hessian: np.ndarray | None = None,
    search_backend: SearchBackend = "metal",
    buf_size_rows: int = 128,
    max_pins: int = 4,
) -> LDLQResult:
    """Reverse 16-row LDLQ over an EXL3 inner-domain matrix."""

    if inner.ndim != 2 or inner.shape[0] % 16 != 0 or inner.shape[1] % 16 != 0:
        raise ValueError(f"expected inner matrix with 16-multiple dims, got {inner.shape}")
    rows, cols = inner.shape
    if l_factor.shape != (rows, rows):
        raise ValueError(f"L factor shape {l_factor.shape} does not match rows {rows}")
    if buf_size_rows < 16 or buf_size_rows % 16 != 0:
        raise ValueError(f"buf_size_rows must be a positive 16-multiple, got {buf_size_rows}")
    if hessian is not None and hessian.shape != (rows, rows):
        raise ValueError(f"Hessian shape {hessian.shape} does not match rows {rows}")

    weight = inner.astype(np.float32, copy=True)
    l = l_factor.astype(np.float32, copy=True)
    l[np.diag_indices_from(l)] = np.float32(0.0)

    tiles_k = rows // 16
    tiles_n = cols // 16
    packed_size = 256 * k // 16
    packed = np.empty((tiles_k, tiles_n, packed_size), dtype=np.uint16)
    states = np.empty((tiles_k, tiles_n, 256), dtype=np.uint16)
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

        for bj in range(chunk_rows, 0, -16):
            bi = bj - 16
            bb_err = b_weight[bj:] - b_reconstructed[bj:]
            bb_l = b_l[bj:, i + bi : i + bj]
            compensation = b_prod_cache[bi:bj]
            if bb_err.size:
                compensation += bb_l.T @ bb_err
            rows_to_quantize = b_weight[bi:bj] + compensation
            block_packed, block_states, block_reconstructed = quantize_inner_matrix_direct(
                rows_to_quantize,
                k=k,
                cb=cb,
                search_backend=search_backend,
                max_pins=max_pins,
            )
            tk = (i + bi) // 16
            packed[tk : tk + 1] = block_packed
            states[tk : tk + 1] = block_states
            b_reconstructed[bi:bj] = block_reconstructed

        b_err = b_weight - b_reconstructed
        prod_cache += b_l.T @ b_err
        j = i

    delta = reconstructed - weight
    mse = float(np.mean(delta * delta))
    ref_rms = float(np.sqrt(np.mean(weight * weight))) + 1e-20
    stats: dict[str, float | bool] = {
        "inner_mse": mse,
        "inner_rel_rms": float(np.sqrt(mse) / ref_rms),
        "pack_roundtrip": True,
    }
    if hessian is not None:
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
    max_pins: int = 4,
) -> DirectLayerResult:
    """Quantize one full linear module with activation-aware LDLQ."""

    fixture = build_layer_fixture(source_dir, oracle_dir, module_key)
    source = fixture.source
    oracle = fixture.oracle
    ref_layer = oracle.layer
    if ref_layer.in_features % HAD_DIM != 0 or ref_layer.out_features % HAD_DIM != 0:
        raise ValueError("LDLQ layer conversion requires 128-multiple dimensions")

    suh, suh_zero_count = scale_full_for_mode(ref_layer.suh, ref_layer.in_features, scale_mode)
    svh, svh_zero_count = scale_full_for_mode(ref_layer.svh, ref_layer.out_features, scale_mode)
    in_blocks = ref_layer.in_features // HAD_DIM
    out_blocks = ref_layer.out_features // HAD_DIM
    target_inner = np.empty((ref_layer.in_features, ref_layer.out_features), dtype=np.float32)
    source_public = np.empty_like(target_inner, dtype=np.float32)

    for ib in range(in_blocks):
        in_start = ib * HAD_DIM
        su_slice = None if suh is None else suh[in_start : in_start + HAD_DIM]
        for ob in range(out_blocks):
            out_start = ob * HAD_DIM
            sv_slice = None if svh is None else svh[out_start : out_start + HAD_DIM]
            public_block = read_source_public_block(
                source_dir,
                source,
                in_start=in_start,
                out_start=out_start,
                rows=HAD_DIM,
                cols=HAD_DIM,
            )
            r0 = in_start
            r1 = in_start + HAD_DIM
            c0 = out_start
            c1 = out_start + HAD_DIM
            source_public[r0:r1, c0:c1] = public_block
            target_inner[r0:r1, c0:c1] = public_block_to_inner_with_scale_slices(
                public_block,
                su=su_slice,
                sv=sv_slice,
            )

    inner_acts = public_activations_to_inner(fixture.activations, suh)
    prepared = prepare_hessian_for_ldl(capture_hessian(inner_acts), sigma_reg=sigma_reg)
    ldl = block_ldl(prepared.hessian, block_size=16, sigma_reg=sigma_reg)
    quantized = ldlq_inner_matrix(
        target_inner,
        ldl.l,
        k=ref_layer.k,
        cb=oracle.cb,
        hessian=prepared.hessian,
        search_backend=search_backend,
        buf_size_rows=buf_size_rows,
        max_pins=max_pins,
    )

    out_layer = EXL3Layer(
        key=module_key,
        in_features=ref_layer.in_features,
        out_features=ref_layer.out_features,
        k=ref_layer.k,
        trellis=quantized.packed,
        suh=suh,
        svh=svh,
        mcg=ref_layer.mcg,
        mul1=ref_layer.mul1,
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

    x = fixture.activations.astype(np.float32)
    source_y = x @ source_public
    converted_y = x @ reconstructed_public
    public_delta = reconstructed_public - source_public
    output_delta = converted_y - source_y
    public_sse = float(np.sum(public_delta * public_delta))
    public_ref_ss = float(np.sum(source_public * source_public))
    output_sse = float(np.sum(output_delta * output_delta))
    output_ref_ss = float(np.sum(source_y * source_y))

    stats = dict(quantized.stats)
    stats.update(
        {
            "public_mse": mse_from_sse(public_sse, int(source_public.size)),
            "public_rel_rms": rel_rms_from_sse(
                public_sse,
                public_ref_ss,
                int(source_public.size),
            ),
            "output_mse": mse_from_sse(output_sse, int(source_y.size)),
            "output_rel_rms": rel_rms_from_sse(output_sse, output_ref_ss, int(source_y.size)),
            "suh_zero_replacements": float(suh_zero_count),
            "svh_zero_replacements": float(svh_zero_count),
            "hessian_diag_mean": prepared.diag_mean,
            "ldl_retries": float(ldl.retries),
        }
    )
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
