"""Hessian capture and NumPy LDLQ primitives for EXL3 conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
    quantize_inner_matrix_direct_mlx,
    rel_rms_from_sse,
)
from ponyexl3.convert import reuse, timing
from ponyexl3.convert.fixtures import SearchBackend, build_layer_fixture
from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.hadamard import HAD_DIM, had_r_128, preapply_had_left, preapply_had_right
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.reconstruct import reconstruct_public_weights
from ponyexl3.ref.signs import unpack_signs_or_pass


_MLX_ACTIVATION_HADAMARD_MIN_ROWS = 192


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


class LDLQGroupIncompatible(ValueError):
    """Raised when candidate sibling linears cannot share batched search."""


@dataclass(frozen=True)
class _LDLQGroupPrepared:
    """CPU-prepared state for one member before MLX buffers are allocated."""

    key: str
    fixture: Any
    basis: Any
    prepared: PreparedHessian
    ldl: BlockLDLResult
    ref_layer: EXL3Layer
    k: int
    cb: CodebookMode
    l_factor: np.ndarray
    activations_mlx_hadamard: bool
    hessian_stats: dict[str, float | bool]


@dataclass
class _LDLQGroupState:
    """Mutable MLX state for one member of a batched-search LDLQ group."""

    key: str
    fixture: Any
    basis: Any
    prepared: PreparedHessian
    ldl: BlockLDLResult
    ref_layer: EXL3Layer
    weight: Any
    l_factor: Any
    reconstructed: Any
    prod_cache: Any
    packed: Any
    activations_mlx_hadamard: bool
    hessian_stats: dict[str, float | bool]


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


def apply_hessian_shrinkage(hessian: np.ndarray, *, shrinkage: float = 0.0) -> np.ndarray:
    """Shrink off-diagonal covariance toward a diagonal Hessian estimate."""

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"expected square Hessian, got {hessian.shape}")
    if not np.isfinite(shrinkage) or shrinkage < 0.0 or shrinkage > 1.0:
        raise ValueError(f"hessian_shrinkage must be in [0, 1], got {shrinkage}")
    if shrinkage == 0.0:
        return hessian.astype(np.float32, copy=False)
    h = hessian.astype(np.float32, copy=True)
    diag_idx = np.diag_indices_from(h)
    diag = h[diag_idx].copy()
    h *= np.float32(1.0 - shrinkage)
    h[diag_idx] = diag
    return h.astype(np.float32, copy=False)


def hessian_offdiag_rel(hessian: np.ndarray) -> float:
    """Return off-diagonal RMS normalized by diagonal RMS."""

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"expected square Hessian, got {hessian.shape}")
    n = hessian.shape[0]
    if n == 0:
        return 0.0
    h = hessian.astype(np.float32, copy=False)
    diag = np.diag(h).astype(np.float64, copy=False)
    diag_ss = float(np.sum(diag * diag, dtype=np.float64))
    total_ss = float(np.sum(h * h, dtype=np.float64))
    offdiag_ss = max(0.0, total_ss - diag_ss)
    offdiag_count = max(1, n * n - n)
    diag_rms = float(np.sqrt(diag_ss / max(1, n)))
    offdiag_rms = float(np.sqrt(offdiag_ss / offdiag_count))
    return float(offdiag_rms / (diag_rms + 1e-20))


def _prepare_activation_hessian(
    inner_activations: np.ndarray,
    *,
    sigma_reg: float,
    hessian_shrinkage: float,
) -> tuple[PreparedHessian, dict[str, float | bool]]:
    raw_hessian = capture_hessian(inner_activations)
    raw_offdiag_rel = hessian_offdiag_rel(raw_hessian)
    hessian = apply_hessian_shrinkage(raw_hessian, shrinkage=hessian_shrinkage)
    shrunk_offdiag_rel = hessian_offdiag_rel(hessian)
    prepared = prepare_hessian_for_ldl(hessian, sigma_reg=sigma_reg)
    return prepared, {
        "hessian_shrinkage": float(hessian_shrinkage),
        "hessian_offdiag_rel_unshrunk": raw_offdiag_rel,
        "hessian_offdiag_rel": shrunk_offdiag_rel,
    }


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


def _public_activations_to_inner_for_backend(
    activations: np.ndarray,
    suh: np.ndarray | None,
    *,
    search_backend: SearchBackend,
) -> tuple[np.ndarray, bool]:
    if (
        suh is not None
        and search_backend == "metal"
        and activations.shape[0] >= _MLX_ACTIVATION_HADAMARD_MIN_ROWS
    ):
        try:
            import mlx.core as mx

            from ponyexl3.mlx.hadamard import had_r_128_mlx

            x = mx.array(activations.astype(np.float32, copy=False), dtype=mx.float32)
            scale = mx.array(suh.astype(np.float32, copy=False), dtype=mx.float32)
            out = had_r_128_mlx(x, pre_scale=scale, r_scale=1.0).astype(mx.float32)
            mx.eval(out)
            return np.array(out).astype(np.float32, copy=False), True
        except Exception:
            pass
    return public_activations_to_inner(activations, suh), False


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

    if search_backend == "metal" and not collect_states:
        return _ldlq_inner_matrix_mlx(
            inner,
            l_factor,
            k=k,
            cb=cb,
            hessian=hessian,
            buf_size_rows=buf_size_rows,
            feedback_rows=feedback_rows,
            compute_proxy=compute_proxy,
        )

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


def _ldlq_inner_matrix_mlx(
    inner: np.ndarray,
    l_factor: np.ndarray,
    *,
    k: int,
    cb: CodebookMode,
    hessian: np.ndarray | None,
    buf_size_rows: int,
    feedback_rows: int,
    compute_proxy: bool,
) -> LDLQResult:
    """MLX-resident production LDLQ loop for Metal/no-state conversion."""

    import mlx.core as mx

    rows, cols = inner.shape
    l_np = l_factor.astype(np.float32, copy=True)
    l_np[np.diag_indices_from(l_np)] = np.float32(0.0)

    weight = mx.array(inner.astype(np.float32, copy=False), dtype=mx.float32)
    l = mx.array(l_np, dtype=mx.float32)
    reconstructed = mx.zeros_like(weight)
    prod_cache = mx.zeros_like(weight)

    tiles_k = rows // 16
    tiles_n = cols // 16
    packed_size = 256 * k // 16
    packed_mx = mx.zeros((tiles_k, tiles_n, packed_size), dtype=mx.uint16)

    j = rows
    while j > 0:
        i = max(0, j - buf_size_rows)
        b_weight = weight[i:j]
        b_reconstructed = reconstructed[i:j]
        b_prod_cache = prod_cache[i:j]
        b_l = l[i:j]
        chunk_rows = j - i

        for bj in range(chunk_rows, 0, -feedback_rows):
            bi = max(0, bj - feedback_rows)
            bb_err = b_weight[bj:] - b_reconstructed[bj:]
            compensation = b_prod_cache[bi:bj]
            if bb_err.size:
                bb_l = b_l[bj:, i + bi : i + bj]
                compensation = compensation + mx.matmul(bb_l.T, bb_err)
            rows_to_quantize = b_weight[bi:bj] + compensation
            block_packed_mx, block_reconstructed = quantize_inner_matrix_direct_mlx(
                rows_to_quantize,
                k=k,
                cb=cb,
            )
            tk = (i + bi) // 16
            packed_mx = mx.slice_update(
                packed_mx,
                block_packed_mx,
                start_indices=mx.array([tk, 0, 0], dtype=mx.int32),
                axes=(0, 1, 2),
            )
            b_reconstructed = mx.slice_update(
                b_reconstructed,
                block_reconstructed,
                start_indices=mx.array([bi, 0], dtype=mx.int32),
                axes=(0, 1),
            )
            # Force the accumulators every feedback group. Otherwise the deferred
            # slice_updates into packed_mx pile up into one enormous final eval for
            # huge layers (lm_head ~1.3B params), which balloons MLX's buffer pool
            # past RAM mid-layer (observed ~108 GB on the 27B lm_head before OOM).
            mx.eval(packed_mx, b_reconstructed)

        b_err = b_weight - b_reconstructed
        reconstructed = mx.slice_update(
            reconstructed,
            b_reconstructed,
            start_indices=mx.array([i, 0], dtype=mx.int32),
            axes=(0, 1),
        )
        if i > 0:
            prod_update = mx.matmul(b_l[:, :i].T, b_err)
            prod_cache = mx.slice_update(
                prod_cache,
                prod_cache[:i] + prod_update,
                start_indices=mx.array([0, 0], dtype=mx.int32),
                axes=(0, 1),
            )
        # Materialize this row-buffer's accumulators and hand the freed buffers
        # back to the OS, so steady-state memory stays at the live working set
        # rather than the running total of every group's discarded scratch.
        mx.eval(reconstructed, prod_cache)
        mx.clear_cache()
        j = i

    mx.eval(reconstructed, packed_mx)
    packed = np.array(packed_mx).astype(np.uint16, copy=False)
    reconstructed_np = np.array(reconstructed).astype(np.float32, copy=False)
    weight_np = inner.astype(np.float32, copy=False)
    delta = reconstructed_np - weight_np
    mse = float(np.mean(delta * delta))
    ref_rms = float(np.sqrt(np.mean(weight_np * weight_np))) + 1e-20
    stats: dict[str, float | bool] = {
        "inner_mse": mse,
        "inner_rel_rms": float(np.sqrt(mse) / ref_rms),
        "pack_roundtrip": True,
        "ldlq_feedback_rows": float(feedback_rows),
        "mlx_ldlq": True,
        "mlx_packed_deferred": True,
    }
    if hessian is not None and compute_proxy:
        proxy = hessian_proxy_stats(delta, hessian, reference=weight_np)
        stats.update(
            {
                "hessian_proxy_sse": proxy.proxy_sse,
                "hessian_proxy_mse": proxy.proxy_mse,
                "hessian_proxy_rel_rms": proxy.proxy_rel_rms,
            }
        )

    return LDLQResult(
        packed=packed,
        states=None,
        reconstructed=reconstructed_np,
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
    hessian_shrinkage: float = 0.0,
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

    reuse_key = None
    if reuse.active() and calibration_activations is not None:
        reuse_key = reuse.make_key(
            source_dir=source_dir,
            oracle_dir=oracle_dir,
            module_key=module_key,
            scale_mode=scale_mode,
            sigma_reg=sigma_reg,
            hessian_shrinkage=hessian_shrinkage,
            buf_size_rows=buf_size_rows,
            feedback_rows=feedback_rows,
            max_pins=max_pins,
            skip_g_scale=skip_g_scale,
            regularization_seed=regularization_seed,
            quant_bits=quant_bits,
            search_backend=search_backend,
            acts_fp=reuse.activations_fingerprint(calibration_activations),
        )
        cached = reuse.lookup(reuse_key)
        if cached is not None:
            return cached

    with timing.phase("fixture"):
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

    with timing.phase("basis"):
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

    with timing.phase("act_hadamard"):
        inner_acts, activations_mlx_hadamard = _public_activations_to_inner_for_backend(
            fixture.activations,
            suh,
            search_backend=search_backend,
        )
    with timing.phase("hessian"):
        prepared, hessian_stats = _prepare_activation_hessian(
            inner_acts,
            sigma_reg=sigma_reg,
            hessian_shrinkage=hessian_shrinkage,
        )
    with timing.phase("ldl"):
        ldl = block_ldl(prepared.hessian, block_size=16, sigma_reg=sigma_reg)
    with timing.phase("ldlq_loop"):
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

    timing.begin("metrics")
    x = fixture.activations.astype(np.float32)
    public_ref_ss = float(np.sum(source_public * source_public))

    stats = dict(quantized.stats)
    stats.update(
        {
            "hessian_diag_mean": prepared.diag_mean,
            "ldl_retries": float(ldl.retries),
            "oracle_metrics": bool(compare_oracle),
            "fast_metrics": bool(fast_metrics),
            "activations_mlx_hadamard": activations_mlx_hadamard,
        }
    )
    stats.update(hessian_stats)
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
    timing.end("metrics")
    stats.update(basis.stats)
    result = DirectLayerResult(
        module_key=module_key,
        search_backend=search_backend,
        scale_mode=scale_mode,
        layer=out_layer,
        activations=x,
        source_output=source_y.astype(np.float32),
        converted_output=converted_y.astype(np.float32),
        stats=stats,
    )
    if reuse_key is not None:
        reuse.store(reuse_key, result)
    return result


def _group_prep_worker_count(
    group_size: int,
    *,
    search_backend: SearchBackend,
    scale_mode: ScaleMode,
    skip_g_scale: bool,
) -> int:
    if group_size <= 1:
        return 1
    # Metal prep can launch Hadamard/GSS kernels before LDLQ search. Keep those
    # paths serial so worker threads do not compete for the same GPU queue.
    if search_backend == "metal" and scale_mode in ("oracle", "oracle_safe"):
        return 1
    if search_backend == "metal" and scale_mode == "computed" and not skip_g_scale:
        return 1
    return min(group_size, 4)


def _prepare_ldlq_group_member(
    source_dir: str | Path,
    oracle_dir: str | Path,
    key: str,
    *,
    activations: np.ndarray | None,
    scale_mode: ScaleMode,
    search_backend: SearchBackend,
    sigma_reg: float,
    hessian_shrinkage: float,
    max_pins: int,
    skip_g_scale: bool,
    regularization_seed: int,
    quant_bits: int | None,
) -> _LDLQGroupPrepared:
    fixture = build_layer_fixture(source_dir, oracle_dir, key, activations=activations)
    oracle = fixture.oracle
    ref_layer = oracle.layer
    if ref_layer.in_features % HAD_DIM != 0 or ref_layer.out_features % HAD_DIM != 0:
        raise LDLQGroupIncompatible(f"{key}: dimensions must be 128-multiple")
    k = ref_layer.k if quant_bits is None else int(quant_bits)
    if k < 1 or k > 8:
        raise LDLQGroupIncompatible(f"{key}: EXL3 trellis K must be in [1, 8], got {k}")
    basis = prepare_layer_quantization_basis(
        source_dir,
        fixture.source,
        ref_layer,
        oracle.cb,
        scale_mode=scale_mode,
        search_backend=search_backend,
        max_pins=max_pins,
        skip_g_scale=skip_g_scale,
        regularization_seed=regularization_seed,
        quant_bits=k,
    )
    inner_acts, activations_mlx_hadamard = _public_activations_to_inner_for_backend(
        fixture.activations,
        basis.suh,
        search_backend=search_backend,
    )
    prepared, hessian_stats = _prepare_activation_hessian(
        inner_acts,
        sigma_reg=sigma_reg,
        hessian_shrinkage=hessian_shrinkage,
    )
    ldl = block_ldl(prepared.hessian, block_size=16, sigma_reg=sigma_reg)
    l_factor = ldl.l.astype(np.float32, copy=True)
    l_factor[np.diag_indices_from(l_factor)] = np.float32(0.0)
    return _LDLQGroupPrepared(
        key=key,
        fixture=fixture,
        basis=basis,
        prepared=prepared,
        ldl=ldl,
        ref_layer=ref_layer,
        k=k,
        cb=oracle.cb,
        l_factor=l_factor,
        activations_mlx_hadamard=activations_mlx_hadamard,
        hessian_stats=hessian_stats,
    )


def ldlq_quantize_group(
    source_dir: str | Path,
    oracle_dir: str | Path,
    module_keys: Sequence[str],
    *,
    search_backend: SearchBackend = "metal",
    scale_mode: ScaleMode = "oracle_safe",
    sigma_reg: float = 0.025,
    hessian_shrinkage: float = 0.0,
    buf_size_rows: int = 128,
    feedback_rows: int = 16,
    max_pins: int = 4,
    calibration_activations: np.ndarray | None = None,
    calibration_activations_by_module: Mapping[str, np.ndarray | None] | None = None,
    skip_g_scale: bool = False,
    regularization_seed: int = 0,
    quant_bits_by_module: dict[str, int | None] | None = None,
) -> list[DirectLayerResult]:
    """Quantize compatible sibling linears with batched Metal search calls.

    Each module keeps its own source scales, Hessian, LDL factor, compensation,
    and reconstructed matrix. At each reverse-LDLQ feedback step we concatenate
    the current rows across modules, launch one larger trellis search, then split
    the packed/reconstructed result back to the independent module states. This
    avoids the unsafe assumption that sibling modules share identical input
    scales while still reducing the small-kernel gaps between modules.
    """

    keys = list(module_keys)
    if len(keys) < 2:
        raise LDLQGroupIncompatible("need at least two modules for grouped LDLQ")
    if search_backend != "metal":
        raise LDLQGroupIncompatible("grouped LDLQ is only enabled for Metal")
    if buf_size_rows < 16 or buf_size_rows % 16 != 0:
        raise ValueError(f"buf_size_rows must be a positive 16-multiple, got {buf_size_rows}")
    if feedback_rows < 16 or feedback_rows % 16 != 0:
        raise ValueError(f"feedback_rows must be a positive 16-multiple, got {feedback_rows}")
    if feedback_rows > buf_size_rows:
        raise ValueError("feedback_rows must be <= buf_size_rows")

    reuse_keys: list[tuple[Any, ...] | None] = [None] * len(keys)
    if reuse.active():
        for idx, key in enumerate(keys):
            acts = calibration_activations
            if calibration_activations_by_module is not None and key in calibration_activations_by_module:
                acts = calibration_activations_by_module[key]
            planned_k = None if quant_bits_by_module is None else quant_bits_by_module.get(key)
            reuse_keys[idx] = reuse.make_key(
                source_dir=source_dir,
                oracle_dir=oracle_dir,
                module_key=key,
                scale_mode=scale_mode,
                sigma_reg=sigma_reg,
                hessian_shrinkage=hessian_shrinkage,
                buf_size_rows=buf_size_rows,
                feedback_rows=feedback_rows,
                max_pins=max_pins,
                skip_g_scale=skip_g_scale,
                regularization_seed=regularization_seed,
                quant_bits=planned_k,
                search_backend=search_backend,
                acts_fp=reuse.activations_fingerprint(acts),
            )
        cached_results = [reuse.lookup(rkey) for rkey in reuse_keys]
        if cached_results and all(item is not None for item in cached_results):
            return [item for item in cached_results if item is not None]

    import mlx.core as mx

    prep_inputs: list[tuple[str, np.ndarray | None, int | None]] = []
    for key in keys:
        acts = calibration_activations
        if calibration_activations_by_module is not None and key in calibration_activations_by_module:
            acts = calibration_activations_by_module[key]
        planned_k = None if quant_bits_by_module is None else quant_bits_by_module.get(key)
        prep_inputs.append((key, acts, planned_k))

    prep_workers = _group_prep_worker_count(
        len(keys),
        search_backend=search_backend,
        scale_mode=scale_mode,
        skip_g_scale=skip_g_scale,
    )
    if prep_workers <= 1:
        prepared_members = [
            _prepare_ldlq_group_member(
                source_dir,
                oracle_dir,
                key,
                activations=acts,
                scale_mode=scale_mode,
                search_backend=search_backend,
                sigma_reg=sigma_reg,
                hessian_shrinkage=hessian_shrinkage,
                max_pins=max_pins,
                skip_g_scale=skip_g_scale,
                regularization_seed=regularization_seed,
                quant_bits=planned_k,
            )
            for key, acts, planned_k in prep_inputs
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=prep_workers,
            thread_name_prefix="ponyexl3-ldlq-prep",
        ) as executor:
            futures = [
                executor.submit(
                    _prepare_ldlq_group_member,
                    source_dir,
                    oracle_dir,
                    key,
                    activations=acts,
                    scale_mode=scale_mode,
                    search_backend=search_backend,
                    sigma_reg=sigma_reg,
                    hessian_shrinkage=hessian_shrinkage,
                    max_pins=max_pins,
                    skip_g_scale=skip_g_scale,
                    regularization_seed=regularization_seed,
                    quant_bits=planned_k,
                )
                for key, acts, planned_k in prep_inputs
            ]
            prepared_members = [future.result() for future in futures]

    states: list[_LDLQGroupState] = []
    first_cb: CodebookMode | None = None
    first_rows: int | None = None
    first_k: int | None = None
    for member in prepared_members:
        ref_layer = member.ref_layer
        if first_cb is None:
            first_cb = member.cb
        elif member.cb != first_cb:
            raise LDLQGroupIncompatible(f"{member.key}: codebook differs from group")
        if first_k is None:
            first_k = member.k
        elif member.k != first_k:
            raise LDLQGroupIncompatible(f"{member.key}: K differs from group")
        if first_rows is None:
            first_rows = int(ref_layer.in_features)
        elif int(ref_layer.in_features) != first_rows:
            raise LDLQGroupIncompatible(f"{member.key}: in_features differs from group")
        rows = int(ref_layer.in_features)
        out_features = int(ref_layer.out_features)
        packed_size = 256 * member.k // 16
        states.append(
            _LDLQGroupState(
                key=member.key,
                fixture=member.fixture,
                basis=member.basis,
                prepared=member.prepared,
                ldl=member.ldl,
                ref_layer=ref_layer,
                weight=mx.array(member.basis.target_inner.astype(np.float32, copy=False), dtype=mx.float32),
                l_factor=mx.array(member.l_factor, dtype=mx.float32),
                reconstructed=mx.zeros((rows, out_features), dtype=mx.float32),
                prod_cache=mx.zeros((rows, out_features), dtype=mx.float32),
                packed=mx.zeros(
                    (rows // 16, out_features // 16, packed_size),
                    dtype=mx.uint16,
                ),
                activations_mlx_hadamard=member.activations_mlx_hadamard,
                hessian_stats=member.hessian_stats,
            )
        )

    if not states:
        raise LDLQGroupIncompatible("empty grouped LDLQ request")
    if first_cb is None or first_rows is None or first_k is None:
        raise LDLQGroupIncompatible("invalid grouped LDLQ request")

    rows = first_rows
    total_out = sum(int(state.ref_layer.out_features) for state in states)
    j = rows
    while j > 0:
        i = max(0, j - buf_size_rows)
        if (j - i) % 16 != 0:
            raise ValueError("LDLQ row chunks must stay 16-aligned")
        chunk_rows = j - i
        b_weights = [state.weight[i:j] for state in states]
        b_reconstructed = [state.reconstructed[i:j] for state in states]
        b_prod_cache = [state.prod_cache[i:j] for state in states]
        b_l = [state.l_factor[i:j] for state in states]

        for bj in range(chunk_rows, 0, -feedback_rows):
            bi = max(0, bj - feedback_rows)
            if (bj - bi) % 16 != 0:
                raise ValueError("LDLQ feedback groups must stay 16-aligned")
            rows_to_quantize = []
            for index, state in enumerate(states):
                bb_err = b_weights[index][bj:] - b_reconstructed[index][bj:]
                compensation = b_prod_cache[index][bi:bj]
                if bb_err.size:
                    bb_l = b_l[index][bj:, i + bi : i + bj]
                    compensation = compensation + mx.matmul(bb_l.T, bb_err)
                rows_to_quantize.append(b_weights[index][bi:bj] + compensation)

            search_input = mx.concatenate(rows_to_quantize, axis=1)
            block_packed_mx, block_reconstructed_cat = quantize_inner_matrix_direct_mlx(
                search_input,
                k=first_k,
                cb=first_cb,
            )
            tk = (i + bi) // 16
            col = 0
            tile_col = 0
            for index, state in enumerate(states):
                out_features = int(state.ref_layer.out_features)
                out_tiles = out_features // 16
                block_reconstructed = block_reconstructed_cat[:, col : col + out_features]
                b_reconstructed[index] = mx.slice_update(
                    b_reconstructed[index],
                    block_reconstructed,
                    start_indices=mx.array([bi, 0], dtype=mx.int32),
                    axes=(0, 1),
                )
                state.packed = mx.slice_update(
                    state.packed,
                    block_packed_mx[:, tile_col : tile_col + out_tiles],
                    start_indices=mx.array([tk, 0, 0], dtype=mx.int32),
                    axes=(0, 1, 2),
                )
                col += out_features
                tile_col += out_tiles
            # Force per feedback group so the per-state packed slice_updates don't
            # defer into one giant final eval that balloons the buffer pool.
            mx.eval(*[state.packed for state in states], *b_reconstructed)

        for index, state in enumerate(states):
            b_err = b_weights[index] - b_reconstructed[index]
            state.reconstructed = mx.slice_update(
                state.reconstructed,
                b_reconstructed[index],
                start_indices=mx.array([i, 0], dtype=mx.int32),
                axes=(0, 1),
            )
            if i > 0:
                prod_update = mx.matmul(b_l[index][:, :i].T, b_err)
                state.prod_cache = mx.slice_update(
                    state.prod_cache,
                    state.prod_cache[:i] + prod_update,
                    start_indices=mx.array([0, 0], dtype=mx.int32),
                    axes=(0, 1),
                )
        # Materialize this row-buffer's accumulators and release the freed pool.
        mx.eval(
            *[state.reconstructed for state in states],
            *[state.prod_cache for state in states],
        )
        mx.clear_cache()
        j = i

    mx.eval(*[state.reconstructed for state in states], *[state.packed for state in states])
    out: list[DirectLayerResult] = []
    for state in states:
        ref_layer = state.ref_layer
        recon_part = np.array(state.reconstructed).astype(np.float32, copy=False)
        packed = np.array(state.packed).astype(np.uint16, copy=False)
        target_part = state.basis.target_inner
        delta = recon_part - target_part
        mse = float(np.mean(delta * delta))
        ref_rms = float(np.sqrt(np.mean(target_part * target_part))) + 1e-20
        stats = dict(state.basis.stats)
        stats.update(
            {
                "inner_mse": mse,
                "inner_rel_rms": float(np.sqrt(mse) / ref_rms),
                "pack_roundtrip": True,
                "ldlq_feedback_rows": float(feedback_rows),
                "hessian_diag_mean": state.prepared.diag_mean,
                "ldl_retries": float(state.ldl.retries),
                "oracle_metrics": False,
                "fast_metrics": True,
                "mlx_ldlq": True,
                "activations_mlx_hadamard": state.activations_mlx_hadamard,
                "batched_group_size": float(len(keys)),
                "batched_group_out_features": float(total_out),
                "batched_search_group_size": float(len(keys)),
                "batched_search_group_out_features": float(total_out),
                "batched_prep_workers": float(prep_workers),
                "mlx_packed_deferred": True,
                "public_mse": float("nan"),
                "public_rel_rms": float("nan"),
                "output_mse": float("nan"),
                "output_rel_rms": float("nan"),
            }
        )
        stats.update(state.hessian_stats)
        layer = EXL3Layer(
            key=state.key,
            in_features=ref_layer.in_features,
            out_features=ref_layer.out_features,
            k=first_k,
            trellis=packed,
            suh=state.basis.suh,
            svh=state.basis.svh,
            mcg=ref_layer.mcg,
            mul1=ref_layer.mul1,
        )
        layer.validate()
        empty = np.empty((0, 0), dtype=np.float32)
        out.append(
            DirectLayerResult(
                module_key=state.key,
                search_backend=search_backend,
                scale_mode=scale_mode,
                layer=layer,
                activations=state.fixture.activations.astype(np.float32),
                source_output=empty,
                converted_output=empty,
                stats=stats,
            )
        )
    for result, rkey in zip(out, reuse_keys):
        if rkey is not None:
            reuse.store(rkey, result)
    return out


def ldlq_layer_summary(result: DirectLayerResult) -> dict[str, Any]:
    """JSON-friendly summary for LDLQ layer pilots."""

    summary = direct_layer_summary(result)
    summary["quantizer"] = "ldlq"
    return summary
