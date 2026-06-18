"""EXL3 weight regularization and global scale search helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ponyexl3.ref.hadamard import HAD_DIM, preapply_had_left, preapply_had_right


CODEBOOK_SCALE = 1.24371088


@dataclass(frozen=True)
class RegularizedWeights:
    """Regularized inner-domain weights plus EXL3 reconstruction scales."""

    inner: np.ndarray
    suh: np.ndarray
    svh: np.ndarray
    stats: dict[str, float | bool]


@dataclass(frozen=True)
class GlobalScaleSearchResult:
    """Golden-section result for the EXL3 global scale."""

    scale: float
    mse: float
    evaluations: int


def block_rms_np(
    x: np.ndarray,
    axis: int,
    *,
    keepdims: bool = False,
    blocksize: int = 32,
) -> np.ndarray:
    """Compute upstream-style blockwise ``sqrt(mean(square(x), axis))``."""

    arr = x.astype(np.float32, copy=False)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"axis {axis} out of range for shape {arr.shape}")
    if blocksize <= 0:
        raise ValueError(f"blocksize must be positive, got {blocksize}")

    n = arr.shape[axis]
    sq: np.ndarray | None = None
    for start in range(0, n, blocksize):
        stop = min(start + blocksize, n)
        block = np.take(arr, range(start, stop), axis=axis)
        block_sq = np.asarray(
            np.sum(block * block, axis=axis, keepdims=keepdims, dtype=np.float32),
            dtype=np.float32,
        )
        sq = block_sq if sq is None else sq + block_sq
    if sq is None:
        raise ValueError("cannot compute RMS over an empty axis")
    return np.sqrt(sq / np.float32(n), dtype=np.float32)


def _random_signs(size: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.standard_normal(size, dtype=np.float32)
    return np.where(values < 0.0, -1.0, 1.0).astype(np.float32)


def _choose_apply_out_scales(
    apply_out_scales: bool | None,
    hessian_diag: np.ndarray | None,
) -> tuple[bool, float]:
    if apply_out_scales is not None:
        return bool(apply_out_scales), float("nan")
    if hessian_diag is None:
        return True, float("nan")

    diag = np.sqrt(np.maximum(hessian_diag.astype(np.float32, copy=False), 0.0))
    total = float(np.sum(diag, dtype=np.float64))
    if total <= 1e-30:
        return True, 0.0
    ordered = np.sort(diag)[::-1]
    cutoff = max(1, ordered.shape[0] // 50)
    skew = float(np.sum(ordered[:cutoff], dtype=np.float64) / total)
    return skew < 0.15, skew


def regularize_public_weight(
    public_weight: np.ndarray,
    *,
    seed: int = 0,
    apply_out_scales: bool | None = None,
    hessian_diag: np.ndarray | None = None,
) -> RegularizedWeights:
    """
    Port the scale/sign/Hadamard part of upstream ``regularize``.

    ``public_weight`` is in EXL3 public layout ``(in_features, out_features)``.
    The returned ``inner`` matrix is the target for trellis search, while
    ``suh`` and ``svh`` are the scale vectors needed to reconstruct the public
    matrix after quantization.
    """

    if public_weight.ndim != 2:
        raise ValueError(f"expected 2D public weight, got {public_weight.shape}")
    rows, cols = public_weight.shape
    if rows % HAD_DIM != 0 or cols % HAD_DIM != 0:
        raise ValueError(f"public weight shape must be 128-multiple, got {public_weight.shape}")

    rng = np.random.default_rng(seed)
    weight = public_weight.astype(np.float32, copy=True)
    suh = _random_signs(rows, rng).reshape(rows)
    svh = _random_signs(cols, rng).reshape(cols)
    apply_out, input_skew = _choose_apply_out_scales(apply_out_scales, hessian_diag)

    out_scales = block_rms_np(weight, axis=0, keepdims=True).reshape(cols)
    out_mean = float(np.mean(out_scales, dtype=np.float64))
    output_all_zero = out_mean <= 1e-30
    if not output_all_zero:
        out_scales = (out_scales / np.float32(out_mean)).astype(np.float32, copy=False)
    else:
        out_scales = np.zeros_like(out_scales, dtype=np.float32)
        if apply_out_scales is not None:
            apply_out = True
    zero_out = np.abs(out_scales) < 1e-30
    zero_out_count = int(np.sum(zero_out))

    if apply_out:
        out_scales = out_scales.copy()
        out_scales[zero_out] = np.float32(0.1)
        svh = (svh * out_scales + np.float32(1e-10)).astype(np.float32, copy=False)

    weight /= svh.reshape(1, cols)
    svh = svh.copy()
    svh[zero_out] = np.float32(0.0)
    weight = preapply_had_right(weight).astype(np.float32, copy=False)

    in_scales = block_rms_np(weight, axis=1, keepdims=True).reshape(rows)
    zero_in = np.abs(in_scales) < 1e-30
    zero_in_count = int(np.sum(zero_in))
    if zero_in_count:
        in_scales = in_scales.copy()
        in_scales[zero_in] = np.float32(0.1)
    suh = (suh * in_scales / np.float32(-CODEBOOK_SCALE) + np.float32(1e-10)).astype(
        np.float32,
        copy=False,
    )
    weight /= suh.reshape(rows, 1)
    weight = preapply_had_left(weight).astype(np.float32, copy=False)

    stats: dict[str, float | bool] = {
        "regularize_apply_out_scales": bool(apply_out),
        "regularize_input_skew": input_skew,
        "regularize_output_all_zero": bool(output_all_zero),
        "regularize_zero_out_scales": float(zero_out_count),
        "regularize_zero_in_scales": float(zero_in_count),
        "regularize_suh_abs_mean": float(np.mean(np.abs(suh), dtype=np.float64)),
        "regularize_svh_abs_mean": float(np.mean(np.abs(svh), dtype=np.float64)),
        "regularize_g_scale": 1.0,
        "regularize_g_scale_mse": float("nan"),
        "regularize_g_scale_evals": 0.0,
    }
    return RegularizedWeights(
        inner=weight.astype(np.float32, copy=False),
        suh=suh.astype(np.float32, copy=False),
        svh=svh.astype(np.float32, copy=False),
        stats=stats,
    )


def sample_tile_matrix(weight: np.ndarray, *, width: int = 3) -> np.ndarray:
    """Sample a wrapped diagonal of 16x16 tiles for global scale search."""

    if weight.ndim != 2 or weight.shape[0] % 16 != 0 or weight.shape[1] % 16 != 0:
        raise ValueError(f"expected 16-multiple 2D weight, got {weight.shape}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    tiles_k = weight.shape[0] // 16
    tiles_n = weight.shape[1] // 16
    tiles: list[np.ndarray] = []
    for i in range(max(tiles_k, tiles_n)):
        for w in range(width):
            row = (i % tiles_k) * 16
            col = ((i + w) % tiles_n) * 16
            tiles.append(weight[row : row + 16, col : col + 16].astype(np.float32, copy=True))
    return np.concatenate(tiles, axis=0)


def g_scale_gss(
    score_fn: Callable[[float], float],
    *,
    low: float = 0.1,
    high: float = 1.9,
    tol: float = 0.01,
) -> GlobalScaleSearchResult:
    """Golden-section search for the global scale minimizing ``score_fn``."""

    if low <= 0.0 or high <= low:
        raise ValueError(f"invalid search interval: low={low}, high={high}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi
    a = float(low)
    b = float(high)
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = float(score_fn(float(x1)))
    f2 = float(score_fn(float(x2)))
    evaluations = 2
    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = float(score_fn(float(x1)))
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = float(score_fn(float(x2)))
        evaluations += 1

    return GlobalScaleSearchResult(
        scale=float((a + b) / 2.0),
        mse=float((f1 + f2) / 2.0),
        evaluations=evaluations,
    )


def apply_global_scale(
    regularized: RegularizedWeights,
    result: GlobalScaleSearchResult,
) -> RegularizedWeights:
    """Apply upstream's final ``weight *= g`` and ``suh /= g`` step."""

    stats = dict(regularized.stats)
    stats.update(
        {
            "regularize_g_scale": result.scale,
            "regularize_g_scale_mse": result.mse,
            "regularize_g_scale_evals": float(result.evaluations),
        }
    )
    return RegularizedWeights(
        inner=(regularized.inner * np.float32(result.scale)).astype(np.float32, copy=False),
        suh=(regularized.suh / np.float32(result.scale)).astype(np.float32, copy=False),
        svh=regularized.svh.astype(np.float32, copy=False),
        stats=stats,
    )
