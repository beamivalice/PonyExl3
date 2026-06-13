"""Shared MLX ↔ numpy parity helpers for EXL3 correctness tests."""

from __future__ import annotations

import numpy as np
import mlx.core as mx


def mlx_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("mlx.core") is not None
    except ImportError:
        return False


def assert_allclose_np(
    got: np.ndarray,
    ref: np.ndarray,
    *,
    atol: float = 1e-2,
    rtol: float = 0.0,
    label: str = "",
) -> None:
    got32 = got.astype(np.float32)
    ref32 = ref.astype(np.float32)
    prefix = f"{label}: " if label else ""
    both_nan = np.isnan(got32) & np.isnan(ref32)
    diff = np.abs(got32 - ref32)
    diff[both_nan] = 0.0
    # numpy-style combined tolerance: |got - ref| <= atol + rtol * |ref|
    bound = atol + rtol * np.abs(ref32)
    excess = diff - bound
    excess[both_nan] = -np.inf
    max_excess = float(np.nanmax(excess))
    assert max_excess <= 0, (
        f"{prefix}max |diff| = {float(np.nanmax(diff))} exceeds "
        f"atol + rtol*|ref| by {max_excess} (atol={atol}, rtol={rtol})"
    )


def assert_allclose_mlx(
    got: mx.array,
    ref: np.ndarray,
    *,
    atol: float = 1e-2,
    rtol: float = 0.0,
    label: str = "",
) -> None:
    mx.eval(got)
    assert_allclose_np(np.array(got), ref, atol=atol, rtol=rtol, label=label)


def compare_arrays(
    got: np.ndarray,
    ref: np.ndarray,
    *,
    atol: float = 1e-2,
    rtol: float = 0.0,
) -> dict[str, float | bool]:
    got32 = got.astype(np.float32)
    ref32 = ref.astype(np.float32)
    diff = np.abs(got32 - ref32)
    return {
        "max_abs": float(np.nanmax(diff)),
        "mean_abs": float(np.nanmean(diff)),
        "close": bool(np.allclose(got32, ref32, atol=atol, rtol=rtol)),
    }
