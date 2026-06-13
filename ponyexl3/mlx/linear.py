"""MLX linear forward — re-exports and numpy comparison helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ponyexl3.mlx._parity import mlx_available
from ponyexl3.mlx.forward import linear_forward_mlx
from ponyexl3.ref.forward import linear_forward_reconstruct
from ponyexl3.ref.layer import EXL3Layer

__all__ = [
    "compare_numpy_vs_mlx",
    "linear_forward_mlx",
    "mlx_available",
]


def compare_numpy_vs_mlx(
    layer: EXL3Layer,
    x: np.ndarray,
    *,
    atol: float = 2.5,
    rtol: float = 0.1,
) -> dict[str, Any]:
    """Compare MLX forward against numpy golden path."""
    y_ref = linear_forward_reconstruct(layer, x)
    y_mlx = np.array(linear_forward_mlx(layer, x))
    diff = np.abs(y_mlx.astype(np.float32) - y_ref.astype(np.float32))
    max_abs = float(diff.max())
    denom = np.abs(y_ref.astype(np.float32))
    mask = denom > 1e-6
    max_rel = float((diff[mask] / denom[mask]).max()) if np.any(mask) else 0.0
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_abs": float(diff.mean()),
        "ok": bool(max_abs <= atol and max_rel <= rtol),
    }
