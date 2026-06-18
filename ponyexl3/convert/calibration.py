"""Calibration activation loading for converter pilots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def validate_activation_matrix(
    activations: np.ndarray,
    *,
    expected_features: int | None = None,
) -> np.ndarray:
    """Validate a 2D activation matrix and return float32 rows."""

    arr = np.asarray(activations)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D calibration activations, got {arr.shape}")
    if arr.shape[0] <= 0:
        raise ValueError("calibration activations must contain at least one row")
    if expected_features is not None and arr.shape[1] != expected_features:
        raise ValueError(
            f"calibration activations feature dim {arr.shape[1]} "
            f"!= expected {expected_features}"
        )
    out = arr.astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise ValueError("calibration activations contain non-finite values")
    return out


def load_calibration_activations(
    path: str | Path,
    *,
    expected_features: int | None = None,
) -> np.ndarray:
    """
    Load pre-captured calibration activations.

    Supported formats:
    - ``.npy``: one 2D array
    - ``.npz``: ``activations`` array if present, otherwise the first array
    - ``.safetensors``: ``activations`` tensor if present, otherwise the first tensor
    """

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.asarray(np.load(p))
    elif suffix == ".npz":
        with np.load(p) as data:
            if not data.files:
                raise ValueError(f"{p} does not contain any arrays")
            key = "activations" if "activations" in data.files else data.files[0]
            arr = np.asarray(data[key])
    elif suffix == ".safetensors":
        tensors = load_file(str(p))
        if not tensors:
            raise ValueError(f"{p} does not contain any tensors")
        key = "activations" if "activations" in tensors else sorted(tensors)[0]
        arr = np.asarray(tensors[key])
    else:
        raise ValueError(f"unsupported calibration activation format: {p.suffix}")
    return validate_activation_matrix(arr, expected_features=expected_features)
