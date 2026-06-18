"""Calibration activation loading for converter pilots."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

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


def load_calibration_activations_map(path: str | Path) -> dict[str, np.ndarray]:
    """
    Load pre-captured per-module calibration activations.

    Supported formats:
    - ``.npz``: one 2D array per module key
    - directory: ``*.npy`` files, keyed by filename stem
    - ``.safetensors``: one 2D tensor per module key
    """

    p = Path(path)
    if p.is_dir():
        out = {
            item.stem: np.asarray(np.load(item))
            for item in sorted(p.glob("*.npy"))
            if item.is_file()
        }
    elif p.suffix.lower() == ".npz":
        with np.load(p) as data:
            out = {str(key): np.asarray(data[key]) for key in data.files}
    elif p.suffix.lower() == ".safetensors":
        tensors = load_file(str(p))
        out = {str(key): np.asarray(value) for key, value in tensors.items()}
    else:
        raise ValueError(f"unsupported calibration activation map format: {p.suffix}")
    if not out:
        raise ValueError(f"{p} does not contain any calibration activation arrays")
    for key, arr in out.items():
        validate_activation_matrix(arr)
        if not key:
            raise ValueError("calibration activation map contains an empty module key")
    return out


def activation_for_module(
    activations: np.ndarray | None,
    activations_by_module: Mapping[str, np.ndarray] | None,
    module_key: str,
) -> np.ndarray | None:
    """Return per-module calibration activations, falling back to the global matrix."""

    if activations_by_module is not None and module_key in activations_by_module:
        return activations_by_module[module_key]
    return activations
