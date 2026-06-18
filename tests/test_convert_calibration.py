"""Calibration activation loading for conversion pilots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ponyexl3.convert.calibration import load_calibration_activations, validate_activation_matrix


def test_load_calibration_activations_npy_npz_and_safetensors(tmp_path: Path):
    activations = np.arange(24, dtype=np.float32).reshape(3, 8)
    npy_path = tmp_path / "acts.npy"
    npz_path = tmp_path / "acts.npz"
    st_path = tmp_path / "acts.safetensors"
    np.save(npy_path, activations)
    np.savez(npz_path, ignored=np.zeros((1, 1), dtype=np.float32), activations=activations)
    save_file({"activations": activations}, str(st_path))

    np.testing.assert_array_equal(
        load_calibration_activations(npy_path, expected_features=8),
        activations,
    )
    np.testing.assert_array_equal(
        load_calibration_activations(npz_path, expected_features=8),
        activations,
    )
    np.testing.assert_array_equal(
        load_calibration_activations(st_path, expected_features=8),
        activations,
    )


def test_validate_activation_matrix_rejects_bad_shape():
    with pytest.raises(ValueError, match="feature dim"):
        validate_activation_matrix(np.zeros((2, 7), dtype=np.float32), expected_features=8)
    with pytest.raises(ValueError, match="2D"):
        validate_activation_matrix(np.zeros((2, 3, 4), dtype=np.float32))
