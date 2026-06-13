"""Test helpers for EXL3 correctness suites."""

from __future__ import annotations

import numpy as np
import pytest


def require_finite(*arrays: object, label: str = "") -> None:
    """Skip when synthetic data produced non-finite reference values."""
    for arr in arrays:
        data = np.asarray(arr)
        if not np.isfinite(data).all():
            pytest.skip(f"{label + ': ' if label else ''}non-finite reference")
