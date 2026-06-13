"""MLX checkpoint weight loading helpers."""

from __future__ import annotations

import os

import mlx.core as mx


def load_safetensors(path: str | os.PathLike[str]) -> dict[str, mx.array]:
    """Load a safetensors (or npz) file as a string-keyed weight dict."""
    raw = mx.load(path)
    if isinstance(raw, tuple):
        raw = raw[0]
    if not isinstance(raw, dict):
        raise TypeError(f"expected weight dict from {path!r}, got {type(raw).__name__}")
    return raw
