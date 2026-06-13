"""Decode-once caches — re-exports from ``layer_state``."""

from __future__ import annotations

from ponyexl3.mlx.layer_state import (
    clear_layer_caches,
    inner_weight_mlx,
    stripe_weight_mlx,
)

clear_inner_weight_cache = clear_layer_caches

__all__ = [
    "clear_inner_weight_cache",
    "clear_layer_caches",
    "inner_weight_mlx",
    "stripe_weight_mlx",
]
