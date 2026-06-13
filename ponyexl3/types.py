"""Shared typing aliases for PonyExl3."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

JsonDict = dict[str, Any]
FloatArray = NDArray[np.floating[Any]]
UInt16Array = NDArray[np.uint16]

# mlx_lm model graph objects are dynamic; annotate boundaries explicitly.
MlxLmModel = Any
ExLlamaModel = Any
Tokenizer = Any
DraftModule = Any
KvCache = Any


class StoredTensorInfo(TypedDict):
    shape: list[int]
    n_bytes: int


class Exl3LayerInfo(TypedDict):
    key: str
    bits_per_weight: float | int | None
    stored_tensors: dict[str, StoredTensorInfo]
    mcg: bool
    mul1: bool


class LayerMeta(TypedDict):
    key: str
    k: int
    bits_per_weight: float | int | None
    in_features: int
    out_features: int
    in_tiles: int
    out_tiles: int
    n_tiles: int
    trellis_shape: tuple[int, int, int]
    trellis_bytes: int
    weight_fp16_bytes: int
    mcg: bool
    mul1: bool
