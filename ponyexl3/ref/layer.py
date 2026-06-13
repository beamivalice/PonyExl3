"""EXL3 linear layer container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codebook import codebook_mode_from_flags


@dataclass
class EXL3Layer:
    """One EXL3-quantized linear layer as stored on disk."""

    key: str
    in_features: int
    out_features: int
    k: int
    trellis: np.ndarray  # (in_tiles, out_tiles, packed_size) uint16
    suh: np.ndarray | None = None  # float16 (in,) or packed int16 (in//16,)
    svh: np.ndarray | None = None
    bias: np.ndarray | None = None
    mcg: bool = False
    mul1: bool = False

    @property
    def codebook_mode(self) -> int:
        return int(codebook_mode_from_flags(mcg=self.mcg, mul1=self.mul1))

    @property
    def packed_tile_size(self) -> int:
        return 256 * self.k // 16

    def validate(self) -> None:
        in_tiles = self.in_features // 16
        out_tiles = self.out_features // 16
        if self.in_features % 16 != 0 or self.out_features % 16 != 0:
            raise ValueError("in/out features must be multiples of 16")
        if self.trellis.shape != (in_tiles, out_tiles, self.packed_tile_size):
            raise ValueError(
                f"trellis shape {self.trellis.shape} != "
                f"expected {(in_tiles, out_tiles, self.packed_tile_size)}"
            )
