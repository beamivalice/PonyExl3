"""EXL3 → MLX inference for Apple Silicon (PonyExl3).

Golden CPU reference in ``ponyexl3.ref``; MLX runtime in ``ponyexl3.mlx``.
Ported from ``exllamav3/mlxport`` — see ``README.md``.
"""

from ponyexl3.ref import (
    CodebookMode,
    EXL3Layer,
    decode_3inst,
    linear_forward_reconstruct,
    list_exl3_layers,
    load_exl3_layer,
)

__all__ = [
    "CodebookMode",
    "EXL3Layer",
    "decode_3inst",
    "linear_forward_reconstruct",
    "list_exl3_layers",
    "load_exl3_layer",
]
