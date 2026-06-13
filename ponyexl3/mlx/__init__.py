"""MLX ops for EXL3 (optional — requires mlx on Apple Silicon)."""

from .forward import linear_forward_mlx, linear_forward_public_mlx
from .hadamard import had_r_128_mlx, preapply_had_left_mlx, preapply_had_right_mlx
from .linear import compare_numpy_vs_mlx, mlx_available
from .reconstruct import reconstruct_inner_mlx, reconstruct_public_mlx
from .signs import unpack_sign_bitfield_mlx, unpack_signs_or_pass_mlx

__all__ = [
    "compare_numpy_vs_mlx",
    "had_r_128_mlx",
    "linear_forward_mlx",
    "linear_forward_public_mlx",
    "mlx_available",
    "preapply_had_left_mlx",
    "preapply_had_right_mlx",
    "reconstruct_inner_mlx",
    "reconstruct_public_mlx",
    "unpack_sign_bitfield_mlx",
    "unpack_signs_or_pass_mlx",
]
