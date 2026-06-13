"""Full EXL3 weight reconstruction — inner + outer Hadamard/scales."""

from __future__ import annotations

import numpy as np

from .codebook import codebook_mode_from_flags
from .decode import decode_packed_trellis
from .hadamard import preapply_had_left, preapply_had_right
from .signs import unpack_signs_or_pass


def reconstruct_inner(
    trellis: np.ndarray,
    k: int,
    *,
    mcg: bool = False,
    mul1: bool = False,
    n_offset: int = 0,
    n_count: int | None = None,
) -> np.ndarray:
    cb = codebook_mode_from_flags(mcg=mcg, mul1=mul1)
    return decode_packed_trellis(trellis, k, cb, n_offset=n_offset, n_count=n_count)


def reconstruct_public_weights(
    trellis: np.ndarray,
    suh: np.ndarray | None,
    svh: np.ndarray | None,
    k: int,
    *,
    mcg: bool = False,
    mul1: bool = False,
) -> np.ndarray:
    """
  Reconstruct the public weight matrix W used for matmul.

  Matches LinearEXL3.get_weight_tensor() without loading from packed su/sv.
  """
    w = reconstruct_inner(trellis, k, mcg=mcg, mul1=mul1).astype(np.float32)
    suh_u = unpack_signs_or_pass(suh)
    svh_u = unpack_signs_or_pass(svh)
    if suh_u is not None:
        w = preapply_had_left(w)
        w *= suh_u.reshape(-1, 1).astype(np.float32)
    if svh_u is not None:
        w = preapply_had_right(w)
        w *= svh_u.reshape(1, -1).astype(np.float32)
    return w.astype(np.float16)
