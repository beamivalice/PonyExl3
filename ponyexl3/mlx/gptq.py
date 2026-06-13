"""GPTQ (Frantar et al.) for re-quantizing decoded EXL3 weights to mlx
affine w4/w8 — the activation-aware upgrade over the RTN ``w4a16`` engine.

Solver runs in numpy fp32 (Accelerate BLAS); the quantization grid mirrors
mlx's affine semantics exactly (q ∈ [0, 2^bits-1], deq = q*scale + bias,
groups along the input dim, little-endian sub-word packing — verified
round-trip vs mx.dequantize), so the artifact loads straight into
``nn.QuantizedLinear`` parts.

Error model context (Phase 20/26 probes): plain RTN w4g64 on W_pub adds
~0.090 rel-RMS on top of ~0.05-0.10 trellis noise (the documented Phase 5
disaster). GPTQ minimizes ||X(W - Ŵ)|| instead of ||W - Ŵ||.
"""

from __future__ import annotations

import numpy as np


def quant_grid(w: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row affine (scale, bias) for one group block ``w (out, g)``,
    fp16-cast like mlx's stored scales."""
    lo = w.min(axis=1, keepdims=True)
    hi = w.max(axis=1, keepdims=True)
    scale = ((hi - lo) / (2**bits - 1)).astype(np.float16).astype(np.float32)
    scale[scale == 0] = 1.0
    bias = lo.astype(np.float16).astype(np.float32)
    return scale, bias


def quant_col(col: np.ndarray, scale: np.ndarray, bias: np.ndarray, bits: int):
    q = np.clip(np.round((col[:, None] - bias) / scale), 0, 2**bits - 1)
    return q[:, 0].astype(np.uint32), (q * scale + bias)[:, 0].astype(np.float32)


def prepare_hinv_u(H: np.ndarray, damp: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Damped inverse-Hessian upper-Cholesky factor (MODIFIES H)."""
    in_f = H.shape[0]
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    H[np.diag_indices(in_f)] += damp * float(np.mean(np.diag(H)))
    Hinv_full = np.linalg.inv(H)
    U = np.linalg.cholesky(Hinv_full).T.copy()
    return U, dead


def gptq_quantize(
    W: np.ndarray,  # (out, in) fp32 — MODIFIED in place
    H: np.ndarray | None = None,  # (in, in) fp32 — MODIFIED in place
    *,
    bits: int = 4,
    group_size: int = 64,
    blocksize: int = 128,
    damp: float = 0.01,
    hinv_u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (q (out,in) uint32 values, scales (out,in/g), biases) fp32.

    Act-order: permute W's columns and H by descending diag(H) BEFORE
    calling (caller-side), and either invert the permutation on the
    dequantized result (eval) or permute the runtime input to match
    (mlx's qmm has no g_idx, so permuted groups can't be stored in
    storage order). Measured +1% over plain GPTQ here — not used.

    Pass a precomputed ``U`` (from :func:`prepare_hinv_u`) to share one
    Hessian factorization across members with the same input (qkv+z,
    gate+up, q/k/v) — the inv+chol dominates solve time at in=17408."""
    out_f, in_f = W.shape
    if hinv_u is None:
        if H is None:
            raise ValueError("H required when hinv_u is not provided")
        hinv_u, dead = prepare_hinv_u(H, damp)
    else:
        dead = np.diag(H) == 0 if H is not None else None
    if dead is not None:
        W[:, dead] = 0.0

    Q = np.zeros((out_f, in_f), dtype=np.uint32)
    scales = np.zeros((out_f, in_f // group_size), dtype=np.float32)
    biases = np.zeros((out_f, in_f // group_size), dtype=np.float32)

    for b0 in range(0, in_f, blocksize):
        b1 = min(b0 + blocksize, in_f)
        Werr = np.zeros((out_f, b1 - b0), dtype=np.float32)
        for j in range(b0, b1):
            if j % group_size == 0:
                g = j // group_size
                sc, bi = quant_grid(W[:, j : j + group_size], bits)
                scales[:, g : g + 1] = sc
                biases[:, g : g + 1] = bi
            g = j // group_size
            q, deq = quant_col(
                W[:, j], scales[:, g : g + 1], biases[:, g : g + 1], bits
            )
            Q[:, j] = q
            err = (W[:, j] - deq) / hinv_u[j, j]
            if j + 1 < b1:
                W[:, j + 1 : b1] -= np.outer(err, hinv_u[j, j + 1 : b1])
            Werr[:, j - b0] = err
            W[:, j] = deq
        if b1 < in_f:
            W[:, b1:] -= Werr @ hinv_u[b0:b1, b1:]
    return Q, scales, biases


def pack_q(Q: np.ndarray, bits: int) -> np.ndarray:
    """uint32 little-endian sub-word packing, mlx layout (verified)."""
    out_f, in_f = Q.shape
    per = 32 // bits
    packed = np.zeros((out_f, in_f // per), dtype=np.uint32)
    for s in range(per):
        packed |= (Q[:, s::per] & ((1 << bits) - 1)) << (s * bits)
    return packed
