"""EXL3 quantized linear as an MLX ``nn.Module`` — activations stay on device.

Unlike the functional forwards in ``dispatch.py`` (which accept numpy and force a
sync per call), ``EXL3Linear`` takes and returns ``mx.array`` so a whole model
forward builds one lazy graph. Routing per batch size:

- layer fits in memory      → decode-once cached ``W`` + compiled ``matmul``
- huge layer (lm_head), M=1 → fused Metal GEMV (no weight materialize)
- huge layer, M ≤ 144       → fused Metal GEMM (v5 grid-parallel batch)
- huge layer, M > 144       → striped decode + ``matmul`` per 512-col chunk
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ponyexl3.mlx.gemv_metal import (
    decode_full_mlx,
    inner_gemm_mlx,
    inner_gemv_had_mlx,
    inner_gemv_mlx,
    inner_gemv_post_mlx,
)
from ponyexl3.mlx.layer_state import (
    inner_weight_mlx,
    layer_runtime_mlx,
    stripe_weight_mlx,
)
from ponyexl3.mlx.ops import prefill_matmul_mlx
from ponyexl3.mlx.stripe import DEFAULT_STRIPE_COLS
from ponyexl3.ref.layer import EXL3Layer

import os

_FUSE_HAD = os.environ.get("EXL3_FUSE_HAD", "0") == "1"
# The persistent fp16 W cache for non-huge layers is OFF by default
# (EXL3_WCACHE=1 restores it): measured on 35B-A3B AND 27B, the
# trellis-direct prefill ladder (fused GEMM / transient decode_full+matmul)
# matches or beats the cached matmul, while the cache costs 2.56 GB resident
# on 35B (more at 27B shapes). Decode (rows=1) and verify (rows<=8) never
# read it. Phase 26 measurement; supersedes the Phase 5 "cached W wins"
# lesson — the kernels caught up.
_WCACHE = os.environ.get("EXL3_WCACHE", "0") == "1"

HUGE_WEIGHT_BYTES = 64 * 1024 * 1024
# Above this batch, decode-once (v13) + native matmul beats the fused GEMM,
# which re-reads the trellis once per M_TILE (8) rows (measured ~64-row
# crossover on M5 Max for 27B-scale layers).
FUSED_GEMM_ROW_LIMIT = 64
# Don't materialize transient fp16 W beyond this (lm_head-scale layers keep
# the striped path).
DECODE_FULL_MAX_BYTES = 1536 * 1024 * 1024


class EXL3Linear(nn.Module):
    """One EXL3-quantized linear layer (trellis stays on device, no fp16 ``weight``)."""

    def __init__(self, layer: EXL3Layer):
        super().__init__()
        layer.validate()
        # Underscore attrs are invisible to the MLX parameter tree, so
        # ``load_weights(strict=True)`` never expects a ``weight`` here.
        self._exl3 = layer
        self._rt = layer_runtime_mlx(layer)
        self._huge = layer.in_features * layer.out_features * 2 > HUGE_WEIGHT_BYTES
        self.in_features = layer.in_features
        self.out_features = layer.out_features

    def _extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"k={self._exl3.k}, huge={self._huge}"
        )

    def warm(self) -> "EXL3Linear":
        """Pre-decode the cached fp16 ``W`` (no-op for huge layers and
        under ``EXL3_WCACHE=0``)."""
        if _WCACHE and not self._huge:
            mx.eval(inner_weight_mlx(self._exl3))
        return self

    def release_source(self) -> "EXL3Linear":
        """Drop the host-side numpy trellis once the device runtime owns the
        weights. On unified memory that numpy is dead weight competing with the
        KV cache (~30% of a 27B's footprint → ~225k tokens of context). The
        device runtime is pinned in the cache first so the stripe / lm_head
        path still resolves without re-deriving from it. No-op under
        ``EXL3_WCACHE`` (which reconstructs the fp16 ``W`` from the numpy)."""
        if _WCACHE or getattr(self._exl3, "trellis", None) is None:
            return self
        from ponyexl3.mlx.layer_state import pin_runtime

        pin_runtime(self._exl3, self._rt)
        self._exl3.trellis = None
        return self

    def __call__(self, x: mx.array) -> mx.array:
        rt = self._rt
        in_shape = x.shape
        rows = 1
        for d in in_shape[:-1]:
            rows *= d
        x2d = x.reshape(rows, self.in_features).astype(mx.float16)

        if rows == 1:
            # Trellis-direct GEMV for every layer: reads ~4 bpw instead of the
            # 16-bpw decoded W. The post-Hadamard + svh fuse into the kernel
            # epilogue (v18) when the shape allows. (Fusing the PRE-Hadamard
            # was measured SLOWER — its threadgroup x buffer tanks occupancy;
            # EXL3_FUSE_HAD=1 re-enables that experiment.)
            if _FUSE_HAD and rt.suh is not None:
                y = inner_gemv_had_mlx(x2d, rt.suh, rt.trellis, rt.k, rt.cb)
                if y is not None:
                    y = rt.finish_y(y.reshape(1, self.out_features).astype(mx.float16))
                    if rt.bias is not None:
                        y = y + rt.bias
                    return y.reshape(in_shape[:-1] + (self.out_features,))
            xh = rt.prepare_xh(x2d)
            if rt.svh is not None:
                y = inner_gemv_post_mlx(xh, rt.svh, rt.trellis, rt.k, rt.cb)
            else:
                y = None
            if y is None:
                y = inner_gemv_mlx(xh.reshape(-1), rt.trellis, rt.k, rt.cb)
                y = rt.finish_y(y.reshape(1, self.out_features).astype(mx.float16))
            else:
                y = y.astype(mx.float16)
        elif rows <= 16:
            # Small batches (batched serving, speculative verify) ride the
            # barrier-free simd GEMM — the cached-matmul path below pays an
            # fp32 weight cast per call (measured 8x slower at rows=2).
            # Rows 9-16 take the v20 devx kernel's second row group (Phase
            # 28b; falls back to the staged kernel when devx is off or k=7).
            xh = rt.prepare_xh(x2d)
            y = rt.finish_y(inner_gemm_mlx(xh, rt.trellis, rt.k, rt.cb).astype(mx.float16))
        elif _WCACHE and not self._huge:
            w = inner_weight_mlx(self._exl3)
            y = prefill_matmul_mlx(x2d, w, rt.suh, rt.svh)
        elif rows <= FUSED_GEMM_ROW_LIMIT:
            # The fused GEMM re-reads the trellis once per 8 batch rows; only
            # worth it below the decode-once crossover.
            xh = rt.prepare_xh(x2d)
            y = rt.finish_y(inner_gemm_mlx(xh, rt.trellis, rt.k, rt.cb).astype(mx.float16))
        elif self.in_features * self.out_features * 2 <= DECODE_FULL_MAX_BYTES:
            # Prefill: one-dispatch full decode (transient fp16 W), then the
            # GEMM runs on MLX's native matmul (tensor units).
            w = decode_full_mlx(rt.trellis, rt.k, rt.cb)
            y = rt.finish_y((rt.prepare_xh(x2d) @ w).astype(mx.float16))
        else:
            # Audit v2 P1: full-model() consumers hit this with M>64 on
            # lm_head (485 stripes, ~600 ms cold). Cache stripes so repeated
            # full-sequence forwards (teacher-forcing, eval harnesses) pay
            # once; the generate path never reaches here.
            xh32 = rt.prepare_xh(x2d).astype(mx.float32)
            chunks = []
            for n0 in range(0, self.out_features, DEFAULT_STRIPE_COLS):
                n = min(DEFAULT_STRIPE_COLS, self.out_features - n0)
                w = stripe_weight_mlx(self._exl3, n0, n, use_cache=True)
                chunks.append((xh32 @ w.astype(mx.float32)).astype(mx.float16))
            y = rt.finish_y(mx.concatenate(chunks, axis=1))

        if rt.bias is not None:
            y = y + rt.bias
        return y.reshape(in_shape[:-1] + (self.out_features,))
