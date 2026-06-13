"""MLX-native fused linear ops — ``matmul``, ``mx.compile`` prefill blocks."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from ponyexl3.mlx.hadamard import had_r_128_mlx

_compiled_post_had: Callable[[mx.array, mx.array], mx.array] | None = None
_compiled_prefill_signed: Callable[[mx.array, mx.array, mx.array, mx.array], mx.array] | None = None


def _get_compiled_post_had():
    global _compiled_post_had
    if _compiled_post_had is None:

        @mx.compile
        def _fn(y: mx.array, svh: mx.array) -> mx.array:
            return had_r_128_mlx(y, post_scale=svh, r_scale=1.0)

        _compiled_post_had = _fn
    return _compiled_post_had


def _get_compiled_prefill_signed():
    global _compiled_prefill_signed
    if _compiled_prefill_signed is None:

        @mx.compile
        def _fn(x: mx.array, w: mx.array, suh: mx.array, svh: mx.array) -> mx.array:
            xh = had_r_128_mlx(x.astype(mx.float32), pre_scale=suh, r_scale=1.0).astype(
                mx.float16
            )
            y = (xh.astype(mx.float32) @ w.astype(mx.float32)).astype(mx.float16)
            y = had_r_128_mlx(y.astype(mx.float32), post_scale=svh, r_scale=1.0).astype(
                mx.float16
            )
            return y

        _compiled_prefill_signed = _fn
    return _compiled_prefill_signed


def inner_matmul_mlx(
    xh: mx.array,
    w: mx.array,
    *,
    svh: mx.array | None = None,
    use_compile: bool = True,
) -> mx.array:
    """``xh @ w`` (fp32 MLX GEMM), fp16 cast, then optional post-Hadamard (matches reconstruct)."""
    y = (xh.astype(mx.float32) @ w.astype(mx.float32)).astype(mx.float16)
    if svh is None:
        return y
    y32 = y.astype(mx.float32)
    if use_compile:
        y32 = _get_compiled_post_had()(y32, svh)
    else:
        y32 = had_r_128_mlx(y32, post_scale=svh, r_scale=1.0)
    return y32.astype(mx.float16)


def prefill_matmul_mlx(
    x2d: mx.array,
    w: mx.array,
    suh: mx.array | None,
    svh: mx.array | None,
    *,
    use_compile: bool = True,
) -> mx.array:
    """Compiled block: pre-Hadamard + ``matmul`` + post-Hadamard (when signs present)."""
    if use_compile and suh is not None and svh is not None:
        return _get_compiled_prefill_signed()(x2d, w, suh, svh)

    xh = had_r_128_mlx(x2d.astype(mx.float32), pre_scale=suh, r_scale=1.0).astype(mx.float16)
    return inner_matmul_mlx(xh, w, svh=svh, use_compile=use_compile)
