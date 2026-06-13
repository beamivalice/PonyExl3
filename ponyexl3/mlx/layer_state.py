"""Per-layer MLX runtime state — device-resident trellis, signs, decoded weights."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import math
import os
from dataclasses import dataclass

import numpy as np
import mlx.core as mx

from ponyexl3.mlx.decode import decode_inner_cols_mlx
from ponyexl3.mlx.hadamard import had_r_128_mlx
from ponyexl3.mlx.reconstruct import reconstruct_inner_mlx
from ponyexl3.mlx.signs import unpack_signs_or_pass_mlx
from ponyexl3.ref.codebook import CodebookMode, codebook_mode_from_flags
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.signs import unpack_signs_or_pass


def _layer_key(layer: EXL3Layer) -> tuple[Any, ...]:
    # ``id(layer)`` disambiguates per object without reading the trellis, so
    # the host-side numpy can be released post-load (EXL3Linear.release_source)
    # while the runtime cache still keys correctly. Each EXL3Layer lives for
    # the model's lifetime (held by its EXL3Linear), so the id is stable.
    return (
        id(layer),
        layer.key,
        layer.in_features,
        layer.out_features,
        layer.k,
        layer.mcg,
        layer.mul1,
    )


_HAD_SCALE = 1.0 / math.sqrt(128)
# fp16 rotations (the turboquant lesson: never promote activations to fp32 —
# pony/docs/turboquant_vs_pony.md; exllamav3's CUDA rotations are fp16 too).
# EXL3_HAD_F32=1 restores the fp32 path for debugging.
_HAD_F32 = os.environ.get("EXL3_HAD_F32", "0") == "1"

_compiled_had: dict[str, Callable[..., mx.array]] = {}


def _had_fn(kind: str) -> Callable[..., mx.array]:
    """mx.compile'd pre/post Hadamard blocks — one fused dispatch, fp16."""
    fn = _compiled_had.get(kind)
    if fn is None:
        if _HAD_F32:
            if kind == "pre_scaled":

                @mx.compile
                def fn_pre(x: mx.array, s: mx.array) -> mx.array:
                    return had_r_128_mlx(
                        x.astype(mx.float32), pre_scale=s, r_scale=1.0
                    ).astype(mx.float16)

                fn = fn_pre
            elif kind == "post_scaled":

                @mx.compile
                def fn_post(y: mx.array, s: mx.array) -> mx.array:
                    return had_r_128_mlx(
                        y.astype(mx.float32), post_scale=s, r_scale=1.0
                    ).astype(mx.float16)

                fn = fn_post
            else:

                @mx.compile
                def fn_plain(x: mx.array) -> mx.array:
                    return had_r_128_mlx(x.astype(mx.float32), r_scale=1.0).astype(
                        mx.float16
                    )

                fn = fn_plain

        elif kind == "pre_scaled":

            @mx.compile
            def fn_pre(x: mx.array, s: mx.array) -> mx.array:
                rows, n = x.shape
                xs = (x.astype(mx.float16) * s.astype(mx.float16)).reshape(
                    rows, n // 128, 128
                )
                return mx.hadamard_transform(xs, scale=_HAD_SCALE).reshape(rows, n)

            fn = fn_pre

        elif kind == "post_scaled":

            @mx.compile
            def fn_post(y: mx.array, s: mx.array) -> mx.array:
                rows, n = y.shape
                yh = mx.hadamard_transform(
                    y.astype(mx.float16).reshape(rows, n // 128, 128),
                    scale=_HAD_SCALE,
                )
                return yh.reshape(rows, n) * s.astype(mx.float16)

            fn = fn_post

        else:

            @mx.compile
            def fn_plain(x: mx.array) -> mx.array:
                rows, n = x.shape
                xs = x.astype(mx.float16).reshape(rows, n // 128, 128)
                return mx.hadamard_transform(xs, scale=_HAD_SCALE).reshape(rows, n)

            fn = fn_plain

        _compiled_had[kind] = fn
    return fn


@dataclass(frozen=True)
class EXL3LayerRuntime:
    """MLX arrays kept on device for one EXL3 layer."""

    key: tuple[Any, ...]
    trellis: mx.array
    suh: mx.array | None
    svh: mx.array | None
    bias: mx.array | None
    k: int
    cb: CodebookMode

    def prepare_xh(self, x2d: mx.array) -> mx.array:
        if self.suh is None:
            return _had_fn("plain")(x2d)
        return _had_fn("pre_scaled")(x2d, self.suh)

    def finish_y(self, y: mx.array) -> mx.array:
        if self.svh is None:
            return _had_fn("plain")(y)
        return _had_fn("post_scaled")(y, self.svh)


_runtime_cache: dict[tuple[Any, ...], EXL3LayerRuntime] = {}
_inner_cache: dict[tuple[Any, ...], mx.array] = {}
_stripe_cache: dict[tuple[Any, ...], mx.array] = {}


def layer_runtime_mlx(layer: EXL3Layer) -> EXL3LayerRuntime:
    """Trellis + signs + bias uploaded once per layer buffer."""
    key = _layer_key(layer)
    if key in _runtime_cache:
        return _runtime_cache[key]

    suh_np = unpack_signs_or_pass(layer.suh)
    svh_np = unpack_signs_or_pass(layer.svh)
    rt = EXL3LayerRuntime(
        key=key,
        trellis=mx.array(layer.trellis),  # already uint16 — no redundant host copy
        suh=None if suh_np is None else unpack_signs_or_pass_mlx(mx.array(suh_np)),
        svh=None if svh_np is None else unpack_signs_or_pass_mlx(mx.array(svh_np)),
        bias=None if layer.bias is None else mx.array(layer.bias.astype(np.float16)),
        k=layer.k,
        cb=codebook_mode_from_flags(mcg=layer.mcg, mul1=layer.mul1),
    )
    _runtime_cache[key] = rt
    return rt


def pin_runtime(layer: EXL3Layer, rt: EXL3LayerRuntime) -> None:
    """Seed the runtime cache for ``layer`` with an already-built ``rt`` so a
    later ``layer_runtime_mlx(layer)`` (e.g. the lm_head stripe path) hits the
    cache instead of re-deriving from ``layer.trellis`` — which lets
    EXL3Linear.release_source drop that host-side numpy."""
    _runtime_cache[_layer_key(layer)] = rt


def inner_weight_mlx(layer: EXL3Layer, *, use_cache: bool = True) -> mx.array:
    """Full decoded inner ``W``, cached per trellis buffer."""
    key = _layer_key(layer)
    if use_cache and key in _inner_cache:
        return _inner_cache[key]
    rt = layer_runtime_mlx(layer)
    w = reconstruct_inner_mlx(
        layer.trellis,
        rt.k,
        mcg=layer.mcg,
        mul1=layer.mul1,
    )
    if use_cache:
        _inner_cache[key] = w
    return w


def stripe_weight_mlx(
    layer: EXL3Layer,
    n_offset: int,
    n_count: int,
    *,
    use_cache: bool = True,
) -> mx.array:
    """Decoded ``W[:, n_offset:n_offset+n_count]``, cached per stripe."""
    key = (_layer_key(layer), n_offset, n_count)
    if use_cache and key in _stripe_cache:
        return _stripe_cache[key]
    rt = layer_runtime_mlx(layer)
    w = decode_inner_cols_mlx(rt.trellis, rt.k, rt.cb, n_offset, n_count)
    if use_cache:
        _stripe_cache[key] = w
    return w


def warm_layer_mlx(
    layer: EXL3Layer,
    *,
    inner: bool = True,
    stripes: bool = False,
    stripe_cols: int = 512,
) -> EXL3LayerRuntime:
    """
    Pre-upload trellis/signs and optionally decode weights before the first forward.

    For layers above 64 MiB decoded, warms stripe caches instead of full ``W``.
    """
    from ponyexl3.mlx.stripe import DEFAULT_STRIPE_COLS

    if stripe_cols % 128 != 0:
        raise ValueError("stripe_cols must be a multiple of 128")
    cols = stripe_cols or DEFAULT_STRIPE_COLS

    rt = layer_runtime_mlx(layer)
    nbytes = layer.in_features * layer.out_features * 2
    huge = nbytes > 64 * 1024 * 1024

    if inner and not huge:
        inner_weight_mlx(layer)
    if stripes or (inner and huge):
        for n0 in range(0, layer.out_features, cols):
            n1 = min(n0 + cols, layer.out_features)
            stripe_weight_mlx(layer, n0, n1 - n0)
    return rt


def clear_layer_caches() -> None:
    _runtime_cache.clear()
    _inner_cache.clear()
    _stripe_cache.clear()


# Back-compat alias used by tests
clear_inner_weight_cache = clear_layer_caches
