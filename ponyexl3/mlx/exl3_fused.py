"""Fused EXL3 sibling groups — several same-K linears in one kernel launch.

mlx_lm modules call sibling projections separately on the same input
(``q_proj(x); k_proj(x); v_proj(x)``). Each sibling here is a thin view onto a
shared :class:`FusedEXL3Group` that computes all outputs in ONE prepare + ONE
trellis GEMV + ONE finish when the first sibling is called, then serves the
rest from a single-slot cache keyed on input identity. Per decoder layer this
collapses ~9 dispatch chains into ~3.

Exactness: the group concatenates member trellises along out_tiles (the
decode is per-tile, unaffected), stacks ``suh`` rows so each member's xh is
its own Hadamard transform, and concatenates ``svh`` (member out_features are
multiples of 128, so post-Hadamard blocks never straddle a member boundary).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ponyexl3.mlx.gemv_metal import (
    decode_full_mlx,
    inner_gem_fused_mlx,
    inner_gemv_had_mlx,
    inner_gemv_post_mlx,
)
from ponyexl3.ref.codebook import codebook_mode_from_flags
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.signs import unpack_signs_or_pass

_compiled_fused_prep: Callable[[mx.array, mx.array], mx.array] | None = None
_compiled_fused_finish: Callable[[mx.array, mx.array], mx.array] | None = None


import os

_FUSE_HAD = os.environ.get("EXL3_FUSE_HAD", "0") == "1"
_HAD_SCALE = 1.0 / 128.0**0.5


def _fused_prep():
    """had((x ⊙ suh_i)) for every member i in one fp16 dispatch."""
    global _compiled_fused_prep
    if _compiled_fused_prep is None:

        @mx.compile
        def _fn(x2d: mx.array, suh_stack: mx.array) -> mx.array:
            # x2d: (batch, in); suh_stack: (n_sub, in) -> (batch, n_sub, in)
            xs = x2d.astype(mx.float16)[:, None, :] * suh_stack.astype(mx.float16)[None]
            batch, n_sub, in_features = xs.shape
            xh = mx.hadamard_transform(
                xs.reshape(batch * n_sub, in_features // 128, 128), scale=_HAD_SCALE
            )
            return xh.reshape(batch, n_sub, in_features)

        _compiled_fused_prep = _fn
    return _compiled_fused_prep


def _fused_finish():
    global _compiled_fused_finish
    if _compiled_fused_finish is None:

        @mx.compile
        def _fn(y2d: mx.array, svh_cat: mx.array) -> mx.array:
            rows, n = y2d.shape
            yh = mx.hadamard_transform(
                y2d.astype(mx.float16).reshape(rows, n // 128, 128), scale=_HAD_SCALE
            )
            return yh.reshape(rows, n) * svh_cat.astype(mx.float16)

        _compiled_fused_finish = _fn
    return _compiled_fused_finish


def fusable(layers: list[EXL3Layer]) -> bool:
    first = layers[0]
    return (
        len(layers) >= 2
        and all(l.in_features == first.in_features for l in layers)
        and all(l.k == first.k for l in layers)
        and all((l.mcg, l.mul1) == (first.mcg, first.mul1) for l in layers)
        and all(l.out_features % 128 == 0 for l in layers)
        and all(l.suh is not None and l.svh is not None for l in layers)
        and all(l.bias is None for l in layers)
    )


class FusedEXL3Group(nn.Module):
    """Owner of the concatenated trellis + stacked signs for sibling linears."""

    def __init__(self, layers: list[EXL3Layer]):
        super().__init__()
        if not fusable(layers):
            raise ValueError("layers not fusable: " + ", ".join(l.key for l in layers))
        for l in layers:
            l.validate()
        self._keys = [l.key for l in layers]
        self._k = layers[0].k
        self._cb = codebook_mode_from_flags(mcg=layers[0].mcg, mul1=layers[0].mul1)
        self.in_features = layers[0].in_features
        self._out_features = [l.out_features for l in layers]

        self._trellis = mx.array(
            np.concatenate([l.trellis for l in layers], axis=1).astype(np.uint16)
        )
        suh_rows = [unpack_signs_or_pass(l.suh) for l in layers]
        svh_rows = [unpack_signs_or_pass(l.svh) for l in layers]
        if any(s is None for s in suh_rows) or any(s is None for s in svh_rows):
            raise ValueError("fused group member missing suh/svh")
        self._suh_stack = mx.array(
            np.stack(suh_rows).astype(np.float16)  # type: ignore[arg-type]
        )
        self._svh_cat = mx.array(
            np.concatenate(svh_rows).astype(np.float16)  # type: ignore[arg-type]
        )
        self._tile_sub = mx.array(
            np.repeat(
                np.arange(len(layers), dtype=np.uint32),
                [l.out_features // 16 for l in layers],
            )
        )
        # output split points (mx.split takes boundaries, not sizes)
        self._split_at = [int(v) for v in np.cumsum(self._out_features[:-1])]
        mx.eval(self._trellis, self._suh_stack, self._svh_cat, self._tile_sub)

        self._cache_x: mx.array | None = None
        self._cache_out: tuple[mx.array, ...] | None = None

    def _extra_repr(self) -> str:
        return f"members={self._keys}, k={self._k}"

    def forward_all(self, x: mx.array) -> tuple[mx.array, ...]:
        from ponyexl3.mlx.exl3_linear import FUSED_GEMM_ROW_LIMIT

        in_shape = x.shape
        rows = 1
        for d in in_shape[:-1]:
            rows *= d
        x2d = x.reshape(rows, self.in_features).astype(mx.float16)

        if rows == 1 and _FUSE_HAD:
            y = inner_gemv_had_mlx(
                x2d,
                self._suh_stack,
                self._trellis,
                self._k,
                self._cb,
                self._tile_sub,
                n_sub=len(self._out_features),
            )
            if y is not None:
                y = _fused_finish()(y.astype(mx.float16), self._svh_cat)
                outs = mx.split(y, self._split_at, axis=-1)
                return tuple(
                    o.reshape(in_shape[:-1] + (n,))
                    for o, n in zip(outs, self._out_features)
                )

        xh = _fused_prep()(x2d, self._suh_stack)  # (rows, n_sub, in)
        if rows == 1:
            # post-Hadamard + svh fused into the GEMV epilogue (v18)
            y = inner_gemv_post_mlx(
                xh.reshape(-1, self.in_features),
                self._svh_cat,
                self._trellis,
                self._k,
                self._cb,
                self._tile_sub,
                n_sub=len(self._out_features),
            )
            if y is not None:
                outs = mx.split(y.astype(mx.float16), self._split_at, axis=-1)
                return tuple(
                    o.reshape(in_shape[:-1] + (n,))
                    for o, n in zip(outs, self._out_features)
                )
        if rows <= FUSED_GEMM_ROW_LIMIT:
            y = inner_gem_fused_mlx(xh, self._trellis, self._k, self._cb, self._tile_sub)
        else:
            # Prefill: decode each member once (contiguous out-tile range of
            # the concatenated trellis) and run native matmuls.
            parts = []
            tn0 = 0
            for i, n in enumerate(self._out_features):
                tn_count = n // 16
                w = decode_full_mlx(
                    self._trellis, self._k, self._cb, tn_offset=tn0, tn_count=tn_count
                )
                parts.append(xh[:, i, :] @ w)
                tn0 += tn_count
            y = mx.concatenate(parts, axis=-1)
        y = _fused_finish()(y.astype(mx.float16), self._svh_cat)

        outs = mx.split(y, self._split_at, axis=-1)
        return tuple(
            o.reshape(in_shape[:-1] + (n,))
            for o, n in zip(outs, self._out_features)
        )

    def cached(self, x: mx.array) -> tuple[mx.array, ...]:
        if self._cache_out is None or self._cache_x is not x:
            self._cache_out = self.forward_all(x)
            self._cache_x = x
        return self._cache_out

    def sibling(self, idx: int) -> "FusedEXL3Sibling":
        return FusedEXL3Sibling(self, idx)


class FusedEXL3Sibling(nn.Module):
    """Drop-in replacement for one member linear; delegates to the group."""

    def __init__(self, group: FusedEXL3Group, idx: int):
        super().__init__()
        self._group = group
        self._idx = idx
        self.in_features = group.in_features
        self.out_features = group._out_features[idx]  # pyright: ignore[reportPrivateUsage]

    def _extra_repr(self) -> str:
        return f"{self._group._keys[self._idx]} (member {self._idx})"  # pyright: ignore[reportPrivateUsage]

    def __call__(self, x: mx.array) -> mx.array:
        return self._group.cached(x)[self._idx]


class FusedPlainPair(nn.Module):
    """Two tiny same-input fp16 Linears (DeltaNet's in_proj_b/in_proj_a) fused
    into one matmul; siblings serve from a single-slot input-identity cache
    (pony's qkv_fusion pattern, applied to the unquantized tail)."""

    def __init__(self, a: nn.Linear, b: nn.Linear):
        super().__init__()
        self._w = mx.concatenate([a.weight, b.weight], axis=0)
        self._split = int(a.weight.shape[0])
        self._cache_x = None
        self._cache_out = None

    def outs(self, x: mx.array) -> tuple[mx.array, mx.array]:
        if self._cache_out is None or self._cache_x is not x:
            y = x @ self._w.T
            self._cache_out = (y[..., : self._split], y[..., self._split :])
            self._cache_x = x
        return self._cache_out


class FusedPlainSibling(nn.Module):
    def __init__(self, pair: FusedPlainPair, idx: int):
        super().__init__()
        self._pair = pair
        self._idx = idx

    def __call__(self, x: mx.array) -> mx.array:
        return self._pair.outs(x)[self._idx]
