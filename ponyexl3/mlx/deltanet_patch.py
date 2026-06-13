"""Compiled-glue patch for mlx_lm's qwen3_5 GatedDeltaNet.

The recurrent layer's forward issues ~9 small dependent ops between the
projections and ``gated_delta_update`` (concat/conv/silu/split/reshape/rms x2)
and ~3 after it — at 30-48 layers that serial dispatch chain IS the decode
wall (measured, Phase 12/15f). This patch (pony's ``integration/qkv_fusion``
pattern) re-implements ``__call__`` with the glue wrapped in two ``mx.compile``
graphs. Bit-exact ops, masked/sharded paths fall back to the original.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ponyexl3.types import KvCache


@lru_cache(maxsize=None)
def _pre_fn(
    key_dim: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    groups: int,
    n_keep: int,
) -> Any:
    inv = head_k_dim**-0.5

    @mx.compile
    def _fn(qkv: mx.array, conv_state: mx.array, conv_w: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        new_state = conv_input[:, -n_keep:, :]
        out = nn.silu(mx.conv1d(conv_input, conv_w, groups=groups))
        B, S = out.shape[0], out.shape[1]
        q, k, v = mx.split(out, [key_dim, 2 * key_dim], axis=-1)
        q2: mx.array = q.reshape(B, S, num_k_heads, head_k_dim)
        k2: mx.array = k.reshape(B, S, num_k_heads, head_k_dim)
        v2: mx.array = v.reshape(B, S, num_v_heads, head_v_dim)
        q = (inv**2) * mx.fast.rms_norm(q2, None, 1e-6)
        k = inv * mx.fast.rms_norm(k2, None, 1e-6)
        v = v2
        return q, k, v, new_state

    return _fn


@lru_cache(maxsize=None)
def _post_fn(eps: float) -> Any:
    @mx.compile
    def _fn(out: mx.array, z: mx.array, w: mx.array) -> mx.array:
        x = mx.fast.rms_norm(out, w, eps)
        gate = nn.silu(z.astype(mx.float32))
        y = (gate * x.astype(mx.float32)).astype(out.dtype)
        B, S = y.shape[0], y.shape[1]
        y2: mx.array = y.reshape(B, S, -1)
        return y2

    return _fn


# Speculative-decode trace hook: while a verify forward runs, the spec loop
# sets a sink list here; ``patched`` appends per-layer scan inputs so a
# partial acceptance can repair the recurrent caches by re-running ONLY
# ``gated_delta_update`` on truncated slices (no EXL3 projections, no conv,
# no out_proj). See generate._DeltaNetTrace.
_trace_sink: list[Any] | None = None


def set_trace_sink(sink: list[Any] | None) -> None:
    global _trace_sink
    _trace_sink = sink


def install_deltanet_glue() -> None:
    from mlx_lm.models import qwen3_5 as q5
    from mlx_lm.models.gated_delta import gated_delta_update

    cls = q5.GatedDeltaNet
    if getattr(cls, "_exl3_glue", False):
        return
    cls._exl3_glue = True  # type: ignore[attr-defined]
    orig = cls.__call__

    def patched(
        self: Any,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: KvCache | None = None,
    ) -> mx.array:
        if (
            mask is not None
            or self.sharding_group is not None
            or (cache is not None and cache.lengths is not None)
        ):
            if _trace_sink is not None:
                _trace_sink.append(("module", self, orig, inputs, cache))
            return orig(self, inputs, mask=mask, cache=cache)
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        if S > 1:
            # MLX's (S,5120)@(5120,32) gemm at S=2..8 is ~8x its S=1 cost
            # (routing cliff); one fused (.,64) matmul for b|a halves the
            # verify's pair (measured 81 vs 177 µs at S=4). S=1 keeps the
            # separate calls so plain-decode bits are untouched.
            wba = getattr(self, "_exl3_wba", None)
            if wba is None:
                wba = mx.concatenate(
                    [self.in_proj_b.weight, self.in_proj_a.weight], axis=0
                ).T
                self._exl3_wba = wba
            ba = inputs @ wba
            b = ba[..., : self.num_v_heads]
            a = ba[..., self.num_v_heads :]
        else:
            b = self.in_proj_b(inputs)
            a = self.in_proj_a(inputs)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        pre = _pre_fn(
            self.key_dim,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_dim,
            self.conv_kernel_size - 1,
        )
        q, k, v, new_conv = pre(qkv, conv_state, self.conv1d.weight)
        if cache is not None:
            cache[0] = mx.contiguous(new_conv)
        state = cache[1] if cache else None
        if _trace_sink is not None:
            _trace_sink.append(
                ("scan", self, cache, inputs, conv_state, qkv, q, k, v, a, b, state)
            )
        out, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, None,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        y = _post_fn(self.norm.eps)(out, z, self.norm.weight)
        return self.out_proj(y)

    cls.__call__ = patched
