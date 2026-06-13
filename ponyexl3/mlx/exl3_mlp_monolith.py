"""Dense MLP monolith — single module for gate+up+SwiGLU+down.

Default path (``EXL3_MLP_KERNEL=fast``): reuses the production stack —
``FusedEXL3Group.forward_all`` (v12 simd + optional post-fuse) plus
``EXL3Linear`` down — wrapped in one ``nn.Module`` so load can replace
``Qwen3NextMLP`` without changing numerics or speed.

Experimental path (``EXL3_MLP_KERNEL=moe``): MoE gate+up Metal kernel +
mapped down (parity OK on 27B; **slower** than fast on dense MLP — the
expert-scale kernels do not beat v12 post-fused GEMV at hidden=17408).

Enable replacement at load: ``EXL3_MLP_MONO=1``.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ponyexl3.mlx.exl3_fused import FusedEXL3Group, FusedEXL3Sibling
from ponyexl3.mlx.exl3_linear import EXL3Linear
from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU
from ponyexl3.types import MlxLmModel


def _mlp_kernel() -> str:
    return os.environ.get("EXL3_MLP_KERNEL", "fast").lower()


def _mlp_mono_enabled() -> bool:
    return os.environ.get("EXL3_MLP_MONO", "0") == "1"


class EXL3MLPMonolith(nn.Module):
    """Drop-in replacement for ``Qwen3NextMLP`` on EXL3 checkpoints."""

    def __init__(
        self,
        group: FusedEXL3Group,
        down: EXL3Linear,
        *,
        switch: EXL3SwitchGLU | None = None,
    ):
        super().__init__()
        self._group = group
        self._down = down
        self._switch = switch
        self.in_features = group.in_features
        self.hidden_dims = group._out_features[0]  # pyright: ignore[reportPrivateUsage]

    @classmethod
    def from_fused_gate_up(
        cls, group: FusedEXL3Group, down: EXL3Linear
    ) -> "EXL3MLPMonolith":
        switch = None
        if _mlp_kernel() == "moe":
            rt = down._rt  # pyright: ignore[reportPrivateUsage]
            if rt.suh is None or rt.svh is None:
                raise ValueError(f"{down._exl3.key}: down_proj missing suh/svh")  # pyright: ignore[reportPrivateUsage]
            if group._k == 7:  # pyright: ignore[reportPrivateUsage]
                raise ValueError("k=7 gate/up group unsupported by MoE kernels")
            switch = EXL3SwitchGLU(
                gu_trellis=group._trellis,  # pyright: ignore[reportPrivateUsage]
                gu_suh=group._suh_stack.reshape(1, 2, group.in_features),  # pyright: ignore[reportPrivateUsage]
                gu_svh=group._svh_cat.reshape(1, -1),  # pyright: ignore[reportPrivateUsage]
                dn_trellis=rt.trellis,
                dn_suh=rt.suh.reshape(1, -1),
                dn_svh=rt.svh.reshape(1, -1),
                k=group._k,  # pyright: ignore[reportPrivateUsage]
                cb=group._cb,  # pyright: ignore[reportPrivateUsage]
            )
            mx.eval(
                switch._gu_trellis,  # pyright: ignore[reportPrivateUsage]
                switch._gu_suh,  # pyright: ignore[reportPrivateUsage]
                switch._gu_svh,  # pyright: ignore[reportPrivateUsage]
                switch._dn_trellis,  # pyright: ignore[reportPrivateUsage]
                switch._dn_suh,  # pyright: ignore[reportPrivateUsage]
                switch._dn_svh,  # pyright: ignore[reportPrivateUsage]
            )
        return cls(group, down, switch=switch)

    def _fast(self, x: mx.array) -> mx.array:
        in_shape = x.shape
        rows = 1
        for d in in_shape[:-1]:
            rows *= d
        d = in_shape[-1]
        x2d = x.reshape(rows, d)
        gate, up = self._group.forward_all(x2d)
        h = nn.silu(gate) * up
        return self._down(h).reshape(in_shape)

    def _moe(self, x: mx.array) -> mx.array:
        assert self._switch is not None
        in_shape = x.shape
        rows = 1
        for d in in_shape[:-1]:
            rows *= d
        d = in_shape[-1]
        x2d = x.reshape(rows, d)
        sel = mx.zeros((rows,), dtype=mx.int32)

        if rows == 1:
            y = self._switch._decode_fused(x2d, sel)  # pyright: ignore[reportPrivateUsage]
            return y.reshape(in_shape)

        outs = []
        for r in range(rows):
            y = self._switch._decode_fused(x2d[r : r + 1], sel[r : r + 1])  # pyright: ignore[reportPrivateUsage]
            outs.append(y)
        return mx.concatenate(outs, axis=0).reshape(in_shape)

    def __call__(self, x: mx.array) -> mx.array:
        if self._switch is not None:
            return self._moe(x)
        return self._fast(x)


def _mlp_gate_up_group(mlp: Any) -> FusedEXL3Group | None:
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if not isinstance(gate, FusedEXL3Sibling) or not isinstance(up, FusedEXL3Sibling):
        return None
    if gate._group is not up._group or gate._idx != 0 or up._idx != 1:  # pyright: ignore[reportPrivateUsage]
        return None
    return gate._group  # pyright: ignore[reportPrivateUsage]


def install_mlp_monoliths(model: MlxLmModel) -> int:
    """Replace each eligible dense ``mlp`` with :class:`EXL3MLPMonolith`."""
    if not _mlp_mono_enabled():
        return 0

    n = 0
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or isinstance(mlp, EXL3MLPMonolith):
            continue
        down = getattr(mlp, "down_proj", None)
        if not isinstance(down, EXL3Linear):
            continue
        group = _mlp_gate_up_group(mlp)
        if group is None:
            continue
        try:
            mono = EXL3MLPMonolith.from_fused_gate_up(group, down)
        except ValueError:
            continue
        layer.mlp = mono
        n += 1
    return n
