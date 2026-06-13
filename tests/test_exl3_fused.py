"""FusedEXL3Group parity vs individual EXL3Linear members."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.exl3_linear import EXL3Linear
from ponyexl3.mlx.exl3_fused import FusedEXL3Group, fusable
from ponyexl3.ref.synthetic import make_exl3_layer

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]


def _group_layers(k=4, outs=(384, 256, 128), in_features=256, seed0=70):
    layers = []
    for i, out in enumerate(outs):
        l = make_exl3_layer(
            k=k, in_features=in_features, out_features=out, seed=seed0 + i, mcg=True
        )
        l.key = f"fused.test.{k}.{i}"
        layers.append(l)
    return layers


@pytest.mark.parametrize("rows", [1, 8, 19])
def test_group_matches_members(rows: int):
    layers = _group_layers()
    group = FusedEXL3Group(layers)
    members = [EXL3Linear(l) for l in layers]
    x = (mlx.random.normal((rows, 256), key=mlx.random.key(rows)) * 0.5).astype(
        mlx.float16
    )
    outs = group.forward_all(x)
    for o, m, l in zip(outs, members, layers):
        ref = np.array(m(x)).astype(np.float32)
        got = np.array(o).astype(np.float32)
        scale = np.abs(ref).max() + 1e-9
        assert o.shape == (rows, l.out_features)
        assert np.abs(got - ref).max() / scale < 5e-3, l.key


def test_single_slot_cache_computes_once():
    layers = _group_layers(outs=(128, 128))
    group = FusedEXL3Group(layers)
    s0, s1 = group.sibling(0), group.sibling(1)
    calls = {"n": 0}
    orig = group.forward_all

    def counting(x):
        calls["n"] += 1
        return orig(x)

    group.forward_all = counting
    x = mlx.zeros((1, 256), dtype=mlx.float16)
    s0(x)
    s1(x)
    assert calls["n"] == 1
    # new input object -> recompute
    s0(mlx.zeros((1, 256), dtype=mlx.float16))
    assert calls["n"] == 2


def test_fusable_rejects_mixed_k():
    a = _group_layers(k=4, outs=(128,))[0]
    b = _group_layers(k=3, outs=(128,), seed0=80)[0]
    assert not fusable([a, b])
