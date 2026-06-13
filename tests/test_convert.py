"""Converter M1 gates: reference trellis search + pack round-trip."""

import numpy as np
import pytest

from ponyexl3.convert.reference_search import quantize_tile_reference
from ponyexl3.ref.trellis import pack_trellis_tile, unpack_trellis_tile


@pytest.mark.parametrize("k", [2, 3])
def test_search_pack_roundtrip(k):
    rng = np.random.default_rng(7)
    w = rng.standard_normal(256).astype(np.float32)
    states, decoded = quantize_tile_reference(w, k=k)
    # tail-biting transition invariant (checkpoint-validated convention)
    s = states.astype(np.uint32)
    nxt = np.roll(s, -1)
    assert ((((s << k) | (nxt & ((1 << k) - 1))) & 0xFFFF) == nxt).all()
    # bit-exact round-trip through the inference-side pack/unpack
    packed = pack_trellis_tile((states & ((1 << k) - 1)).astype(np.uint16), k)
    assert (unpack_trellis_tile(packed, k).astype(np.uint16) == states).all()
    # quantization quality sanity (QTIP-class MSE on unit Gaussian)
    mse = float(((decoded - w) ** 2).mean())
    assert mse < {2: 0.11, 3: 0.032}[k]
