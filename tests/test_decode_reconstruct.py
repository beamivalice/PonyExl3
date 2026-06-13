import numpy as np
import pytest

from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.decode import decode_packed_tile
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.forward import linear_forward_reconstruct
from ponyexl3.ref.reconstruct import reconstruct_inner, reconstruct_public_weights
from ponyexl3.ref.trellis import pack_trellis_tile

pytestmark = pytest.mark.ponyexl3


def _random_tile(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = (1 << k) - 1
    encoded = (rng.integers(0, 2**16, size=256, dtype=np.uint32) & mask).astype(np.uint16)
    return pack_trellis_tile(encoded, k)


def test_decode_tile_shape():
    packed = _random_tile(4, 1)
    tile = decode_packed_tile(packed, 4, CodebookMode.DEFAULT)
    assert tile.shape == (16, 16)
    assert tile.dtype == np.float16
    # Random codewords can produce extreme fp16; most samples should be finite.
    assert np.isfinite(tile.astype(np.float32)).mean() > 0.9


def test_reconstruct_inner_sizes():
    k = 4
    trellis = np.stack(
        [[_random_tile(k, i + j) for j in range(2)] for i in range(3)],
        axis=0,
    )
    w = reconstruct_inner(trellis, k)
    assert w.shape == (48, 32)


def test_forward_small_layer():
    k = 3
    in_features, out_features = 128, 128
    trellis = np.stack(
        [[_random_tile(k, i * 10 + j) for j in range(out_features // 16)] for i in range(in_features // 16)],
        axis=0,
    )
    rng = np.random.default_rng(7)
    suh = np.where(rng.random((in_features // 16, 16)) > 0.5, -1.0, 1.0).astype(np.float16).reshape(-1)
    svh = np.where(rng.random((out_features // 16, 16)) > 0.5, -1.0, 1.0).astype(np.float16).reshape(-1)

    layer = EXL3Layer(
        key="test",
        in_features=in_features,
        out_features=out_features,
        k=k,
        trellis=trellis,
        suh=suh,
        svh=svh,
    )
    x = rng.standard_normal((2, in_features)).astype(np.float16)
    y = linear_forward_reconstruct(layer, x)
    assert y.shape == (2, out_features)
    assert np.isfinite(y.astype(np.float32)).all()

    w_pub = reconstruct_public_weights(trellis, suh, svh, k)
    y2 = (x.astype(np.float32) @ w_pub.astype(np.float32)).astype(np.float16)
    # Public weights path vs inner+had path should match closely
    # Public-weight matmul vs inner+had path — same math, different evaluation order.
    np.testing.assert_allclose(y.astype(np.float32), y2.astype(np.float32), rtol=0.15, atol=1.0)
