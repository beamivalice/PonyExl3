"""Metal trellis search gates for the converter path."""

from __future__ import annotations

import math

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.convert.direct import quantize_inner_matrix_direct
from ponyexl3.convert.metal_search import _scratch_bytes_per_tile, quantize_tiles_mlx_np
from ponyexl3.convert.mlx_trellis import pack_trellis_mlx
from ponyexl3.convert.reference_search import quantize_tile_reference
from ponyexl3.ref.codebook import CodebookMode, decode_3inst
from ponyexl3.ref.trellis import pack_trellis, pack_trellis_tile, unpack_trellis_tile

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]


def _assert_tail_biting(states: np.ndarray, k: int) -> None:
    first = int(states[0])
    last = int(states[-1])
    assert (first >> k) == (last & ((1 << (16 - k)) - 1))


def _valid_tail_biting_indices(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    encoded = rng.integers(0, 65535, size=256, dtype=np.uint32)
    mask = (1 << k) - 1
    for i in range(256):
        x = int(encoded[i]) & mask
        for shift in range(1, int(math.ceil(16 / k))):
            j = (i + 256 - shift) % 256
            y = int(encoded[j]) & mask
            x |= y << (k * shift)
        encoded[i] = x & 0xFFFF
    return encoded.astype(np.uint16)


@pytest.mark.parametrize(
    ("k", "cb"),
    [
        (2, CodebookMode.MCG),
        (3, CodebookMode.MCG),
        (4, CodebookMode.DEFAULT),
        (4, CodebookMode.MCG),
        (4, CodebookMode.MUL1),
    ],
)
def test_quantize_tiles_mlx_random_tile_matches_reference_quality(
    k: int, cb: CodebookMode
) -> None:
    scale = 1.0 if cb == CodebookMode.MUL1 else 1.24371088
    rng = np.random.default_rng(100 + 10 * k + int(cb))
    tile = (rng.standard_normal((1, 256)) * scale).astype(np.float32)

    q_tiles, states = quantize_tiles_mlx_np(tile, k, cb)
    _ref_states, ref_tile = quantize_tile_reference(tile[0], k, cb)

    _assert_tail_biting(states[0], k)
    packed = pack_trellis_tile((states[0] & ((1 << k) - 1)).astype(np.uint16), k)
    assert np.array_equal(unpack_trellis_tile(packed, k), states[0])

    metal_mse = float(np.mean((q_tiles[0] - tile[0]) ** 2))
    ref_mse = float(np.mean((ref_tile - tile[0]) ** 2))
    assert metal_mse <= ref_mse * 1.10


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8])
def test_quantize_tiles_mlx_recovers_ideal_tail_biting_tile(k: int) -> None:
    encoded = _valid_tail_biting_indices(k, seed=k)
    decoded = np.array(
        [decode_3inst(int(code), CodebookMode.MCG) for code in encoded],
        dtype=np.float32,
    ).reshape(1, 256)

    q_tiles, states = quantize_tiles_mlx_np(decoded, k, CodebookMode.MCG)

    _assert_tail_biting(states[0], k)
    assert np.array_equal(states[0], encoded)
    assert np.array_equal(q_tiles[0], decoded[0])


def test_quantize_tiles_mlx_chunks_large_batches_by_scratch_budget() -> None:
    rng = np.random.default_rng(333)
    tiles = rng.standard_normal((13, 256)).astype(np.float32)
    q_tiles, states = quantize_tiles_mlx_np(
        tiles,
        2,
        CodebookMode.MCG,
        max_scratch_bytes=3 * _scratch_bytes_per_tile(2),
    )

    assert q_tiles.shape == tiles.shape
    assert states.shape == tiles.shape
    assert np.isfinite(q_tiles).all()
    for row in states:
        _assert_tail_biting(row, 2)


def test_quantize_tiles_mlx_rejects_k1_until_low_k_strategy_exists() -> None:
    tile = np.zeros((1, 256), dtype=np.float32)
    with pytest.raises(ValueError, match=r"supports K in \[2, 8\]"):
        quantize_tiles_mlx_np(tile, 1, CodebookMode.MCG)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
def test_pack_trellis_mlx_matches_numpy_reference(k: int) -> None:
    rng = np.random.default_rng(800 + k)
    encoded = rng.integers(0, 1 << k, size=(3, 5, 256), dtype=np.uint16)

    packed = pack_trellis_mlx(mlx.array(encoded), k)
    mlx.eval(packed)

    np.testing.assert_array_equal(np.array(packed), pack_trellis(encoded, k))


def test_quantize_inner_matrix_direct_metal_no_states_matches_debug_path() -> None:
    rng = np.random.default_rng(909)
    inner = rng.standard_normal((32, 48)).astype(np.float32)

    packed_debug, states_debug, reconstructed_debug = quantize_inner_matrix_direct(
        inner,
        k=4,
        cb=CodebookMode.MCG,
        search_backend="metal",
        return_states=True,
    )
    packed_fast, states_fast, reconstructed_fast = quantize_inner_matrix_direct(
        inner,
        k=4,
        cb=CodebookMode.MCG,
        search_backend="metal",
        return_states=False,
    )

    assert states_debug is not None
    assert states_fast is None
    np.testing.assert_array_equal(packed_fast, packed_debug)
    np.testing.assert_array_equal(reconstructed_fast, reconstructed_debug)
