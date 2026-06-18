import numpy as np
import pytest

from ponyexl3.ref.trellis import pack_trellis, pack_trellis_tile, unpack_trellis, unpack_trellis_tile

pytestmark = pytest.mark.ponyexl3


def _expected_states(encoded: np.ndarray, k: int) -> np.ndarray:
    """Independent reconstruction of the 16-bit sliding-window states.

    state_t = the 16 bits of the circular tile bitstream ending at bit
    (t+1)*K — i.e. ``state_t = ((state_{t-1} << K) | fresh_t) & 0xFFFF`` —
    matching ``dq`` in ``exllamav3_ext/quant/exl3_dq.cuh``.
    """
    fresh = encoded.astype(np.uint32) & ((1 << k) - 1)
    states = np.zeros(256, dtype=np.uint16)
    state = 0
    for t in list(range(256)) * 2:  # second lap converges the circular seam
        state = ((state << k) | int(fresh[t])) & 0xFFFF
        states[t] = np.uint16(state)
    return states


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
def test_pack_unpack_roundtrip_tile(k):
    rng = np.random.default_rng(k)
    mask = (1 << k) - 1
    encoded = (rng.integers(0, 2**16, size=256, dtype=np.uint32) & mask).astype(np.uint16)
    packed = pack_trellis_tile(encoded, k)
    decoded = unpack_trellis_tile(packed, k)
    # low K bits of each state are that weight's fresh bits
    np.testing.assert_array_equal(decoded & mask, encoded)
    # full 16-bit windows match an independent reconstruction of the stream
    np.testing.assert_array_equal(decoded, _expected_states(encoded, k))


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
def test_unpack_states_shift_consistent(k):
    """Consecutive states overlap by 16-K bits (bitshift trellis property)."""
    rng = np.random.default_rng(1000 + k)
    mask = (1 << k) - 1
    encoded = (rng.integers(0, 2**16, size=256, dtype=np.uint32) & mask).astype(np.uint16)
    states = unpack_trellis_tile(pack_trellis_tile(encoded, k), k).astype(np.uint32)
    keep = (1 << (16 - k)) - 1
    for t in range(256):
        assert states[(t + 1) % 256] >> k == (states[t] & keep), f"t={t}"


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
def test_pack_unpack_roundtrip_tensor(k):
    rng = np.random.default_rng(99 + k)
    mask = (1 << k) - 1
    encoded = (rng.integers(0, 2**16, size=(2, 3, 256), dtype=np.uint32) & mask).astype(np.uint16)
    packed = pack_trellis(encoded, k)
    decoded = unpack_trellis(packed, k)
    np.testing.assert_array_equal(decoded & mask, encoded)
