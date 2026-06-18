"""EXL3 trellis pack/unpack — port of exllamav3_ext/quant/pack.cu (CPU reference)."""

from __future__ import annotations

import numpy as np


def _funnel_shift_r(b: np.uint32, a: np.uint32, shift: int) -> np.uint32:
    merged = (np.uint64(a) << 32) | np.uint64(b)
    return np.uint32(merged >> shift)


def pack_trellis_tile(encoded: np.ndarray, k: int) -> np.ndarray:
    """
  Pack one 16x16 tile of K-bit codewords.

  encoded: (256,) uint16 — values use only the low K bits.
  returns: (256 * k // 16,) uint16 bitstream tile
  """
    if encoded.shape != (256,):
        raise ValueError(f"expected encoded shape (256,), got {encoded.shape}")
    if not 1 <= k <= 8:
        raise ValueError(f"K must be in [1, 8], got {k}")

    packed_size = 256 * k // 16
    s_unpacked = encoded.astype(np.uint16, copy=False)
    s_packed = np.zeros(packed_size, dtype=np.uint16)

    spans = 16
    length = 256 // spans
    for t in range(spans):
        i = length * t
        j = k * t
        buf = np.uint32(0)
        bit_pos = 32
        for _ in range(length):
            v = np.uint32(s_unpacked[i]) & np.uint32((1 << k) - 1)
            bit_pos -= k
            buf |= v << bit_pos
            if bit_pos <= 16:
                s_packed[j] = np.uint16(buf >> 16)
                buf = np.uint32(buf << 16)
                bit_pos += 16
                j += 1
            i += 1

    packed_u32 = s_packed.view(np.uint32)
    packed_u32[:] = (packed_u32 << 16) | (packed_u32 >> 16)  # SWAP16 per uint32
    return s_packed


def _pack_trellis_div16(encoded: np.ndarray, k: int) -> np.ndarray:
    """Vectorized pack path for K values that evenly divide 16."""

    vals_per_word = 16 // k
    words_per_span = k
    mask = np.uint32((1 << k) - 1)
    vals = (encoded.astype(np.uint32, copy=False) & mask).reshape(
        *encoded.shape[:-1],
        16,
        words_per_span,
        vals_per_word,
    )
    shifts = (np.arange(vals_per_word - 1, -1, -1, dtype=np.uint32) * np.uint32(k)).reshape(
        *((1,) * (vals.ndim - 1)),
        vals_per_word,
    )
    words = np.sum(vals << shifts, axis=-1, dtype=np.uint32).astype(np.uint16, copy=False)
    flat = words.reshape(*encoded.shape[:-1], 256 * k // 16).copy()
    pairs = flat.reshape(*flat.shape[:-1], flat.shape[-1] // 2, 2)
    pairs[...] = pairs[..., ::-1]
    return flat


def _pack_trellis_general(encoded: np.ndarray, k: int) -> np.ndarray:
    """Vectorized pack path for K values that do not evenly divide 16."""

    mask = np.uint32((1 << k) - 1)
    vals = (encoded.astype(np.uint32, copy=False) & mask).reshape(
        *encoded.shape[:-1],
        16,
        16,
    )
    words = np.zeros((*encoded.shape[:-1], 16, k), dtype=np.uint32)
    for i in range(16):
        bit_start = i * k
        word = bit_start // 16
        offset = bit_start % 16
        v = vals[..., i]
        if offset + k <= 16:
            words[..., word] |= v << np.uint32(16 - offset - k)
        else:
            high_bits = 16 - offset
            low_bits = k - high_bits
            words[..., word] |= v >> np.uint32(low_bits)
            words[..., word + 1] |= (v & np.uint32((1 << low_bits) - 1)) << np.uint32(
                16 - low_bits
            )
    flat = words.astype(np.uint16, copy=False).reshape(*encoded.shape[:-1], 256 * k // 16).copy()
    pairs = flat.reshape(*flat.shape[:-1], flat.shape[-1] // 2, 2)
    pairs[...] = pairs[..., ::-1]
    return flat


def unpack_trellis_tile(packed: np.ndarray, k: int) -> np.ndarray:
    """
  Unpack one trellis tile to 256 uint16 codewords.

  Each codeword is the full 16-bit sliding window of the tile bitstream ending
  at that weight's K fresh bits (bitshift trellis), matching ``dq``/``dq2`` in
  ``exllamav3_ext/quant/exl3_dq.cuh``.

  packed: (256 * k // 16,) uint16
  returns: (256,) uint16
  """
    packed_size = 256 * k // 16
    if packed.shape != (packed_size,):
        raise ValueError(f"expected packed shape ({packed_size},), got {packed.shape}")

    s_packed = packed.astype(np.uint16, copy=True)
    words = s_packed.view(np.uint32)
    mask_words = (k * 256) // 32

    decoded = np.zeros(128, dtype=np.uint32)
    for t in range(128):
        b0 = t * 2 * k + k - 16 + 256 * k
        b1 = b0 + k
        b2 = b1 + 16
        i0 = b0 // 32
        i1 = (b2 - 1) // 32
        s1 = (i1 + 1) * 32 - b2
        a = words[i0 % mask_words]
        b = words[i1 % mask_words]
        w1 = _funnel_shift_r(b, a, s1)
        w0 = (w1 >> k) & 0xFFFF
        w1 &= 0xFFFF
        decoded[t] = (np.uint32(w1) << 16) | np.uint32(w0)

    # Codewords are full 16-bit sliding windows of the bitstream (exl3_dq.cuh:
    # ``__funnelshift_r(b, a, s0) & 0xffff``), NOT the low K bits per weight.
    return decoded.view(np.uint16).reshape(256).copy()


def pack_trellis(encoded: np.ndarray, k: int) -> np.ndarray:
    """Pack a trellis tensor of shape (tiles_k, tiles_n, 256)."""
    if encoded.ndim != 3 or encoded.shape[-1] != 256:
        raise ValueError(f"expected shape (tiles_k, tiles_n, 256), got {encoded.shape}")
    if k in (1, 2, 4, 8):
        return _pack_trellis_div16(encoded, k)
    return _pack_trellis_general(encoded, k)


def _unpack_trellis_div16(packed: np.ndarray, k: int) -> np.ndarray:
    """Vectorized unpack path for K values that evenly divide 16."""

    vals_per_word = 16 // k
    words_per_span = k
    mask = np.uint16((1 << k) - 1)
    flat = packed.astype(np.uint16, copy=True)
    pairs = flat.reshape(*flat.shape[:-1], flat.shape[-1] // 2, 2)
    pairs[...] = pairs[..., ::-1]
    words = flat.reshape(*flat.shape[:-1], 16, words_per_span)
    shifts = (np.arange(vals_per_word - 1, -1, -1, dtype=np.uint16) * np.uint16(k)).reshape(
        *((1,) * words.ndim),
        vals_per_word,
    )
    vals = ((words[..., None] >> shifts) & mask).reshape(*packed.shape[:-1], 256)
    out = np.zeros_like(vals, dtype=np.uint16)
    for lag in range(vals_per_word):
        out |= (np.roll(vals, lag, axis=-1).astype(np.uint16) << np.uint16(lag * k))
    return out


def _unpack_trellis_general(packed: np.ndarray, k: int) -> np.ndarray:
    """Vectorized unpack path for K values that do not evenly divide 16."""

    flat = packed.astype(np.uint16, copy=True)
    pairs = flat.reshape(*flat.shape[:-1], flat.shape[-1] // 2, 2)
    pairs[...] = pairs[..., ::-1]
    words = flat.reshape(*flat.shape[:-1], 16, k).astype(np.uint32, copy=False)
    fresh = np.empty((*packed.shape[:-1], 16, 16), dtype=np.uint32)
    mask = np.uint32((1 << k) - 1)
    for i in range(16):
        bit_start = i * k
        word = bit_start // 16
        offset = bit_start % 16
        if offset + k <= 16:
            fresh[..., i] = (words[..., word] >> np.uint32(16 - offset - k)) & mask
        else:
            high_bits = 16 - offset
            low_bits = k - high_bits
            high = (words[..., word] & np.uint32((1 << high_bits) - 1)) << np.uint32(low_bits)
            low = words[..., word + 1] >> np.uint32(16 - low_bits)
            fresh[..., i] = (high | low) & mask

    fresh_flat = fresh.reshape(*packed.shape[:-1], 256)
    out = np.zeros_like(fresh_flat, dtype=np.uint16)
    state = np.zeros(fresh_flat.shape[:-1], dtype=np.uint32)
    for t in list(range(256)) * 2:
        state = ((state << np.uint32(k)) | fresh_flat[..., t]) & np.uint32(0xFFFF)
        out[..., t] = state.astype(np.uint16, copy=False)
    return out


def unpack_trellis(packed: np.ndarray, k: int) -> np.ndarray:
    """Unpack trellis tensor of shape (tiles_k, tiles_n, 256 * k // 16)."""
    if packed.ndim != 3:
        raise ValueError(f"expected 3D packed trellis, got {packed.shape}")
    packed_size = 256 * k // 16
    if packed.shape[-1] != packed_size:
        raise ValueError(f"expected last dim {packed_size}, got {packed.shape[-1]}")
    if k in (1, 2, 4, 8):
        return _unpack_trellis_div16(packed, k)
    return _unpack_trellis_general(packed, k)
