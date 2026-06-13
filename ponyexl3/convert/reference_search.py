"""Reference trellis search: exact Viterbi over the bitshift trellis (numpy).

Mirrors exllamav3's quantize_tiles_kernel semantics: states are 16-bit
windows, each step appends K bits (state' = ((state << K) | j) & 0xFFFF),
branch cost = (decode_3inst(state') - w[t])^2, tail-biting closure via
pinned re-passes. Slow (~0.5 s/tile) — this is the parity oracle for the
Metal kernel, not the production path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ponyexl3.ref.codebook import CodebookMode, decode_3inst

_LUT_CACHE: dict[int, np.ndarray] = {}


def _decode_lut(cb: CodebookMode) -> np.ndarray:
    lut = _LUT_CACHE.get(int(cb))
    if lut is None:
        lut = np.empty(65536, dtype=np.float32)
        for s in range(65536):
            lut[s] = float(decode_3inst(s, cb))
        _LUT_CACHE[int(cb)] = lut
    return lut


def _forward(
    w: np.ndarray, k: int, lut: np.ndarray, init_state: int | None
) -> tuple[int, NDArray[np.uint16], float]:
    """One Viterbi pass; returns (best_final_state, backpointers)."""
    n_states = 65536
    kk = 1 << k
    cost = np.zeros(n_states, dtype=np.float64)
    if init_state is not None:
        cost[:] = np.float64(1e30)
        cost[init_state] = 0.0
    s = np.arange(n_states, dtype=np.uint32)
    # Checkpoint-validated transition (256/256 on real tiles, incl. the
    # tail-biting wrap): s' = ((s << K) | fresh) & 0xFFFF — new bits enter
    # LOW; fresh bits of step t are s_t & (2^K - 1). Predecessors of s'
    # are the kk states whose low 16-K bits equal s' >> K.
    pred = ((s >> k)[:, None] | (np.arange(kk, dtype=np.uint32) << (16 - k))[None, :])
    back = np.zeros((256, n_states), dtype=np.uint16)
    lutd = lut.astype(np.float64)
    for t in range(256):
        c_in = cost[pred]  # (n_states, kk)
        amin = c_in.argmin(axis=1)
        back[t] = pred[s, amin].astype(np.uint16)
        d = lutd - float(w[t])
        cost = c_in[s, amin] + d * d
    fin = int(cost.argmin())
    return fin, back, float(cost[fin])


def quantize_tile_reference(
    w: np.ndarray, k: int, cb: CodebookMode = CodebookMode.DEFAULT, max_pins: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """w: (256,) fp32 in kernel order -> (states (256,) u16, decoded f32).

    Tail-biting: codeword 0's window overlaps codeword 255's, so a valid
    path needs pre-state(step 0) == states[255]. Free pass proposes a
    closure; pinned passes enforce it (upstream's roll/pre_state scheme).
    """
    lut = _decode_lut(cb)

    def backtrack(fin: int, back: NDArray[np.uint16]) -> tuple[NDArray[np.uint16], int]:
        states: NDArray[np.uint16] = np.zeros(256, dtype=np.uint16)
        s = fin
        for t in range(255, -1, -1):
            states[t] = s
            s = int(back[t, s])
        return states, s  # s = pre-state of step 0

    fin, back, _ = _forward(w, k, lut, None)
    states, pre0 = backtrack(fin, back)
    pin = int(states[255])
    for _ in range(max_pins):
        fin, back, _ = _forward(w, k, lut, pin)
        states, pre0 = backtrack(fin, back)
        if pre0 == pin and int(states[255]) == pin:
            break
        pin = int(states[255])
    decoded = lut[states].astype(np.float32)
    return states, decoded
