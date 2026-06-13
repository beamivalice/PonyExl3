"""Procedural EXL3 codebook — port of exllamav3_ext/quant/codebook.cuh (CPU reference)."""

from __future__ import annotations

import numpy as np
from enum import IntEnum


class CodebookMode(IntEnum):
    DEFAULT = 0
    MCG = 1
    MUL1 = 2


LOP3_IMM_6A = 0x6A
K_INV = np.float16(np.frombuffer(np.uint16(0x1EEE).tobytes(), dtype=np.float16)[0])
K_BIAS = np.float16(np.frombuffer(np.uint16(0xC931).tobytes(), dtype=np.float16)[0])

MCG_MULT = np.uint32(0xCBAC1FED)
MUL1_MULT = np.uint32(0x83DCD12D)


def lop3_b32(a: int | np.uint32, b: int | np.uint32, c: int | np.uint32, imm: int) -> int:
    """Bitwise LOP3 from the PTX 8-bit truth-table immediate.

    PTX convention: ``imm`` equals the operation applied to (0xF0, 0xCC, 0xAA),
    so the lookup index is ``(a << 2) | (b << 1) | c`` — a is the HIGH bit.
    Imm 0x6A therefore computes ``c ^ (a & b)``, which forces the fp16 exponent
    bits to a finite range (never NaN/inf) in the 3INST decode.
    """
    a_u = int(np.uint32(a))
    b_u = int(np.uint32(b))
    c_u = int(np.uint32(c))
    out = 0
    for bit in range(32):
        ab = (a_u >> bit) & 1
        bb = (b_u >> bit) & 1
        cb = (c_u >> bit) & 1
        idx = (ab << 2) | (bb << 1) | cb
        ob = (imm >> idx) & 1
        out |= ob << bit
    return int(np.uint32(out))


def _half2_add_u32(x: int) -> np.float16:
    x_u = int(np.uint32(x))
    lo = np.float16(np.frombuffer(np.uint16(x_u & 0xFFFF).tobytes(), dtype=np.float16)[0])
    hi = np.float16(np.frombuffer(np.uint16((x_u >> 16) & 0xFFFF).tobytes(), dtype=np.float16)[0])
    # Half-precision add, matching CUDA __hadd (codebook.cuh) — NOT a float add
    # rounded back to fp16, which double-rounds on rare ties.
    return lo + hi


def vabsdiff4_add(x: int, y: int = 0, acc: int = 0x6400) -> int:
    """PTX vabsdiff4.u32.u32.u32.add — sum byte |a-b| lanes plus accumulator."""
    s = int(np.uint32(acc))
    x_u = int(np.uint32(x))
    y_u = int(np.uint32(y))
    for lane in range(4):
        ai = (x_u >> (8 * lane)) & 0xFF
        bi = (y_u >> (8 * lane)) & 0xFF
        s += abs(ai - bi)
    return int(np.uint32(s & 0xFFFFFFFF))


def decode_3inst(codeword: int, cb: CodebookMode | int) -> np.float16:
    """
  Map a 16-bit trellis codeword to one fp16 weight sample.

  Mirrors decode_3inst<cb> in codebook.cuh.
  """
    mode = CodebookMode(cb)
    x = int(np.uint32(codeword & 0xFFFF))

    if mode == CodebookMode.DEFAULT:
        x = int(np.uint32((np.uint64(x) * np.uint64(89226354) + np.uint64(64248484)) & np.uint64(0xFFFFFFFF)))
        x = lop3_b32(x, 0x8FFF8FFF, 0x3B603B60, LOP3_IMM_6A)
        return _half2_add_u32(x)

    if mode == CodebookMode.MCG:
        x = int(np.uint32((np.uint64(x) * np.uint64(int(MCG_MULT))) & np.uint64(0xFFFFFFFF)))
        x = lop3_b32(x, 0x8FFF8FFF, 0x3B603B60, LOP3_IMM_6A)
        return _half2_add_u32(x)

    if mode == CodebookMode.MUL1:
        x = int(np.uint32((np.uint64(x) * np.uint64(int(MUL1_MULT))) & np.uint64(0xFFFFFFFF)))
        summed = vabsdiff4_add(x, 0, 0x6400)
        h = np.float16(np.frombuffer(np.uint16(summed & 0xFFFF).tobytes(), dtype=np.float16)[0])
        return np.float16(float(h) * float(K_INV) + float(K_BIAS))

    raise ValueError(f"unknown codebook mode {mode}")


def codebook_mode_from_flags(mcg: bool = False, mul1: bool = False) -> CodebookMode:
    if mcg:
        return CodebookMode.MCG
    if mul1:
        return CodebookMode.MUL1
    return CodebookMode.DEFAULT
