"""MLX vectorized codebook ops + Metal ``decode_3inst`` batch."""

from __future__ import annotations

import mlx.core as mx

from ponyexl3.ref.codebook import CodebookMode, MCG_MULT, MUL1_MULT

_LOP3_B = mx.array(0x8FFF8FFF, dtype=mx.uint32)
_LOP3_C = mx.array(0x3B603B60, dtype=mx.uint32)
_LOP3_IMM = 0x6A


def lop3_b32_mlx(a: mx.array, b: mx.array, c: mx.array, imm: int) -> mx.array:
    """Vectorized LOP3 — one uint32 result per element.

    PTX convention: lookup index is ``(a << 2) | (b << 1) | c`` (a is the high
    bit), so ``imm`` equals the operation applied to (0xF0, 0xCC, 0xAA).
    """
    imm_u = mx.array(imm, dtype=mx.uint32)
    ar = mx.arange(32, dtype=mx.uint32)
    bits_a = (a[..., None] >> ar) & 1
    bits_b = (b[..., None] >> ar) & 1
    bits_c = (c[..., None] >> ar) & 1
    idx = (bits_a << 2) | (bits_b << 1) | bits_c
    ob = (imm_u >> idx) & 1
    shifts = (mx.array(1, dtype=mx.uint32) << ar).astype(mx.uint32)
    return mx.sum(ob * shifts, axis=-1).astype(mx.uint32)


def _vabsdiff4_add_mlx(x: mx.array, y: mx.array | None = None, acc: int = 0x6400) -> mx.array:
    x = x.astype(mx.uint32)
    y = mx.array(0, dtype=mx.uint32) if y is None else y.astype(mx.uint32)
    s = mx.array(acc, dtype=mx.uint32)
    for lane in range(4):
        ai = (x >> (8 * lane)) & 0xFF
        bi = (y >> (8 * lane)) & 0xFF
        s = s + mx.abs(ai.astype(mx.int32) - bi.astype(mx.int32)).astype(mx.uint32)
    return s


def decode_u32_mlx(codewords: mx.array, cb: CodebookMode | int) -> mx.array:
    """Map codewords to uint32 words ready for Metal half-finalize."""
    cb = int(CodebookMode(cb))
    x = codewords.astype(mx.uint32) & 0xFFFF
    if cb == CodebookMode.DEFAULT:
        x = ((x.astype(mx.uint64) * 89226354 + 64248484) & 0xFFFFFFFF).astype(mx.uint32)
        return lop3_b32_mlx(x, _LOP3_B, _LOP3_C, _LOP3_IMM)
    if cb == CodebookMode.MCG:
        x = ((x.astype(mx.uint64) * int(MCG_MULT)) & 0xFFFFFFFF).astype(mx.uint32)
        return lop3_b32_mlx(x, _LOP3_B, _LOP3_C, _LOP3_IMM)
    if cb == CodebookMode.MUL1:
        x = ((x.astype(mx.uint64) * int(MUL1_MULT)) & 0xFFFFFFFF).astype(mx.uint32)
        return _vabsdiff4_add_mlx(x)
    raise ValueError(f"unknown codebook mode {cb}")
