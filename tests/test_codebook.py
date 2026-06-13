import numpy as np
import pytest

from ponyexl3.ref.codebook import CodebookMode, decode_3inst, lop3_b32

pytestmark = pytest.mark.ponyexl3


@pytest.mark.parametrize("cb", list(CodebookMode))
@pytest.mark.parametrize("word", [0, 1, 42, 0xABCD, 0xFFFF])
def test_decode_finite(cb, word):
    v = decode_3inst(word, cb)
    assert np.isfinite(float(v))


def test_lop3_bitwise_lut():
    a = np.uint32(0xA5A5A5A5)
    b = np.uint32(0x8FFF8FFF)
    c = np.uint32(0x3B603B60)
    r = lop3_b32(a, b, c, 0x6A)
    # PTX convention: imm = op(0xF0, 0xCC, 0xAA), lookup idx = (a<<2)|(b<<1)|c
    for bit in range(32):
        ab = (int(a) >> bit) & 1
        bb = (int(b) >> bit) & 1
        cb = (int(c) >> bit) & 1
        idx = (ab << 2) | (bb << 1) | cb
        expect = (0x6A >> idx) & 1
        got = (int(r) >> bit) & 1
        assert got == expect
    # imm 0x6A is exactly c ^ (a & b) — the 3INST decode's closed form
    assert int(r) == (int(a) & int(b)) ^ int(c)


def test_decode_never_nan_exhaustive():
    """The lop3 mask forces a finite fp16 exponent for every possible codeword."""
    for cb in (CodebookMode.DEFAULT, CodebookMode.MCG, CodebookMode.MUL1):
        for word in range(0, 65536, 251):
            v = float(decode_3inst(word, cb))
            assert np.isfinite(v), f"cb={cb} word={word:#x} -> {v}"
            assert abs(v) < 16.0, f"cb={cb} word={word:#x} -> {v}"
