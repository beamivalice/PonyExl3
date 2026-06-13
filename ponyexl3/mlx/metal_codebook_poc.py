"""
Proof-of-concept: procedural codebook decode in a custom Metal kernel.

Run: python -m ponyexl3.mlx.metal_codebook_poc

This validates mlx.fast.metal_kernel for porting decode_3inst into fused GEMV.
Requires mlx on Apple Silicon.
"""

from __future__ import annotations

import numpy as np

from ponyexl3.ref.codebook import CodebookMode, decode_3inst, lop3_b32


METAL_DECODE_DEFAULT = r"""
    uint idx = thread_position_in_grid.x;
    uint x = inp[idx];
    x = x * 89226354u + 64248484u;
    // lop3.b32 imm 0x6A == c ^ (a & b) under the PTX (a<<2)|(b<<1)|c convention
    uint r = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  // half2 add: treat as two fp16 in one word (approximation for POC)
    ushort lo = ushort(r & 0xFFFFu);
    ushort hi = ushort((r >> 16) & 0xFFFFu);
    out[idx] = T(as_type<half>(lo) + as_type<half>(hi));
"""


def run_poc(n: int = 4096) -> None:
    import mlx.core as mx

    codewords = np.random.randint(0, 65536, size=n, dtype=np.uint32)
    ref = np.array([decode_3inst(int(w), CodebookMode.DEFAULT) for w in codewords], dtype=np.float32)

    kernel = mx.fast.metal_kernel(
        name="exl3_decode_default",
        input_names=["inp"],
        output_names=["out"],
        source=METAL_DECODE_DEFAULT,
    )
    out = kernel(
        inputs=[mx.array(codewords)],
        template=[("T", mx.float32)],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]
    got = np.array(out)
    diff = np.abs(ref - got)
    print(f"metal codebook POC: max_err={diff.max():.6f} mean_err={diff.mean():.6f}")
    if diff.max() > 0.05:
        raise SystemExit("Metal POC diverged from CPU reference — refine kernel")


def verify_cpu_lop3_sample() -> None:
    x = np.uint32(12345)
    y = lop3_b32(int(x), int(np.uint32(0x8FFF8FFF)), int(np.uint32(0x3B603B60)), 0x6A)
    z = decode_3inst(12345, CodebookMode.DEFAULT)
    assert np.isfinite(float(z))
    assert isinstance(y, (int, np.integer))


if __name__ == "__main__":
    verify_cpu_lop3_sample()
    try:
        run_poc()
    except ImportError:
        print("mlx not installed — skipping Metal kernel POC")
