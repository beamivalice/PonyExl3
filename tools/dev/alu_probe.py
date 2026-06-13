#!/usr/bin/env python3
"""M5 ALU rate probe: u32 imul vs fp32 fma vs 16-bit mul decomposition.

Each thread runs ITERS iterations of CHAINS independent dependent-chains of
the op under test (ILP to fill the pipeline); the result is written so the
compiler can't elide. Reported: effective Gops/s.
"""

from __future__ import annotations

import time


import mlx.core as mx

ITERS = 4096
CHAINS = 8
THREADS = 1 << 16


def make_kernel(name: str, body: str, acc_t: str = "uint") -> object:
    src = f"""
    uint tid = thread_position_in_grid.x;
    {acc_t} acc[{CHAINS}];
    for (uint j = 0u; j < {CHAINS}u; j++) {{
        acc[j] = ({acc_t})(seed[0] + tid + j);
    }}
    for (uint it = 0u; it < {ITERS}u; it++) {{
        for (uint j = 0u; j < {CHAINS}u; j++) {{
{body}
        }}
    }}
    {acc_t} s = ({acc_t})0;
    for (uint j = 0u; j < {CHAINS}u; j++) {{
        s += acc[j];
    }}
    out[tid] = {"float(s.x) + float(s.y)" if acc_t == "half2" else "float(s)"};
"""
    return mx.fast.metal_kernel(
        name=name, input_names=["seed"], output_names=["out"], source=src
    )


VARIANTS = {
    # one 32-bit integer multiply (+add to keep the chain live)
    "imul32 (cw*C)": ("acc[j] = acc[j] * 0xCBAC1FEDu + 1u;", "uint"),
    # 16-bit decomposition of the same product for a 16-bit operand:
    # masks model the real decode (cw arrives masked to 16 bits)
    "imul16x2 decomp": (
        "acc[j] = (acc[j] & 0xFFFFu) * 0x1FEDu + (((acc[j] & 0xFFFFu) * 0xCBACu) << 16) + 1u;",
        "uint",
    ),
    # fp32 fma reference
    "fp32 fma": ("acc[j] = fma(acc[j], 1.0001f, 1.0f);", "float"),
    # dual-issue check: if half2 fma runs ~2x fp32 fma, fp16 pairing pays
    "half2 fma": ("acc[j] = fma(acc[j], half2(1.0009765625h), half2(0.001h));", "half2"),
    "half fma": ("acc[j] = fma(acc[j], 1.0009765625h, 0.001h);", "half"),
    # the bit ops from decode for scale
    "and-xor": ("acc[j] = (acc[j] & 0x8FFF8FFFu) ^ 0x3B603B60u;", "uint"),
    # half2 add as in decode
    "half2-add+cvt": (
        "half2 h = as_type<half2>(acc[j]); acc[j] = uint(float(h.x + h.y)) + acc[j];",
        "uint",
    ),
    # 64-bit funnel shift as in extraction
    "u64 var-shift": (
        "ulong m = ((ulong)acc[j] << 32) | (ulong)(acc[j] ^ 7u); acc[j] = uint(m >> (acc[j] & 31u));",
        "uint",
    ),
}


def main() -> int:
    seed = mx.array([1], dtype=mx.uint32)
    kerns = {
        name: make_kernel(
            "probe_" + "".join(c for c in name if c.isalnum()), body, acc_t
        )
        for name, (body, acc_t) in VARIANTS.items()
    }

    def run(kern):
        return kern(
            inputs=[seed],
            grid=(THREADS, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(THREADS,)],
            output_dtypes=[mx.float32],
        )[0]

    # warm everything (compile + clock ramp), then interleave reps
    # round-robin so GPU power state confounds no single variant.
    for _ in range(3):
        for kern in kerns.values():
            mx.eval(run(kern))
    mx.synchronize()
    times = {name: 0.0 for name in kerns}
    reps = 8
    for _ in range(reps):
        for name, kern in kerns.items():
            tic = time.perf_counter()
            mx.eval(run(kern))
            mx.synchronize()
            times[name] += time.perf_counter() - tic
    ops = THREADS * ITERS * CHAINS
    for name, total in times.items():
        dt = total / reps
        print(f"{name:18s} {ops/dt/1e12:7.3f} Tops/s   ({dt*1000:6.2f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
