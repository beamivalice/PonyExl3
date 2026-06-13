#!/usr/bin/env python3
"""Affine-requant error: rotated (trellis) domain vs public folded domain.

EXL3's incoherence processing makes the inner W~ near-Gaussian/outlier-free;
folding the rotations back into W_pub restores the hard-to-quantize
distribution. If 4-bit affine error in the rotated domain is materially
lower, a fast engine should quantize W~ and keep the runtime rotations.

Both error norms are output-comparable: the finish transform (H_out, svh)
is orthogonal up to scaling, so inner-domain Frobenius error maps 1:1.
"""

from __future__ import annotations

import argparse


import mlx.core as mx
import numpy as np


def rel_rms(err: mx.array, ref: mx.array) -> float:
    return float(mx.sqrt(mx.mean(err.astype(mx.float32) ** 2))) / float(
        mx.sqrt(mx.mean(ref.astype(mx.float32) ** 2))
    )


def rtn_err(w: mx.array, bits: int, group: int) -> float:
    """Quantize along the last axis (rows = out features, like mlx qmv)."""
    wq, scales, biases = mx.quantize(w, group_size=group, bits=bits)
    wd = mx.dequantize(wq, scales, biases, group_size=group, bits=bits)
    return rel_rms(wd - w, w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--layers", default="2,30,60")
    args = ap.parse_args()

    from ponyexl3.mlx.layer_state import inner_weight_mlx
    from ponyexl3.mlx.model import load_model
    from ponyexl3.mlx.native import public_weight_chunks

    model, _ = load_model(args.model, engine="exl3", warm=False, verbose=False)
    layers = model.language_model.model.layers

    def exl3_of(mod):
        return mod._exl3 if hasattr(mod, "_exl3") else None

    for li in (int(x) for x in args.layers.split(",")):
        blk = layers[li]
        mods = {"down_proj": blk.mlp.down_proj, "gate_proj": blk.mlp.gate_proj}
        if getattr(blk, "is_linear", False):
            mods["out_proj"] = blk.linear_attn.out_proj
        for name, mod in mods.items():
            layer = exl3_of(mod)
            if layer is None:
                continue
            w_inner = inner_weight_mlx(layer).T  # (out, in) fp16, rotated domain
            mx.eval(w_inner)
            w_pub = mx.concatenate(list(public_weight_chunks(layer)), axis=1).T
            mx.eval(w_pub)
            row = f"L{li:2d} {name:9s} k={layer.k}"
            for dom, w in (("rot", w_inner), ("pub", w_pub)):
                e4_64 = rtn_err(w, 4, 64)
                e4_32 = rtn_err(w, 4, 32)
                e5_64 = rtn_err(w, 5, 64)
                row += f" | {dom}: w4g64 {e4_64:.4f} w4g32 {e4_32:.4f} w5g64 {e5_64:.4f}"
            print(row)
            del w_inner, w_pub
            mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
