#!/usr/bin/env python3
"""GPTQ pilot on one 27B layer: does activation-aware w4g64 beat RTN's 0.090?

Calibrates H = X^T X on wikitext through the exact exl3 model, decodes the
layer's W_pub, and compares RTN vs GPTQ in weight space AND output space
(held-out activations) — the latter is what KLD feels.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


import numpy as np
import mlx.core as mx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--seqs", type=int, default=24)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--cache", default="/tmp/gptq_pilot.npz")
    ap.add_argument("--solve-only", action="store_true")
    args = ap.parse_args()

    import os

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    if args.solve_only and os.path.exists(args.cache):
        z = np.load(args.cache)
        H_np, X_held, W = z["H"], z["X"], z["W"]
        run_solvers(H_np, X_held, W)
        return 0
    from datasets import load_dataset

    from mlx_lm.utils import load_tokenizer
    from ponyexl3.mlx.exl3_linear import EXL3Linear
    from ponyexl3.mlx.gptq import gptq_quantize
    from ponyexl3.mlx.model import load_model
    from ponyexl3.mlx.native import public_weight_chunks

    model, _ = load_model(args.model, engine="exl3", verbose=False)
    tokenizer = load_tokenizer(Path(args.model))
    lm = model.language_model
    target = lm.model.layers[args.layer].mlp.down_proj
    in_f = target.in_features

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer.encode(text)[: args.seqs * args.seqlen]
    seqs = [
        ids[i : i + args.seqlen] for i in range(0, len(ids) - args.seqlen, args.seqlen)
    ][: args.seqs]
    print(f"calibration: {len(seqs)} seqs x {args.seqlen}")

    H = mx.zeros((in_f, in_f), dtype=mx.float32)
    held = []
    state = {"H": H, "mode": "acc"}
    orig = EXL3Linear.__call__

    def wrapped(mod, x):
        if mod is target:
            x2 = x.reshape(-1, in_f).astype(mx.float32)
            if state["mode"] == "acc":
                state["H"] = state["H"] + x2.T @ x2
            else:
                held.append(x.reshape(-1, in_f).astype(mx.float16))
        return orig(mod, x)

    EXL3Linear.__call__ = wrapped
    tic = time.perf_counter()
    try:
        for i, s in enumerate(seqs):
            state["mode"] = "acc" if i < len(seqs) - 4 else "held"
            cache = lm.make_cache()
            mx.eval(lm.model(mx.array([s]), cache=cache))
            mx.eval(state["H"])
            del cache
    finally:
        EXL3Linear.__call__ = orig
    print(f"calibration forward: {time.perf_counter() - tic:.1f}s")

    H_np = np.array(state["H"], dtype=np.float32)
    X_held = np.array(mx.concatenate(held, axis=0), dtype=np.float32)
    W = np.concatenate(
        [np.array(c, dtype=np.float32) for c in public_weight_chunks(target._exl3)],
        axis=1,
    ).T  # (out, in)
    print("W", W.shape, "H", H_np.shape, "held x", X_held.shape)

    np.savez(args.cache, H=H_np, X=X_held.astype(np.float16), W=W.astype(np.float16))
    run_solvers(H_np, X_held, W)
    return 0


def run_solvers(H_np, X_held, W):
    from ponyexl3.mlx.gptq import gptq_quantize
    import time

    H_np = H_np.astype(np.float32)
    X_held = X_held.astype(np.float32)
    W = W.astype(np.float32)
    y_ref = X_held @ W.T

    def report(name, W_hat):
        werr = np.sqrt(np.mean((W_hat - W) ** 2)) / np.sqrt(np.mean(W**2))
        y = X_held @ W_hat.T
        yerr = np.sqrt(np.mean((y - y_ref) ** 2)) / np.sqrt(np.mean(y_ref**2))
        print(f"{name:12s} weight rel-RMS {werr:.4f}   OUTPUT rel-RMS {yerr:.4f}")

    # RTN baseline via mx (identical grid semantics)
    wq, sc, bi = mx.quantize(mx.array(W.astype(np.float16)), group_size=64, bits=4)
    rtn = np.array(
        mx.dequantize(wq, sc, bi, group_size=64, bits=4), dtype=np.float32
    )
    report("RTN w4g64", rtn)

    def deq_of(Q, scales, biases):
        deq = np.zeros_like(W)
        g = 64
        for j in range(0, W.shape[1], g):
            deq[:, j : j + g] = (
                Q[:, j : j + g] * scales[:, j // g : j // g + 1]
                + biases[:, j // g : j // g + 1]
            )
        return deq

    def deq_of_g(Q, scales, biases, g):
        deq = np.zeros_like(W)
        for j in range(0, W.shape[1], g):
            deq[:, j : j + g] = (
                Q[:, j : j + g] * scales[:, j // g : j // g + 1]
                + biases[:, j // g : j // g + 1]
            )
        return deq

    for name, g, damp, act in (
        ("w4g64 d=.10", 64, 0.10, False),
        ("w4g64 d=.10 ACT", 64, 0.10, True),
        ("w4g32 d=.10", 32, 0.10, False),
        ("w8g64 d=.10", 64, 0.10, False),
    ):
        bits = 8 if "w8" in name else 4
        tic = time.perf_counter()
        if act:
            perm = np.argsort(-np.diag(H_np))
            inv = np.argsort(perm)
            Q, scales, biases = gptq_quantize(
                W[:, perm].copy(), H_np[perm][:, perm].copy(),
                bits=bits, group_size=g, damp=damp,
            )
            deq = deq_of_g(Q, scales, biases, g)[:, inv]
        else:
            Q, scales, biases = gptq_quantize(
                W.copy(), H_np.copy(), bits=bits, group_size=g, damp=damp
            )
            deq = deq_of_g(Q, scales, biases, g)
        dt = time.perf_counter() - tic
        report(f"GPTQ {name} ({dt:.0f}s)", deq)


if __name__ == "__main__":
    raise SystemExit(main())
