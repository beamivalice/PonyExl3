#!/usr/bin/env python3
"""Decompose the spec-verify forward cost: inner model vs lm_head at S=1 vs S=4.

Also times per-layer-type contributions at S=1 vs S=4 by hooking the two
decoder layer classes.
"""

from __future__ import annotations

import argparse
import time


import mlx.core as mx


def bench(fn, reps=20, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    tic = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    return (time.perf_counter() - tic) / reps * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()

    from ponyexl3.mlx.model import load_model

    model, config = load_model(args.model, engine="exl3", verbose=False)
    lm = model.language_model
    S = args.rows

    # prime a cache with some context
    cache = lm.make_cache()
    toks = mx.array([[1000 + i for i in range(64)]])
    mx.eval(lm.model(toks, cache=cache))

    one = mx.array([[1234]])
    multi = mx.array([[1234 + i for i in range(S)]])

    t_model_1 = bench(lambda: lm.model(one, cache=cache))
    t_model_S = bench(lambda: lm.model(multi, cache=cache))

    h1 = lm.model(one, cache=cache)
    hS = lm.model(multi, cache=cache)
    mx.eval(h1, hS)
    t_head_1 = bench(lambda: lm.lm_head(h1))
    t_head_S = bench(lambda: lm.lm_head(hS))

    print(f"inner model  S=1: {t_model_1:7.2f} ms   S={S}: {t_model_S:7.2f} ms  (x{t_model_S/t_model_1:.2f})")
    print(f"lm_head      M=1: {t_head_1:7.2f} ms   M={S}: {t_head_S:7.2f} ms  (x{t_head_S/t_head_1:.2f})")

    # per-layer-type split via class hooks
    from mlx_lm.models import qwen3_5 as q5

    acc = {"linear": 0.0, "full": 0.0, "n": 0}

    def timed(cls, key):
        orig = cls.__call__

        def wrapped(self, *a, **kw):
            mx.synchronize()
            tic = time.perf_counter()
            out = orig(self, *a, **kw)
            mx.eval(out)
            acc[key] += time.perf_counter() - tic
            return out

        return orig, wrapped

    o1, w1 = timed(q5.GatedDeltaNet, "linear")
    q5.GatedDeltaNet.__call__ = w1
    o2, w2 = timed(q5.Attention, "full")
    q5.Attention.__call__ = w2
    try:
        for label, inp in (("S=1", one), (f"S={S}", multi)):
            acc["linear"] = acc["full"] = 0.0
            reps = 10
            for _ in range(reps):
                mx.eval(lm.model(inp, cache=cache))
            print(
                f"{label}: deltanet(48) {acc['linear']*1000/reps:7.2f} ms   "
                f"attention(16) {acc['full']*1000/reps:7.2f} ms   (synced per-module)"
            )
    finally:
        q5.GatedDeltaNet.__call__ = o1
        q5.Attention.__call__ = o2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
