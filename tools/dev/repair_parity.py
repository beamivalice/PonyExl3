#!/usr/bin/env python3
"""Bitwise parity of DeltaNet cache repair strategies after a partial accept.

A = scan-only repair (slice verify-forward q/k/v, re-run gated_delta_update)
B = legacy module replay (re-run full GatedDeltaNet on truncated layer input)
C = ground truth: feed the accepted tokens sequentially (S=1 steps)

Compares cache[0] (conv state) and cache[1] (recurrent state) across all
DeltaNet layers, bitwise and max-abs.
"""

from __future__ import annotations

import argparse
import os


import mlx.core as mx
import numpy as np


def snap_states(cache):
    out = []
    for c in cache:
        if hasattr(c, "cache") and isinstance(getattr(c, "cache"), list):
            out.append([None if a is None else np.array(a) for a in c.cache])
        else:
            out.append(None)
    return out


def cmp_states(x, y, label):
    n_exact = 0
    n_total = 0
    worst = 0.0
    worst_layer = -1
    for i, (sx, sy) in enumerate(zip(x, y)):
        if sx is None or sy is None:
            continue
        for j, (ax, ay) in enumerate(zip(sx, sy)):
            if ax is None:
                continue
            n_total += 1
            if np.array_equal(ax, ay):
                n_exact += 1
            else:
                d = float(np.max(np.abs(ax.astype(np.float64) - ay.astype(np.float64))))
                if d > worst:
                    worst, worst_layer = d, i
    print(f"{label}: {n_exact}/{n_total} arrays bit-identical; worst abs diff {worst:.3e} (cache idx {worst_layer})")
    return n_exact == n_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--keep", type=int, default=2, help="accepted tokens (1..3)")
    args = ap.parse_args()

    from ponyexl3.mlx.generate import _DeltaNetTrace, _snapshot_recurrent
    from ponyexl3.mlx.model import load_model

    model, _ = load_model(args.model, engine="exl3", verbose=False)
    lm = model.language_model

    prompt = mx.array([[151644, 872, 198, 9707, 11, 1246, 525, 498, 30, 151645]])
    verify = mx.array([[3838, 374, 279, 6722]])  # 4 tokens
    keep = args.keep

    def fresh_cache():
        cache = lm.make_cache()
        mx.eval(lm.model(prompt, cache=cache))
        return cache

    def all_evaled(cache):
        for c in cache:
            if hasattr(c, "cache") and isinstance(c.cache, list):
                mx.eval(*[a for a in c.cache if a is not None])

    # ---- C: ground truth — sequential steps
    cache_c = fresh_cache()
    for i in range(keep):
        mx.eval(lm.model(verify[:, i : i + 1], cache=cache_c))
    truth = snap_states(cache_c)

    # ---- A/B: verify forward then repair
    for mode, label in (("scan", "A(scan)"), ("module", "B(module)")):
        os.environ["EXL3_SPEC_REPAIR"] = mode
        cache = fresh_cache()
        snap = _snapshot_recurrent(cache)
        with _DeltaNetTrace() as trace:
            h = lm.model(verify, cache=cache)
        mx.eval(h)
        discard = verify.shape[1] - keep
        for c, s in zip(cache, snap):
            if s is not None:
                c.cache = list(s)
            else:
                c.trim(discard)
        trace.repair(keep)
        all_evaled(cache)
        got = snap_states(cache)
        ok = cmp_states(got, truth, f"{label} vs C(sequential)")
        if mode == "scan":
            scan_states = got
        else:
            cmp_states(scan_states, got, "A(scan) vs B(module)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
