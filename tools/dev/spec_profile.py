#!/usr/bin/env python3
"""Per-phase timing of the speculative-decode cycle (verify / repair / draft).

Synchronizes between phases (mx.eval), so the total runs slower than the
pipelined loop — the point is the BUDGET, not the wall.

  python tools/spec_profile.py MODEL --mtp MTP.safetensors -n 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


import mlx.core as mx
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--mtp", required=True)
    ap.add_argument("-p", "--prompt", default="Write a Python class implementing an LRU cache with get/put methods and unit tests.")
    ap.add_argument("-n", "--max-tokens", type=int, default=200)
    ap.add_argument("--draft", type=int, default=3)
    args = ap.parse_args()

    from mlx_lm.utils import load_tokenizer
    from ponyexl3.mlx.generate import _DeltaNetTrace, _snapshot_recurrent
    from ponyexl3.mlx.model import load_model
    from ponyexl3.mlx.mtp import load_mtp

    model, config = load_model(args.model, engine="exl3", verbose=False)
    mtp = load_mtp(args.model, config, args.mtp)
    tokenizer = load_tokenizer(Path(args.model))
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )

    from mlx_lm.models.cache import KVCache

    num_draft = args.draft
    lm = model.language_model
    embed = lm.model.embed_tokens
    cache = lm.make_cache()
    mtp_cache = KVCache()

    toks = mx.array([list(prompt_ids)])
    h_all = lm.model(toks, cache=cache)
    mx.eval(h_all)
    logits = lm.lm_head(h_all[:, -1:, :])
    pending = mx.argmax(logits[0, -1]).reshape(1).astype(mx.int32)
    mx.eval(pending)

    catch_h = h_all
    catch_t = mx.concatenate([toks[0, 1:].astype(mx.int32), pending])[None]

    def draft_phase(catch_t, catch_h):
        h_mtp = mtp(embed(catch_t), catch_h, mtp_cache)
        h_chain = h_mtp[:, -1:, :]
        drafts = []
        for j in range(num_draft):
            d_logits = lm.lm_head(mtp.head_input(h_chain))
            dj = mx.argmax(d_logits[0, -1]).reshape(1).astype(mx.int32)
            drafts.append(dj)
            if j < num_draft - 1:
                h_chain = mtp(embed(dj[None]), h_chain, mtp_cache)
        mx.eval(*drafts)
        return drafts

    t = {"verify": 0.0, "host": 0.0, "repair": 0.0, "draft": 0.0}
    n_cycles = 0
    n_partial = 0
    repair_times = []

    drafts = draft_phase(catch_t, catch_h)
    n_spec_mtp = num_draft - 1
    emitted = 0
    while emitted < args.max_tokens:
        tic = time.perf_counter()
        verify_tokens = mx.concatenate([pending] + drafts)
        snap = _snapshot_recurrent(cache)
        with _DeltaNetTrace() as trace:
            h_ver = lm.model(verify_tokens[None], cache=cache)
        preds = mx.argmax(lm.lm_head(h_ver)[0], axis=-1)
        mx.eval(preds)
        t["verify"] += time.perf_counter() - tic

        tic = time.perf_counter()
        preds_np = np.array(preds)
        verify_np = np.array(verify_tokens)
        m = 0
        while m < num_draft and preds_np[m] == verify_np[m + 1]:
            m += 1
        bonus = int(preds_np[m])
        accepted = [int(v) for v in verify_np[: m + 1]]
        t["host"] += time.perf_counter() - tic

        tic = time.perf_counter()
        h_acc = h_ver
        if m < num_draft:
            n_partial += 1
            discard = int(verify_tokens.shape[0]) - (m + 1)
            for c, s in zip(cache, snap):
                if s is not None:
                    c.cache = list(s)
                else:
                    c.trim(discard)
            trace.repair(m + 1)
            mx.eval(*[a for c in cache for a in getattr(c, "cache", []) if isinstance(a, mx.array)])
            repair_times.append(time.perf_counter() - tic)
        mtp_cache.trim(n_spec_mtp)
        t["repair"] += time.perf_counter() - tic

        tic = time.perf_counter()
        pending = mx.array([bonus], dtype=mx.int32)
        catch_h = h_acc[:, : m + 1, :]
        catch_t = mx.concatenate([mx.array(accepted[1:], dtype=mx.int32), pending])[None]
        drafts = draft_phase(catch_t, catch_h)
        t["draft"] += time.perf_counter() - tic

        n_cycles += 1
        emitted += m + 1

    total = sum(t.values())
    print(f"cycles={n_cycles} partial={n_partial} tokens={emitted} tok/cycle={emitted/n_cycles:.2f}")
    for k, v in t.items():
        print(f"  {k:8s} {v*1000/n_cycles:7.2f} ms/cycle  ({100*v/total:4.1f}%)")
    if repair_times:
        print(f"  repair (partial cycles only): mean {1000*np.mean(repair_times):.2f} ms  n={len(repair_times)}")
    print(f"  synced total: {total*1000/n_cycles:.1f} ms/cycle -> {emitted/total:.1f} tok/s (pipelined loop is faster)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
