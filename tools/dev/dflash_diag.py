#!/usr/bin/env python3
"""DFlash drafter diagnostics: aux-layer shift × output-slice alignment.

Drafts one block after a prompt and compares against the target's true
greedy continuation. A healthy drafter matches most early positions; a
misaligned one matches ~none (random tokens)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


import mlx.core as mx
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--dflash", required=True)
    args = ap.parse_args()

    from mlx_lm.utils import load_tokenizer
    from ponyexl3.mlx.dflash import DFlashDraft
    from ponyexl3.mlx.eagle3 import AuxTrace
    from ponyexl3.mlx.model import load_model

    model, _ = load_model(args.model, engine="exl3", verbose=False)
    drafter = DFlashDraft(args.dflash)
    tokenizer = load_tokenizer(Path(args.model))
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write a Python class implementing an LRU cache with get/put methods and unit tests."}],
        add_generation_prompt=True,
    )

    lm = model.language_model
    embed = lm.model.embed_tokens

    # ground truth: pending + 16 greedy tokens
    cache = lm.make_cache()
    h = lm.model(mx.array([prompt_ids]), cache=cache)
    y = mx.argmax(lm.lm_head(h[:, -1:, :])[0, -1]).reshape(1).astype(mx.int32)
    truth = [int(y.item())]
    for _ in range(16):
        h = lm.model(y[:, None], cache=cache)
        y = mx.argmax(lm.lm_head(h)[0, -1]).reshape(1).astype(mx.int32)
        truth.append(int(y.item()))
    print("truth:", truth)
    print("text :", tokenizer.decode(truth))

    base = (1, 16, 31, 46, 61)
    for shift in (0, -1, 1):
        ids = tuple(max(0, i + shift) for i in base)
        drafter.aux_ids = ids
        drafter.make_caches()
        cache = lm.make_cache()
        with AuxTrace(lm.model, ids) as aux:
            h = lm.model(mx.array([prompt_ids]), cache=cache)
            feats = drafter.fuse(aux.take())
        drafter.update_kv(feats)
        pending = mx.array([truth[0]], dtype=mx.int32)
        for ar in ("0", "1"):
            os.environ["EXL3_DFLASH_AR"] = ar
            drafts = np.array(drafter.draft_block(pending, embed, lm.lm_head, 15))
            ref = truth[1:16]
            hits = sum(int(a == b) for a, b in zip(drafts.tolist(), ref))
            prefix = 0
            for a, b in zip(drafts.tolist(), ref):
                if a != b:
                    break
                prefix += 1
            print(
                f"shift={shift:+d} ar={ar}: prefix={prefix:2d} hits={hits:2d}/15  drafts={drafts.tolist()}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
