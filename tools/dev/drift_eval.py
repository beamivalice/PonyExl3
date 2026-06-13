#!/usr/bin/env python3
"""Drift ladder for target-engine variants, referenced to the 8bpw export.

The 8.00bpw EXL3 model is near-lossless (and decodes at the same speed as
4.15bpw — Phase 26), so it stands in for the bf16 base. For each candidate
engine this reports, on wikitext-2 TEST (disjoint from GPTQ calibration):

- mean per-token KLD(ref || cand)
- top-1 agreement vs the reference
- top-1 agreement vs the exl3-4.15 model (the quality the user knows)

Run reference first (writes logits), then candidates.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import mlx.core as mx


def get_seqs(tokenizer, n, seqlen):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer.encode(text)[: n * seqlen]
    return [ids[i : i + seqlen] for i in range(0, len(ids) - seqlen, seqlen)][:n]


def forward_logits(model, seq):
    lm = model.language_model
    cache = lm.make_cache()
    h = lm.model(mx.array([seq]), cache=cache)
    logits = lm.lm_head(h)[0]
    mx.eval(logits)
    return logits.astype(mx.float16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--engine", default="exl3")
    ap.add_argument("--ref-dir", default="/tmp/drift_ref")
    ap.add_argument("--save-ref", action="store_true")
    ap.add_argument("--tag", default=None, help="store argmax under this tag")
    ap.add_argument("--vs-tag", default=None, help="also report agreement vs this tag")
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--tokenizer-from", default=None)
    args = ap.parse_args()

    from mlx_lm.utils import load_tokenizer
    from ponyexl3.mlx.model import load_model

    model, _ = load_model(args.model, engine=args.engine, warm=False, verbose=False)
    tokenizer = load_tokenizer(Path(args.tokenizer_from or args.model))
    seqs = get_seqs(tokenizer, args.seqs, args.seqlen)
    os.makedirs(args.ref_dir, exist_ok=True)

    if args.save_ref:
        for i, s in enumerate(seqs):
            logits = forward_logits(model, s)
            mx.save_safetensors(
                os.path.join(args.ref_dir, f"ref_{i}.safetensors"), {"l": logits}
            )
        print(f"reference saved: {len(seqs)} x {args.seqlen}")
        return 0

    kls, agree_ref, agree_tag = [], [], []
    am_all = []
    for i, s in enumerate(seqs):
        logits = forward_logits(model, s).astype(mx.float32)
        ref = mx.load(os.path.join(args.ref_dir, f"ref_{i}.safetensors"))["l"].astype(
            mx.float32
        )
        lp_ref = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
        lp_c = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        kl = mx.sum(mx.exp(lp_ref) * (lp_ref - lp_c), axis=-1)
        am_ref = mx.argmax(ref, axis=-1)
        am_c = mx.argmax(logits, axis=-1)
        mx.eval(kl, am_ref, am_c)
        kls.append(np.array(kl))
        agree_ref.append(np.array(am_ref == am_c))
        am_all.append(np.array(am_c))
        del logits, ref, lp_ref, lp_c
        mx.clear_cache()

    kl = np.concatenate(kls)
    ar = np.concatenate(agree_ref)
    print(
        f"engine={args.engine}: KLD(ref||cand) mean {kl.mean():.4f} "
        f"median {np.median(kl):.4f} p95 {np.percentile(kl, 95):.4f} | "
        f"top1-vs-8bpw {100 * ar.mean():.2f}%"
    )
    if args.tag:
        np.save(os.path.join(args.ref_dir, f"argmax_{args.tag}.npy"), np.stack(am_all))
    if args.vs_tag:
        other = np.load(os.path.join(args.ref_dir, f"argmax_{args.vs_tag}.npy"))
        same = (np.stack(am_all) == other).mean()
        print(f"top1 agreement vs {args.vs_tag}: {100 * same:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
