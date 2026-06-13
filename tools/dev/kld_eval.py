#!/usr/bin/env python3
"""Behavioral accuracy: KLD(reference ‖ candidate) on wikitext-2 test.

The drift audit (docs/drifts_investigation.md) established KLD on real text as
the correct cross-engine quality metric — bit-exact logits are unattainable
across independently-scheduled engines, and random-token argmax is one ulp
deep. This measures how much an EXL3 quant diverges from its bf16 original.

Two phases keep each model in its own process (the bf16 reference is 50-70 GB):

    # 1. reference (bf16 original, loaded via mlx_lm)
    python tools/dev/kld_eval.py --mode ref \
      --model /path/to/Qwen3.6-27B --loader mlxlm --ref-dir /tmp/kld_27b

    # 2. candidate (EXL3, loaded via ponyexl3) — prints KLD + top-1 agreement
    python tools/dev/kld_eval.py --mode cand \
      --model /path/to/Qwen3.6-27B-exl3-4.15bpw --loader ponyexl3 \
      --ref-dir /tmp/kld_27b --tokenizer /path/to/Qwen3.6-27B
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np


def get_seqs(tokenizer, n: int, seqlen: int) -> list[list[int]]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer.encode(text)
    return [ids[i : i + seqlen] for i in range(0, n * seqlen, seqlen)][:n]


def load_any(model_dir: str, loader: str, engine: str):
    """Return (model, forward_logits_fn, tokenizer_or_None)."""
    if loader == "mlxlm":
        from mlx_lm import load

        model, tokenizer = load(model_dir)

        def fwd(seq):
            out = model(mx.array([seq]))
            logits = out[0] if not hasattr(out, "logits") else out.logits[0]
            return logits.astype(mx.float32)

        return model, fwd, tokenizer

    from ponyexl3.mlx.model import load_model

    model, _ = load_model(model_dir, engine=engine, warm=False, verbose=False)
    lm = model.language_model

    def fwd(seq):
        cache = lm.make_cache()
        h = lm.model(mx.array([seq]), cache=cache)
        return lm.lm_head(h)[0].astype(mx.float32)

    return model, fwd, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("ref", "cand"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--loader", choices=("mlxlm", "ponyexl3"), required=True)
    ap.add_argument("--engine", default="exl3")
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--tokenizer", default=None, help="tokenizer dir (default: --model)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.ref_dir, exist_ok=True)

    from mlx_lm.utils import load_tokenizer

    tok_dir = args.tokenizer or args.model
    tokenizer = load_tokenizer(Path(tok_dir))
    seqs = get_seqs(tokenizer, args.seqs, args.seqlen)

    model, fwd, _ = load_any(args.model, args.loader, args.engine)
    peak0 = mx.get_peak_memory()

    if args.mode == "ref":
        for i, s in enumerate(seqs):
            logits = fwd(s).astype(mx.float16)
            mx.eval(logits)
            mx.save_safetensors(os.path.join(args.ref_dir, f"ref_{i}.safetensors"), {"l": logits})
            mx.clear_cache()
        meta = {"model": args.model, "seqs": len(seqs), "seqlen": args.seqlen}
        Path(os.path.join(args.ref_dir, "meta.json")).write_text(json.dumps(meta))
        print(f"reference saved: {len(seqs)} x {args.seqlen} from {args.model}")
        print(f"peak {mx.get_peak_memory() / 1024**3:.1f} GB")
        return 0

    # candidate: KLD(ref || cand) + agreement
    kls, agree, cand_nll, ref_nll = [], [], [], []
    for i, s in enumerate(seqs):
        cand = fwd(s)
        ref = mx.load(os.path.join(args.ref_dir, f"ref_{i}.safetensors"))["l"].astype(mx.float32)
        V = min(ref.shape[-1], cand.shape[-1])
        ref, cand = ref[..., :V], cand[..., :V]
        lp_ref = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
        lp_c = cand - mx.logsumexp(cand, axis=-1, keepdims=True)
        p_ref = mx.exp(lp_ref)
        kl = mx.sum(p_ref * (lp_ref - lp_c), axis=-1)            # KLD(ref||cand) per token
        am_ref, am_c = mx.argmax(ref, axis=-1), mx.argmax(cand, axis=-1)
        mx.eval(kl, am_ref, am_c)
        kls.append(np.array(kl))
        agree.append(np.array(am_ref == am_c))
        del cand, ref, lp_ref, lp_c, p_ref
        mx.clear_cache()

    kl = np.concatenate(kls)
    ag = np.concatenate(agree)
    out = {
        "label": args.label or Path(args.model).name,
        "kld_mean": round(float(kl.mean()), 5),
        "kld_median": round(float(np.median(kl)), 5),
        "kld_p95": round(float(np.percentile(kl, 95)), 5),
        "top1_agree_pct": round(100 * float(ag.mean()), 2),
        "tokens": int(kl.size),
        "peak_gb": round(mx.get_peak_memory() / 1024**3, 2),
    }
    print("KLD " + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
