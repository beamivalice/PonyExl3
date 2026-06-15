#!/usr/bin/env python3
"""Compare EXL3 engines end-to-end: greedy token agreement + logit drift.

Loads engines sequentially (frees the previous one) and reports, for each
engine vs the exact ``exl3`` reference:
  - greedy token agreement over N steps (same forced prefix each step)
  - max / RMS logit difference at the first step

Usage:
  python tools/compare_engines.py MODEL -p "prompt" -n 128 --engines fold16 w8a16
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from ponyexl3.cli._generate_common import require_metal, validate_exl3_model_dir


def run_engine(
    model_dir: str,
    engine: str,
    prompt_ids: list[int],
    steps: int,
    forced: list[int] | None = None,
) -> tuple[list[int], np.ndarray]:
    """Greedy (forced=None) or teacher-forced decode.

    Returns (argmax_tokens, first_step_logits). With ``forced``, the model is
    fed the forced sequence regardless of its own argmax, so every engine sees
    identical prefixes at every step — drift cannot compound.
    """
    import mlx.core as mx

    from ponyexl3.mlx.model import load_model

    model, _ = load_model(model_dir, engine=engine)
    lm = model.language_model
    cache = lm.make_cache()
    toks = mx.array([prompt_ids])
    h = lm.model(toks, cache=cache)
    logits = lm.lm_head(h[:, -1:, :]).astype(mx.float32)
    mx.eval(logits)

    first_logits = logits[0, -1, :]
    tokens = []
    for i in range(steps):
        y = mx.argmax(logits[:, -1, :], axis=-1)
        tokens.append(int(y.item()))
        feed = y if forced is None else mx.array([forced[i]])
        h = lm.model(feed[:, None], cache=cache)
        logits = lm.lm_head(h).astype(mx.float32)
        mx.eval(logits)

    out = (tokens, np.array(first_logits))
    del model, lm, cache, h, logits, first_logits
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("-p", "--prompt", default="Explain why the sky is blue.")
    ap.add_argument("-n", "--steps", type=int, default=128)
    ap.add_argument("--engines", nargs="+", default=["fold16"])
    ap.add_argument("--raw", action="store_true", help="skip the chat template")
    args = ap.parse_args()

    validate_exl3_model_dir(args.model)
    require_metal()
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")

    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(Path(args.model))
    if not args.raw and getattr(tokenizer, "chat_template", None):
        prompt_ids = list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}], add_generation_prompt=True
            )
        )
    else:
        prompt_ids = list(tokenizer.encode(args.prompt))
    if not prompt_ids:
        raise SystemExit("prompt is empty after encoding")

    print(f"reference engine: exl3 ({args.steps} greedy steps, then teacher-forced)")
    ref_tokens, ref_logits = run_engine(args.model, "exl3", prompt_ids, args.steps)

    for engine in args.engines:
        if engine == "exl3":
            print("exl3: skipping (reference engine)")
            continue
        toks, logits = run_engine(
            args.model, engine, prompt_ids, args.steps, forced=ref_tokens
        )
        agree = sum(a == b for a, b in zip(toks, ref_tokens))
        dl = np.abs(logits - ref_logits)
        print(
            f"{engine}: forced argmax agreement {agree}/{args.steps}"
            f" | step-0 logits: max|d|={dl.max():.4f} rms={float(np.sqrt((dl**2).mean())):.5f}"
            f" (ref logit scale rms={float(np.sqrt((ref_logits**2).mean())):.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
