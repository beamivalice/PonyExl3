#!/usr/bin/env python3
"""Interleaved A/B of spec-decode repair modes (scan vs module replay).

Loads the model once, alternates EXL3_SPEC_REPAIR per run (the trace reads
it at each cycle), checks every run's text against plain greedy.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--mtp", required=True)
    ap.add_argument("-p", "--prompt", default="Write a Python class implementing an LRU cache with get/put methods and unit tests.")
    ap.add_argument("-n", "--max-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--draft", type=int, default=3)
    args = ap.parse_args()

    from mlx_lm.utils import load_tokenizer
    from ponyexl3.mlx.generate import generate_text
    from ponyexl3.mlx.model import load_model
    from ponyexl3.mlx.mtp import load_mtp

    model, config = load_model(args.model, engine="exl3", verbose=False)
    mtp = load_mtp(args.model, config, args.mtp)
    tokenizer = load_tokenizer(Path(args.model))

    def run(mtp_mod, mode=None):
        if mode is not None:
            os.environ["EXL3_SPEC_REPAIR"] = mode
        text, stats = generate_text(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens, mtp=mtp_mod, num_draft=args.draft,
        )
        return text, stats

    ref_text, ref_stats = run(None)
    print(f"plain      : {ref_stats.gen_tokens / ref_stats.decode_s:6.2f} tok/s")

    results = {"scan": [], "module": []}
    for rep in range(args.reps):
        for mode in ("scan", "module"):
            text, stats = run(mtp, mode)
            dc = stats.gen_tokens / stats.decode_s
            ok = "✓" if text == ref_text else "✗ TEXT DIFFERS"
            results[mode].append(dc)
            print(
                f"{mode:7s} #{rep}: {dc:6.2f} tok/s  "
                f"({stats.gen_tokens / stats.spec_cycles:.2f} tok/cycle) {ok}"
            )

    for mode, vals in results.items():
        print(f"{mode:7s} mean: {sum(vals)/len(vals):6.2f} tok/s  (n={len(vals)}: {', '.join(f'{v:.1f}' for v in vals)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
