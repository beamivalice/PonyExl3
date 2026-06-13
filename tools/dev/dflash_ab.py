#!/usr/bin/env python3
"""Interleaved A/B: DFlash drafter precision (bf16 vs w8 vs w4 body).

Output is token-identical in every variant (the verify gates it) — the
metric at risk is ACCEPTANCE (tok/cycle), which is also thermal-immune,
unlike tok/s. One process, variants interleaved per rep.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROMPTS = {
    "lru-code": (
        "Write a Python class implementing an LRU cache with get/put "
        "methods and unit tests.",
        256,
    ),
    "reasoning": (
        "If a train leaves at 3pm traveling 60mph and another leaves at 4pm "
        "at 80mph on the same track, when does the second catch the first? "
        "Think step by step.",
        256,
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--dflash", required=True)
    ap.add_argument("--draft", type=int, default=7)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    from mlx_lm.utils import load_tokenizer
    from ponyexl3.mlx.dflash import DFlashDraft
    from ponyexl3.mlx.generate import generate_text
    from ponyexl3.mlx.model import load_model

    model, _ = load_model(args.model, engine="exl3", verbose=False)
    tokenizer = load_tokenizer(Path(args.model))

    drafters = {}
    head = None
    for name, q in (("bf16", None), ("w8", 8), ("w4", 4)):
        d = DFlashDraft(args.dflash)
        if head is None:
            d.quantize_draft(model.language_model.lm_head)
            head = d._draft_head
        else:
            d._draft_head = head  # share the w4 lm_head copy
        if q is not None:
            d.quantize_body(bits=q, group_size=64)
        drafters[name] = d

    refs = {}
    for rep in range(args.reps):
        for vname, drafter in drafters.items():
            for pname, (prompt, n) in PROMPTS.items():
                text, stats = generate_text(
                    model, tokenizer, prompt,
                    max_tokens=n, dflash=drafter, num_draft=args.draft,
                )
                key = pname
                if key not in refs:
                    refs[key] = text
                gate = "✓" if text == refs[key] else "✗ DIFFERS"
                tc = stats.gen_tokens / stats.spec_cycles
                dc = stats.gen_tokens / stats.decode_s
                print(
                    f"{vname:5s} {pname:10s} #{rep}: {tc:5.2f} tok/cycle "
                    f"({stats.spec_accepted}/{stats.spec_drafted})  {dc:5.1f} tok/s  {gate}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
