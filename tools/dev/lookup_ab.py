#!/usr/bin/env python3
"""Interleaved A/B: plain decode vs n-gram lookup decoding, one process.

Per prompt: alternate plain/lookup runs (thermal-fair), gate every lookup
run's text against the plain reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROMPTS = {
    "doc-edit": (
        "Here is a Python file:\n\n```python\n{src}\n```\n\n"
        "Add a one-line docstring to every function that lacks one and "
        "output the complete modified file in a single python code block.",
        512,
    ),
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
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    from mlx_lm.utils import load_tokenizer
    from ponyexl3.mlx.generate import generate_text
    from ponyexl3.mlx.model import load_model

    model, _ = load_model(args.model, engine="exl3", verbose=False)
    tokenizer = load_tokenizer(Path(args.model))
    src = (Path(__file__).resolve().parents[1] / "mlx" / "hadamard.py").read_text()

    for name, (tmpl, n) in PROMPTS.items():
        prompt = tmpl.format(src=src) if "{src}" in tmpl else tmpl
        results = {"plain": [], "lookup": []}
        ref_text = None
        accept = ""
        for rep in range(args.reps):
            for mode in ("plain", "lookup"):
                text, stats = generate_text(
                    model, tokenizer, prompt, max_tokens=n, lookup=(mode == "lookup")
                )
                dc = stats.gen_tokens / stats.decode_s
                results[mode].append(dc)
                if mode == "plain" and ref_text is None:
                    ref_text = text
                gate = "" if text == ref_text else "  ✗ TEXT DIFFERS"
                if mode == "lookup" and stats.spec_cycles:
                    accept = f" [{stats.spec_accepted}/{stats.spec_drafted} drafts, {stats.spec_cycles} cycles]"
                print(f"  {name:10s} {mode:6s} #{rep}: {dc:6.2f} tok/s{gate}")
        p = sum(results["plain"]) / len(results["plain"])
        l = sum(results["lookup"]) / len(results["lookup"])
        print(f"  {name:10s} mean: plain {p:.2f} / lookup {l:.2f}  ({(l/p-1)*100:+.1f}%){accept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
