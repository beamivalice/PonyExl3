#!/usr/bin/env python3
"""Run end-to-end text generation on an EXL3 checkpoint via MLX.

Usage:
  ponyexl3-generate /path/to/exl3/model -p "Hello!" -n 128
  ponyexl3-generate MODEL -p "..." --temp 0.7 --raw --no-warm
"""

from __future__ import annotations

import argparse
import sys

from ponyexl3.cli._generate_common import add_generate_arguments, load_generate_stack


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="EXL3 model directory")
    add_generate_arguments(ap)
    args = ap.parse_args()

    from ponyexl3.mlx.generate import generate_text

    stack = load_generate_stack(args)

    _, stats = generate_text(
        stack.model,
        stack.tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        prefill_chunk=args.prefill_chunk,
        use_chat_template=not args.raw,
        extra_eos=stack.extra_eos,
        on_segment=lambda s: print(s, end="", flush=True),
        mtp=stack.mtp,
        num_draft=stack.draft,
        lookup=args.lookup,
        eagle3=stack.eagle3,
        dflash=stack.dflash,
    )
    print()
    if not args.quiet:
        print(f"[gen] {stats.summary()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
