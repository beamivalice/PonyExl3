#!/usr/bin/env python3
"""Run end-to-end text generation on an EXL3 checkpoint via MLX.

Usage:
  ponyexl3-generate /path/to/exl3/model -p "Hello!" -n 128
  ponyexl3-generate MODEL -p "..." --temp 0.7 --raw --no-warm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="EXL3 model directory")
    ap.add_argument("-p", "--prompt", default="Why is the sky blue?")
    ap.add_argument("-n", "--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--raw", action="store_true", help="skip the chat template")
    ap.add_argument(
        "--engine",
        default="exl3",
        choices=("exl3", "fold16", "w8a16", "w4a16", "w4gptq"),
        help="exl3=exact trellis GEMV (default — fastest AND smallest since v9); "
        "fold16=exact fp16 fold; w8a16/w4a16=lossy requantization "
        "(validate before trusting)",
    )
    ap.add_argument(
        "--mtp",
        default="auto",
        help="MTP draft weights for speculative decoding: 'auto' (look in the "
        "model dir), 'off', or a path to the MTP safetensors file/dir",
    )
    ap.add_argument(
        "--draft", type=int, default=3, help="draft tokens per speculative cycle"
    )
    ap.add_argument(
        "--lookup",
        action="store_true",
        help="draft-free n-gram lookup speculation (greedy only, no MTP "
        "weights needed; token-identical to plain decoding)",
    )
    ap.add_argument(
        "--draft-w4",
        action="store_true",
        help="quantize the DRAFT side (MTP body + a w4 lm_head copy, "
        "+0.45 GB) — drafts are verify-gated, output stays token-identical",
    )
    ap.add_argument(
        "--eagle3",
        default=None,
        help="EAGLE-3 draft head directory (SpecForge export, bf16) — "
        "takes precedence over --mtp; output stays token-identical",
    )
    ap.add_argument(
        "--dflash",
        default=None,
        help="DFlash block-drafter directory (bf16 or exl3) — takes precedence "
        "over --eagle3/--mtp; default --draft becomes 7 (rows-8 verify); "
        "output stays token-identical",
    )
    ap.add_argument(
        "--dflash-quant",
        default="w8",
        choices=("bf16", "w8", "w4"),
        help="drafter-body precision (bf16 mode only): w8 measured draft-"
        "identical to bf16 at half the memory (default); w4 saves more but "
        "acceptance jitters ±8%% by prompt. Output is identical regardless.",
    )
    ap.add_argument("--no-warm", action="store_true", help="skip weight-cache warmup")
    ap.add_argument("-q", "--quiet", action="store_true", help="suppress load progress")
    args = ap.parse_args()

    from mlx_lm.utils import load_tokenizer

    from ponyexl3.mlx.generate import generate_text
    from ponyexl3.mlx.model import describe, load_model

    tic = time.perf_counter()
    model, config = load_model(
        args.model,
        engine=args.engine,
        warm=not args.no_warm,
        verbose=False,
        report_errors=args.engine in ("w8a16", "w4a16"),
    )
    if not args.quiet:
        print(
            f"[load] {time.perf_counter() - tic:.1f}s engine={args.engine} — {describe(model)}",
            file=sys.stderr,
        )

    dflash = None
    if args.dflash and args.engine in ("exl3", "w4gptq") and args.temp <= 0.0:
        from ponyexl3.mlx.dflash import DFlashDraft

        dflash = DFlashDraft(args.dflash)
        if args.dflash_quant != "bf16":
            import os as _os

            _bits = 8 if args.dflash_quant == "w8" else 4
            _src = _os.path.getsize(_os.path.join(args.dflash, "model.safetensors"))
            dflash.quantize_body(
                bits=_bits,
                cache_path=_os.path.join(
                    args.dflash, ".pony_cache", f"body_w{_bits}g64_{_src}.safetensors"
                ),
            )
        if args.draft_w4:
            dflash.quantize_draft(model.language_model.lm_head, cache_dir=args.model)
        if "--draft" not in sys.argv and args.draft == 3:
            args.draft = 7
        if not args.quiet:
            print(
                f"[dflash] block drafter loaded (k={args.draft}) — "
                "speculative decoding on"
                + (" (w4 draft head)" if args.draft_w4 else ""),
                file=sys.stderr,
            )

    eagle3 = None
    if dflash is None and args.eagle3 and args.engine == "exl3" and args.temp <= 0.0:
        from ponyexl3.mlx.eagle3 import Eagle3Draft

        eagle3 = Eagle3Draft(args.eagle3)
        if args.draft_w4:
            eagle3.quantize_draft()
        if not args.quiet:
            print(
                "[eagle3] draft head loaded — speculative decoding on"
                + (" (w4 draft side)" if args.draft_w4 else ""),
                file=sys.stderr,
            )

    mtp = None
    if dflash is None and eagle3 is None and args.mtp != "off" and args.engine == "exl3" and args.temp <= 0.0:
        from ponyexl3.mlx.mtp import load_mtp

        mtp = load_mtp(
            args.model, config, None if args.mtp == "auto" else args.mtp
        )
        if mtp is not None and args.draft_w4:
            from ponyexl3.mlx.mtp import quantize_draft

            quantize_draft(mtp, model.language_model.lm_head, cache_dir=args.model)
        if not args.quiet and mtp is not None:
            print(
                "[mtp] draft head loaded — speculative decoding on"
                + (" (w4 draft side)" if args.draft_w4 else ""),
                file=sys.stderr,
            )

    tokenizer = load_tokenizer(Path(args.model))

    extra_eos = ()
    text_cfg = config.get("text_config", config)
    eos = text_cfg.get("eos_token_id")
    if eos is not None:
        extra_eos = tuple(eos) if isinstance(eos, list) else (eos,)

    _, stats = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        prefill_chunk=args.prefill_chunk,
        use_chat_template=not args.raw,
        extra_eos=extra_eos,
        on_segment=lambda s: print(s, end="", flush=True),
        mtp=mtp,
        num_draft=args.draft,
        lookup=args.lookup,
        eagle3=eagle3,
        dflash=dflash,
    )
    print()
    if not args.quiet:
        print(f"[gen] {stats.summary()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
