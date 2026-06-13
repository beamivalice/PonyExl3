#!/usr/bin/env python3
"""One (model, config) throughput measurement → a JSON RESULT line.

Drives ``generate_text`` and reports prefill / decode tok/s from its stats.
Run one config per process (clean engine env + drafter load); a driver
interleaves configs and keeps the peak (thermally-clean) reading.

Modes:
  --mode prefill   long realistic prompt, few generated tokens → prefill_tps
  --mode decode    reasoning prompt, many generated tokens     → decode_tps
                   (drafter flags only affect decode)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# A ~realistic generation prompt — structured output (good, honest spec
# acceptance: copy-ish in the code, novel in the prose).
_DECODE_PROMPT = (
    "Implement an LRU cache class in Python with get and put in O(1). "
    "Then explain the data structures you chose, walk through an eviction "
    "step by step, and analyze the time and space complexity."
)

# Long prefill prompt: a real passage repeated to a few thousand tokens.
_PREFILL_SEED = (
    "The trellis quantization format stores weights as bit-packed transition "
    "indices decoded on the fly inside the GEMV kernel. Each 16x16 tile holds "
    "256 codewords; a procedural codebook maps each 16-bit state to a weight "
    "with three instructions, so no lookup table touches memory. A blockwise "
    "Walsh-Hadamard transform on both activation dimensions spreads outliers "
    "before quantization, and per-feature sign flips restore the distribution "
    "on decode. The result keeps four-bit memory traffic while preserving the "
    "numerics of the original sixteen-bit matmul to within a small tolerance. "
)


def _build_prefill_prompt(target_tokens: int, tokenizer) -> str:
    text = _PREFILL_SEED
    while len(tokenizer.encode(text)) < target_tokens:
        text += _PREFILL_SEED
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=("prefill", "decode"), default="decode")
    ap.add_argument("--engine", default="exl3")
    ap.add_argument("--mtp", default="off")
    ap.add_argument("--dflash", default=None)
    ap.add_argument("--eagle3", default=None)
    ap.add_argument("--lookup", action="store_true")
    ap.add_argument("--draft-w4", action="store_true")
    ap.add_argument("--draft", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--warmup-tokens", type=int, default=16)
    ap.add_argument("--prefill-tokens", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm.utils import load_tokenizer

    from ponyexl3.mlx.generate import generate_text
    from ponyexl3.mlx.model import load_model

    t0 = time.perf_counter()
    # warm=False: the persistent fp16 W cache is retired by default
    # (EXL3_WCACHE=0); decode/verify read the 4-bit trellis directly, so
    # warming only burns time. Explicit here so the reported footprint is
    # the honest 4-bit-resident number.
    model, config = load_model(args.model, engine=args.engine, warm=False, verbose=False)
    load_s = time.perf_counter() - t0
    tokenizer = load_tokenizer(Path(args.model))

    text_cfg = config.get("text_config", config)
    eos = text_cfg.get("eos_token_id")
    extra_eos = tuple(eos) if isinstance(eos, list) else ((eos,) if eos is not None else ())

    # Drafters (decode only) — mirrors the generate CLI wiring.
    mtp = dflash = eagle3 = None
    if args.mode == "decode":
        if args.dflash:
            from ponyexl3.mlx.dflash import DFlashDraft

            dflash = DFlashDraft(args.dflash)
            if args.draft_w4:
                dflash.quantize_draft(model.language_model.lm_head, cache_dir=args.model)
            if args.draft == 3:
                args.draft = 7
        elif args.eagle3:
            from ponyexl3.mlx.eagle3 import Eagle3Draft

            eagle3 = Eagle3Draft(args.eagle3)
            if args.draft_w4:
                eagle3.quantize_draft()
        elif args.mtp != "off":
            from ponyexl3.mlx.mtp import load_mtp, quantize_draft

            mtp = load_mtp(args.model, config, None if args.mtp == "auto" else args.mtp)
            if mtp is not None and args.draft_w4:
                quantize_draft(mtp, model.language_model.lm_head, cache_dir=args.model)

    if args.mode == "prefill":
        prompt = _build_prefill_prompt(args.prefill_tokens, tokenizer)
        max_tokens, warmup = 8, 4
        use_chat_template = False
    else:
        prompt = _DECODE_PROMPT
        max_tokens, warmup = args.max_tokens, args.warmup_tokens
        use_chat_template = True

    common = dict(
        temp=0.0,
        prefill_chunk=args.prefill_chunk,
        use_chat_template=use_chat_template,
        extra_eos=extra_eos,
        mtp=mtp,
        num_draft=args.draft,
        lookup=args.lookup,
        eagle3=eagle3,
        dflash=dflash,
    )

    # Warmup (JIT + clocks), discarded.
    generate_text(model, tokenizer, prompt, max_tokens=warmup, **common)
    mx.synchronize() if hasattr(mx, "synchronize") else None

    # Resident footprint after load+warmup, and reset peak so the reported
    # peak reflects the INFERENCE working set — not the one-time load
    # transient (mmap source + device staging spikes ~2x during conversion).
    mx.clear_cache()
    active_gb = round(mx.get_active_memory() / 1024**3, 2)
    mx.reset_peak_memory()

    _, stats = generate_text(model, tokenizer, prompt, max_tokens=max_tokens, **common)
    peak_gb = round(mx.get_peak_memory() / 1024**3, 2)

    out = {
        "label": args.label,
        "mode": args.mode,
        "engine": args.engine,
        "load_s": round(load_s, 2),
        "active_gb": active_gb,
        "peak_gb": peak_gb,
        "prompt_tokens": stats.prompt_tokens,
        "gen_tokens": stats.gen_tokens,
        "prefill_tps": round(stats.prompt_tokens / stats.prefill_s, 1) if stats.prefill_s else None,
        "decode_tps": round(stats.gen_tokens / stats.decode_s, 1) if stats.decode_s else None,
        "accept": stats.spec_accepted,
        "drafted": stats.spec_drafted,
        "tok_per_cycle": round(stats.gen_tokens / stats.spec_cycles, 2) if stats.spec_cycles else None,
    }
    print("RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
