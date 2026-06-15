"""Shared argparse and model/drafter setup for generate CLIs."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ponyexl3.types import DraftModule, MlxLmModel, Tokenizer

PREFILL_BENCH_SIZES = (1024, 2048, 4096, 8192, 16384, 32768)


def add_generate_arguments(
    ap: argparse.ArgumentParser,
    *,
    with_prompt: bool = True,
    with_max_tokens: bool = True,
) -> None:
    if with_prompt:
        ap.add_argument("-p", "--prompt", default="Why is the sky blue?")
    if with_max_tokens:
        ap.add_argument("-n", "--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--raw", action="store_true", help="skip the chat template")
    ap.add_argument(
        "--engine",
        default="exl3",
        choices=("exl3", "fold16", "w8a16", "w4a16", "w4gptq"),
        help="exl3=exact trellis GEMV (default); fold16=exact fp16 fold; "
        "w8a16/w4a16=lossy requantization",
    )
    ap.add_argument(
        "--mtp",
        default="auto",
        help="MTP draft weights: 'auto', 'off', or a path",
    )
    ap.add_argument("--draft", type=int, default=3, help="draft tokens per speculative cycle")
    ap.add_argument(
        "--lookup",
        action="store_true",
        help="draft-free n-gram lookup speculation (greedy only)",
    )
    ap.add_argument(
        "--draft-w4",
        action="store_true",
        help="quantize the draft side (verify-gated; output unchanged)",
    )
    ap.add_argument("--eagle3", default=None, help="EAGLE-3 draft head directory")
    ap.add_argument("--dflash", default=None, help="DFlash block-drafter directory")
    ap.add_argument(
        "--dflash-quant",
        default="w8",
        choices=("bf16", "w8", "w4"),
        help="DFlash body precision when drafter is bf16",
    )
    ap.add_argument("--no-warm", action="store_true", help="skip weight-cache warmup")
    ap.add_argument("-q", "--quiet", action="store_true", help="suppress load progress")


@dataclass
class GenerateStack:
    model: MlxLmModel
    config: dict[str, Any]
    tokenizer: Tokenizer
    mtp: DraftModule | None
    eagle3: DraftModule | None
    dflash: DraftModule | None
    extra_eos: tuple[int, ...]
    draft: int


def load_generate_stack(args: argparse.Namespace) -> GenerateStack:
    from mlx_lm.utils import load_tokenizer

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

    dflash: DraftModule | None = None
    draft = args.draft
    if args.dflash and args.engine in ("exl3", "w4gptq"):
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
        if "--draft" not in sys.argv and draft == 3:
            draft = 7
        if not args.quiet:
            print(
                f"[dflash] block drafter loaded (k={draft}) — speculative decoding on"
                + (" (w4 draft head)" if args.draft_w4 else ""),
                file=sys.stderr,
            )

    eagle3: DraftModule | None = None
    if dflash is None and args.eagle3 and args.engine == "exl3":
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

    mtp: DraftModule | None = None
    if dflash is None and eagle3 is None and args.mtp != "off" and args.engine == "exl3":
        from ponyexl3.mlx.mtp import load_mtp

        mtp = load_mtp(args.model, config, None if args.mtp == "auto" else args.mtp)
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

    text_cfg = config.get("text_config", config)
    eos = text_cfg.get("eos_token_id")
    extra_eos: tuple[int, ...] = ()
    if eos is not None:
        extra_eos = tuple(eos) if isinstance(eos, list) else (eos,)

    return GenerateStack(
        model=model,
        config=config,
        tokenizer=tokenizer,
        mtp=mtp,
        eagle3=eagle3,
        dflash=dflash,
        extra_eos=extra_eos,
        draft=draft,
    )


def resolve_prompt_file(path: str | None) -> Path:
    if path is not None:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"prompt file not found: {p}")
        return p
    for candidate in (
        Path.cwd() / "README.md",
        Path(__file__).resolve().parents[2] / "README.md",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("no README.md in cwd or project root; pass --prompt-file")


def encode_prompt_text(text: str, tokenizer: Tokenizer, *, raw: bool) -> list[int]:
    if raw:
        return list(tokenizer.encode(text))
    return list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
        )
    )


def build_prefill_prompt_ids(
    text: str,
    target_tokens: int,
    tokenizer: Tokenizer,
    *,
    raw: bool,
) -> list[int]:
    base = encode_prompt_text(text, tokenizer, raw=raw)
    if not base:
        raise ValueError("prompt file encodes to zero tokens")
    ids: list[int] = []
    while len(ids) < target_tokens:
        need = target_tokens - len(ids)
        ids.extend(base if need >= len(base) else base[:need])
    return ids[:target_tokens]
