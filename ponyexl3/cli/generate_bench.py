#!/usr/bin/env python3
"""Prefill/decode throughput sweep at fixed context lengths.

Runs generation with prefill sizes 1k–32k (default) and 128 generated tokens
per row. Prompt text comes from ``--prompt-file`` (default: ``README.md`` in
cwd or the project root). Accepts the same engine and speculative-decode flags
as ``ponyexl3-generate``.

Usage:
  ponyexl3-generate-bench /path/to/model
  ponyexl3-generate-bench MODEL --dflash /path/to/dflash --raw
  ponyexl3-generate-bench MODEL --prompt-file docs/conversion_tool.md --json
"""

from __future__ import annotations

import argparse
import json
import sys

from ponyexl3.cli._generate_common import (
    PREFILL_BENCH_SIZES,
    add_generate_arguments,
    build_prefill_prompt_ids,
    check_context_limit,
    load_generate_stack,
    max_position_embeddings,
    resolve_prompt_file,
    validate_generate_cli_args,
)


def _run_row(
    stack,
    prompt_ids: list[int],
    args: argparse.Namespace,
    *,
    expected_prefill: int,
) -> dict[str, object]:
    import mlx.core as mx

    from ponyexl3.mlx.generate import generate_text

    try:
        _, stats = generate_text(
            stack.model,
            stack.tokenizer,
            prompt="",
            prompt_ids=prompt_ids,
            max_tokens=args.gen_tokens,
            temp=args.temp,
            prefill_chunk=args.prefill_chunk,
            use_chat_template=False,
            extra_eos=stack.extra_eos,
            mtp=stack.mtp,
            num_draft=stack.draft,
            lookup=args.lookup,
            eagle3=stack.eagle3,
            dflash=stack.dflash,
            max_context=max_position_embeddings(stack.config),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if stats.prompt_tokens != expected_prefill:
        raise SystemExit(
            f"internal error: expected prefill {expected_prefill} tokens, "
            f"got {stats.prompt_tokens}"
        )

    if hasattr(mx, "synchronize"):
        mx.synchronize()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    prefill_tps = stats.prompt_tokens / stats.prefill_s if stats.prefill_s else 0.0
    decode_tps = stats.gen_tokens / stats.decode_s if stats.decode_s else 0.0
    row: dict[str, object] = {
        "prefill_tokens": stats.prompt_tokens,
        "gen_tokens": stats.gen_tokens,
        "prefill_s": round(stats.prefill_s, 3),
        "decode_s": round(stats.decode_s, 3),
        "prefill_tps": round(prefill_tps, 1),
        "decode_tps": round(decode_tps, 1),
        "finish_reason": stats.finish_reason,
    }
    if stats.spec_cycles:
        row["spec_accepted"] = stats.spec_accepted
        row["spec_drafted"] = stats.spec_drafted
        row["tok_per_cycle"] = round(stats.gen_tokens / stats.spec_cycles, 2)
    return row


def _print_table(rows: list[dict[str, object]]) -> None:
    headers = (
        "prefill",
        "prefill tok/s",
        "decode tok/s",
        "gen",
        "tok/cycle",
    )
    print(
        f"{headers[0]:>8}  {headers[1]:>12}  {headers[2]:>12}  "
        f"{headers[3]:>5}  {headers[4]:>9}"
    )
    print("-" * 56)
    for row in rows:
        cycle = row.get("tok_per_cycle")
        cycle_s = f"{cycle:.2f}" if cycle is not None else "-"
        print(
            f"{int(row['prefill_tokens']):>8}  "
            f"{float(row['prefill_tps']):>12.1f}  "
            f"{float(row['decode_tps']):>12.1f}  "
            f"{int(row['gen_tokens']):>5}  {cycle_s:>9}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="EXL3 model directory")
    add_generate_arguments(ap, with_prompt=False, with_max_tokens=False)
    ap.add_argument(
        "--prompt-file",
        default=None,
        help="text file for prompt body (default: README.md in cwd or project root)",
    )
    ap.add_argument(
        "--prefill-sizes",
        default=",".join(str(n) for n in PREFILL_BENCH_SIZES),
        help="comma-separated prefill token counts (default: 1k,2k,4k,8k,16k,32k)",
    )
    ap.add_argument(
        "--gen-tokens",
        type=int,
        default=128,
        help="generated tokens per prefill size (default: 128)",
    )
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="run a discarded 1k-prefill warmup before the sweep",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    if not args.raw:
        print(
            "[bench] tip: pass --raw for exact prefill token counts "
            "(chat template adds overhead)",
            file=sys.stderr,
        )

    try:
        prompt_path = resolve_prompt_file(args.prompt_file)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read prompt file {prompt_path}: {exc}") from exc
    if not prompt_text.strip():
        raise SystemExit(f"prompt file is empty: {prompt_path}")

    try:
        sizes = [int(x.strip()) for x in args.prefill_sizes.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --prefill-sizes: {exc}") from exc
    if not sizes or any(n <= 0 for n in sizes):
        raise SystemExit("--prefill-sizes must be positive integers")

    validate_generate_cli_args(args, bench=True)

    if not args.quiet:
        print(f"[bench] prompt file: {prompt_path}", file=sys.stderr)

    stack = load_generate_stack(args)
    for n in sizes:
        check_context_limit(
            n,
            args.gen_tokens,
            stack.config,
            label=f"prefill={n}",
        )

    if args.warmup:
        try:
            warm_ids = build_prefill_prompt_ids(
                prompt_text, 1024, stack.tokenizer, raw=args.raw
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        _run_row(stack, warm_ids, args, expected_prefill=1024)
        if not args.quiet:
            print("[bench] warmup done", file=sys.stderr)

    rows: list[dict[str, object]] = []
    for n in sizes:
        try:
            prompt_ids = build_prefill_prompt_ids(
                prompt_text, n, stack.tokenizer, raw=args.raw
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not args.quiet:
            print(f"[bench] prefill={n} gen={args.gen_tokens} ...", file=sys.stderr)
        row = _run_row(stack, prompt_ids, args, expected_prefill=n)
        rows.append(row)

    if args.json:
        print(json.dumps({"prompt_file": str(prompt_path), "rows": rows}, indent=2))
    else:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
