#!/usr/bin/env python3
"""Compare MLX EXL3 forward logits against native exllamav3 reference .npz.

Reference files are produced by export_reference.py on CUDA (flash_attn_nc,
fixed seed, last-position logits). This script replays the stored input_ids
through pony's MLX loader and reports bit-exact / near-exact agreement.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ponyexl3.types import MlxLmModel

import numpy as np
from typing import Any


def forward_logits(model: MlxLmModel, input_ids: np.ndarray) -> np.ndarray:
    import mlx.core as mx

    lm = model.language_model
    cache = lm.make_cache()
    ids = mx.array(input_ids.astype(np.int64))
    h = lm.model(ids, cache=cache)
    logits = lm.lm_head(h[:, -1:, :]).astype(mx.float32)
    mx.eval(logits)
    return np.array(logits[0, 0, :], dtype=np.float32)


def compare(ref_logits: np.ndarray, cand_logits: np.ndarray) -> dict[str, Any]:
    diff = cand_logits - ref_logits
    abs_diff = np.abs(diff)
    ref_f32: np.ndarray = np.asarray(ref_logits, dtype=np.float32)
    cand_f32: np.ndarray = np.asarray(cand_logits, dtype=np.float32)
    ref_u32: np.ndarray = ref_f32.view(np.uint32)
    cand_u32: np.ndarray = cand_f32.view(np.uint32)
    exact = np.array_equal(ref_u32, cand_u32)
    top1_ref = int(ref_logits.argmax())
    top1_cand = int(cand_logits.argmax())
    return {
        "bit_exact": bool(exact),
        "max_abs": float(abs_diff.max()),
        "rms": float(np.sqrt((diff**2).mean())),
        "top1_ref": top1_ref,
        "top1_cand": top1_cand,
        "top1_match": top1_ref == top1_cand,
        "n_mismatch_bits": int(np.sum(ref_u32 != cand_u32)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path, help=".npz from export_reference.py")
    ap.add_argument("-m", "--model-dir", type=str, default=None,
                    help="override checkpoint path (default: infer from npz)")
    ap.add_argument("--engine", default="exl3", choices=("exl3", "fold16"))
    ap.add_argument("--warm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save", type=Path, default=None, help="write candidate logits .npz")
    args = ap.parse_args()

    ref = np.load(args.reference, allow_pickle=True)
    input_ids = ref["input_ids"]
    ref_logits = ref["logits"].astype(np.float32)
    if ref_logits.ndim == 2:
        ref_logits = ref_logits[-1]

    model_dir = args.model_dir
    if model_dir is None:
        stem = args.reference.stem
        for suffix in ("_win", "_linux", "_mac", "_cuda"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        name = stem
        for candidate in (
            os.path.expanduser(f"~/llm/models/{name}"),
            os.path.expanduser(f"~/models/{name}"),
        ):
            if os.path.isdir(candidate):
                model_dir = candidate
                break
        if model_dir is None:
            raise SystemExit(
                f"could not resolve model dir for {name!r}; pass -m/--model-dir"
            )

    from ponyexl3.mlx.model import describe, load_model

    print(f"reference: {args.reference}")
    print(f"model:     {model_dir}")
    print(f"engine:    {args.engine}")
    print(f"seq_len:   {input_ids.shape[-1]}  vocab: {ref_logits.shape[-1]}")
    if "attn_mode" in ref:
        print(f"ref attn:  {ref['attn_mode']}")

    model, _ = load_model(
        model_dir,
        engine=args.engine,
        warm=args.warm,
        verbose=False,
    )
    print(f"loaded:    {describe(model)}")

    cand_logits = forward_logits(model, input_ids)
    if cand_logits.shape != ref_logits.shape:
        raise SystemExit(
            f"logit shape mismatch: ref {ref_logits.shape} vs cand {cand_logits.shape}"
        )

    stats = compare(ref_logits, cand_logits)
    print()
    print(f"bit-exact:  {stats['bit_exact']}")
    print(f"max |d|:    {stats['max_abs']:.6g}")
    print(f"rms:        {stats['rms']:.6g}")
    print(f"top-1:      ref={stats['top1_ref']} cand={stats['top1_cand']} match={stats['top1_match']}")
    if not stats["bit_exact"]:
        print(f"fp32 words differing: {stats['n_mismatch_bits']}/{ref_logits.size}")

    if args.save:
        np.savez(
            args.save,
            input_ids=input_ids,
            logits=cand_logits[np.newaxis, :],
            engine=np.array(args.engine),
            model_dir=np.array(model_dir),
        )
        print(f"wrote candidate logits to {args.save}")

    return 0 if stats["bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
