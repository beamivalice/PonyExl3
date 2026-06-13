#!/usr/bin/env python3
"""Verify an MLX-exported .npz against native exllamav3 on CUDA (or either direction).

CUDA team: compare your forward against the Mac ``*_mac.npz`` pony export.
Mac team: compare MLX against the Windows ``*_win.npz`` CUDA export.

Usage (CUDA host):
  python exl3/reference/verify_reference.py \
    exl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_mac.npz \
    -m /path/to/Qwen3.6-35B-A3B-exl3-4.00bpw

Usage (Mac):
  python exl3/reference/verify_reference.py \
    exl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win.npz \
    -m /path/to/Qwen3.6-35B-A3B-exl3-4.00bpw --mlx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ponyexl3.types import ExLlamaModel

import numpy as np

if __package__ in (None, ""):
    from _cuda_common import (
        forward_params,
        input_ids_to_torch,
        load_exllama_model,
        load_input_ids,
        mask_logits,
    )
    from compare_reference import compare, forward_logits
else:
    from ponyexl3.reference._cuda_common import (
        forward_params,
        input_ids_to_torch,
        load_exllama_model,
        load_input_ids,
        mask_logits,
    )
    from ponyexl3.reference.compare_reference import compare, forward_logits


def _cuda_logits(model: ExLlamaModel, input_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    import torch

    params = forward_params(attn_mode="flash_attn_nc")
    with torch.inference_mode():
        out = model.forward(input_ids_to_torch(input_ids), params)
        mask_logits(out, vocab_size)
        row = out[0, -1, :vocab_size].float().cpu().numpy()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path, help=".npz with input_ids + logits")
    ap.add_argument("-m", "--model-dir", type=str, required=True)
    ap.add_argument("--mlx", action="store_true",
                    help="recompute candidate with MLX instead of CUDA")
    ap.add_argument("--engine", default="exl3")
    args = ap.parse_args()

    ref = np.load(args.reference, allow_pickle=True)
    input_ids = load_input_ids(args.reference)
    ref_logits = ref["logits"].astype(np.float32)
    if ref_logits.ndim == 2:
        ref_logits = ref_logits[-1]
    vocab_size = int(ref["vocab_size"]) if "vocab_size" in ref else ref_logits.shape[-1]

    print(f"reference file: {args.reference}")
    print(f"  runtime tag: {ref['runtime'] if 'runtime' in ref else ref.get('attn_mode', '?')}")
    print(f"  platform:    {ref['platform'] if 'platform' in ref else '?'}")
    print(f"  model_dir:   {ref['model_dir'] if 'model_dir' in ref else '?'}")
    print(f"  seq_len:     {input_ids.shape[-1]}  vocab: {vocab_size}")
    print(f"  ref top1:    {int(ref_logits.argmax())}")

    if args.mlx:
        from ponyexl3.mlx.model import load_model

        model, _cfg = load_model(args.model_dir, engine=args.engine, warm=True, verbose=False)
        cand = forward_logits(model, input_ids)
        label = f"mlx/{args.engine}"
    else:
        model, _cfg = load_exllama_model(args.model_dir, seq_len=input_ids.shape[-1])
        cand = _cuda_logits(model, input_ids, vocab_size)
        label = "cuda/flash_attn_nc"

    stats = compare(ref_logits, cand)
    print(f"\ncandidate: {label}")
    print(f"  cand top1:   {stats['top1_cand']}")
    print(f"  bit-exact:   {stats['bit_exact']}")
    print(f"  max |d|:     {stats['max_abs']:.6g}")
    print(f"  rms:         {stats['rms']:.6g}")
    if not stats["bit_exact"]:
        print(f"  fp32 diffs:  {stats['n_mismatch_bits']}/{ref_logits.size}")
    return 0 if stats["bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
