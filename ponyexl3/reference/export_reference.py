#!/usr/bin/env python3
"""Export deterministic model logits to .npz for cross-platform bit-exact comparison.

Uses fixed random input_ids (torch.manual_seed(0)) and flash_attn_nc forward pass,
matching the compare_q / conversion reference path.

Run on CUDA host with exllamav3 installed.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

if __package__ in (None, ""):
    from _cuda_common import (
        forward_params,
        load_exllama_model,
        make_input_ids,
        mask_logits,
        save_npz,
        standard_metadata,
    )
else:
    from ponyexl3.reference._cuda_common import (
        forward_params,
        load_exllama_model,
        make_input_ids,
        mask_logits,
        save_npz,
        standard_metadata,
    )


def main(args: argparse.Namespace) -> int:
    model, config = load_exllama_model(args.model_dir, seq_len=args.seq_len)
    vocab_size = config.vocab_size

    input_ids_t = make_input_ids(vocab_size, args.seq_len, args.seed)
    params = forward_params(attn_mode=args.attn_mode)

    output = model.forward(input_ids_t, params)
    mask_logits(output, vocab_size)

    logits_rows = output[0, -args.logit_rows :, :vocab_size].float().cpu().numpy()

    payload = standard_metadata(
        model_dir=args.model_dir,
        input_ids=input_ids_t.cpu().numpy(),
        seed=args.seed,
        seq_len=args.seq_len,
        attn_mode=args.attn_mode,
        vocab_size=np.int64(vocab_size),
        hidden_size=np.int64(config.hidden_size),
        logits=logits_rows,
    )
    save_npz(args.output, payload, compressed=False)
    print(f" -- input_ids shape: {tuple(input_ids_t.shape)}")
    print(f" -- logits shape: {tuple(logits_rows.shape)} dtype float32")
    print(f" -- seed: {args.seed}, attn_mode: {args.attn_mode}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-s", "--seq_len", type=int, default=512)
    parser.add_argument(
        "-r",
        "--logit_rows",
        type=int,
        default=1,
        help="Number of final positions to export (1 row float32 ~970 KiB per row)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-mode", default="flash_attn_nc")
    args = parser.parse_args()
    with torch.inference_mode():
        raise SystemExit(main(args))
