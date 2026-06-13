#!/usr/bin/env python3
"""Export MLX pony EXL3 logits in the same .npz schema as export_reference.py.

Run on Apple Silicon. Prefer ``--from-npz`` with the CUDA reference so both
runtimes replay identical ``input_ids``.
"""

from __future__ import annotations

from typing import Any

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from ponyexl3.reference._cuda_common import load_input_ids, save_npz, standard_metadata
from ponyexl3.reference.compare_reference import forward_logits


def _read_config(model_dir: str) -> dict[str, Any]:
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _make_input_ids_torch(vocab_size: int, seq_len: int, seed: int) -> np.ndarray:
    import torch

    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model-dir", type=str, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--from-npz", type=Path, default=None,
                    help="reuse input_ids (+ seed/seq_len) from CUDA or prior export")
    ap.add_argument("-s", "--seq-len", type=int, default=512)
    ap.add_argument("-r", "--logit-rows", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default="exl3", choices=("exl3", "fold16"))
    ap.add_argument("--platform", default="mac", help="tag stored in npz (mac, mlx, ...)")
    args = ap.parse_args()

    import mlx.core as mx

    if not mx.metal.is_available():
        raise SystemExit("Metal required")

    vocab_size = hidden_size = None
    if args.from_npz:
        input_ids = load_input_ids(args.from_npz)
        ref = np.load(args.from_npz, allow_pickle=True)
        seed = int(ref["seed"]) if "seed" in ref else args.seed
        seq_len = int(input_ids.shape[-1])
        if "vocab_size" in ref:
            vocab_size = int(ref["vocab_size"])
        if "hidden_size" in ref:
            hidden_size = int(ref["hidden_size"])
    else:
        seed = args.seed
        seq_len = args.seq_len
        cfg = _read_config(args.model_dir)
        text = cfg.get("text_config", cfg)
        vocab_size = int(text.get("vocab_size", cfg.get("vocab_size", 0)))
        hidden_size = int(text.get("hidden_size", cfg.get("hidden_size", 0)))
        try:
            input_ids = _make_input_ids_torch(vocab_size, seq_len, seed)
        except ImportError:
            raise SystemExit(
                "without --from-npz, install torch so input_ids match CUDA torch.manual_seed"
            )

    if vocab_size is None or hidden_size is None:
        cfg = _read_config(args.model_dir)
        text = cfg.get("text_config", cfg)
        if vocab_size is None:
            vocab_size = int(text.get("vocab_size", cfg.get("vocab_size", 0)))
        if hidden_size is None:
            hidden_size = int(text.get("hidden_size", cfg.get("hidden_size", 0)))

    from ponyexl3.mlx.model import describe, load_model

    model, _ = load_model(
        args.model_dir,
        engine=args.engine,
        warm=True,
        verbose=False,
    )
    print(f"loaded: {describe(model)}")

    logits_1d = forward_logits(model, input_ids)
    logits_rows = logits_1d[np.newaxis, :] if args.logit_rows == 1 else logits_1d

    payload = standard_metadata(
        model_dir=args.model_dir,
        input_ids=input_ids,
        seed=seed,
        seq_len=seq_len,
        attn_mode="mlx",
        vocab_size=np.int64(vocab_size),
        hidden_size=np.int64(hidden_size),
        runtime=np.array("mlx"),
        engine=np.array(args.engine),
        platform=np.array(args.platform),
        logits=logits_rows.astype(np.float32),
    )
    save_npz(args.output, payload, compressed=False)
    print(f" -- input_ids shape: {tuple(input_ids.shape)}")
    top1 = int(np.argmax(logits_rows))
    print(f" -- logits shape: {tuple(logits_rows.shape)} top1={top1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
