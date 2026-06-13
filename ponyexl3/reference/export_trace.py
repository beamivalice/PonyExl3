#!/usr/bin/env python3
"""Export per-module activations along the exllamav3 forward path.

Replays ``Model.forward`` module-by-module (same order as ``forward_ls``) and
stores the output tensor after each top-level module. Use this to bisect where
MLX first diverges from CUDA.

Typical use: export last-row hidden states for every block, then compare on Mac.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ponyexl3.types import ExLlamaModel

if __package__ in (None, ""):
    from _cuda_common import (
        activation_to_np,
        forward_params,
        input_ids_to_torch,
        load_exllama_model,
        load_input_ids,
        make_input_ids,
        module_key_to_npz,
        parse_row_slice,
        save_npz,
        standard_metadata,
    )
else:
    from ponyexl3.reference._cuda_common import (
        activation_to_np,
        forward_params,
        input_ids_to_torch,
        load_exllama_model,
        load_input_ids,
        make_input_ids,
        module_key_to_npz,
        parse_row_slice,
        save_npz,
        standard_metadata,
    )


def forward_with_trace(
    model: ExLlamaModel,
    input_ids_t: Any,
    params: dict[str, Any],
    *,
    row_slice: slice,
    modules: set[str] | None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    x = model.prepare_inputs(input_ids_t, dict(params))
    trace: dict[str, np.ndarray] = {}
    module_keys: list[str] = []

    for module, instance, _idx in model.fwd_modules:
        key = module.key
        if modules is not None and key not in modules:
            params["layer_instance"] = instance
            if module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                x = x[..., -num:, :].contiguous()
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            continue

        params["layer_instance"] = instance
        if module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
            x = x[..., -num:, :].contiguous()
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
        trace[module_key_to_npz(key)] = activation_to_np(x, row_slice)
        module_keys.append(key)

    return trace, module_keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model-dir", type=str, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("-s", "--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-npz", type=Path, default=None,
                    help="reuse input_ids (and seed/seq_len if present) from another export")
    ap.add_argument("--row-slice", default="last:1",
                    help="sequence rows to store: last:1 (default), last:8, all, or start:stop")
    ap.add_argument("--modules", nargs="*", default=None,
                    help="only store these exact module keys (still runs full forward)")
    ap.add_argument("--attn-mode", default="flash_attn_nc")
    ap.add_argument("--activate-all-experts", action="store_true",
                    help="deterministic MoE routing (calibration mode)")
    args = ap.parse_args()

    if args.from_npz:
        input_ids = load_input_ids(args.from_npz)
        ref = np.load(args.from_npz, allow_pickle=True)
        seed = int(ref["seed"]) if "seed" in ref else args.seed
        seq_len = int(input_ids.shape[-1])
    else:
        seed = args.seed
        seq_len = args.seq_len
        input_ids = np.zeros((1, seq_len), dtype=np.int64)

    model, config = load_exllama_model(args.model_dir, seq_len=seq_len)
    if args.from_npz:
        input_ids_t = input_ids_to_torch(input_ids)
    else:
        input_ids_t = make_input_ids(config.vocab_size, seq_len, seed)
        input_ids = input_ids_t.cpu().numpy()

    row_slice = parse_row_slice(args.row_slice, seq_len)
    module_filter = set(args.modules) if args.modules else None
    params = forward_params(
        attn_mode=args.attn_mode,
        activate_all_experts=args.activate_all_experts,
    )

    trace, module_keys = forward_with_trace(
        model,
        input_ids_t,
        params,
        row_slice=row_slice,
        modules=module_filter,
    )

    payload = standard_metadata(
        model_dir=args.model_dir,
        input_ids=input_ids,
        seed=seed,
        seq_len=seq_len,
        attn_mode=args.attn_mode,
        vocab_size=np.int64(config.vocab_size),
        hidden_size=np.int64(config.hidden_size),
        row_slice=np.array(args.row_slice),
        module_keys=np.array(module_keys),
        activate_all_experts=np.bool_(args.activate_all_experts),
    )
    payload.update(trace)

    save_npz(args.output, payload)
    print(f" -- traced {len(module_keys)} modules, row_slice={args.row_slice!r}")
    for key in module_keys:
        arr = trace[module_key_to_npz(key)]
        print(f"    {key}: {tuple(arr.shape)} float32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
