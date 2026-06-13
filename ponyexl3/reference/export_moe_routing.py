#!/usr/bin/env python3
"""Export MoE expert indices and routing weights from an exllamav3 forward.

Captures ``selected_experts`` and ``routing_weights`` for each BlockSparseMLP
layer at the requested sequence positions. Needed when MLX MoE gather diverges
but dense paths match.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ponyexl3.types import ExLlamaModel

if __package__ in (None, ""):
    from _cuda_common import (
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


def _install_moe_hooks(
    model: ExLlamaModel,
    bucket: dict[str, Any],
    row_slice: slice,
) -> list[Any]:
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

    hooks = []
    for module in model:
        if not isinstance(module, BlockSparseMLP) or module.routing_fn is None:
            continue
        key = module.key
        orig = module.routing_fn

        def routing_hook(
            bsz: Any,
            cfg: Any,
            z: Any,
            params: Any,
            _key: str = key,
            _orig: Any = orig,
        ) -> tuple[Any, Any]:
            experts, weights = _orig(bsz, cfg, z, params)
            e = experts.detach().cpu().numpy()
            w = weights.detach().float().cpu().numpy()
            if e.ndim == 2 and e.shape[0] > 1:
                e = e.reshape(1, -1, e.shape[-1])[:, row_slice, :].reshape(-1, e.shape[-1])
                w = w.reshape(1, -1, w.shape[-1])[:, row_slice, :].reshape(-1, w.shape[-1])
            elif e.ndim == 2:
                e = e[row_slice, :]
                w = w[row_slice, :]
            bucket[_key] = {
                "selected_experts": np.ascontiguousarray(e),
                "routing_weights": np.ascontiguousarray(w),
            }
            return experts, weights

        module.routing_fn = routing_hook
        hooks.append((module, orig))

    return hooks


def _restore_hooks(hooks: list[tuple[Any, Any]]) -> None:
    for module, orig in hooks:
        module.routing_fn = orig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model-dir", type=str, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("-s", "--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-npz", type=Path, default=None)
    ap.add_argument("--row-slice", default="last:1")
    ap.add_argument("--attn-mode", default="flash_attn_nc")
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
    row_slice = parse_row_slice(args.row_slice, seq_len)
    params = forward_params(attn_mode=args.attn_mode)

    if args.from_npz:
        input_ids_t = input_ids_to_torch(input_ids)
    else:
        input_ids_t = make_input_ids(config.vocab_size, seq_len, seed)
        input_ids = input_ids_t.cpu().numpy()

    bucket: dict[str, dict[str, Any]] = {}
    hooks = _install_moe_hooks(model, bucket, row_slice)

    with __import__("torch").inference_mode():
        model.forward(input_ids_t, params)

    _restore_hooks(hooks)

    payload = standard_metadata(
        model_dir=args.model_dir,
        input_ids=input_ids,
        seed=seed,
        seq_len=seq_len,
        attn_mode=args.attn_mode,
        row_slice=np.array(args.row_slice),
        moe_layer_keys=np.array(sorted(bucket.keys())),
    )
    for key, entry in sorted(bucket.items()):
        pfx = module_key_to_npz(key)
        payload[f"{pfx}__experts"] = entry["selected_experts"]
        payload[f"{pfx}__weights"] = entry["routing_weights"]

    save_npz(args.output, payload)
    print(f" -- captured routing for {len(bucket)} MoE layers")
    for key in sorted(bucket.keys()):
        e = bucket[key]["selected_experts"]
        print(f"    {key}: experts{tuple(e.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
