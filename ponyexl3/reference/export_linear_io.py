#!/usr/bin/env python3
"""Export EXL3 linear layer inputs/outputs from a real exllamav3 forward.

For each requested module key, captures the actual activation ``x`` seen during
forward and the CUDA output ``y`` (fast kernel). Optionally also runs the
``reconstruct=True`` path on the same ``x`` for decode-vs-kernel ground truth.

Use for per-layer CUDA↔MLX parity without running the full model on Mac.
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
        list_exl3_module_keys,
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
        list_exl3_module_keys,
        load_exllama_model,
        load_input_ids,
        make_input_ids,
        module_key_to_npz,
        parse_row_slice,
        save_npz,
        standard_metadata,
    )


def _is_exl3_linear(module: Any) -> bool:
    return getattr(module, "quant_type", None) == "exl3" and getattr(module, "inner", None) is not None


def _install_hooks(
    model: ExLlamaModel,
    wanted: set[str],
    bucket: dict[str, Any],
    row_slice: slice,
) -> list[Any]:
    hooks = []

    for module in model:
        if not _is_exl3_linear(module) or module.key not in wanted:
            continue
        key = module.key
        orig = module.forward

        def wrapped(
            self: Any,
            x: Any,
            params: dict[str, Any] | None,
            out_dtype: Any = None,
            _key: str = key,
            _orig: Any = orig,
        ) -> Any:
            y = _orig(x, params, out_dtype)
            out = y[0] if isinstance(y, tuple) else y
            entry = bucket.setdefault(_key, {})
            entry["x"] = activation_to_np(x, row_slice)
            entry["y"] = activation_to_np(out, row_slice)
            return y

        module.forward = wrapped.__get__(module, type(module))
        hooks.append((module, orig, key))

    return hooks


def _restore_hooks(hooks: list[tuple[Any, Any, str]]) -> None:
    for module, orig, _key in hooks:
        module.forward = orig


def _add_reconstruct(bucket: dict[str, Any], model: ExLlamaModel, params: dict[str, Any]) -> None:
    from exllamav3.modules.quant.exl3 import LinearEXL3
    import torch

    recon_params = dict(params)
    recon_params["reconstruct"] = True

    for key, entry in bucket.items():
        module = model.find_module(key)
        inner = getattr(module, "inner", module)
        if not isinstance(inner, LinearEXL3):
            continue
        x_np = entry["x"]
        # x may be [1, rows, in] or [rows, in]
        if x_np.ndim == 3:
            x = torch.from_numpy(x_np).to(inner.trellis.device, dtype=torch.float16)
            rows = x.shape[1]
            x2d = x.reshape(rows, -1)
            y = inner.forward(x2d, recon_params)
            y = y.reshape(1, rows, -1)
        else:
            x = torch.from_numpy(x_np).to(inner.trellis.device, dtype=torch.float16)
            y = inner.forward(x, recon_params)
        entry["y_reconstruct"] = activation_to_np(y, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model-dir", type=str, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("modules", nargs="+", help="EXL3 module keys to capture")
    ap.add_argument("-s", "--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-npz", type=Path, default=None)
    ap.add_argument("--row-slice", default="last:1")
    ap.add_argument("--reconstruct", action="store_true",
                    help="also run reconstruct=True on captured x")
    ap.add_argument("--attn-mode", default="flash_attn_nc")
    ap.add_argument("--validate-keys", action="store_true",
                    help="error if a module key is missing from checkpoint")
    args = ap.parse_args()

    known = set(list_exl3_module_keys(args.model_dir))
    wanted = set(args.modules)
    if args.validate_keys:
        missing = sorted(wanted - known)
        if missing:
            raise SystemExit(f"unknown EXL3 keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    input_ids: np.ndarray
    if args.from_npz:
        input_ids = load_input_ids(args.from_npz)
        ref = np.load(args.from_npz, allow_pickle=True)
        seed = int(ref["seed"]) if "seed" in ref else args.seed
        seq_len = int(input_ids.shape[-1])
    else:
        seed = args.seed
        seq_len = args.seq_len
        input_ids = np.empty((1, 0), dtype=np.int64)  # set below

    model, config = load_exllama_model(args.model_dir, seq_len=seq_len)
    row_slice = parse_row_slice(args.row_slice, seq_len)
    params = forward_params(attn_mode=args.attn_mode)

    bucket: dict[str, dict[str, Any]] = {}
    hooks = _install_hooks(model, wanted, bucket, row_slice)

    if args.from_npz:
        input_ids_t = input_ids_to_torch(input_ids)
    else:
        input_ids_t = make_input_ids(config.vocab_size, seq_len, seed)
        input_ids = input_ids_t.cpu().numpy()

    with __import__("torch").inference_mode():
        model.forward(input_ids_t, params)
        if args.reconstruct and bucket:
            _add_reconstruct(bucket, model, params)

    _restore_hooks(hooks)

    if sorted(bucket.keys()) != sorted(wanted & set(bucket.keys())):
        got = set(bucket.keys())
        miss = sorted(wanted - got)
        if miss:
            print(f" warning: no activations captured for: {miss}")

    payload = standard_metadata(
        model_dir=args.model_dir,
        input_ids=input_ids,
        seed=seed,
        seq_len=seq_len,
        attn_mode=args.attn_mode,
        row_slice=np.array(args.row_slice),
        module_keys=np.array(sorted(bucket.keys())),
    )
    for key, entry in sorted(bucket.items()):
        pfx = module_key_to_npz(key)
        payload[f"{pfx}__x"] = entry["x"]
        payload[f"{pfx}__y"] = entry["y"]
        if "y_reconstruct" in entry:
            payload[f"{pfx}__y_reconstruct"] = entry["y_reconstruct"]

    save_npz(args.output, payload)
    for key in sorted(bucket.keys()):
        x = bucket[key]["x"]
        y = bucket[key]["y"]
        print(f" -- {key}: x{tuple(x.shape)} y{tuple(y.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
