"""Shared helpers for CUDA-side exllamav3 reference exports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ponyexl3.types import ExLlamaModel

import numpy as np

DEFAULT_ATTN_MODE = "flash_attn_nc"


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run these tools on the exllamav3 GPU host")


def load_exllama_model(
    model_dir: str,
    *,
    seq_len: int = 512,
    progressbar: bool = True,
) -> tuple[ExLlamaModel, ExLlamaModel]:
    """Load exllamav3 ``Model`` with standard reference-load settings."""
    require_cuda()
    from exllamav3 import Config, Model

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load(
        max_output_size=seq_len + 512,
        max_output_factor=7,
        progressbar=progressbar,
    )
    return model, config


def make_input_ids(vocab_size: int, seq_len: int, seed: int) -> "Any":
    import torch

    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)


def load_input_ids(path: str | Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "input_ids" not in data:
        raise KeyError(f"{path} missing input_ids")
    ids = np.asarray(data["input_ids"])
    if ids.ndim == 1:
        ids = ids[np.newaxis, :]
    return ids.astype(np.int64, copy=False)


def input_ids_to_torch(ids: np.ndarray, device: "Any" = None) -> "Any":
    import torch

    t = torch.from_numpy(np.asarray(ids, dtype=np.int64))
    if device is not None:
        t = t.to(device)
    return t


def forward_params(
    *,
    attn_mode: str = DEFAULT_ATTN_MODE,
    activate_all_experts: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {"attn_mode": attn_mode}
    if activate_all_experts:
        params["activate_all_experts"] = True
    return params


def module_key_to_npz(key: str) -> str:
    return key.replace(".", "__")


def npz_to_module_key(key: str) -> str:
    return key.replace("__", ".")


def list_exl3_module_keys(model_dir: str) -> list[str]:
    qpath = os.path.join(model_dir, "quantization_config.json")
    with open(qpath, encoding="utf-8") as f:
        qcfg = json.load(f)
    storage = qcfg.get("tensor_storage", {})
    return sorted(
        k for k, v in storage.items() if v.get("quant_format") == "exl3"
    )


def list_forward_module_keys(model: ExLlamaModel) -> list[str]:
    return [module.key for module, _instance, _idx in model.fwd_modules]


def parse_row_slice(spec: str, seq_len: int) -> slice:
    """Parse ``last``, ``last:N``, ``all``, or ``start:stop`` against seq_len."""
    spec = spec.strip().lower()
    if spec == "all":
        return slice(0, seq_len)
    if spec.startswith("last"):
        rest = spec[4:]
        if not rest:
            return slice(seq_len - 1, seq_len)
        if rest.startswith(":"):
            n = int(rest[1:])
            return slice(max(0, seq_len - n), seq_len)
        raise ValueError(f"bad row slice: {spec!r}")
    if ":" in spec:
        a, b = spec.split(":", 1)
        start = int(a) if a else 0
        stop = int(b) if b else seq_len
        return slice(start, stop)
    idx = int(spec)
    return slice(idx, idx + 1)


def activation_to_np(tensor: Any, row_slice: slice | None = None) -> np.ndarray:
    """Cast activations to float32 numpy, optionally slice sequence dim."""
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim >= 2 and row_slice is not None:
        arr = arr[:, row_slice, ...]
    return np.ascontiguousarray(arr)


def mask_logits(output: "Any", vocab_size: int) -> "Any":
    output[..., vocab_size:] = float("-inf")
    return output


def standard_metadata(
    *,
    model_dir: str,
    input_ids: np.ndarray,
    seed: int,
    seq_len: int,
    attn_mode: str,
    **extra: Any,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "input_ids": input_ids,
        "seed": np.int64(seed),
        "seq_len": np.int64(seq_len),
        "attn_mode": np.array(attn_mode),
        "model_dir": np.array(model_dir),
        "format_version": np.int64(1),
    }
    meta.update(extra)
    return meta


def save_npz(path: str | Path, payload: dict[str, Any], *, compressed: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    size = path.stat().st_size
    print(f" -- Wrote {path}")
    print(f" -- File size: {size:,} bytes ({size / 1024 / 1024:.2f} MiB)")
