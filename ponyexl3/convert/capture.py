"""Calibration activation capture for checkpoint-backed conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np


CaptureDType = Literal["float16", "float32"]


def default_calibration_text() -> Path:
    """Path to the bundled default calibration corpus (a WikiText-2 excerpt).

    Lets the one-command convert pipeline run offline when the user doesn't
    supply ``--calibration-text``. See the NOTICE in
    ``ponyexl3/convert/calibration_data/`` for source and licensing (CC BY-SA).
    """
    from importlib.resources import files

    return Path(str(files("ponyexl3.convert") / "calibration_data" / "wikitext2.txt"))


@dataclass(frozen=True)
class CalibrationCaptureSummary:
    """JSON-friendly summary for a calibration capture run."""

    output: str
    source_dir: str
    module_count: int
    captured_count: int
    rows: int
    seq_len: int
    seqs_run: int
    dtype: CaptureDType
    missing: list[str]


class _ActivationCollector:
    """Collect a fixed number of input rows per module key."""

    def __init__(self, module_keys: Sequence[str], *, rows: int, dtype: CaptureDType):
        self.rows = int(rows)
        self.dtype = dtype
        self.chunks: dict[str, list[np.ndarray]] = {key: [] for key in module_keys}
        self.counts: dict[str, int] = {key: 0 for key in module_keys}
        self.pending: list[tuple[str, Any]] = []

    def complete(self) -> bool:
        return all(count >= self.rows for count in self.counts.values())

    def add(self, key: str, x: Any) -> None:
        need = self.rows - self.counts.get(key, 0)
        if need <= 0:
            return
        flat = x.reshape(-1, x.shape[-1])
        take = min(int(flat.shape[0]), need)
        if take <= 0:
            return
        dtype = mx.float16 if self.dtype == "float16" else mx.float32
        self.pending.append((key, flat[:take].astype(dtype)))
        self.counts[key] = self.counts.get(key, 0) + take

    def flush(self, *extra: Any) -> None:
        arrays = [arr for _, arr in self.pending]
        if arrays or extra:
            mx.eval(*extra, *arrays)
        for key, arr in self.pending:
            self.chunks[key].append(np.array(arr))
        self.pending.clear()

    def arrays(self) -> tuple[dict[str, np.ndarray], list[str]]:
        out: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for key, chunks in self.chunks.items():
            if not chunks:
                missing.append(key)
                continue
            arr = np.concatenate(chunks, axis=0)[: self.rows]
            if arr.shape[0] < self.rows:
                missing.append(key)
                continue
            dtype = np.float16 if self.dtype == "float16" else np.float32
            out[key] = np.ascontiguousarray(arr.astype(dtype, copy=False))
        return out, missing


def _candidate_mlx_paths(module_key: str) -> list[str]:
    if module_key == "lm_head":
        return ["language_model.lm_head", "lm_head"]
    paths = [module_key]
    if module_key.startswith("model.language_model."):
        suffix = module_key.removeprefix("model.language_model.")
        paths.append(f"language_model.model.{suffix}")
    if module_key.startswith("model."):
        suffix = module_key.removeprefix("model.")
        paths.append(f"language_model.{suffix}")
    return list(dict.fromkeys(paths))


def _resolve_path(root: Any, path: str) -> Any:
    cur = root
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _resolve_linear_module(root: Any, module_key: str) -> Any:
    errors: list[str] = []
    for path in _candidate_mlx_paths(module_key):
        try:
            mod = _resolve_path(root, path)
        except (AttributeError, IndexError, TypeError, KeyError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        if isinstance(mod, nn.Linear):
            return mod
        errors.append(f"{path}: resolved to {type(mod).__name__}, not nn.Linear")
    raise KeyError(f"cannot resolve {module_key!r} in MLX model ({'; '.join(errors)})")


def _load_text_tokens(tokenizer: Any, text_path: str | Path, *, min_tokens: int) -> list[int]:
    text = Path(text_path).read_text(encoding="utf-8")
    ids = list(tokenizer.encode(text))
    if not ids:
        raise ValueError(f"{text_path} did not produce any tokens")
    while len(ids) < min_tokens:
        ids.extend(ids)
    return [int(x) for x in ids]


def _save_activation_map(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    metadata: dict[str, str],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.numpy import save_file

        save_file(arrays, str(out), metadata=metadata)
        return
    if suffix == ".npz":
        np.savez(out, **arrays)  # type: ignore[reportArgumentType]
        return
    raise ValueError(f"unsupported calibration map output format: {out.suffix}")


def capture_calibration_activations(
    source_dir: str | Path,
    module_keys: Sequence[str],
    output: str | Path,
    *,
    text_path: str | Path,
    rows: int = 250,
    seq_len: int = 2048,
    max_seqs: int | None = None,
    dtype: CaptureDType = "float16",
    progress: Any | None = None,
) -> CalibrationCaptureSummary:
    """Run a BF16 source forward and save per-module input activations."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if dtype not in ("float16", "float32"):
        raise ValueError(f"unsupported capture dtype: {dtype}")

    from mlx_lm.utils import load

    selected = list(dict.fromkeys(module_keys))
    if not selected:
        raise ValueError("no module keys selected for calibration capture")

    model, tokenizer = load(str(source_dir))
    if hasattr(model, "eval"):
        model.eval()

    linear_keys = [key for key in selected if key != "lm_head"]
    keys_by_id: dict[int, list[str]] = {}
    skipped_resolution: list[str] = []
    for key in linear_keys:
        try:
            mod = _resolve_linear_module(model, key)
        except KeyError:
            skipped_resolution.append(key)
            continue
        keys_by_id.setdefault(id(mod), []).append(key)

    capture_keys = [key for key in selected if key not in skipped_resolution]
    collector = _ActivationCollector(capture_keys, rows=rows, dtype=dtype)
    min_tokens = seq_len if max_seqs is None else seq_len * max(1, max_seqs)
    min_tokens = max(min_tokens, rows)
    tokens = _load_text_tokens(tokenizer, text_path, min_tokens=min_tokens)
    if max_seqs is None:
        max_seqs = max(1, (rows + seq_len - 1) // seq_len)

    lm = getattr(model, "language_model", model)
    text_model = getattr(lm, "model", None)
    if text_model is None:
        raise ValueError("loaded model does not expose language_model.model")

    original_linear_call: Any = nn.Linear.__call__

    def wrapped_linear(mod: Any, x: Any) -> Any:
        for key in keys_by_id.get(id(mod), []):
            collector.add(key, x)
        return original_linear_call(mod, x)

    seqs_run = 0
    nn.Linear.__call__ = wrapped_linear  # type: ignore[method-assign]
    try:
        for start in range(0, len(tokens), seq_len):
            if seqs_run >= max_seqs or collector.complete():
                break
            chunk = tokens[start : start + seq_len]
            if len(chunk) < seq_len:
                break
            input_ids = mx.array([chunk], dtype=mx.int32)
            hidden = text_model(input_ids)
            if "lm_head" in collector.counts:
                collector.add("lm_head", hidden)
            collector.flush(hidden)
            seqs_run += 1
            mx.clear_cache()
            if progress is not None:
                done = sum(1 for count in collector.counts.values() if count >= rows)
                progress(
                    "calibration_seq",
                    {
                        "seqs_run": seqs_run,
                        "captured_modules": done,
                        "module_count": len(collector.counts),
                    },
                )
    finally:
        nn.Linear.__call__ = original_linear_call  # type: ignore[method-assign]

    arrays, missing = collector.arrays()
    missing.extend(key for key in skipped_resolution if key not in missing)
    metadata = {
        "source_dir": str(source_dir),
        "text_path": str(text_path),
        "rows": str(rows),
        "seq_len": str(seq_len),
        "seqs_run": str(seqs_run),
        "dtype": dtype,
    }
    _save_activation_map(output, arrays, metadata=metadata)
    return CalibrationCaptureSummary(
        output=str(output),
        source_dir=str(source_dir),
        module_count=len(selected),
        captured_count=len(arrays),
        rows=rows,
        seq_len=seq_len,
        seqs_run=seqs_run,
        dtype=dtype,
        missing=sorted(missing),
    )
