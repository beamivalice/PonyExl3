"""Process-scoped reuse cache for quantized layer results.

The e2e pipeline ``measure``s every module (to score candidate K/shrinkage) and
then ``convert``s every module (to emit). For the winning ``(K, shrinkage)`` the
two passes produce a bit-identical layer — grouped sibling search is pure
batching (each module keeps its own scales/Hessian/LDL), so the per-module
trellis matches the per-candidate measurement. The convert pass is therefore
redundant: when this cache is enabled, ``measure`` populates it and ``convert``
reuses it, roughly halving the quantization work.

Off by default (the library behaves exactly as before). The e2e CLI enables it
around the measure→convert span. Keys cover every layer-determining input but
NOT the metric flags (``fast_metrics``/``compare_oracle`` do not change the
trellis). Stored results have their large activation/output arrays stripped, and
an LRU byte cap bounds memory — a miss simply recomputes, so eviction is safe.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ponyexl3.convert.direct import DirectLayerResult

_LOCK = threading.Lock()
_EMPTY = np.empty((0, 0), dtype=np.float32)


class _Cache:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
        self.store: "OrderedDict[tuple[Any, ...], DirectLayerResult]" = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0


_CACHE: _Cache | None = None


def enable(max_bytes: int = 8 << 30) -> None:
    """Start a fresh reuse cache (default 8 GiB LRU budget)."""
    global _CACHE
    _CACHE = _Cache(max_bytes)


def disable() -> dict[str, int] | None:
    """Drop the cache and return its hit/miss/byte stats (or None if inactive)."""
    global _CACHE
    cache = _CACHE
    _CACHE = None
    return None if cache is None else _snapshot(cache)


def active() -> bool:
    return _CACHE is not None


def stats() -> dict[str, int] | None:
    return None if _CACHE is None else _snapshot(_CACHE)


def _snapshot(cache: _Cache) -> dict[str, int]:
    return {
        "hits": cache.hits,
        "misses": cache.misses,
        "entries": len(cache.store),
        "bytes": cache.bytes,
    }


def activations_fingerprint(acts: np.ndarray | None) -> str | None:
    """Stable content fingerprint of a calibration activation matrix."""
    if acts is None:
        return None
    arr = np.ascontiguousarray(acts)
    digest = hashlib.blake2b(arr.tobytes(), digest_size=16)
    digest.update(str(arr.shape).encode())
    digest.update(str(arr.dtype).encode())
    return digest.hexdigest()


def make_key(
    *,
    source_dir: Any,
    oracle_dir: Any,
    module_key: str,
    scale_mode: str,
    sigma_reg: float,
    hessian_shrinkage: float,
    buf_size_rows: int,
    feedback_rows: int,
    max_pins: int,
    skip_g_scale: bool,
    regularization_seed: int,
    quant_bits: int | None,
    search_backend: str,
    acts_fp: str | None,
) -> tuple[Any, ...] | None:
    """Build a cache key from every input that determines the trellis output."""
    if acts_fp is None:
        return None
    return (
        str(source_dir),
        str(oracle_dir),
        str(module_key),
        str(scale_mode),
        round(float(sigma_reg), 12),
        round(float(hessian_shrinkage), 12),
        int(buf_size_rows),
        int(feedback_rows),
        int(max_pins),
        bool(skip_g_scale),
        int(regularization_seed),
        None if quant_bits is None else int(quant_bits),
        str(search_backend),
        acts_fp,
    )


def _result_nbytes(result: "DirectLayerResult") -> int:
    layer = result.layer
    total = int(layer.trellis.nbytes)
    if layer.suh is not None:
        total += int(layer.suh.nbytes)
    if layer.svh is not None:
        total += int(layer.svh.nbytes)
    return total


def lookup(key: tuple[Any, ...] | None) -> "DirectLayerResult | None":
    cache = _CACHE
    if cache is None or key is None:
        return None
    with _LOCK:
        result = cache.store.get(key)
        if result is None:
            cache.misses += 1
            return None
        cache.store.move_to_end(key)
        cache.hits += 1
        return result


def store(key: tuple[Any, ...] | None, result: "DirectLayerResult") -> None:
    cache = _CACHE
    if cache is None or key is None:
        return
    stripped = replace(
        result,
        activations=_EMPTY,
        source_output=_EMPTY,
        converted_output=_EMPTY,
    )
    nbytes = _result_nbytes(result)
    with _LOCK:
        if key in cache.store:
            return
        cache.store[key] = stripped
        cache.bytes += nbytes
        cache.store.move_to_end(key)
        while cache.bytes > cache.max_bytes and len(cache.store) > 1:
            _, evicted = cache.store.popitem(last=False)
            cache.bytes -= _result_nbytes(evicted)
