"""Phase timing + GPU-activation attribution for the converter pipeline.

Off by default (zero overhead). Enable with ``PONYEXL3_CONVERT_TIMING=1`` or
``ponyexl3.convert.timing.enable()``.

Model
-----
The converter alternates CPU work (NumPy LDLQ feedback, Hessian, metrics) with
GPU work (the Metal trellis search, MLX Hadamard, trellis decode). MLX blocks on
``mx.eval`` inside those primitives, so the wall time spent in a :func:`gpu` span
*is* the GPU-active time (the CPU is parked on the GPU).

- :func:`phase` wraps a non-nested top-level pipeline phase; phases partition the
  wall clock of one ``ldlq_quantize_layer`` call.
- :func:`gpu` wraps a GPU primitive; its time is attributed to the enclosing
  phase, so each phase reports a wall total *and* the GPU-active slice within it.

``report()`` returns per-phase wall + GPU split and the overall GPU-activation %.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterator
from typing import Any


def _env_enabled() -> bool:
    return os.environ.get("PONYEXL3_CONVERT_TIMING", "") not in ("", "0", "false", "no")


class _State(threading.local):
    def __init__(self) -> None:
        self.enabled: bool = _env_enabled()
        self.wall: dict[str, float] = {}
        self.gpu_by_phase: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.stack: list[str] = []
        self.starts: list[float] = []
        self.gpu_total: float = 0.0
        self.gpu_count: int = 0


_S = _State()


def begin(name: str) -> None:
    """Open a phase manually (pair with :func:`end`); for blocks awkward to indent."""
    if not _S.enabled:
        return
    _S.stack.append(name)
    _S.starts.append(time.perf_counter())


def end(name: str) -> None:
    """Close the phase opened by the matching :func:`begin`."""
    if not _S.enabled:
        return
    if not _S.starts:
        return
    start = _S.starts.pop()
    owner = _S.stack.pop()
    dt = time.perf_counter() - start
    _S.wall[owner] = _S.wall.get(owner, 0.0) + dt
    _S.counts[owner] = _S.counts.get(owner, 0) + 1


def enabled() -> bool:
    return _S.enabled


def enable() -> None:
    _S.enabled = True


def disable() -> None:
    _S.enabled = False


def reset() -> None:
    """Clear all accumulators (call between pipeline phases to compare them)."""
    _S.wall.clear()
    _S.gpu_by_phase.clear()
    _S.counts.clear()
    _S.stack.clear()
    _S.gpu_total = 0.0
    _S.gpu_count = 0


@contextlib.contextmanager
def phase(name: str) -> Iterator[None]:
    """Time a top-level pipeline phase (do not nest phases of the same call)."""
    if not _S.enabled:
        yield
        return
    begin(name)
    try:
        yield
    finally:
        end(name)


@contextlib.contextmanager
def gpu(name: str = "gpu") -> Iterator[None]:
    """Time a GPU primitive; attributed to the enclosing :func:`phase`."""
    if not _S.enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        _S.gpu_total += dt
        _S.gpu_count += 1
        owner = _S.stack[-1] if _S.stack else "(root)"
        _S.gpu_by_phase[owner] = _S.gpu_by_phase.get(owner, 0.0) + dt
        _S.counts[f"gpu:{name}"] = _S.counts.get(f"gpu:{name}", 0) + 1


def report() -> dict[str, Any]:
    """Aggregate snapshot: per-phase wall/GPU split and overall GPU-activation %."""
    total_wall = sum(_S.wall.values())
    phases = {
        name: {
            "wall_s": wall,
            "gpu_s": _S.gpu_by_phase.get(name, 0.0),
            "cpu_s": max(0.0, wall - _S.gpu_by_phase.get(name, 0.0)),
            "calls": _S.counts.get(name, 0),
        }
        for name, wall in sorted(_S.wall.items(), key=lambda kv: -kv[1])
    }
    return {
        "total_wall_s": total_wall,
        "gpu_s": _S.gpu_total,
        "cpu_s": max(0.0, total_wall - _S.gpu_total),
        "gpu_active_pct": (100.0 * _S.gpu_total / total_wall) if total_wall > 0 else 0.0,
        "gpu_calls": _S.gpu_count,
        "phases": phases,
    }


def format_report(rep: dict[str, Any] | None = None, *, label: str = "convert") -> str:
    """Human-readable one-block summary of :func:`report`."""
    rep = report() if rep is None else rep
    lines = [
        f"[timing:{label}] total={rep['total_wall_s']:.2f}s "
        f"gpu={rep['gpu_s']:.2f}s cpu={rep['cpu_s']:.2f}s "
        f"GPU-active={rep['gpu_active_pct']:.1f}%  (gpu_calls={rep['gpu_calls']})"
    ]
    for name, info in rep["phases"].items():
        share = (100.0 * info["wall_s"] / rep["total_wall_s"]) if rep["total_wall_s"] else 0.0
        lines.append(
            f"  {name:<16} wall={info['wall_s']:7.2f}s ({share:4.1f}%) "
            f"gpu={info['gpu_s']:6.2f}s cpu={info['cpu_s']:6.2f}s  x{info['calls']}"
        )
    return "\n".join(lines)
