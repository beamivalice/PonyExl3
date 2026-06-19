"""Cooperative cancellation for the converter's GPU work.

Worker threads in a ``ThreadPoolExecutor`` cannot be interrupted by ``SIGINT`` —
only the main thread receives ``KeyboardInterrupt``. Cancelling the queued futures
stops *pending* work, but a candidate already mid-flight is blocked inside
``mx.eval`` and would run to completion (seconds of GPU at 100%).

This module is the escape hatch: the parallel driver sets the flag on interrupt,
and every Metal trellis launch (:func:`quantize_tiles_mlx`) checks it first and
raises :class:`ConversionCancelled`, so an in-flight candidate aborts at its next
GPU launch (~one feedback group, ~ms) instead of finishing. The flag is process
-global but checked cheaply (``Event.is_set``); drivers must :func:`clear` it at
the start of a run so a previous cancellation never poisons a new one.
"""

from __future__ import annotations

import threading


class ConversionCancelled(Exception):
    """Raised inside GPU primitives when cancellation has been requested."""


_EVENT = threading.Event()


def request() -> None:
    """Ask in-flight GPU work to abort at the next launch."""
    _EVENT.set()


def clear() -> None:
    """Reset the flag — call at the start of a cancellable run."""
    _EVENT.clear()


def is_requested() -> bool:
    return _EVENT.is_set()


def raise_if_requested() -> None:
    if _EVENT.is_set():
        raise ConversionCancelled("conversion cancelled")
