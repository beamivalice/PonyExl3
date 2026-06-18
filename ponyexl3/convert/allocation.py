"""M5 bit-allocation scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class ModuleAllocation:
    """Assigned integer trellis bits for one quantized module."""

    key: str
    bits: int
    priority: float


def allocate_priority_bits(
    module_keys: Sequence[str],
    *,
    target_bpw: float,
    base_bits: int | None = None,
    priorities: dict[str, float] | None = None,
) -> list[ModuleAllocation]:
    """
    Cheap M5a allocation: floor bpw plus one-bit upgrades by priority.

    This is the non-measured upstream tier: it produces deterministic integer
    K assignments under an average bit budget and gives M5b a stable API to
    replace priorities with measured proxy-loss deltas.
    """

    keys = list(module_keys)
    if not keys:
        return []
    if target_bpw <= 0.0:
        raise ValueError(f"target_bpw must be positive, got {target_bpw}")
    floor_bits = int(math.floor(target_bpw)) if base_bits is None else int(base_bits)
    if floor_bits < 1:
        raise ValueError(f"base bits must be >= 1, got {floor_bits}")

    extra_budget = int(round((target_bpw - floor_bits) * len(keys)))
    extra_budget = max(0, min(len(keys), extra_budget))
    pr = priorities or {}
    ranked = sorted(
        enumerate(keys),
        key=lambda item: (-float(pr.get(item[1], 0.0)), item[0]),
    )
    upgraded = {key for _idx, key in ranked[:extra_budget]}
    return [
        ModuleAllocation(
            key=key,
            bits=floor_bits + (1 if key in upgraded else 0),
            priority=float(pr.get(key, 0.0)),
        )
        for key in keys
    ]


def allocation_summary(allocations: Sequence[ModuleAllocation]) -> dict[str, float | int]:
    """Return a JSON-friendly summary for an allocation plan."""

    if not allocations:
        return {"module_count": 0, "average_bits": 0.0, "min_bits": 0, "max_bits": 0}
    bits = [item.bits for item in allocations]
    return {
        "module_count": len(bits),
        "average_bits": float(sum(bits) / len(bits)),
        "min_bits": int(min(bits)),
        "max_bits": int(max(bits)),
    }
