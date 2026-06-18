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
    weight: int = 1


def allocate_priority_bits(
    module_keys: Sequence[str],
    *,
    target_bpw: float,
    base_bits: int | None = None,
    priorities: dict[str, float] | None = None,
    weights: dict[str, int] | None = None,
    fixed_bits: dict[str, int] | None = None,
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

    wt = {key: max(1, int((weights or {}).get(key, 1))) for key in keys}
    total_weight = sum(wt.values())
    fixed = {key: int(bits) for key, bits in (fixed_bits or {}).items() if key in wt}
    for key, bits in fixed.items():
        if bits < 1:
            raise ValueError(f"fixed bits for {key!r} must be >= 1, got {bits}")
    target_weighted_bits = int(round(target_bpw * total_weight))
    base_weighted_bits = sum(fixed.get(key, floor_bits) * wt[key] for key in keys)
    extra_budget = max(0, target_weighted_bits - base_weighted_bits)
    extra_budget = min(extra_budget, sum(wt[key] for key in keys if key not in fixed))
    pr = priorities or {}
    ranked = sorted(
        enumerate(keys),
        key=lambda item: (-float(pr.get(item[1], 0.0)), item[0]),
    )
    upgraded: set[str] = set()
    spent = 0
    for _idx, key in ranked:
        if key in fixed:
            continue
        cost = wt[key]
        if spent + cost > extra_budget:
            continue
        upgraded.add(key)
        spent += cost
    return [
        ModuleAllocation(
            key=key,
            bits=fixed.get(key, floor_bits + (1 if key in upgraded else 0)),
            priority=float(pr.get(key, 0.0)),
            weight=wt[key],
        )
        for key in keys
    ]


def allocation_summary(allocations: Sequence[ModuleAllocation]) -> dict[str, float | int]:
    """Return a JSON-friendly summary for an allocation plan."""

    if not allocations:
        return {"module_count": 0, "average_bits": 0.0, "min_bits": 0, "max_bits": 0}
    bits = [item.bits for item in allocations]
    weights = [max(1, int(item.weight)) for item in allocations]
    total_weight = sum(weights)
    return {
        "module_count": len(bits),
        "average_bits": float(
            sum(bit * weight for bit, weight in zip(bits, weights, strict=True))
            / total_weight
        ),
        "min_bits": int(min(bits)),
        "max_bits": int(max(bits)),
        "total_weight": int(total_weight),
    }


def default_module_priority(key: str) -> float:
    """Cheap M5a quality priority used before measured proxy deltas exist."""

    if key == "lm_head" or key.endswith(".lm_head"):
        return 100.0
    if ".self_attn.o_proj" in key or ".attention.o_proj" in key:
        return 80.0
    if any(part in key for part in (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")):
        return 70.0
    if ".mlp.down_proj" in key:
        return 60.0
    if any(part in key for part in (".mlp.gate_proj", ".mlp.up_proj")):
        return 50.0
    return 0.0
