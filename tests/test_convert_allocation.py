"""M5 priority bit-allocation scaffold."""

from __future__ import annotations

from ponyexl3.convert.allocation import allocate_priority_bits, allocation_summary


def test_allocate_priority_bits_spends_fractional_budget_by_priority():
    modules = ["a", "b", "c", "d", "e"]
    allocation = allocate_priority_bits(
        modules,
        target_bpw=4.4,
        priorities={"d": 10.0, "b": 5.0},
    )

    assert [item.key for item in allocation] == modules
    assert {item.key: item.bits for item in allocation} == {
        "a": 4,
        "b": 5,
        "c": 4,
        "d": 5,
        "e": 4,
    }
    assert allocation_summary(allocation) == {
        "module_count": 5,
        "average_bits": 4.4,
        "min_bits": 4,
        "max_bits": 5,
    }


def test_allocate_priority_bits_empty_plan():
    assert allocate_priority_bits([], target_bpw=4.0) == []
    assert allocation_summary([]) == {
        "module_count": 0,
        "average_bits": 0.0,
        "min_bits": 0,
        "max_bits": 0,
    }
