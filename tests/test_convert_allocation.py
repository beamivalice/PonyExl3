"""M5 priority bit-allocation scaffold."""

from __future__ import annotations

from ponyexl3.convert.allocation import (
    allocate_priority_bits,
    allocation_summary,
    default_module_priority,
)


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
        "total_weight": 5,
    }


def test_allocate_priority_bits_empty_plan():
    assert allocate_priority_bits([], target_bpw=4.0) == []
    assert allocation_summary([]) == {
        "module_count": 0,
        "average_bits": 0.0,
        "min_bits": 0,
        "max_bits": 0,
    }


def test_allocate_priority_bits_uses_parameter_weights():
    modules = ["huge", "small_a", "small_b"]
    allocation = allocate_priority_bits(
        modules,
        target_bpw=4.2,
        priorities={"huge": 100.0, "small_a": 10.0, "small_b": 5.0},
        weights={"huge": 10, "small_a": 1, "small_b": 1},
    )

    # The huge module is highest priority, but upgrading it would exceed the
    # weighted fractional budget. The two small modules fit.
    assert {item.key: item.bits for item in allocation} == {
        "huge": 4,
        "small_a": 5,
        "small_b": 5,
    }
    assert allocation_summary(allocation) == {
        "module_count": 3,
        "average_bits": 4.166666666666667,
        "min_bits": 4,
        "max_bits": 5,
        "total_weight": 12,
    }


def test_allocate_priority_bits_reserves_fixed_costs_first():
    modules = ["head", "small_a", "small_b"]
    allocation = allocate_priority_bits(
        modules,
        target_bpw=4.5,
        priorities={"small_a": 10.0, "small_b": 5.0},
        weights={"head": 4, "small_a": 1, "small_b": 1},
        fixed_bits={"head": 5},
    )

    assert {item.key: item.bits for item in allocation} == {
        "head": 5,
        "small_a": 4,
        "small_b": 4,
    }
    assert allocation_summary(allocation)["average_bits"] == 4.666666666666667


def test_default_module_priority_prefers_output_sensitive_modules():
    assert default_module_priority("lm_head") > default_module_priority(
        "model.layers.0.mlp.down_proj"
    )
    assert default_module_priority("model.layers.0.self_attn.o_proj") > (
        default_module_priority("model.layers.0.mlp.up_proj")
    )
