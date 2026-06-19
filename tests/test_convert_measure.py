"""M5b bounded candidate measurement scaffolding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ponyexl3.convert import measure
from ponyexl3.convert.direct import DirectLayerResult
from ponyexl3.ref.layer import EXL3Layer


def _fake_layer(key: str, k: int) -> EXL3Layer:
    return EXL3Layer(
        key=key,
        in_features=128,
        out_features=128,
        k=k,
        trellis=np.zeros((8, 8, 256 * k // 16), dtype=np.uint16),
        mcg=True,
    )


def test_measure_ldlq_candidates_ranks_bits_and_shrinkage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, int | None, float, np.ndarray | None]] = []
    activations = {"a": np.ones((4, 128), dtype=np.float32)}

    def fake_ldlq(source_dir, oracle_dir, module_key, **kwargs):  # noqa: ARG001
        k = 4 if kwargs["quant_bits"] is None else int(kwargs["quant_bits"])
        shrinkage = float(kwargs["hessian_shrinkage"])
        calls.append((module_key, kwargs["quant_bits"], shrinkage, kwargs["calibration_activations"]))
        score = (6 - k) * 0.1 + shrinkage
        layer = _fake_layer(module_key, k)
        return DirectLayerResult(
            module_key=module_key,
            search_backend="metal",
            scale_mode="computed",
            layer=layer,
            activations=np.zeros((1, 128), dtype=np.float32),
            source_output=np.zeros((1, 128), dtype=np.float32),
            converted_output=np.zeros((1, 128), dtype=np.float32),
            stats={
                "output_rel_rms": score,
                "hessian_proxy_rel_rms": score * 2.0,
                "hessian_shrinkage": shrinkage,
                "hessian_offdiag_rel": 0.25 * (1.0 - shrinkage),
                "pack_roundtrip": True,
            },
        )

    monkeypatch.setattr(measure, "ldlq_quantize_layer", fake_ldlq)

    summary = measure.measure_ldlq_candidates(
        tmp_path / "source",
        tmp_path / "oracle",
        ["a"],
        candidate_bits=[4, 5],
        hessian_shrinkages=[0.0, 0.2],
        calibration_activations_by_module=activations,
        scale_mode="computed",
    )

    assert summary["candidate_count"] == 4
    assert summary["candidate_bits"] == [4, 5]
    assert summary["hessian_shrinkages"] == [0.0, 0.2]
    assert summary["best_by_module"] == [
        {
            "module": "a",
            "candidate_bits": 5,
            "k": 5,
            "hessian_shrinkage": 0.0,
            "score_metric": "output_rel_rms",
            "score": pytest.approx(0.1),
        }
    ]
    assert len(calls) == 4
    assert all(call[3] is activations["a"] for call in calls)


def test_measure_ldlq_candidates_uses_bit_plan_when_no_candidate_bits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    seen_bits: list[int | None] = []

    def fake_ldlq(source_dir, oracle_dir, module_key, **kwargs):  # noqa: ARG001
        seen_bits.append(kwargs["quant_bits"])
        k = int(kwargs["quant_bits"])
        layer = _fake_layer(module_key, k)
        return DirectLayerResult(
            module_key=module_key,
            search_backend="metal",
            scale_mode="computed",
            layer=layer,
            activations=np.zeros((1, 128), dtype=np.float32),
            source_output=np.zeros((1, 128), dtype=np.float32),
            converted_output=np.zeros((1, 128), dtype=np.float32),
            stats={"output_rel_rms": 0.0, "pack_roundtrip": True},
        )

    monkeypatch.setattr(measure, "ldlq_quantize_layer", fake_ldlq)

    summary = measure.measure_ldlq_candidates(
        tmp_path / "source",
        tmp_path / "oracle",
        ["a", "b"],
        bit_plan={"a": 4, "b": 5},
    )

    assert seen_bits == [4, 5]
    assert [item["k"] for item in summary["best_by_module"]] == [4, 5]


def test_measure_ldlq_candidates_rejects_bad_inputs(tmp_path: Path):
    with pytest.raises(ValueError, match="no modules"):
        measure.measure_ldlq_candidates(tmp_path / "source", tmp_path / "oracle", [])
    with pytest.raises(ValueError, match="shrinkage"):
        measure.measure_ldlq_candidates(
            tmp_path / "source",
            tmp_path / "oracle",
            ["a"],
            hessian_shrinkages=[1.1],
        )
