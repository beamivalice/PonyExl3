"""M5b bounded candidate measurement scaffolding."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ponyexl3.convert import measure
from ponyexl3.convert.direct import DirectLayerResult
from ponyexl3.ref.layer import EXL3Layer

ROOT = Path(__file__).resolve().parents[1]


def _fake_layer(key: str, k: int) -> EXL3Layer:
    return EXL3Layer(
        key=key,
        in_features=128,
        out_features=128,
        k=k,
        trellis=np.zeros((8, 8, 256 * k // 16), dtype=np.uint16),
        mcg=True,
    )


def _measurement_record(
    module: str,
    *,
    k: int,
    score: float,
    weight: int = 1,
    shrinkage: float = 0.0,
) -> dict[str, object]:
    return {
        "module": module,
        "candidate_bits": k,
        "k": k,
        "hessian_shrinkage": shrinkage,
        "score": score,
        "score_metric": "output_rel_rms",
        "summary": {
            "module": module,
            "k": k,
            "shape": [weight, 1],
            "stats": {
                "output_rel_rms": score,
                "hessian_shrinkage": shrinkage,
            },
        },
    }


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


def test_measure_ldlq_candidates_resumes_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "measure.json"
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
            stats={"output_rel_rms": float(6 - k), "pack_roundtrip": True},
        )

    monkeypatch.setattr(measure, "ldlq_quantize_layer", fake_ldlq)

    first = measure.measure_ldlq_candidates(
        tmp_path / "source",
        tmp_path / "oracle",
        ["a"],
        candidate_bits=[4],
        checkpoint_path=checkpoint,
    )
    assert first["candidate_count"] == 1
    assert seen_bits == [4]

    resumed = measure.measure_ldlq_candidates(
        tmp_path / "source",
        tmp_path / "oracle",
        ["a"],
        candidate_bits=[4, 5],
        checkpoint_path=checkpoint,
        resume=True,
    )

    assert seen_bits == [4, 5]
    assert resumed["candidate_count"] == 2
    assert [record["k"] for record in resumed["records"]] == [4, 5]


def test_optimize_measurement_plan_spends_budget_by_measured_gain():
    measurement = {
        "score_metric": "output_rel_rms",
        "records": [
            _measurement_record("huge", k=4, score=0.10, weight=10),
            _measurement_record("huge", k=5, score=0.05, weight=10),
            _measurement_record("small", k=4, score=0.20, weight=1),
            _measurement_record("small", k=5, score=0.01, weight=1),
        ],
    }

    plan = measure.optimize_measurement_plan(measurement, target_bpw=4.1)

    assert plan["bit_plan"] == {"huge": 4, "small": 5}
    assert plan["spent_weighted_bits"] == 45
    assert plan["target_weighted_bits"] == 45
    assert plan["average_bits"] == pytest.approx(45 / 11)
    assert plan["upgrades"][0]["module"] == "small"
    assert plan["layer_bits"] == [r"^huge$:4", r"^small$:5"]


def test_optimize_measurement_plan_selects_best_global_shrinkage():
    measurement = {
        "score_metric": "output_rel_rms",
        "records": [
            _measurement_record("a", k=4, score=0.20, shrinkage=0.0),
            _measurement_record("b", k=4, score=0.20, shrinkage=0.0),
            _measurement_record("a", k=4, score=0.05, shrinkage=0.1),
            _measurement_record("b", k=4, score=0.06, shrinkage=0.1),
        ],
    }

    plan = measure.optimize_measurement_plan(measurement, target_bpw=4.0)

    assert plan["mode"] == "global_hessian_shrinkage"
    assert plan["hessian_shrinkage"] == 0.1
    assert plan["objective"] == pytest.approx(0.055)
    assert len(plan["global_plans"]) == 2


def test_optimize_measurement_plan_forces_fixed_bits():
    measurement = {
        "score_metric": "output_rel_rms",
        "records": [
            _measurement_record("lm_head", k=4, score=0.01, weight=1),
            _measurement_record("lm_head", k=6, score=0.20, weight=1),
            _measurement_record("body", k=4, score=0.30, weight=10),
            _measurement_record("body", k=5, score=0.10, weight=10),
        ],
    }

    plan = measure.optimize_measurement_plan(
        measurement,
        target_bpw=4.2,
        fixed_bits={"lm_head": 6},
    )

    assert plan["bit_plan"]["lm_head"] == 6
    assert plan["fixed_bits"] == {"lm_head": 6}
    assert all(item["module"] != "lm_head" for item in plan["upgrades"])


def test_optimize_measurements_cli_json(tmp_path: Path):
    measurement = {
        "score_metric": "output_rel_rms",
        "records": [
            _measurement_record("a", k=4, score=0.30),
            _measurement_record("a", k=5, score=0.10),
        ],
    }
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(measurement), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ponyexl3.cli.optimize_measurements",
            str(path),
            "--bits",
            "5.0",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["bit_plan"] == {"a": 5}
    assert plan["layer_bits_args"] == ["--layer-bits", r"^a$:5"]


def test_optimize_measurements_cli_head_bits(tmp_path: Path):
    measurement = {
        "score_metric": "output_rel_rms",
        "records": [
            _measurement_record("lm_head", k=4, score=0.01),
            _measurement_record("lm_head", k=6, score=0.20),
        ],
    }
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(measurement), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ponyexl3.cli.optimize_measurements",
            str(path),
            "--bits",
            "6.0",
            "--head-bits",
            "6",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["bit_plan"] == {"lm_head": 6}
    assert plan["fixed_bits"] == {"lm_head": 6}


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
