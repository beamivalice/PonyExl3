"""M4 Hessian/LDLQ converter primitives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ponyexl3.convert.direct import quantize_inner_matrix_direct, write_direct_layer_bundle
from ponyexl3.convert.hessian import (
    block_ldl,
    capture_hessian,
    hessian_proxy_stats,
    ldlq_quantize_layer,
    ldlq_inner_matrix,
    prepare_hessian_for_ldl,
    public_matrix_to_inner,
)
from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.trellis import unpack_trellis


PILOT_MODULE = "model.language_model.layers.0.linear_attn.in_proj_qkv"


def _metal_available() -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return bool(mx.metal.is_available())


def _write_synthetic_source(model_dir: Path, module_key: str, public_weight: np.ndarray) -> None:
    shard = "model.safetensors"
    tensor_key = f"{module_key}.weight"
    weight = public_weight.T.astype(np.float16)
    save_file({tensor_key: weight}, str(model_dir / shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": int((model_dir / shard).stat().st_size)},
                "weight_map": {tensor_key: shard},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_synthetic_oracle(model_dir: Path, module_key: str, k: int = 4) -> None:
    shard = "model.safetensors"
    trellis_key = f"{module_key}.trellis"
    trellis = np.zeros((8, 8, 256 * k // 16), dtype=np.uint16)
    save_file({trellis_key: trellis}, str(model_dir / shard))
    stored = {
        trellis_key: {
            "dtype": "uint16",
            "shape": [int(x) for x in trellis.shape],
            "n_bytes": int(trellis.nbytes),
        }
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": int((model_dir / shard).stat().st_size)},
                "weight_map": {trellis_key: shard},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (model_dir / "quantization_config.json").write_text(
        json.dumps(
            {
                "quant_method": "exl3",
                "tensor_storage": {
                    module_key: {
                        "quant_format": "exl3",
                        "bits_per_weight": float(k),
                        "mcg_multiplier": False,
                        "mul1_multiplier": False,
                        "stored_tensors": stored,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_capture_prepare_block_ldl_identity_blocks():
    rng = np.random.default_rng(101)
    activations = rng.standard_normal((96, 32)).astype(np.float32)
    hessian = capture_hessian(activations)
    prepared = prepare_hessian_for_ldl(hessian)
    result = block_ldl(prepared.hessian, block_size=16)

    assert result.l.shape == (32, 32)
    assert result.hessian.shape == (32, 32)
    assert result.retries == 0
    for start in (0, 16):
        block = result.l[start : start + 16, start : start + 16]
        np.testing.assert_allclose(block, np.eye(16, dtype=np.float32), atol=2e-5)
    np.testing.assert_allclose(np.triu(result.l, k=1), 0.0, atol=2e-5)


def test_ldlq_identity_hessian_matches_direct_quantization():
    rng = np.random.default_rng(102)
    inner = (rng.standard_normal((32, 32)) * 0.05).astype(np.float32)
    direct_packed, direct_states, direct_reconstructed = quantize_inner_matrix_direct(
        inner,
        k=2,
        cb=CodebookMode.DEFAULT,
        search_backend="cpu",
    )
    result = ldlq_inner_matrix(
        inner,
        np.eye(32, dtype=np.float32),
        k=2,
        cb=CodebookMode.DEFAULT,
        search_backend="cpu",
        buf_size_rows=32,
    )

    np.testing.assert_array_equal(result.packed, direct_packed)
    np.testing.assert_array_equal(result.states, direct_states)
    np.testing.assert_array_equal(result.reconstructed, direct_reconstructed)
    assert result.stats["pack_roundtrip"] is True


def test_ldlq_grouped_feedback_identity_hessian_matches_direct_quantization():
    rng = np.random.default_rng(106)
    inner = (rng.standard_normal((64, 32)) * 0.05).astype(np.float32)
    direct_packed, direct_states, direct_reconstructed = quantize_inner_matrix_direct(
        inner,
        k=2,
        cb=CodebookMode.DEFAULT,
        search_backend="cpu",
    )
    result = ldlq_inner_matrix(
        inner,
        np.eye(64, dtype=np.float32),
        k=2,
        cb=CodebookMode.DEFAULT,
        search_backend="cpu",
        buf_size_rows=64,
        feedback_rows=64,
    )

    np.testing.assert_array_equal(result.packed, direct_packed)
    np.testing.assert_array_equal(result.states, direct_states)
    np.testing.assert_array_equal(result.reconstructed, direct_reconstructed)
    assert result.stats["ldlq_feedback_rows"] == 64.0


def test_ldlq_rejects_invalid_feedback_rows():
    inner = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="feedback_rows"):
        ldlq_inner_matrix(
            inner,
            np.eye(32, dtype=np.float32),
            k=2,
            cb=CodebookMode.DEFAULT,
            search_backend="cpu",
            buf_size_rows=32,
            feedback_rows=24,
        )


def test_public_matrix_to_inner_identity_mode_is_noop():
    rng = np.random.default_rng(105)
    public = rng.standard_normal((128, 128)).astype(np.float32)
    inner = public_matrix_to_inner(public, suh=None, svh=None)
    np.testing.assert_allclose(inner, public, rtol=0.0, atol=2e-6)


def test_ldlq_correlated_hessian_emits_roundtrippable_proxy_stats():
    rng = np.random.default_rng(103)
    activations = rng.standard_normal((128, 32)).astype(np.float32)
    activations[:, :16] += 0.4 * activations[:, 16:]
    inner = (rng.standard_normal((32, 32)) * 0.05).astype(np.float32)
    prepared = prepare_hessian_for_ldl(capture_hessian(activations))
    ldl = block_ldl(prepared.hessian, block_size=16)

    direct_packed, _direct_states, direct_reconstructed = quantize_inner_matrix_direct(
        inner,
        k=2,
        cb=CodebookMode.DEFAULT,
        search_backend="cpu",
    )
    result = ldlq_inner_matrix(
        inner,
        ldl.l,
        k=2,
        cb=CodebookMode.DEFAULT,
        hessian=prepared.hessian,
        search_backend="cpu",
        buf_size_rows=32,
    )
    direct_proxy = hessian_proxy_stats(
        direct_reconstructed - inner,
        prepared.hessian,
        reference=inner,
    )

    assert result.packed.shape == direct_packed.shape
    np.testing.assert_array_equal(unpack_trellis(result.packed, 2), result.states)
    assert np.isfinite(result.reconstructed).all()
    assert np.isfinite(float(result.stats["hessian_proxy_rel_rms"]))
    assert float(result.stats["hessian_proxy_rel_rms"]) <= direct_proxy.proxy_rel_rms * 1.25


@pytest.mark.skipif(not _metal_available(), reason="Metal is required for the 128x128 layer gate")
def test_ldlq_layer_identity_synthetic_emits_loadable_layer(tmp_path: Path):
    source_dir = tmp_path / "source"
    oracle_dir = tmp_path / "oracle"
    out_dir = tmp_path / "out"
    source_dir.mkdir()
    oracle_dir.mkdir()
    rng = np.random.default_rng(104)
    public_weight = (rng.standard_normal((128, 128)) * 0.05).astype(np.float32)
    _write_synthetic_source(source_dir, PILOT_MODULE, public_weight)
    _write_synthetic_oracle(oracle_dir, PILOT_MODULE)

    result = ldlq_quantize_layer(
        source_dir,
        oracle_dir,
        PILOT_MODULE,
        search_backend="metal",
        scale_mode="identity",
        buf_size_rows=128,
    )
    assert result.layer.in_features == 128
    assert result.layer.out_features == 128
    assert result.layer.trellis.shape == (8, 8, 64)
    assert result.stats["pack_roundtrip"] is True
    assert np.isfinite(result.converted_output).all()
    assert np.isfinite(float(result.stats["hessian_proxy_rel_rms"]))
    assert np.isfinite(float(result.stats["oracle_hessian_proxy_rel_rms"]))
    assert np.isfinite(float(result.stats["hessian_proxy_rel_rms_over_oracle"]))
    assert np.isfinite(float(result.stats["oracle_output_rel_rms"]))
    assert float(result.stats["hessian_proxy_rel_rms_over_oracle"]) < 1.0
    assert float(result.stats["output_rel_rms_over_oracle"]) < 1.0

    loaded = write_direct_layer_bundle(result, out_dir)
    assert loaded.in_features == 128
    assert loaded.out_features == 128
    assert loaded.trellis.shape == result.layer.trellis.shape

    no_oracle = ldlq_quantize_layer(
        source_dir,
        oracle_dir,
        PILOT_MODULE,
        search_backend="metal",
        scale_mode="identity",
        buf_size_rows=128,
        feedback_rows=128,
        compare_oracle=False,
    )
    assert no_oracle.stats["oracle_metrics"] is False
    assert "oracle_output_rel_rms" not in no_oracle.stats


@pytest.mark.skipif(not _metal_available(), reason="Metal is required for the 128x128 layer gate")
def test_ldlq_layer_computed_scales_with_calibration_rows(tmp_path: Path):
    source_dir = tmp_path / "source"
    oracle_dir = tmp_path / "oracle"
    source_dir.mkdir()
    oracle_dir.mkdir()
    rng = np.random.default_rng(106)
    public_weight = (rng.standard_normal((128, 128)) * 0.05).astype(np.float32)
    activations = rng.standard_normal((32, 128)).astype(np.float32)
    _write_synthetic_source(source_dir, PILOT_MODULE, public_weight)
    _write_synthetic_oracle(oracle_dir, PILOT_MODULE)

    result = ldlq_quantize_layer(
        source_dir,
        oracle_dir,
        PILOT_MODULE,
        search_backend="metal",
        scale_mode="computed",
        buf_size_rows=128,
        calibration_activations=activations,
        skip_g_scale=True,
        regularization_seed=3,
    )

    assert result.layer.suh is not None
    assert result.layer.svh is not None
    assert result.stats["regularize_computed_scales"] is True
    assert result.stats["regularize_g_scale_skipped"] is True
    np.testing.assert_allclose(result.activations, activations, rtol=0.0, atol=0.0)
    assert np.isfinite(result.converted_output).all()
