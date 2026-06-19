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
    ldlq_quantize_group,
    ldlq_inner_matrix,
    prepare_hessian_for_ldl,
    public_matrix_to_inner,
    reconstruct_oracle_public_fast,
)
from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.reconstruct import reconstruct_public_weights
from ponyexl3.ref.trellis import unpack_trellis


PILOT_MODULE = "model.language_model.layers.0.linear_attn.in_proj_qkv"


def _metal_available() -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return bool(mx.metal.is_available())


def _write_synthetic_source(model_dir: Path, module_key: str, public_weight: np.ndarray) -> None:
    _write_synthetic_sources(model_dir, {module_key: public_weight})


def _write_synthetic_sources(model_dir: Path, public_weights: dict[str, np.ndarray]) -> None:
    shard = "model.safetensors"
    tensors = {
        f"{module_key}.weight": public_weight.T.astype(np.float16)
        for module_key, public_weight in public_weights.items()
    }
    save_file(tensors, str(model_dir / shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": int((model_dir / shard).stat().st_size)},
                "weight_map": {tensor_key: shard for tensor_key in tensors},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_synthetic_oracle(model_dir: Path, module_key: str, k: int = 4) -> None:
    _write_synthetic_oracles(model_dir, [module_key], k=k)


def _write_synthetic_oracles(model_dir: Path, module_keys: list[str], k: int = 4) -> None:
    shard = "model.safetensors"
    tensors = {
        f"{module_key}.trellis": np.zeros((8, 8, 256 * k // 16), dtype=np.uint16)
        for module_key in module_keys
    }
    save_file(tensors, str(model_dir / shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": int((model_dir / shard).stat().st_size)},
                "weight_map": {trellis_key: shard for trellis_key in tensors},
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
                        "stored_tensors": {
                            f"{module_key}.trellis": {
                                "dtype": "uint16",
                                "shape": [int(x) for x in tensors[f"{module_key}.trellis"].shape],
                                "n_bytes": int(tensors[f"{module_key}.trellis"].nbytes),
                            }
                        },
                    }
                    for module_key in module_keys
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


def test_ldlq_can_skip_diagnostic_state_and_proxy():
    rng = np.random.default_rng(107)
    inner = (rng.standard_normal((32, 32)) * 0.05).astype(np.float32)
    activations = rng.standard_normal((64, 32)).astype(np.float32)
    prepared = prepare_hessian_for_ldl(capture_hessian(activations))

    result = ldlq_inner_matrix(
        inner,
        np.eye(32, dtype=np.float32),
        k=2,
        cb=CodebookMode.DEFAULT,
        hessian=prepared.hessian,
        search_backend="cpu",
        buf_size_rows=32,
        collect_states=False,
        compute_proxy=False,
    )

    assert result.states is None
    assert "hessian_proxy_rel_rms" not in result.stats
    assert result.packed.shape == (2, 2, 32)
    assert result.reconstructed.shape == inner.shape


@pytest.mark.skipif(not _metal_available(), reason="Metal is required for MLX LDLQ parity")
def test_ldlq_mlx_no_state_matches_debug_state_path():
    rng = np.random.default_rng(108)
    inner = (rng.standard_normal((32, 32)) * 0.05).astype(np.float32)
    l_factor = np.eye(32, dtype=np.float32)

    debug = ldlq_inner_matrix(
        inner,
        l_factor,
        k=4,
        cb=CodebookMode.MCG,
        search_backend="metal",
        buf_size_rows=32,
        collect_states=True,
        compute_proxy=False,
    )
    fast = ldlq_inner_matrix(
        inner,
        l_factor,
        k=4,
        cb=CodebookMode.MCG,
        search_backend="metal",
        buf_size_rows=32,
        collect_states=False,
        compute_proxy=False,
    )

    assert debug.states is not None
    assert fast.states is None
    assert fast.stats["mlx_ldlq"] is True
    np.testing.assert_array_equal(fast.packed, debug.packed)
    np.testing.assert_array_equal(fast.reconstructed, debug.reconstructed)


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


@pytest.mark.skipif(not _metal_available(), reason="Metal is required for batched-search LDLQ")
def test_ldlq_group_batched_search_matches_individual_with_distinct_scales(tmp_path: Path):
    source_dir = tmp_path / "source"
    oracle_dir = tmp_path / "oracle"
    source_dir.mkdir()
    oracle_dir.mkdir()
    keys = [
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    ]
    rng = np.random.default_rng(109)
    row_scale = np.linspace(0.55, 1.45, 128, dtype=np.float32).reshape(128, 1)
    weights = {
        keys[0]: (rng.standard_normal((128, 128)) * 0.05).astype(np.float32),
        keys[1]: (rng.standard_normal((128, 128)) * 0.05 * row_scale).astype(np.float32),
    }
    activations = {
        keys[0]: rng.standard_normal((48, 128)).astype(np.float32),
        keys[1]: (rng.standard_normal((48, 128)) * 0.75).astype(np.float32),
    }
    _write_synthetic_sources(source_dir, weights)
    _write_synthetic_oracles(oracle_dir, keys)

    grouped = ldlq_quantize_group(
        source_dir,
        oracle_dir,
        keys,
        search_backend="metal",
        scale_mode="computed",
        buf_size_rows=64,
        feedback_rows=32,
        calibration_activations_by_module=activations,
        skip_g_scale=True,
        regularization_seed=5,
    )
    singles = [
        ldlq_quantize_layer(
            source_dir,
            oracle_dir,
            key,
            search_backend="metal",
            scale_mode="computed",
            buf_size_rows=64,
            feedback_rows=32,
            calibration_activations=activations[key],
            skip_g_scale=True,
            regularization_seed=5,
            compare_oracle=False,
            fast_metrics=True,
        )
        for key in keys
    ]

    assert [result.module_key for result in grouped] == keys
    assert grouped[0].layer.suh is not None
    assert grouped[1].layer.suh is not None
    assert not np.array_equal(grouped[0].layer.suh, grouped[1].layer.suh)
    for batched, single in zip(grouped, singles, strict=True):
        assert batched.layer.suh is not None
        assert single.layer.suh is not None
        assert batched.layer.svh is not None
        assert single.layer.svh is not None
        np.testing.assert_array_equal(batched.layer.trellis, single.layer.trellis)
        np.testing.assert_allclose(batched.layer.suh, single.layer.suh, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(batched.layer.svh, single.layer.svh, rtol=0.0, atol=0.0)
        assert batched.stats["batched_search_group_size"] == 2.0
        assert batched.stats["batched_prep_workers"] == 2.0
        assert batched.stats["mlx_ldlq"] is True


def test_reconstruct_oracle_public_fast_matches_reference():
    """The Metal-decoded oracle reconstruct must match the Python reference.

    Guards the ~4000x ``--oracle-metrics`` speedup: only the inner trellis decode
    is swapped (Metal vs per-element Python), so the public weights must be
    identical across codebook modes and packed-sign / float scales.
    """

    if not _metal_available():
        pytest.skip("Metal required")

    rng = np.random.default_rng(7)
    k = 4
    in_features = out_features = 128
    in_tiles, out_tiles = in_features // 16, out_features // 16
    trellis = rng.integers(
        0, 1 << 16, size=(in_tiles, out_tiles, 256 * k // 16), dtype=np.uint16
    )
    suh_packed = rng.integers(-(1 << 15), 1 << 15, size=in_features // 16, dtype=np.int16)
    svh_packed = rng.integers(-(1 << 15), 1 << 15, size=out_features // 16, dtype=np.int16)
    suh_float = (rng.standard_normal(in_features) * 0.1).astype(np.float16)
    svh_float = (rng.standard_normal(out_features) * 0.1).astype(np.float16)

    cases = [
        (False, False, suh_packed, svh_packed),
        (True, False, suh_packed, svh_packed),
        (False, True, suh_float, svh_float),
        (False, False, None, None),
    ]
    for mcg, mul1, suh, svh in cases:
        layer = EXL3Layer(
            key="m",
            in_features=in_features,
            out_features=out_features,
            k=k,
            trellis=trellis,
            suh=suh,
            svh=svh,
            mcg=mcg,
            mul1=mul1,
        )
        layer.validate()
        fast = reconstruct_oracle_public_fast(layer)
        ref = reconstruct_public_weights(trellis, suh, svh, k, mcg=mcg, mul1=mul1)
        np.testing.assert_array_equal(fast, ref)
