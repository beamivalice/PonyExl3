"""Converter M1 gates: reference trellis search + pack round-trip."""

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ponyexl3.convert.fixtures import (
    bf16_to_float32,
    read_source_public_block,
    resolve_source_linear,
    run_tile_pilot,
)
from ponyexl3.convert.direct import (
    direct_quantize_layer,
    direct_quantize_window,
    write_direct_layer_bundle,
    write_direct_window_bundle,
)
from ponyexl3.convert.reference_search import quantize_tile_reference
from ponyexl3.ref.reconstruct import reconstruct_public_weights
from ponyexl3.ref.trellis import pack_trellis_tile, unpack_trellis_tile


SOURCE_35B = Path("/Users/beam/llm/models/Qwen/Qwen3.6-35B-A3B")
ORACLE_35B = Path("/Users/beam/llm/models/Exl3/Qwen3.6-35B-A3B-exl3-4.00bpw")
PILOT_MODULE = "model.language_model.layers.0.linear_attn.in_proj_qkv"
PILOT_EXPERT_GATE = "model.language_model.layers.0.mlp.experts.0.gate_proj"


def _metal_available() -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return bool(mx.metal.is_available())


@pytest.mark.parametrize("k", [2, 3])
def test_search_pack_roundtrip(k):
    rng = np.random.default_rng(7)
    w = rng.standard_normal(256).astype(np.float32)
    states, decoded = quantize_tile_reference(w, k=k)
    # tail-biting transition invariant (checkpoint-validated convention)
    s = states.astype(np.uint32)
    nxt = np.roll(s, -1)
    assert ((((s << k) | (nxt & ((1 << k) - 1))) & 0xFFFF) == nxt).all()
    # bit-exact round-trip through the inference-side pack/unpack
    packed = pack_trellis_tile((states & ((1 << k) - 1)).astype(np.uint16), k)
    assert (unpack_trellis_tile(packed, k).astype(np.uint16) == states).all()
    # quantization quality sanity (QTIP-class MSE on unit Gaussian)
    mse = float(((decoded - w) ** 2).mean())
    assert mse < {2: 0.11, 3: 0.032}[k]


def test_bf16_to_float32_known_values():
    words = np.array([0x3F80, 0xC000, 0x0000, 0x3F00], dtype=np.uint16)
    got = bf16_to_float32(words)
    np.testing.assert_array_equal(got, np.array([1.0, -2.0, 0.0, 0.5], dtype=np.float32))


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
                        "mcg_multiplier": True,
                        "mul1_multiplier": False,
                        "stored_tensors": stored,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_direct_layer_identity_synthetic_emits_loadable_layer(tmp_path: Path):
    module_key = PILOT_MODULE
    source_dir = tmp_path / "source"
    oracle_dir = tmp_path / "oracle"
    out_dir = tmp_path / "out"
    source_dir.mkdir()
    oracle_dir.mkdir()
    rng = np.random.default_rng(123)
    public_weight = (rng.standard_normal((128, 128)) * 0.05).astype(np.float32)
    _write_synthetic_source(source_dir, module_key, public_weight)
    _write_synthetic_oracle(oracle_dir, module_key)

    result = direct_quantize_layer(
        source_dir,
        oracle_dir,
        module_key,
        search_backend="metal" if _metal_available() else "cpu",
        scale_mode="identity",
    )
    assert result.layer.in_features == 128
    assert result.layer.out_features == 128
    assert result.layer.trellis.shape == (8, 8, 64)
    assert result.stats["pack_roundtrip"] is True
    assert np.isfinite(result.converted_output).all()

    loaded = write_direct_layer_bundle(result, out_dir)
    assert loaded.in_features == 128
    assert loaded.out_features == 128
    assert loaded.trellis.shape == result.layer.trellis.shape


@pytest.mark.skipif(
    not (SOURCE_35B / "model.safetensors.index.json").is_file(),
    reason="local Qwen3.6-35B-A3B source checkpoint not present",
)
def test_qwen_source_linear_adapters_are_lightweight():
    dense = resolve_source_linear(SOURCE_35B, PILOT_MODULE)
    assert dense.source_tensor_key == f"{PILOT_MODULE}.weight"
    assert dense.in_features == 2048
    assert dense.out_features == 8192
    dense_block = read_source_public_block(SOURCE_35B, dense, in_start=0, out_start=0)
    assert dense_block.shape == (128, 128)
    assert np.isfinite(dense_block).all()

    gate = resolve_source_linear(SOURCE_35B, PILOT_EXPERT_GATE)
    assert gate.source_tensor_key.endswith(".mlp.experts.gate_up_proj")
    assert gate.layout == "qwen_gate"
    assert gate.in_features == 2048
    assert gate.out_features == 512
    gate_block = read_source_public_block(SOURCE_35B, gate, in_start=0, out_start=0)
    assert gate_block.shape == (128, 128)
    assert np.isfinite(gate_block).all()


@pytest.mark.skipif(
    not (
        (SOURCE_35B / "model.safetensors.index.json").is_file()
        and (ORACLE_35B / "quantization_config.json").is_file()
    ),
    reason="local Qwen3.6-35B-A3B source/oracle checkpoints not present",
)
def test_one_tile_pilot_compares_with_oracle():
    result = run_tile_pilot(SOURCE_35B, ORACLE_35B, PILOT_MODULE, tile_k=0, tile_n=0)
    assert result.search_backend == "cpu"
    assert result.k == 4
    assert result.stats["converted_pack_roundtrip"] is True
    assert result.stats["oracle_pack_roundtrip"] is True
    assert np.isfinite(result.converted_tile).all()
    assert np.isfinite(result.oracle_tile).all()
    assert np.isfinite(result.target_tile).all()
    assert result.stats["converted_target_mse"] < 0.05


@pytest.mark.skipif(
    not (
        (SOURCE_35B / "model.safetensors.index.json").is_file()
        and (ORACLE_35B / "quantization_config.json").is_file()
        and _metal_available()
    ),
    reason="local Qwen3.6-35B-A3B checkpoints or Metal are not present",
)
def test_one_tile_pilot_metal_backend_compares_with_oracle():
    result = run_tile_pilot(
        SOURCE_35B,
        ORACLE_35B,
        PILOT_MODULE,
        tile_k=0,
        tile_n=0,
        search_backend="metal",
    )
    assert result.search_backend == "metal"
    assert result.k == 4
    assert result.stats["converted_pack_roundtrip"] is True
    assert result.stats["oracle_pack_roundtrip"] is True
    assert np.isfinite(result.converted_tile).all()
    assert result.stats["converted_target_mse"] < 0.05


@pytest.mark.skipif(
    not (
        (SOURCE_35B / "model.safetensors.index.json").is_file()
        and (ORACLE_35B / "quantization_config.json").is_file()
        and _metal_available()
    ),
    reason="local Qwen3.6-35B-A3B checkpoints or Metal are not present",
)
def test_direct_window_emits_loadable_mini_layer(tmp_path):
    result = direct_quantize_window(
        SOURCE_35B,
        ORACLE_35B,
        PILOT_MODULE,
        search_backend="metal",
    )
    assert result.layer.in_features == 128
    assert result.layer.out_features == 128
    assert result.layer.trellis.shape == (8, 8, 64)
    assert result.stats["pack_roundtrip"] is True
    assert result.stats["public_rel_rms"] < 0.12
    assert result.stats["output_rel_rms"] < 0.12

    loaded = write_direct_window_bundle(result, tmp_path)
    assert loaded.key == PILOT_MODULE
    assert loaded.in_features == 128
    assert loaded.out_features == 128
    assert loaded.k == result.layer.k
    loaded_public = reconstruct_public_weights(
        loaded.trellis,
        loaded.suh,
        loaded.svh,
        loaded.k,
        mcg=loaded.mcg,
        mul1=loaded.mul1,
    ).astype(np.float32)
    np.testing.assert_array_equal(loaded_public, result.reconstructed_public)


@pytest.mark.skipif(
    not (
        (SOURCE_35B / "model.safetensors.index.json").is_file()
        and (ORACLE_35B / "quantization_config.json").is_file()
    ),
    reason="local Qwen3.6-35B-A3B source/oracle checkpoints not present",
)
def test_tile_pilot_rejects_noninvertible_oracle_scale_block():
    with pytest.raises(ValueError, match="zero scales"):
        run_tile_pilot(SOURCE_35B, ORACLE_35B, PILOT_EXPERT_GATE, tile_k=0, tile_n=0)
