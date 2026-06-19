"""M4 module-set driver and multi-layer bundle gates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ponyexl3.convert import driver as convert_driver
from ponyexl3.convert.direct import DirectLayerResult
from ponyexl3.convert.direct import write_exl3_layers_bundle
from ponyexl3.convert.driver import (
    convert_module_set,
    layer_module_keys,
    model_module_keys,
    plain_tensor_keys,
    supported_model_module_keys,
)
from ponyexl3.ref.layer import EXL3Layer


def _layer(key: str, *, out_tiles: int = 8, k: int = 4) -> EXL3Layer:
    return EXL3Layer(
        key=key,
        in_features=128,
        out_features=out_tiles * 16,
        k=k,
        trellis=np.zeros((8, out_tiles, 256 * k // 16), dtype=np.uint16),
        mcg=True,
    )


def test_write_multi_layer_bundle_with_manifest_and_assets(tmp_path: Path):
    assets = tmp_path / "assets"
    out_dir = tmp_path / "out"
    assets.mkdir()
    (assets / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (assets / "tokenizer.json").write_text("{}", encoding="utf-8")
    layers = [
        _layer("model.language_model.layers.0.linear_attn.in_proj_qkv"),
        _layer("model.language_model.layers.0.linear_attn.in_proj_z"),
    ]

    loaded = write_exl3_layers_bundle(
        layers,
        out_dir,
        asset_dir=assets,
        manifest={"completed": [{"module": layers[0].key}], "skipped": []},
    )

    assert [layer.key for layer in loaded] == [layer.key for layer in layers]
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "tokenizer.json").is_file()
    qcfg = json.loads((out_dir / "quantization_config.json").read_text(encoding="utf-8"))
    assert sorted(qcfg["tensor_storage"]) == sorted(layer.key for layer in layers)
    index = json.loads((out_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert len(index["weight_map"]) == 2
    manifest = json.loads((out_dir / "ponyexl3_convert_manifest.json").read_text(encoding="utf-8"))
    assert manifest["layer_count"] == 2
    assert manifest["tensor_count"] == 2
    assert sorted(manifest["asset_files"]) == ["config.json", "tokenizer.json"]


def test_layer_module_keys_excludes_routed_experts_by_default(tmp_path: Path):
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    layers = [
        _layer("model.language_model.layers.0.linear_attn.in_proj_qkv"),
        _layer("model.language_model.layers.0.mlp.experts.0.gate_proj"),
        _layer("model.language_model.layers.1.linear_attn.in_proj_qkv"),
    ]
    write_exl3_layers_bundle(layers, oracle_dir)

    assert layer_module_keys(oracle_dir, 0) == [
        "model.language_model.layers.0.linear_attn.in_proj_qkv"
    ]
    assert layer_module_keys(oracle_dir, 0, include_routed_experts=True) == [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
    ]


def test_model_module_keys_puts_lm_head_last(tmp_path: Path):
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    layers = [
        _layer("lm_head"),
        _layer("model.layers.0.self_attn.q_proj"),
        _layer("model.layers.0.self_attn.k_proj"),
    ]
    write_exl3_layers_bundle(layers, oracle_dir)

    assert model_module_keys(oracle_dir) == [
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.q_proj",
        "lm_head",
    ]


def test_module_set_resume_loads_existing_layer(tmp_path: Path):
    out_dir = tmp_path / "out"
    layer = _layer("model.language_model.layers.0.linear_attn.in_proj_qkv")
    write_exl3_layers_bundle([layer], out_dir)

    result = convert_module_set(
        tmp_path / "missing_source",
        tmp_path / "missing_oracle",
        [layer.key],
        out_dir=out_dir,
        resume=True,
    )

    assert len(result.loaded_layers) == 1
    assert result.loaded_layers[0].key == layer.key
    assert result.completed[0]["resumed"] is True
    manifest = json.loads((out_dir / "ponyexl3_convert_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"][0]["resumed"] is True


def test_module_set_bit_plan_overrides_emitted_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    key = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    out_dir = tmp_path / "out"

    def fake_convert_one(*args, **kwargs):  # noqa: ARG001
        quant_bits = kwargs["quant_bits"]
        layer = _layer(key, k=quant_bits)
        return DirectLayerResult(
            module_key=key,
            search_backend="cpu",
            scale_mode="identity",
            layer=layer,
            activations=np.zeros((1, layer.in_features), dtype=np.float32),
            source_output=np.zeros((1, layer.out_features), dtype=np.float32),
            converted_output=np.zeros((1, layer.out_features), dtype=np.float32),
            stats={"output_rel_rms": 0.0, "public_rel_rms": 0.0},
        )

    monkeypatch.setattr(convert_driver, "_convert_one", fake_convert_one)

    result = convert_driver.convert_module_set(
        tmp_path / "source",
        tmp_path / "oracle",
        [key],
        quantizer="direct",
        out_dir=out_dir,
        search_backend="cpu",
        scale_mode="identity",
        bit_plan={key: 5},
    )

    assert result.loaded_layers[0].k == 5
    assert result.completed[0]["k"] == 5
    manifest = json.loads((out_dir / "ponyexl3_convert_manifest.json").read_text(encoding="utf-8"))
    assert manifest["bit_plan"] == {key: 5}


def test_module_set_incremental_output_resumes_missing_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    keys = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
    ]
    out_dir = tmp_path / "out"
    calls: list[str] = []

    def fake_convert_one(*args, **kwargs):  # noqa: ARG001
        key = str(args[3])
        calls.append(key)
        layer = _layer(key)
        return DirectLayerResult(
            module_key=key,
            search_backend="cpu",
            scale_mode="identity",
            layer=layer,
            activations=np.zeros((1, layer.in_features), dtype=np.float32),
            source_output=np.zeros((1, layer.out_features), dtype=np.float32),
            converted_output=np.zeros((1, layer.out_features), dtype=np.float32),
            stats={"output_rel_rms": 0.0, "public_rel_rms": 0.0},
        )

    monkeypatch.setattr(convert_driver, "_convert_one", fake_convert_one)

    first = convert_driver.convert_module_set(
        tmp_path / "source",
        tmp_path / "oracle",
        [keys[0]],
        quantizer="direct",
        out_dir=out_dir,
        search_backend="cpu",
        scale_mode="identity",
        incremental_output=True,
    )
    assert [item["module"] for item in first.completed] == [keys[0]]

    second = convert_driver.convert_module_set(
        tmp_path / "source",
        tmp_path / "oracle",
        keys,
        quantizer="direct",
        out_dir=out_dir,
        search_backend="cpu",
        scale_mode="identity",
        resume=True,
        incremental_output=True,
    )

    assert calls == keys
    assert second.completed[0]["resumed"] is True
    qcfg = json.loads((out_dir / "quantization_config.json").read_text(encoding="utf-8"))
    assert sorted(qcfg["tensor_storage"]) == sorted(keys)
    index = json.loads((out_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"][f"{keys[0]}.trellis"].startswith("ponyexl3-layer-")
    manifest = json.loads((out_dir / "ponyexl3_convert_manifest.json").read_text(encoding="utf-8"))
    assert manifest["incremental_output"] is True
    assert manifest["layer_count"] == 2


def test_module_set_resume_recomputes_existing_layer_when_planned_k_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    key = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    out_dir = tmp_path / "out"
    write_exl3_layers_bundle([_layer(key, k=4)], out_dir)
    calls: list[str] = []

    def fake_convert_one(*args, **kwargs):  # noqa: ARG001
        calls.append(str(args[3]))
        layer = _layer(key, k=kwargs["quant_bits"])
        return DirectLayerResult(
            module_key=key,
            search_backend="cpu",
            scale_mode="identity",
            layer=layer,
            activations=np.zeros((1, layer.in_features), dtype=np.float32),
            source_output=np.zeros((1, layer.out_features), dtype=np.float32),
            converted_output=np.zeros((1, layer.out_features), dtype=np.float32),
            stats={"output_rel_rms": 0.0, "public_rel_rms": 0.0},
        )

    monkeypatch.setattr(convert_driver, "_convert_one", fake_convert_one)

    result = convert_driver.convert_module_set(
        tmp_path / "source",
        tmp_path / "oracle",
        [key],
        quantizer="direct",
        out_dir=out_dir,
        search_backend="cpu",
        scale_mode="identity",
        resume=True,
        incremental_output=True,
        bit_plan={key: 5},
    )

    assert calls == [key]
    assert result.completed[0].get("resumed") is not True
    assert result.completed[0]["k"] == 5
    assert convert_driver.load_exl3_layer(str(out_dir), key).k == 5


def test_module_set_batches_fast_ldlq_siblings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    keys = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    calls: list[list[str]] = []

    def fake_group(*args, **kwargs):  # noqa: ARG001
        group_keys = list(args[2])
        calls.append(group_keys)
        out = []
        for key in group_keys:
            layer = _layer(key)
            out.append(
                DirectLayerResult(
                    module_key=key,
                    search_backend="metal",
                    scale_mode="oracle_safe",
                    layer=layer,
                    activations=np.zeros((1, layer.in_features), dtype=np.float32),
                    source_output=np.empty((0, 0), dtype=np.float32),
                    converted_output=np.empty((0, 0), dtype=np.float32),
                    stats={
                        "output_rel_rms": float("nan"),
                        "public_rel_rms": float("nan"),
                        "batched_group_size": float(len(group_keys)),
                    },
                )
            )
        return out

    monkeypatch.setattr(convert_driver, "ldlq_quantize_group", fake_group)

    result = convert_driver.convert_module_set(
        tmp_path / "source",
        tmp_path / "oracle",
        keys,
        quantizer="ldlq",
        search_backend="metal",
        compare_oracle=False,
        fast_metrics=True,
    )

    assert calls == [keys]
    assert [item["module"] for item in result.completed] == keys
    assert all(item["stats"]["batched_group_size"] == 3.0 for item in result.completed)


_EXL3_DIR = Path(
    os.environ.get("PONYEXL3_MODELS_DIR", Path.home() / "llm/models/exl3")
)
MINICPM_SOURCE = Path(
    os.environ.get("PONYEXL3_MODEL_MINICPM5_SOURCE", Path.home() / "llm/models/MiniCPM5-1B")
)
MINICPM_ORACLE = Path(
    os.environ.get(
        "PONYEXL3_MODEL_MINICPM5",
        _EXL3_DIR / "MiniCPM5-1B-exl3-4.00bpw",
    )
)


@pytest.mark.skipif(
    not (
        (MINICPM_SOURCE / "model.safetensors.index.json").is_file()
        and (MINICPM_ORACLE / "quantization_config.json").is_file()
    ),
    reason="local MiniCPM5 source/oracle checkpoints not present",
)
def test_minicpm5_model_discovery_matches_oracle():
    layer0 = layer_module_keys(MINICPM_ORACLE, 0)
    assert layer0 == [
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    ]

    all_modules = model_module_keys(MINICPM_ORACLE)
    assert "lm_head" in all_modules
    assert len(all_modules) == 169
    plain = plain_tensor_keys(MINICPM_ORACLE)
    assert "model.embed_tokens.weight" in plain
    assert "model.norm.weight" in plain
    assert len(plain) == 50

    supported, skipped = supported_model_module_keys(MINICPM_SOURCE, MINICPM_ORACLE)
    assert supported == all_modules
    assert skipped == []
