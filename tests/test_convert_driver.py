"""M4 module-set driver and multi-layer bundle gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ponyexl3.convert.direct import write_exl3_layers_bundle
from ponyexl3.convert.driver import convert_module_set, layer_module_keys
from ponyexl3.ref.layer import EXL3Layer


def _layer(key: str, *, out_tiles: int = 8) -> EXL3Layer:
    return EXL3Layer(
        key=key,
        in_features=128,
        out_features=out_tiles * 16,
        k=4,
        trellis=np.zeros((8, out_tiles, 64), dtype=np.uint16),
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
