"""Clone-and-replace EXL3 layer splice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ponyexl3.convert.direct import write_exl3_layers_bundle
from ponyexl3.convert.heads import is_output_head
from ponyexl3.convert.replace import replace_layers_into_checkpoint
from ponyexl3.ref.codebook import MUL1_MULT
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.loader import load_exl3_layer


def _layer(key: str, *, k: int, mul1: bool = True) -> EXL3Layer:
    in_tiles, out_tiles = 8, 8
    return EXL3Layer(
        key=key,
        in_features=in_tiles * 16,
        out_features=out_tiles * 16,
        k=k,
        trellis=np.arange(in_tiles * out_tiles * (256 * k // 16), dtype=np.uint16).reshape(
            in_tiles, out_tiles, 256 * k // 16
        ),
        suh=np.ones(in_tiles * 16, dtype=np.float16),
        svh=np.ones(out_tiles * 16, dtype=np.float16),
        mcg=False,
        mul1=mul1,
    )


def test_is_output_head_names():
    assert is_output_head("head")
    assert is_output_head("lm_head")
    assert is_output_head("model.lm_head")
    assert not is_output_head("layers.0.attn.q_head_norm")
    assert not is_output_head("hc_head")


def test_replace_layers_into_checkpoint_swaps_head_k(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "quantization_config": {"quant_method": "exl3", "bits": 2.52, "head_bits": 6},
            }
        ),
        encoding="utf-8",
    )
    write_exl3_layers_bundle(
        [_layer("layers.0.attn.wq_a", k=3), _layer("head", k=6)],
        src,
        asset_dir=assets,
    )

    replacement = _layer("head", k=4)
    replacement.trellis.fill(7)
    summary = replace_layers_into_checkpoint(
        src,
        dst,
        [replacement],
        bits=2.50,
        head_bits=4,
    )

    assert summary["replaced"] == ["head"]
    assert summary["head_bits"] == 4
    loaded = load_exl3_layer(str(dst), "head")
    assert loaded.k == 4
    assert loaded.trellis.shape[-1] == 64
    assert np.all(loaded.trellis == 7)
    body = load_exl3_layer(str(dst), "layers.0.attn.wq_a")
    assert body.k == 3

    qcfg = json.loads((dst / "quantization_config.json").read_text(encoding="utf-8"))
    head = qcfg["tensor_storage"]["head"]
    assert head["bits_per_weight"] == 4
    assert int(head["mul1_multiplier"]) == int(MUL1_MULT)
    assert head["stored_tensors"]["head.trellis"]["shape"] == [8, 8, 64]
    assert qcfg["bits"] == 2.5
    assert qcfg["head_bits"] == 4

    cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    assert cfg["quantization_config"]["bits"] == 2.5
    assert cfg["quantization_config"]["head_bits"] == 4

    index = json.loads((dst / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"]["head.trellis"].startswith("ponyexl3-layer-")
    assert index["weight_map"]["layers.0.attn.wq_a.trellis"] == "model.safetensors"
    from safetensors import safe_open

    with safe_open(str(dst / "model.safetensors"), framework="numpy") as handle:
        leftover = [key for key in handle.keys() if key.startswith("head.")]
    assert leftover == []
    assert set(summary["stripped_from_shards"]["model.safetensors"]) == {
        "head.suh",
        "head.svh",
        "head.trellis",
    }
