"""Source-only quantization plan discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

from ponyexl3.convert.discovery import (
    discover_exl3_module_keys,
    generate_quantization_config,
    is_plan_only_checkpoint,
    write_quantization_plan,
)
from ponyexl3.convert.fixtures import load_oracle_linear
from ponyexl3.ref.loader import list_exl3_layers
from ponyexl3.ref.codebook import CodebookMode


def _write_linear_source(
    model_dir: Path,
    module_key: str,
    *,
    in_features: int = 128,
    out_features: int = 128,
) -> None:
    shard = model_dir / "model.safetensors"
    tensor_key = f"{module_key}.weight"
    weight = np.ones((out_features, in_features), dtype=np.float16)
    tensors: dict[str, np.ndarray] = {}
    if shard.is_file():
        from safetensors import safe_open

        with safe_open(str(shard), framework="np") as st:
            tensors = {key: np.array(st.get_tensor(key)) for key in st.keys()}
    tensors[tensor_key] = weight
    save_file(tensors, str(shard))
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"] if index_path.is_file() else {}
    weight_map[tensor_key] = shard.name
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": int(shard.stat().st_size)},
                "weight_map": weight_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_discover_exl3_module_keys_excludes_embeddings(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_linear_source(source, "model.embed_tokens", in_features=256, out_features=128)
    _write_linear_source(source, "model.layers.0.self_attn.q_proj")

    keys = discover_exl3_module_keys(source)
    assert keys == ["model.layers.0.self_attn.q_proj"]


def test_write_quantization_plan_is_plan_only(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_linear_source(source, "model.layers.0.self_attn.q_proj")
    _write_linear_source(
        source,
        "model.embed_tokens",
        in_features=256,
        out_features=128,
    )
    (source / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")

    plan_dir = tmp_path / "plan"
    summary = write_quantization_plan(source, plan_dir, bits=4.0, head_bits=6)
    assert summary["exl3_modules"] == 1
    assert summary["plain_tensors"] == 1
    assert (plan_dir / "quantization_config.json").is_file()
    assert (plan_dir / "config.json").is_file()
    assert is_plan_only_checkpoint(plan_dir)

    qcfg = json.loads((plan_dir / "quantization_config.json").read_text(encoding="utf-8"))
    assert qcfg["quant_method"] == "exl3"
    assert qcfg["tensor_storage"]["model.layers.0.self_attn.q_proj"]["quant_format"] == "exl3"
    assert "model.embed_tokens" in qcfg["tensor_storage"]
    assert qcfg["tensor_storage"]["model.embed_tokens"].get("quant_format") != "exl3"


def test_cli_auto_source_plan_uses_requested_four_bits(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_linear_source(source, "model.layers.0.self_attn.q_proj")
    work_dir = tmp_path / "work"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ponyexl3.cli.convert",
            "--in-dir",
            str(source),
            "--work-dir",
            str(work_dir),
            "--direct-layer",
            "--model-modules",
            "--allocation-dry-run",
            "--bits",
            "4.00",
            "--head-bits",
            "4",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["bit_plan"] == {"model.layers.0.self_attn.q_proj": 4}
    plan = work_dir / "source_quant_plan" / "quantization_config.json"
    qcfg = json.loads(plan.read_text(encoding="utf-8"))
    assert qcfg["bits"] == 4.0
    assert qcfg["head_bits"] == 4
    assert qcfg["tensor_storage"]["model.layers.0.self_attn.q_proj"]["bits_per_weight"] == 4.0


def test_load_oracle_linear_from_plan_only_stub(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_linear_source(source, "model.layers.0.self_attn.q_proj")
    plan_dir = tmp_path / "plan"
    write_quantization_plan(source, plan_dir)

    oracle = load_oracle_linear(plan_dir, "model.layers.0.self_attn.q_proj")
    assert oracle.layer.k == 4
    assert oracle.layer.in_features == 128
    assert oracle.layer.out_features == 128
    assert oracle.cb == CodebookMode.MCG


@pytest.mark.skipif(
    not (Path(os.path.expanduser("~/llm/models/MiniCPM5-1B")) / "model.safetensors.index.json").is_file(),
    reason="MiniCPM5 source checkpoint not available",
)
def test_minicpm5_plan_matches_oracle_module_set():
    source = Path(os.path.expanduser("~/llm/models/MiniCPM5-1B"))
    oracle = Path(os.path.expanduser("~/llm/models/exl3/MiniCPM5-1B-exl3-4.00bpw"))
    cfg = generate_quantization_config(source)
    planned = sorted(
        key
        for key, info in cfg["tensor_storage"].items()
        if info.get("quant_format") == "exl3"
    )
    oracle_keys = sorted(info["key"] for info in list_exl3_layers(str(oracle)))
    assert planned == oracle_keys
    assert int(cfg["tensor_storage"]["lm_head"]["bits_per_weight"]) == 6
