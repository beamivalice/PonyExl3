"""Gemma4 architecture mapping, MoE key patterns, and fusion parity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from ponyexl3.mlx.exl3_fused import FusedEXL3Sibling
from ponyexl3.mlx.model import (
    _ARCHITECTURES,
    _gemma4_expert_match,
    _is_routed_expert_key,
    _qwen_expert_match,
    load_model,
)

pytestmark = [
    pytest.mark.ponyexl3,
    pytest.mark.skipif(not mlx.metal.is_available(), reason="Metal required"),
]

GEMMA4_MODEL = os.environ.get(
    "PONYEXL3_MODEL_GEMMA4",
    "/Users/beam/llm/models/gemma-4-26B-A4B-exl3-4.10bpw",
)
_HAS_GEMMA4 = Path(GEMMA4_MODEL).joinpath("config.json").is_file()
_skip_no_gemma = pytest.mark.skipif(
    not _HAS_GEMMA4,
    reason="Gemma4 EXL3 checkpoint not available",
)


def _last_logits(model, prompt_ids: list[int]) -> np.ndarray:
    lm = model.language_model
    cache = lm.make_cache()
    toks = mlx.array([prompt_ids])
    h = lm.model(toks, cache=cache)
    logits = lm.lm_head(h[:, -1:, :])
    mlx.eval(logits)
    return np.array(logits).astype(np.float32)


def test_gemma4_architecture_mapping():
    assert _ARCHITECTURES["gemma4"] == "gemma4"
    assert _ARCHITECTURES["gemma4_text"] == "gemma4_text"


def test_expert_key_patterns():
    qwen = "model.language_model.layers.3.mlp.experts.7.gate_proj"
    gemma = "model.language_model.layers.3.experts.7.gate_proj"
    dense = "model.language_model.layers.3.mlp.gate_proj"

    assert _qwen_expert_match(qwen) is not None
    assert _gemma4_expert_match(gemma) is not None
    assert _qwen_expert_match(gemma) is None
    assert _gemma4_expert_match(qwen) is None
    assert _is_routed_expert_key(qwen)
    assert _is_routed_expert_key(gemma)
    assert not _is_routed_expert_key(dense)


@_skip_no_gemma
def test_load_gemma4_checkpoint():
    cfg = json.loads(Path(GEMMA4_MODEL).joinpath("config.json").read_text())
    assert cfg["model_type"] == "gemma4"
    assert cfg["text_config"]["enable_moe_block"] is True

    model, config = load_model(GEMMA4_MODEL, engine="exl3", warm=False, verbose=False)
    assert config["model_type"] == "gemma4"
    layer0 = model.layers[0]
    assert layer0.enable_moe
    from ponyexl3.mlx.exl3_moe import EXL3Gemma4MoEBlock

    assert isinstance(layer0.router, EXL3Gemma4MoEBlock)
    from ponyexl3.mlx.exl3_moe import _Gemma4ExpertsShim

    assert isinstance(layer0.experts, _Gemma4ExpertsShim)
    assert layer0.experts._block is layer0.router  # pyright: ignore[reportPrivateUsage]
    assert layer0.router.switch_mlp.num_experts == cfg["text_config"]["num_experts"]

    from ponyexl3.mlx.exl3_linear import EXL3Linear

    assert isinstance(layer0.mlp.gate_proj, EXL3Linear)


@_skip_no_gemma
def test_gemma4_moe_block_parity():
    """EXL3Gemma4MoEBlock logits match legacy Router+Experts wiring."""
    prompt = list(range(3, 48))

    os.environ["PONYEXL3_GEMMA4_LEGACY_MOE"] = "1"
    os.environ["EXL3_FUSE_MIN_MB"] = "999999"
    ref = _last_logits(
        load_model(GEMMA4_MODEL, engine="exl3", warm=True, verbose=False)[0],
        prompt,
    )

    os.environ.pop("PONYEXL3_GEMMA4_LEGACY_MOE", None)
    os.environ.pop("EXL3_FUSE_MIN_MB", None)
    got = _last_logits(
        load_model(GEMMA4_MODEL, engine="exl3", warm=True, verbose=False)[0],
        prompt,
    )

    scale = float(np.abs(ref).max()) + 1e-9
    max_abs = float(np.abs(got - ref).max())
    rel_rms = float(np.sqrt(np.mean((got - ref) ** 2)) / scale)
    assert max_abs / scale < 5e-3, f"max rel err {max_abs / scale:.2e}"
    assert rel_rms < 2e-3, f"rel rms {rel_rms:.2e}"


@_skip_no_gemma
def test_gemma4_sibling_fusion_parity():
    """Fused siblings must match unfused EXL3Linear logits (last prompt token)."""
    prompt = list(range(2, 42))  # 40 tokens — spans sliding + one full-attn layer

    os.environ["EXL3_FUSE_MIN_MB"] = "999999"
    model_u, _ = load_model(GEMMA4_MODEL, engine="exl3", warm=True, verbose=False)
    ref = _last_logits(model_u, prompt)

    os.environ.pop("EXL3_FUSE_MIN_MB", None)
    model_f, _ = load_model(GEMMA4_MODEL, engine="exl3", warm=True, verbose=False)
    got = _last_logits(model_f, prompt)

    scale = float(np.abs(ref).max()) + 1e-9
    max_abs = float(np.abs(got - ref).max())
    rel_rms = float(np.sqrt(np.mean((got - ref) ** 2)) / scale)
    assert max_abs / scale < 5e-3, f"max rel err {max_abs / scale:.2e}"
    assert rel_rms < 2e-3, f"rel rms {rel_rms:.2e}"

    layer0 = model_f.layers[0]
    assert isinstance(layer0.self_attn.q_proj, FusedEXL3Sibling)
    layer5 = model_f.layers[5]
    assert layer5.layer_type == "full_attention"
    assert isinstance(layer5.self_attn.q_proj, FusedEXL3Sibling)
    assert isinstance(layer5.self_attn.k_proj, FusedEXL3Sibling)
    # shared MLP gate+up stays unfused at the 40 MB Gemma4 threshold
    assert not isinstance(layer0.mlp.gate_proj, FusedEXL3Sibling)
