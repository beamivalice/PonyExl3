"""CLI argument handling and helper tests (no checkpoints)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ponyexl3.cli._generate_common import build_prefill_prompt_ids, resolve_prompt_file


ROOT = Path(__file__).resolve().parents[1]


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [100 + (i % 50) for i in range(max(1, len(text) // 4))]


def test_collect_eos_gemma4():
    from ponyexl3.cli._generate_common import collect_eos_token_ids

    cfg = {
        "eos_token_id": [1, 106],
        "text_config": {"eos_token_id": 1},
    }
    assert collect_eos_token_ids(cfg) == (1, 106)


def test_collect_eos_dense():
    from ponyexl3.cli._generate_common import collect_eos_token_ids

    assert collect_eos_token_ids({"eos_token_id": 151643}) == (151643,)


    with pytest.raises(FileNotFoundError, match="prompt file not found"):
        resolve_prompt_file("/no/such/prompt.txt")


def test_generate_bench_invalid_prefill_sizes():
    proc = subprocess.run(
        [sys.executable, "-m", "ponyexl3.cli.generate_bench", "/tmp", "--prefill-sizes", "abc"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "invalid --prefill-sizes" in proc.stderr or "invalid --prefill-sizes" in proc.stdout


def test_generate_bench_missing_prompt_file():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ponyexl3.cli.generate_bench",
            "/tmp",
            "--prompt-file",
            "/no/such/file.txt",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "prompt file not found" in proc.stderr


def test_compare_layer_missing_model():
    proc = subprocess.run(
        [sys.executable, "-m", "ponyexl3.cli.compare_layer", "/nonexistent", "--list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "quantization_config.json" in proc.stderr


def test_compare_layer_requires_module_key():
    proc = subprocess.run(
        [sys.executable, "-m", "ponyexl3.cli.compare_layer", str(ROOT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "module_key required" in proc.stderr


def test_synthetic_layer_rejects_bad_dims():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ponyexl3.cli.generate_synthetic_layer",
            "--in-features",
            "127",
            "--out",
            "/tmp/ponyexl3_test_synthetic.npz",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "multiples of 16" in proc.stderr


def test_build_prefill_repeats_to_target():
    ids = build_prefill_prompt_ids("abcd " * 10, 500, _FakeTokenizer(), raw=True)
    assert len(ids) == 500


def test_build_prefill_rejects_empty_file():
    with pytest.raises(ValueError, match="empty"):
        build_prefill_prompt_ids("   \n", 100, _FakeTokenizer(), raw=True)
