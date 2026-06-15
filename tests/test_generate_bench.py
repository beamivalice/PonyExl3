"""Unit tests for generate-bench helpers (no checkpoint / Metal)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ponyexl3.cli._generate_common import (
    PREFILL_BENCH_SIZES,
    build_prefill_prompt_ids,
    resolve_prompt_file,
)


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [i + 1 for i in range(max(1, len(text) // 8))]


def test_prefill_bench_default_sizes():
    assert PREFILL_BENCH_SIZES == (1024, 2048, 4096, 8192, 16384, 32768)


def test_build_prefill_prompt_ids_exact_length():
    tok = _FakeTokenizer()
    ids = build_prefill_prompt_ids("hello world " * 20, 1024, tok, raw=True)
    assert len(ids) == 1024


def test_resolve_prompt_file_defaults_to_repo_readme():
    root = Path(__file__).resolve().parents[1]
    readme = root / "README.md"
    if not readme.is_file():
        pytest.skip("README.md not present")
    assert resolve_prompt_file(None) == readme
