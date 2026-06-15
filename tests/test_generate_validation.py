"""Tests for generation parameter validation."""

from __future__ import annotations

import pytest

from ponyexl3.cli._generate_common import (
    build_prefill_prompt_ids,
    check_context_limit,
    validate_generate_cli_args,
)
from ponyexl3.mlx.generate import validate_generation_params


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [1 + (len(text) % 7)] * max(1, len(text) // 3)

    def apply_chat_template(self, messages, *, add_generation_prompt: bool = True):
        content = messages[0]["content"]
        return [9000] + self.encode(content)


def test_validate_generation_rejects_empty_prompt():
    with pytest.raises(ValueError, match="empty"):
        validate_generation_params([], max_tokens=8, prefill_chunk=512)


def test_validate_generation_rejects_bad_prefill_chunk():
    with pytest.raises(ValueError, match="prefill_chunk"):
        validate_generation_params([1, 2, 3], max_tokens=8, prefill_chunk=0)


def test_validate_generation_rejects_negative_max_tokens():
    with pytest.raises(ValueError, match="max_tokens"):
        validate_generation_params([1], max_tokens=-1, prefill_chunk=512)


def test_validate_generation_rejects_spec_with_zero_draft():
    with pytest.raises(ValueError, match="num_draft"):
        validate_generation_params(
            [1, 2],
            max_tokens=8,
            prefill_chunk=512,
            num_draft=0,
            using_spec=True,
        )


def test_validate_generation_context_limit():
    with pytest.raises(ValueError, match="max_position_embeddings"):
        validate_generation_params(
            [1] * 100,
            max_tokens=50,
            prefill_chunk=512,
            max_context=120,
        )


def test_build_prefill_concatenates_text_not_token_ids():
    tok = _FakeTokenizer()
    once = tok.encode("abc")
    twice_ids = build_prefill_prompt_ids("abc", len(once) * 2, tok, raw=True)
  # Re-encoding doubled text should match repeated-token-id approach for this toy tokenizer
    doubled = tok.encode("abc" + "abc")
    assert twice_ids == doubled[: len(twice_ids)]


def test_check_context_limit_exits():
    with pytest.raises(SystemExit, match="max_position_embeddings"):
        check_context_limit(1000, 100, {"max_position_embeddings": 1024})
