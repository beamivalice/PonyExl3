"""Output-head module names across HF / EXL3 layouts.

Qwen/Llama store the vocab projection as ``lm_head``. DeepSeek-V4 stores the
same linear as the bare key ``head``. Allocation, calibration, and measured
``--head-bits`` overrides must treat both as the output head.
"""

from __future__ import annotations


def is_output_head(key: str) -> bool:
    """True for the model's vocab projection (not attention/indexer heads)."""

    return key == "lm_head" or key == "head" or key.endswith(".lm_head")
