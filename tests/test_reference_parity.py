"""Full-model logits parity vs native exllamav3 reference .npz (optional).

Two gates (see docs/drifts_investigation.md round-2 addendum):

- tolerance: asserts the measured cross-platform numerics band. CUDA and
  MLX are independently-scheduled exact engines; their drift is fp16
  kernel-rounding chaos (amplified by MoE routing near-ties), bounded and
  characterized on 2026-06-13 at max|d| 0.83 / rel rms 4.6%. A regression
  past 2x that band means a real bug, not numerics.
- bit-exact: xfail aspiration. Unachievable unless both engines adopt a
  common reduction-order spec; kept so a surprise pass gets noticed.

Top-1 argmax is reported but NOT asserted: the fixed-seed random-token
probe's top-2 gap is one fp16 ulp (4.5898 vs 4.5742) — a coin flip.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.ponyexl3

_REF = os.environ.get("PONYEXL3_REFERENCE_NPZ")
_MODEL = os.environ.get("PONYEXL3_MODEL_DIR")

_requires_bundle = pytest.mark.skipif(
    not _REF or not _MODEL or not Path(_REF).is_file() or not Path(_MODEL).is_dir(),
    reason="set PONYEXL3_REFERENCE_NPZ and PONYEXL3_MODEL_DIR",
)


@lru_cache(maxsize=1)
def _ref_and_candidate():
    mlx = pytest.importorskip("mlx.core")
    if not mlx.metal.is_available():
        pytest.skip("Metal required")
    if _REF is None or _MODEL is None:
        pytest.skip("set PONYEXL3_REFERENCE_NPZ and PONYEXL3_MODEL_DIR")

    from ponyexl3.mlx.model import load_model
    from ponyexl3.reference.compare_reference import compare, forward_logits

    ref = np.load(_REF, allow_pickle=True)
    input_ids = ref["input_ids"]
    ref_logits = ref["logits"].astype(np.float32)
    if ref_logits.ndim == 2:
        ref_logits = ref_logits[-1]

    model, _ = load_model(_MODEL, engine="exl3", warm=True, verbose=False)
    cand_logits = forward_logits(model, input_ids)
    return compare(ref_logits, cand_logits), ref_logits


@_requires_bundle
def test_native_reference_within_numerics_band():
    stats, ref_logits = _ref_and_candidate()
    rms_ref = float(np.sqrt((ref_logits.astype(np.float64) ** 2).mean()))
    rel = stats["rms"] / rms_ref
    print(
        f"max|d|={stats['max_abs']:.4g} rel_rms={rel:.4g} "
        f"top1 ref={stats['top1_ref']} cand={stats['top1_cand']}"
    )
    # 2x the characterized 2026-06-13 band (max|d| 0.83, rel 4.6%)
    assert stats["max_abs"] < 1.7, (
        f"logit drift {stats['max_abs']:.3g} exceeds 2x the characterized "
        "cross-platform numerics band — investigate a real regression"
    )
    assert rel < 0.10, f"rel rms {rel:.3g} exceeds 2x the characterized band"


@_requires_bundle
@pytest.mark.xfail(
    reason="bit-exact CUDA<->Metal logits requires a common reduction-order "
    "spec in both engines — aspiration, not a gate",
    strict=False,
)
def test_native_reference_bit_exact():
    stats, _ = _ref_and_candidate()
    assert stats["bit_exact"], (
        f"not bit-exact vs native reference: max|d|={stats['max_abs']:.6g} "
        f"rms={stats['rms']:.6g} mismatched fp32 words={stats['n_mismatch_bits']}"
    )
