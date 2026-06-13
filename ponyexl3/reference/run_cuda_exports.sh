#!/usr/bin/env bash
# Generate the standard reference bundle for MLX parity work.
#
# Usage (on CUDA host, from repo root or this directory):
#   ./ponyexl3/reference/run_cuda_exports.sh \
#     /path/to/Qwen3.6-35B-A3B-exl3-4.00bpw \
#     ./ponyexl3/reference \
#     qwen3.6-35B-A3B-exl3-4.00bpw_win
#
# Requires: python 3.10+, exllamav3 (set EXLLAMAV3_ROOT), torch+cuda, numpy

set -euo pipefail

MODEL_DIR="${1:?model dir}"
OUT_DIR="${2:-./exl3/reference}"
STEM="${3:-reference}"

mkdir -p "$OUT_DIR"
LOGITS="$OUT_DIR/${STEM}.npz"
TRACE="$OUT_DIR/${STEM}_trace.npz"
MOE="$OUT_DIR/${STEM}_moe.npz"

PY="${PYTHON:-python}"

echo "==> module inventory"
"$PY" exl3/reference/list_modules.py "$MODEL_DIR" --exl3-only

echo "==> logits (end-to-end)"
"$PY" exl3/reference/export_reference.py \
  -m "$MODEL_DIR" -o "$LOGITS" -s 512 -r 1 --seed 0

echo "==> forward trace (last token, all blocks)"
"$PY" exl3/reference/export_trace.py \
  -m "$MODEL_DIR" -o "$TRACE" --from-npz "$LOGITS" --row-slice last:1

echo "==> MoE routing (last token)"
"$PY" exl3/reference/export_moe_routing.py \
  -m "$MODEL_DIR" -o "$MOE" --from-npz "$LOGITS" --row-slice last:1

echo "==> sample EXL3 linear I/O (edit MODULES below as needed)"
MODULES=(
  "model.language_model.layers.0.mlp.shared_expert.gate_proj"
  "model.language_model.layers.0.mlp.experts.0.gate_proj"
  "model.language_model.layers.3.self_attn.q_proj"
  "model.language_model.layers.3.linear_attn.in_proj_qkv"
  "lm_head"
)
LINEAR_IO="$OUT_DIR/${STEM}_linear_io.npz"
"$PY" exl3/reference/export_linear_io.py \
  -m "$MODEL_DIR" -o "$LINEAR_IO" \
  --from-npz "$LOGITS" --row-slice last:1 --reconstruct \
  "${MODULES[@]}"

echo ""
echo "Done. Copy to Mac:"
echo "  $LOGITS"
echo "  $TRACE"
echo "  $MOE"
echo "  $LINEAR_IO"
