# PonyExl3

EXL3 quantized LLM inference on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm).

PonyExl3 ports the [ExLlamaV3 EXL3](https://github.com/turboderp-org/exllamav3) format to Metal: weights stay in low-bit trellis form and are decoded on-the-fly inside fused GEMV/GEMM kernels instead of being materialized as full fp16 matrices. A CPU reference implementation (`ponyexl3.ref`) mirrors the CUDA codec for bit-exact validation.

**Platform:** macOS with Apple Silicon (Metal required for inference).  
**Status:** Alpha — inference-focused; HF→EXL3 conversion is experimental (`ponyexl3/convert/`).

---

## Features

- **Exact EXL3 decode path** — fused Metal GEMV at batch size 1; lowest memory footprint
- **Full model loader** — Qwen3.5 / Qwen3.6 dense and MoE architectures via mlx-lm skeleton + `EXL3Linear` swap
- **Speculative decoding** — MTP, DFlash, EAGLE-3, and draft-free n-gram lookup (all verify-gated for token-identical greedy output)
- **Dual implementation + pytest parity** — every MLX primitive has a CPU `ref/` twin
- **Cross-platform reference** — CUDA-exported `.npz` fixtures for logits and per-layer bisection (`ponyexl3/reference/`)

---

## Requirements

- macOS on Apple Silicon (M-series)
- Python ≥ 3.10
- ~15–20 GB RAM for ship-model checkpoints (model-dependent)

---

## Install

```bash
git clone https://github.com/beamster/ponyexl3.git   # when published
cd ponyexl3

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

No Pony monorepo or `PYTHONPATH` setup is needed. Everything imports as `ponyexl3`.

---

## Quick start

Point at a local EXL3 checkpoint directory (safetensors + `quantization_config.json` with `quant_method: exl3`):

```bash
ponyexl3-generate /path/to/Qwen3.6-27B-exl3-4.15bpw \
  -p "Why is the sky blue?" -n 256
```

Greedy decoding (`--temp 0`, default) is the validated path. Use `--raw` to skip the chat template.

### Engines

| Engine | Memory | Speed | Exactness |
|--------|--------|-------|-----------|
| `exl3` (default) | Lowest | Fastest decode | Bit-exact vs trellis |
| `fold16` | Higher | Fast | Exact fp16 fold of public weights |
| `w8a16` / `w4a16` | Medium | Faster matmul | **Lossy** — run `ponyexl3-compare-engines` first |

```bash
ponyexl3-generate /path/to/model --engine fold16 -p "Hello" -n 128
ponyexl3-compare-engines /path/to/model --engines exl3 fold16 w8a16 -n 64
```

### Speculative decoding

All speculative modes verify drafts against the main model — greedy output stays token-identical.

```bash
# MTP draft head (auto-discovers weights in model dir)
ponyexl3-generate /path/to/model --mtp auto --draft 3 -p "..." -n 256

# DFlash block drafter (bf16 or EXL3; default draft width 7)
ponyexl3-generate /path/to/model \
  --dflash /path/to/Qwen3.6-27B-DFlash-bf16 -p "..." -n 256

# Draft-free n-gram lookup (no extra weights; greedy only)
ponyexl3-generate /path/to/model --lookup -p "..." -n 256
```

---

## CLI tools

Installed by `pip install -e .`:

| Command | Purpose |
|---------|---------|
| `ponyexl3-generate` | End-to-end text generation |
| `ponyexl3-compare-layer` | Per-layer correctness ladder (probe → tile → slice → forward) |
| `ponyexl3-compare-engines` | End-to-end engine agreement + logit drift report |

Layer comparison — start with `--probe` on large models (full forward can take tens of minutes per layer on 27B-class checkpoints):

```bash
ponyexl3-compare-layer /path/to/model MODULE --probe
ponyexl3-compare-layer /path/to/model MODULE --mode tile
ponyexl3-compare-layer /path/to/model --list
```

Synthetic fixture (no checkpoint download):

```bash
python -m ponyexl3.cli.generate_synthetic_layer
```

---

## Supported models

| Model | `model_type` | Notes |
|-------|--------------|-------|
| Qwen3.6-27B dense | `qwen3_5` | Primary ship target (~15 GB RAM) |
| Qwen3.6-35B-A3B MoE | `qwen3_5_moe` | `EXL3MoEBlock` / fused expert kernels (~19.5 GB RAM) |
| Qwen3.5-2B | `qwen3_5` | Dev / fast iteration |

Checkpoints must be EXL3-quantized (ExLlamaV3 or compatible converter output) with sidecar `quantization_config.json`.

---

## Testing

```bash
pytest tests/ -q
```

**169 tests** run without any model on disk (synthetic layers + CPU/MLX parity). Optional integration tests are skipped unless env vars are set:

```bash
export PONYEXL3_MODEL_DIR=/path/to/checkpoint
export PONYEXL3_MODEL_27B=/path/to/27b-exl3
export PONYEXL3_MODEL_2B=/path/to/2b-exl3
export PONYEXL3_REFERENCE_NPZ=/path/to/reference.npz   # see below

pytest tests/ -q
```

### Reference parity (CUDA ↔ MLX)

The reference *scripts* ship in `ponyexl3/reference/`; the binary reference
bundle (logits/trace/linear-io/moe `.npz`) is generated on a CUDA host running
exllamav3 and is **not** bundled in this repo (it is model-derived and
host-specific). Produce one with the export tools, then replay and diff on Mac:

```bash
PONYEXL3_MODEL_DIR=/path/to/Qwen3.6-35B-A3B-exl3-4.00bpw \
  python ponyexl3/reference/compare_reference.py /path/to/reference.npz
```

The cross-platform forward has been audited — see
[`docs/drifts_investigation.md`](docs/drifts_investigation.md) (verdict: PASS,
within the fp16 cross-platform floor). The full CUDA export + replay workflow,
including the scale-aware `compare_trace.py` analysis, is in
[`ponyexl3/reference/README.md`](ponyexl3/reference/README.md).

---

## Project layout

```
ponyexl3/
  ref/              CPU/numpy golden codec (trellis, codebook, Hadamard, loader)
  mlx/              MLX runtime — Metal kernels, model loader, generation loop
  reference/        Cross-platform parity export/compare scripts (bundle not shipped)
  convert/          Experimental HF → EXL3 conversion (not v0.1)
  cli/              Installed command-line entry points
tools/
  bench/            Throughput and kernel benchmarks (env-var model paths)
  dev/              Diagnostics, A/B probes, parity repair
tests/              Pytest parity suite
docs/               Dev log and conversion design notes
```

---

## How it works

EXL3 stores each linear layer as independent **16×16 tiles** of K-bit trellis indices plus per-channel sign scales (`suh`, `svh`) and optional Walsh-Hadamard transforms (block size 128). Weights are recovered procedurally via **3-instruction codebooks** — the same inline decode used on CUDA, not a stored LUT.

At inference time PonyExl3 routes by batch size and layer size:

- **Decode (M=1):** fused Metal GEMV — decode trellis bits and dot in one kernel launch
- **Prefill (M>1):** decode-once weight cache + compiled `matmul`, or fused GEMM for huge layers
- **MoE:** fused gate/up/down kernels over selected experts instead of hundreds of tiny matmuls

A custom generation loop (`ponyexl3.mlx.generate`) runs prefill through the text model only and applies `lm_head` to the last position per step — critical for 27B-class models where the head alone is multi-GB.

---

## Environment variables

| Variable | Used by |
|----------|---------|
| `PONYEXL3_MODEL_DIR` | `compare_reference`, `test_reference_parity` |
| `PONYEXL3_MODEL_27B` | 27B integration tests, `tools/bench/*` |
| `PONYEXL3_MODEL_2B` | 2B GEMV layer tests |
| `PONYEXL3_MTP_DIR` | MTP benchmarks |
| `PONYEXL3_REFERENCE_NPZ` | Full-model reference parity test |
| `MODEL` / `MTP` | Legacy aliases for bench scripts |

Runtime tuning knobs (`EXL3_FUSE_POST`, `EXL3_MLP_MONO`, etc.) are documented in [`docs/current_stage.md`](docs/current_stage.md).

---

## Development

```bash
# Metal codebook POC
python -m ponyexl3.mlx.metal_codebook_poc

# Bench scripts (require checkpoint env vars)
export PONYEXL3_MODEL_27B=/path/to/27b-exl3
python tools/bench/benchmark_forward.py "$PONYEXL3_MODEL_27B"
```

Internal bench/dev scripts are described in [`tools/README.md`](tools/README.md).

---

## Attribution

- **EXL3 codec and algorithm** — [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp)
- **Model weights** — Qwen series (Alibaba Cloud); follow each checkpoint's license
- **Runtime** — [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm)

---

## License

Apache-2.0 (see `LICENSE` when published).
