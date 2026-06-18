# Changelog

All notable changes to PonyExl3 are documented here.

## [0.2.1] — 2026-06-19

### Converter (`ponyexl3-convert`)

- **Source-only planning** — `--init-quant-config` writes `quantization_config.json` (+ HF assets) from BF16 weights alone; no turboderp oracle required
- Plan-only dirs work as `--oracle-dir` with `--scale-mode computed` (shape stubs when trellis weights are absent)
- M5a bit budget: `--bits`, `--head-bits`, `--use-bit-allocation`, `--layer-bits REGEX:K`, `--allocation-dry-run`
- Apple Silicon-first two-step convert workflow documented in README

### Housekeeping

- `.work/` gitignored (local converter scratch/logs)

### Tests

- `tests/test_convert_discovery.py` — source plan generation, plan-only oracle stubs, MiniCPM module-set gate

## [0.2.0] — 2026-06-18

### Inference

- MiniCPM5-1B EXL3 support (`model_type` `llama` / LlamaForCausalLM layout)
- ~152 tok/s greedy decode on M5 Max (8k prefill, 128 gen); ~0.9 GB resident

### Converter (`ponyexl3-convert`)

- HF → EXL3 conversion on Metal: trellis search, Hessian/LDLQ, regularization, calibration, allocation
- Full-model conversion validated on MiniCPM5-1B (~7 min direct path on M5 Max)
- MiniCPM5 4.00bpw KLD vs bf16 matches [turboderp oracle](https://huggingface.co/turboderp/MiniCPM5-1B-exl3/tree/4.00bpw) (KLD 0.0422 vs 0.0428; p95 0.136 vs 0.145)
- Module/layer/model scope; manifest output (`ponyexl3_convert_manifest.json`)
- Metal search speedups: vectorized trellis packing, oracle-metrics fast path

### Tests

- Converter gate suite (`tests/test_convert*.py`, 59+ tests)
- MiniCPM5 load integration (`tests/test_minicpm5_model.py`)

## [0.1.7] — 2026-06-13

- Gemma4-26B-A4B EXL3 support (`model_type` `gemma4` / `gemma4_text`)
- Gemma4 MoE: ``EXL3Gemma4MoEBlock`` (compiled router + stacked experts; shared MLP separate)
- Gemma4 routed experts use GeGLU (``gelu_approx``) in MoE kernels, matching exllamav3
- Gemma4 sibling fusion: attn qkv + full-layer qk (40 MB threshold; MLP gate+up unfused)
- Fusion parity test vs unfused logits (`tests/test_gemma4_model.py`)
- Fix Gemma4 generation stop: merge top-level + `text_config` `eos_token_id` (honors `<turn|>`)

## [0.1.6] — 2026-06-13

- CLI validation: model dir, Metal, context limits, empty prompts, spec-flag warnings
- `ponyexl3-generate-bench`: text-repeat prefill padding, cache clear between rows
- Generation guards for `prefill_chunk`, `num_draft`, and `max_position_embeddings`
- CLI edge-case tests (`tests/test_cli.py`, `tests/test_generate_validation.py`)

## [0.1.5] — 2026-06-13

- `ponyexl3-generate-bench`: prefill sweep (1k–32k) with 128-token decode per row
- Shared generate CLI setup (`--mtp`, `--dflash`, `--eagle3`, `--lookup`, engines, etc.)
- Default prompt file: `README.md` (`--prompt-file` to override)

## [0.1.3] — 2026-06-13

- MTP speculative decoding: temperature-aware verify (Leviathan–Chen rejection sampling)
- README benchmark tables (M5 Max, M1 Max, RTX 4090 comparison)

## [0.1.2] — 2026-06-13

- Fix load transient memory / MLX buffer cache growth on 32 GB Macs
- Wired-memory cap via `PONYEXL3_MEM_LIMIT_GB` (92% of device recommended working set)
- M1 Max benchmark numbers in README

## [0.1.1] — 2026-06-13

- First 32 GB memory fix (load peak ~27.5 GB for 27B 4.15bpw)

## [0.1.0] — 2026-06-13

- Initial public release: EXL3 inference on Apple Silicon via MLX
- CPU `ref/` golden codec + MLX Metal runtime
- Model loader for Qwen3.5 / Qwen3.6 dense and MoE
- Speculative decoding: MTP, DFlash, EAGLE-3, n-gram lookup (verify-gated)
- CLIs: `ponyexl3-generate`, `ponyexl3-compare-layer`, `ponyexl3-compare-engines`
- Cross-platform reference export/compare scripts (`ponyexl3/reference/`)
