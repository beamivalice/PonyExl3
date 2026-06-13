# Cross-platform EXL3 reference (CUDA → MLX)

Native **exllamav3** on CUDA exports fixed-seed tensors; PonyExl3 MLX replays
and diffs them to audit cross-platform parity. The generated `.npz` bundle is
model-derived and host-specific — it is **not** committed to this repo; these
scripts produce it on demand.

## CUDA host setup

Run the export scripts on a machine with an NVIDIA GPU and an
[exllamav3](https://github.com/turboderp-org/exllamav3) checkout. PonyExl3
itself does not install on such hosts (no MLX); the CUDA scripts import
`_cuda_common` locally, so only exllamav3 needs to be importable:

```bash
# Linux / macOS host
export EXLLAMAV3_ROOT=/path/to/exllamav3   # optional; auto-detected if installed
export PYTHONPATH=$EXLLAMAV3_ROOT
```

```powershell
# Windows host
$env:EXLLAMAV3_ROOT = "C:\path\to\exllamav3"
$env:PYTHONPATH = "$env:EXLLAMAV3_ROOT"
```

## One-shot bundle

```bash
chmod +x ponyexl3/reference/run_cuda_exports.sh

./ponyexl3/reference/run_cuda_exports.sh \
  "/path/to/Qwen3.6-35B-A3B-exl3-4.00bpw" \
  ./ponyexl3/reference \
  qwen3.6-35B-A3B-exl3-4.00bpw_win
```

(Windows: `run_cuda_exports.ps1` takes the same three arguments.)

Produces four `.npz` files (~2–20 MiB total depending on trace row slice):

| File | Tool | What we need from CUDA |
|------|------|------------------------|
| `{stem}.npz` | `export_reference.py` | End-to-end logits + `input_ids` |
| `{stem}_trace.npz` | `export_trace.py` | Hidden state after every forward module |
| `{stem}_moe.npz` | `export_moe_routing.py` | Expert indices + routing weights per MoE layer |
| `{stem}_linear_io.npz` | `export_linear_io.py` | Real activations through selected EXL3 linears |

Copy all four to Mac `ponyexl3/reference/`.

## Individual tools (CUDA)

### 1. `export_reference.py` — end-to-end logits

Contract matches `exllamav3/eval/compare_q_exllamav3.py`:

- `torch.manual_seed(seed)` → `torch.randint` → `input_ids`
- `model.forward(ids, {"attn_mode": "flash_attn_nc"})`
- mask `output[..., vocab_size:] = -inf`
- export last `logit_rows` positions as float32

```bash
python ponyexl3/reference/export_reference.py \
  -m /path/to/Qwen3.6-35B-A3B-exl3-4.00bpw \
  -o ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win.npz \
  -s 512 -r 1 --seed 0
```

### 2. `list_modules.py` — discover keys

```bash
# fast: EXL3 keys from quantization_config.json only
python ponyexl3/reference/list_modules.py /path/to/model --exl3-only

# slow: also print exllamav3 forward module order (needs GPU load)
python ponyexl3/reference/list_modules.py /path/to/model --json
```

### 3. `export_trace.py` — bisect stack divergence

Runs the same forward as `Model.forward` but saves the output tensor after each
top-level module (`embed_tokens`, each `layers.N`, `norm`, `lm_head`).

```bash
python ponyexl3/reference/export_trace.py \
  -m /path/to/model \
  -o ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win_trace.npz \
  --from-npz ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win.npz \
  --row-slice last:1
```

NPZ keys: metadata + `model__language_model__layers__0` etc. (`.` → `__`).

Row slice options: `last:1` (default, small), `last:8`, `all` (large).

### 4. `export_linear_io.py` — per-EXL3-layer CUDA ground truth

Hooks `LinearEXL3.forward` during a real forward. For each module:

- `{key}__x` — input activation (float32)
- `{key}__y` — CUDA fast-kernel output
- `{key}__y_reconstruct` — optional `reconstruct=True` path (`--reconstruct`)

```bash
python ponyexl3/reference/export_linear_io.py \
  -m /path/to/model \
  -o ponyexl3/reference/sample_linear_io.npz \
  --from-npz ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win.npz \
  --row-slice last:1 --reconstruct \
  model.language_model.layers.0.mlp.gate_proj \
  model.language_model.layers.3.self_attn.q_proj \
  lm_head
```

Edit the module list in `run_cuda_exports.sh` for your checkpoint.

### 5. `export_moe_routing.py` — MoE expert selection

```bash
python ponyexl3/reference/export_moe_routing.py \
  -m /path/to/model \
  -o ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win_moe.npz \
  --from-npz ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win.npz \
  --row-slice last:1
```

Per MoE layer: `{key}__experts` (int64), `{key}__weights` (float32).

## MLX side (Mac) — export for CUDA to verify

```bash
# produce ~1 MiB pony reference (reuses CUDA input_ids)
uv run python ponyexl3/reference/export_reference_mlx.py \
  --from-npz ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win.npz \
  -m /path/to/Qwen3.6-35B-A3B-exl3-4.00bpw \
  -o ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_mac.npz
```

Ship `*_mac.npz` + `*_mac.verify.txt` to CUDA team.

## Verify (either direction)

```bash
# CUDA host: check exllamav3 vs Mac export
python ponyexl3/reference/verify_reference.py \
  qwen3.6-35B-A3B-exl3-4.00bpw_mac.npz \
  -m /path/to/Qwen3.6-35B-A3B-exl3-4.00bpw

# Mac: check MLX vs Windows CUDA export
uv run python ponyexl3/reference/compare_reference.py \
  qwen3.6-35B-A3B-exl3-4.00bpw_win.npz \
  -m /path/to/Qwen3.6-35B-A3B-exl3-4.00bpw
```

## Trace replay / drift analysis (Mac)

`compare_trace.py` replays the trace bundle module-by-module and prints
scale-aware drift tables (max|Δ|, rms, rel = rms(Δ)/rms(h), cos):

```bash
uv run python ponyexl3/reference/compare_trace.py \
  qwen3.6-35B-A3B-exl3-4.00bpw_win_trace.npz \
  -m /path/to/Qwen3.6-35B-A3B-exl3-4.00bpw \
  --reference qwen3.6-35B-A3B-exl3-4.00bpw_win.npz \
  --moe qwen3.6-35B-A3B-exl3-4.00bpw_win_moe.npz \
  --noise-floor 2 --tail-check --fp32-residual
```

Flags: `--noise-floor N` (chunked self-replay — MLX schedule determinism
check), `--tail-check` (CUDA last-layer hidden → MLX norm+lm_head),
`--fp32-residual` (mirror exllamav3's fp32 residual-stream dtype contract),
`--moe` (top-8 expert-set agreement), `--save` (export the MLX trace for
the CUDA side).

## Current baseline (35B-A3B exl3-4.00bpw)

| Metric | CUDA vs MLX |
|--------|-------------|
| top-1 argmax | match (top-2 gap is 1 fp16 ulp — near-tie, not signal) |
| max \|Δ\| logits | ~0.83 (rel 4.6% of logits rms) |
| per-module rel drift | ≤ 1.9% through L12, 4–7% plateau after routing flips |
| bit-exact logits | no — and not an achievable goal cross-platform |

Verdict and full anatomy: `docs/drifts_investigation.md` (round-2
addendum). Drift is cross-implementation fp16 numerics + MoE routing
near-tie chaos; no semantic bug; the one systematic asymmetry (fp32 vs
fp16 residual stream) is mirrorable via `--fp32-residual`.
