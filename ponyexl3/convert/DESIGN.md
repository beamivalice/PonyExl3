# pony-quant: HF → EXL3 converter — engineering roadmap (handoff)

Updated 2026-06-18. Status: **M1 complete, M2a Metal tile pilot complete**
(`tests/test_convert.py`, `tests/test_convert_metal.py`).

Goal: accept a HuggingFace checkpoint (safetensors), emit an EXL3 checkpoint
that `ponyexl3/mlx/model.py` loads unmodified, with per-layer bit
calibration under a global bpw budget. Reference implementation throughout
the exllamav3 source tree (https://github.com/turboderp-org/exllamav3),
`exllamav3/` —
`conversion/{convert_model,measure_model,optimize_model,allocation,
calibration_data,compile}.py`, `modules/quant/exl3_lib/quantize.py` (1151
lines, the math core), `exllamav3_ext/quant/quantize.cu` (the search kernel).

Current checkpoint-backed pilot:

- Source BF16 model:
  `/Users/beam/llm/models/Qwen/Qwen3.6-35B-A3B`
- Oracle EXL3 model:
  `/Users/beam/llm/models/Exl3/Qwen3.6-35B-A3B-exl3-4.00bpw`
- Fast module/tile:
  `model.language_model.layers.0.linear_attn.in_proj_qkv`, tile `[0, 0]`
- Secondary MoE gate pilot:
  `model.language_model.layers.0.mlp.experts.0.gate_proj`; current oracle
  block has zero `svh` scales, so this remains an expected rejected fixture
  until output-space validation replaces raw block inversion for routed paths.
- Latest Metal pilot result on the fast tile:
  converted target MSE `4.659796e-03`, oracle target MSE `8.931810e-03`,
  converted rel-RMS `0.065796`, oracle rel-RMS `0.091093`, pack round-trip
  `True`.

---

## Ground truths already nailed (do NOT re-derive; gates exist)

1. **Window/codeword convention**: codeword t of a tile is the full 16-bit
   window of the tail-biting 256K-bit stream ending at bit (t+1)K. Fresh
   bits per step are the **LOW K bits** of each window; the transition is
   `s' = ((s << K) | fresh) & 0xFFFF`. Verified 256/256 against a real
   checkpoint tile including the wrap (Phase 26).
2. **Packing**: use `ponyexl3/ref/trellis.py::pack_trellis(_tile)` with
   input `states & ((1<<K)-1)`. `ref_pack(low_k(unpack(tile))) == tile`
   bit-exactly on real checkpoints. Never write another packer.
3. **Decode parity**: all reconstruction must go through
   `ref/codebook.py::decode_3inst` (numpy) or `mlx/gemv_metal.py::
   _decode_expr` (Metal) — these are bit-matched to CUDA (lop3 convention
   `(a<<2)|(b<<1)|c`, half-add `__hadd` parity; see Phases 1-3 history).
4. **Lesson (it cost real time)**: a hand-rolled packer and a "corrected"
   search transition formed a self-consistent WRONG pair that round-tripped
   against each other. Real checkpoints are the only arbiter for format
   conventions.

## What exists now (M1 + M2a)

- `convert/reference_search.py` — exact numpy Viterbi, any K∈[1,8], any
  codebook mode; tail-biting via pinned re-passes; ~0.5-2 s/tile. This is
  the **parity oracle** for the Metal kernel, not a production path.
  Quality: MSE 0.075/0.021/0.005/0.0003 at K=2/3/4/6 on unit Gaussians
  (QTIP-class).
- `convert/fixtures.py` — lightweight checkpoint-backed conversion fixture:
  BF16 safetensors slice reader, Qwen dense/MoE source adapters, oracle EXL3
  loading, source→EXL3 inner-block inversion, and one 16×16 tile comparison.
  Data types are in place for `SourceLinear`, `OracleLinear`,
  `QuantizedLinearTensors`, `LayerFixture`, and `TilePilotResult`.
- `convert/metal_search.py` — MLX/Metal trellis search for kernel-order
  16×16 tiles, one tile per threadgroup, K∈[4,8], all three codebooks.
  It mirrors CUDA's compressed-edge DP shape and roll-128 warmup + pinned
  tail-biting pass, then returns full 16-bit states plus reconstructed tiles.
  K=2/3 intentionally raise until the global scratch path lands.
- `cli/convert.py` / `ponyexl3-convert` — current CLI is the oracle-comparable
  one-tile pilot. It accepts `--search-backend cpu|metal`; CPU remains the
  stable default, Metal is the M2a backend.
- `tests/test_convert.py` — transition invariant + bit round-trip +
  MSE bounds, k∈{2,3}, BF16 reader gate, Qwen source adapter gate, CPU
  one-tile oracle gate, guarded Metal one-tile oracle gate, and expected
  zero-scale rejection for the MoE gate block.
- `tests/test_convert_metal.py` — Metal-only gates: random-tile quality
  within `1.10×` CPU reference for all codebooks at K=4, exact recovery of
  ideal tail-biting tiles for K∈{4,5,8}, pack/unpack round-trip, and explicit
  K<4 rejection.
- Verification as of 2026-06-18:
  `python -m pytest tests/test_convert.py tests/test_convert_metal.py -q`
  → 14 passed; `python -m pyright ponyexl3/convert ponyexl3/cli/convert.py
  tests/test_convert.py tests/test_convert_metal.py` → clean.

---

## M2 — Metal trellis-search kernel

Status: **M2a complete for the fast one-tile pilot**. The working primitive is
`quantize_tiles_mlx(_np)` in `convert/metal_search.py`.

Implemented:

- One 16×16 tile per threadgroup through `mx.fast.metal_kernel`.
- K∈[4,8] using threadgroup-resident `half costs[2][edges]`.
- Inline decode for DEFAULT/MCG/MUL1 using the existing codebook formulas.
- Backpointers in device scratch (`temp_edges`) and in-kernel backtracking.
- CUDA-style `roll=128` warmup, then `roll=0` pinned tail-biting solve.
- Returns reconstructed tiles plus full 16-bit states; existing
  `pack_trellis_tile(states & ((1<<K)-1), K)` remains the only pack path.

Remaining M2 work:

- Add the global-cost scratch path for K=2/3 if sub-4-bit conversion is
  needed; K=4 is enough for the current 4.00 bpw Qwen oracle pilot.
- Batch many tiles per launch with bounded/reused scratch and benchmark
  throughput on real 128×128 source blocks.
- Decide the CUDA parity contract: bit-identical indices where no ties occur,
  otherwise equal-MSE/tail-biting/pack-roundtrip. Current tests gate quality
  rather than exact random-tile state identity because CPU and Metal use
  different precision/tie behavior.
- Keep optional A/B of a 128 KB device LUT of all 65536 decoded values vs
  inline ALU decode for throughput only; correctness already comes from the
  inline path.

Original kernel constraints still apply:

- Threadgroup ≤1024 threads (CUDA uses `MIN(1024, 65536>>K)`); cost arrays
  `temp_costs[2][edges]`, edges = 65536>>K. In this Metal version, K≥4 stays
  in threadgroup memory; K<4 needs global scratch.
- Per step t∈[0,256): for each out-edge group, min over 2^K incoming
  branches; branch cost `(decode_3inst(state) − w[t])²` in **half** (CUDA
  uses half costs + `H_INF` sentinels — keep; it's part of parity).
  Decode via the existing `_decode_expr` template ⇒ bit parity by
  construction. Optional A/B: a 128 KB device LUT of all 65536 decoded
  values (Apple SLC-friendly) vs inline ALU decode.
- Backpointers: `temp_edges` (256 × edges × u16 = 2 MB/tile at K=4) in a
  device scratch buffer; batch tiles per launch to bound scratch (512
  tiles → 1 GB; tune).
- Tail-biting: mirror CUDA's `roll`/`pre_state` two-pass pinning (the
  `forward` lambda, quantize.cu:57+). `reference_search.py` implements the
  same scheme — diff against it.
- Backtrack: CUDA does it in-kernel; a first correct version may return
  costs+edges and backtrack in numpy, then move in-kernel.
- **Gates**: (a) encoded indices == `reference_search` on ≥100 random
  tiles, K∈{2,3,4} — ties may legitimately differ; if so compare
  reconstruction-MSE equality and document the tie-break; (b) pack→unpack
  round-trip; (c) throughput ≥2 Mtiles/min @K=4 (then a 2B = 8M tiles ≈
  4 min of search; 27B ≈ 50 min — conversion becomes Hessian/LDLQ-bound).
- **Metal pitfalls already hit in this repo**: `//` comments inside
  multi-line `#define`s get line-spliced (strip via `decode_bs`);
  register cliffs near 32 simdgroup accumulators; `_decode_expr` declares
  `dq_val` in-scope (wrap call sites in braces).

## M3 — Regularize + direct one-linear conversion, then Hessian + LDLQ

Next milestone should stay on the same lightweight pilot before expanding:

1. Port the no-LDL direct path first for
   `model.language_model.layers.0.linear_attn.in_proj_qkv`, using the existing
   oracle `suh`/`svh` fixture path to validate storage and output-space error.
2. Port upstream `regularize`, `block_rms`, Hadamard transforms, MCG/default
   codebook flags, and global scale GSS. Gate with one converted EXL3 layer
   loaded through the existing `EXL3Layer`/`EXL3Linear`.
3. Only after direct one-linear emit is loading and comparing cleanly, add
   Hessian capture and reverse 16-row LDLQ.

- `hessian.py`: layer-sequential driver. Load the HF model with mlx_lm's
  classes (qwen3_5/qwen3_5_moe already mapped); stream calibration rows
  (reuse `exllamav3/conversion/standard_cal_data/*` and the sampling in
  `calibration_data.py`) through embedding → block i, accumulating
  per-linear-INPUT `H += xᵀx` in fp32 on device (H is (in,in):
  2048²=16 MB, 5120²=105 MB — fine). Same-input siblings (q/k/v/z;
  gate/up; ALL experts' gate/up share the block input) share one H —
  exllamav3's capture dict keys by input; mirror it.
- After quantizing block i, RE-FORWARD its output from the quantized
  weights before capturing block i+1 (GPTQ error propagation) —
  `convert_model.py`'s main loop is the ordering reference.
- `regularize` port (quantize.py:835): seeded random sign flips (su/sv),
  128-block Hadamard, JSD accept heuristic. fp32.
- `block_ldl` port (quantize.py:292): Cholesky on H with the `sigma_reg`
  damping retry ladder. numpy/CPU is fine — microseconds per layer.
- `ldlq` port (quantize.py:365): per 16-row block in kernel order
  (`ref/perm.py` has tensor_core_perm), quantize tiles via M2 with
  error feedback through L, two passes like upstream (call site
  quantize.py:1065). Port `g_scale_gss` (golden-section global scale)
  and `block_rms` faithfully — they shape final quality.
- **Gate**: quantize ONE Qwen3.5-2B layer; proxy error
  tr(EHEᵀ)/tr(WHWᵀ) within ~10% of the corresponding layer in the
  upstream-made 4.00bpw checkpoint (dequantize theirs and recompute).

## M4 — driver + emit (~2 days)

- Walk blocks in order; MoE experts are independent Linears — emit
  per-expert tensors exactly like upstream (the stacked layout is an
  inference-side load-time transform).
- Emit safetensors shards `{name}.{trellis,suh,svh,mcg,mul1,bias}` +
  `config.json`/`quantization_config` + tokenizer files. Confirm exact
  tensor names by `safe_open` on an existing checkpoint; our `load_model`
  (strict) IS the format spec — loading with it is the gate.
- **Resume/checkpointing**: hours-overnight jobs; persist per-module
  outputs + a manifest so a crash resumes at the next module (upstream's
  `compile.py` + job-dir scheme is the model).
- **Gates**: full Qwen3.5-2B convert @4.0 bpw → loads strict; teacher-
  forced quality within the class of the existing 2B-exl3-4.00bpw;
  generation sanity; suite stays green.

## M5 — per-layer bit calibration (~2-3 days)

Port both upstream tiers, in this order:
- **M5a priority allocation** (`conversion/allocation.py::
  create_q_strategy`): integer base K = floor(bpw); spend the remaining
  budget one bit at a time by module-group priority (qgroups,
  `q_priority`). Cheap, no extra compute — produces "4.15bpw"-style
  mixes. CLI: `--bpw`, `--head-bits`.
- **M5b measured allocation** (`measure_model.py` + `optimize_model.py`):
  quantize a sample of each module at candidate K, record proxy error,
  optimize the allocation under the budget — the full "calibrate bits per
  layer". Mitigate the K-candidate multiplier by measuring on row subsets
  like upstream. CLI: `--hq`, `--layer-bits regex:K` overrides.

## M6 — polish

Tile batching across MoE experts per launch; progress UI; `--dry-run`
size report; `fallback_quant` port (quantize.py:484 — no-LDL variant for
degenerate H / odd layers); MCG/MUL1 codebook flags (DEFAULT suffices for
new conversions); README for converted models (recommend the Phase-26
runtime env, e.g. EXL3_WCACHE default).

## Runtime expectations (M5 Max, post-M2 target)

| model | search | total (1 pass, incl. Hessian+LDLQ) |
|---|---|---|
| 2B | ~4 min | ~1-2 h |
| 27B | ~50 min | overnight |
| 35B-A3B | ~1 h (experts are small tiles) | overnight |

## Open questions for the implementer

- Chase bit-identical encodings vs CUDA argmin tie-order, or accept
  equal-MSE? (Recommend equal-MSE; document the tie-break.)
- fp16 cost overflow at large K — CUDA clamps via H_INF; watch for
  inf-poisoning interacting with pinned passes.
- mx.linalg Cholesky stability — numpy fallback is free.
