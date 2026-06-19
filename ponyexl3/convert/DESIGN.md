# pony-quant: HF → EXL3 converter — engineering roadmap (handoff)

Updated 2026-06-19. Status: **M1 complete, M2 complete, M3b direct full-layer
emit/load gate complete, M4 complete for selected-module/layer-set LDLQ emit,
post-M4 computed-scale/calibration inputs complete, MiniCPM5 direct full-model
conversion/load/KLD gated, M5a priority allocation wired, and new-converter
GPU-residency/batched-search optimization in progress**
(`tests/test_convert.py`, `tests/test_convert_metal.py`,
`tests/test_convert_hessian.py`, `tests/test_convert_driver.py`,
`tests/test_convert_regularize.py`, `tests/test_convert_calibration.py`,
`tests/test_convert_allocation.py`, `tests/test_minicpm5_model.py`).

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
- Latest M3 direct-window result on the fast 128×128 block:
  public rel-RMS `0.067390`, output rel-RMS `0.070094`, inner MSE
  `5.283837e-03`, public MSE `1.360480e-06`, output MSE `1.704962e-04`,
  mini bundle reloads through `load_exl3_layer`.
- Latest M3 direct full-layer result on `in_proj_qkv`:
  shape `(2048, 8192)`, trellis `(128, 512, 64)`, output rel-RMS
  `0.069103`, public rel-RMS `0.069056`, inner MSE `5.498035e-03`, output
  MSE `2.255773e-03`; `oracle_safe` scale mode replaced `112` zero `svh`
  entries with `1.0`; full local run took `147 s`, so normal pytest uses a
  synthetic full-layer gate.
- Latest M4 LDLQ full-layer result on `in_proj_qkv`:
  shape `(2048, 8192)`, output rel-RMS `0.005657`, oracle output rel-RMS
  `0.075661`, output/oracle ratio `0.074772`, Hessian proxy rel-RMS
  `0.012307`, oracle proxy rel-RMS `0.076510`, proxy/oracle ratio
  `0.160857`, public rel-RMS `0.069466`, inner MSE `5.562147e-03`;
  `oracle_safe` again replaced `112` zero `svh` entries.
- MiniCPM5 gate:
  source `/Users/beam/llm/models/MiniCPM5-1B`, oracle
  `/Users/beam/llm/models/Exl3/MiniCPM5-1B-exl3-4.00bpw`, generated output
  `/Users/beam/llm/models/Exl3/MiniCPM5-1B-ponyexl3-4.00bpw`. The generated
  bundle has `169` EXL3 linears, `50` plain tensors, `557` stored tensors,
  and strict `load_model(warm=False)` succeeds through the `llama` mlx_lm
  mapping. Full direct/oracle-scale conversion took `428 s`; the fast
  `model.layers.0.self_attn.q_proj` gate dropped from `24.6 s` to about
  `0.95 s` after eliminating CPU packed-trellis decode from metric paths.
  KLD proof used mlx-eval with the BF16 original as the reference distribution over
  `4 x 512` tokens. Overall vs original reference:

  | model | KLD | p95 | p99 | ΔPPL | ΔAcc@1 |
  |-------|----:|----:|----:|-----:|-------:|
  | official oracle | 0.042778 | 0.145305 | 0.315975 | +0.243127 | +0.002446 |
  | ponyexl3-4.00bpw | 0.042186 | 0.136010 | 0.333524 | +0.382146 | −0.001957 |

  PonyExl3-converted matches oracle mean KLD and edges p95; acceptance gate is
  converted-vs-original, not converted-vs-oracle.
- MiniCPM5 M5 optimization smoke:
  one-module direct override proved allocation plumbing by forcing
  `model.layers.0.mlp.down_proj` to K=5; it emitted/reloaded a
  `288 x 96 x 80` trellis in `8.3 s` with output rel-RMS `0.034803`.
  More importantly, same-module LDLQ at the oracle K=4 took `56.8 s` and
  reduced output rel-RMS to `0.003690` (`0.046x` the oracle output rel-RMS
  under the fixture Hessian). This confirms the next quality phase should
  prioritize full-model LDLQ/calibration and measured proxy allocation over
  naive extra-bit spending.
- MiniCPM5 speed hardening:
  the `56.8 s` LDLQ smoke was dominated by diagnostic oracle reconstruction,
  not conversion. LDLQ now defaults to production metrics: oracle metrics are
  skipped unless `--oracle-metrics --full-layer-metrics` is provided. The same
  `model.layers.0.mlp.down_proj` exact LDLQ run dropped to `2.8 s` with
  identical output/proxy metrics. Full MiniCPM layer 0 exact LDLQ
  (`7` modules) took `8.5 s` with `--skip-oracle-metrics`. A grouped feedback
  approximation (`--ldlq-feedback-rows 128`) did not materially improve speed
  and degraded the module output rel-RMS to `0.025153`, so exact
  `--ldlq-feedback-rows 16` remains the recommended heavy-job setting.
- M5b LM-head speed fix:
  K6 uses the non-divisor trellis pack path. The old tensor packer fell back
  to a Python tile loop, leaving the GPU idle between large Metal search
  bursts for `lm_head`. `pack_trellis`/`unpack_trellis` now have vectorized
  general-K paths for K3/K5/K6/K7 while keeping the same public API and
  bitstream convention. At one MiniCPM5 `lm_head` LDLQ row group
  `(1, 8160, 256)`, K6 packing dropped from an estimated `~1.04 s` to
  `0.0032 s` (`~328x`). Isolated MiniCPM5 `lm_head` exact LDLQ dropped from
  the previous full-run phase time of `~2m17s` to `43.24 s` with full local
  diagnostics, or `40.30 s` with production `--fast-layer-metrics`. Raising
  Metal search scratch above `256 MB` did not improve the one-group K6 search
  micro-benchmark, so the default scratch budget remains unchanged.
- M5b production metrics mode:
  production conversion uses fast layer metrics by default, skipping
  reconstructed-public/output/proxy diagnostics after each LDLQ module. The
  emitted EXL3 tensors are unchanged. This mostly
  helps large K6 heads and peak memory pressure; representative MiniCPM5
  `down_proj` remained `2.78 s`, essentially unchanged from `2.8 s`.
- Fast oracle metrics:
  `hessian.py::reconstruct_oracle_public_fast(layer)` now uses the MLX/Metal
  packed-trellis decode kernel (`decode_packed_trellis_mlx_layer`) for the hot
  oracle inner decode, then reuses the reference outer Hadamard + scale steps
  verbatim. `oracle_comparison_weights` is rewired to this path, with a pure
  Python fallback when Metal is unavailable. The parity test covers
  DEFAULT/MCG/MUL1 codebooks and packed-sign/float/None scales. Measured on
  a real MiniCPM5 `gate_proj`: oracle public reconstruction dropped from
  `53.6 s` to `0.014 s` (`~3900x`), and a full `--oracle-metrics` layer run
  dropped from `56 s` to `1.78 s` (`~31x`). Results are bit-identical to the
  reference path (`max|diff| = 0`, exact array equality), so oracle diagnostics
  are useful again for long M6 conversion runs.
- New converter GPU-residency step 1:
  production quantization now packs trellis indices on MLX/GPU via
  `convert/mlx_trellis.py::pack_trellis_mlx`. `quantize_inner_matrix_direct`
  still returns full 256-state arrays for parity/debug calls, but when
  `return_states=False` it no longer materializes the state matrix on CPU and
  no longer calls the NumPy packer. Full direct-layer conversion and LDLQ
  conversion use this no-state path. The output is bit-identical to the debug
  path on Metal parity tests; a bounded MiniCPM5 LDLQ smoke for
  `model.layers.0.mlp.down_proj` wrote/reloaded a K4 layer in `3.55 s` wall
  time. This is the first slice of the larger GPU-resident converter design;
  remaining work is sibling batching and broader timing instrumentation.
- New converter GPU-residency step 2:
  production LDLQ (`search_backend=metal`, `collect_states=False`) now uses an
  MLX-resident inner loop for row buffers, reconstruction, compensation GEMMs,
  and prod-cache updates. The old NumPy loop remains active for CPU mode and
  debug/state-collecting Metal parity. `ldlq_inner_matrix` reports
  `mlx_ldlq=True` on the production path. A parity test compares the new
  no-state MLX loop against the debug state-collecting Metal loop bit-for-bit
  on packed trellis and reconstruction. Bounded MiniCPM5 LDLQ smoke for
  `model.layers.0.mlp.down_proj` wrote/reloaded a K4 layer in `3.69 s` wall
  time with `mlx_ldlq=True`.
- New converter GPU-residency step 3:
  sibling projection conversion now batches the hot Metal trellis search
  calls without requiring shared source scales. Each grouped module keeps its
  own source scales, calibration rows, Hessian, LDL factor, compensation, and
  output layer; at each reverse-LDLQ feedback step the current rows are
  concatenated across siblings for one larger Metal search and then split back
  into per-module packed trellis tensors. This is safe for oracle-safe and
  computed-scale conversion because it does not assume identical `suh`.
  Driver grouping is enabled for production LDLQ (`--search-backend metal`,
  fast metrics, no oracle metrics) on `gate_proj/up_proj`, `q_proj/k_proj/v_proj`,
  and `linear_attn.in_proj_qkv/in_proj_z`. CLI progress now reports
  `batch-start` and `batch-fallback` events, and grouped summaries include
  `batched_search_group_size` / `batched_search_group_out_features`. Parity
  test: two synthetic computed-scale siblings with distinct `suh` and distinct
  calibration activations produce bit-identical trellis/scales versus two
  independent LDLQ conversions. Real MiniCPM5 smoke:
  `--only-layer 0 --module-limit 3` batched `gate/up` and wrote `3` layers in
  `5.44 s` wall (`0.85 s` user); full layer-0 smoke batched `gate/up` plus
  `k/q/v`, wrote `7` layers in `7.61 s` wall (`1.16 s` user).
- New converter GPU-residency step 4:
  grouped LDLQ now prepares sibling fixtures, source/basis matrices, Hessians,
  and LDL factors in a bounded thread pool before allocating MLX buffers. This
  overlaps CPU/IO work inside `gate/up`, `q/k/v`, and linear-attention groups
  while preserving each module's independent math. The pool is intentionally
  disabled for computed-scale GSS unless `--skip-g-scale` is used, because GSS
  itself launches Metal search kernels during prep. Group summaries now report
  `batched_prep_workers`. The computed-scale GSS path also switched its sample
  quantization calls to `return_states=False`, avoiding debug state materialization
  during source-only conversion. Real MiniCPM5 smoke after this step: full
  layer 0 wrote `7` layers in `7.32 s` wall with `batched_prep_workers=2`
  for `gate/up` and `3` for `k/q/v`; source-only computed-scale LDLQ for
  `model.layers.0.mlp.down_proj` ran `13` GSS evaluations and wrote a K4 layer
  in `4.49 s` wall.
- Qwen3.6-27B M6 gate setup:
  source `/Users/beam/llm/models/Qwen/Qwen3.6-27B`, oracle
  `/Users/beam/llm/models/Exl3/Qwen3.6-27B-exl3-4.15bpw`. The oracle advertises
  `bits=4.15`, `head_bits=6`, `codebook=mcg`, calibration metadata
  `{rows: 250, cols: 2048}`, and has `401` supported EXL3 linears with no
  source-adapter skips: `285` at K4, `115` at K5, and `lm_head` at K6. For the
  gatekeeper run, do **not** use `--use-bit-allocation`; leaving `quant_bits`
  unset preserves the oracle's exact per-module K plan.
- Calibration capture:
  `ponyexl3-convert --capture-calibration-map` now runs the BF16 source model
  through MLX-LM, hooks the selected `nn.Linear` modules, and saves fixed-row
  input activations as `.safetensors`/`.npz` keyed by PonyExl3 module name.
  The same `--model-modules`, `--layer-modules`, `--only-layer`, and
  `--module-limit` selection flags are supported. A one-row MiniCPM5 smoke
  wrote a loadable map in `1.77 s`.
- Source-only bpw selection:
  when `--oracle-dir` is omitted, `ponyexl3-convert` now writes a
  BF16-source-derived plan into `--work-dir/source_quant_plan` (or a hidden
  sibling of `--out-dir`) and uses that plan for module discovery, shapes,
  codebook flags, and K selection. This avoids inheriting K from an oracle
  `quantization_config.json`. Exact 4.00bpw means passing both `--bits 4.00`
  and `--head-bits 4`; otherwise the default K6 head override intentionally
  raises the weighted average above 4.00. Plan-only conversion cannot use
  `--oracle-metrics`; if the default oracle-safe scale mode is still selected,
  the auto source-plan path switches to `--scale-mode computed`.
  MiniCPM5 source-only smoke: full-model allocation dry-run produced `169`
  modules, weighted average `4.0`, and all modules at K4; a bounded direct
  conversion of `model.layers.0.mlp.down_proj` from the source-generated plan
  wrote a strict-loadable K4 layer in `3.49 s` wall time.

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

## What exists now (M1 + M2 + M3b + M4)

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
  16×16 tiles, one tile per threadgroup, K∈[2,8], all three codebooks.
  It mirrors CUDA's compressed-edge DP shape and roll-128 warmup + pinned
  tail-biting pass, then returns full 16-bit states plus reconstructed tiles.
  K≥4 keeps costs in threadgroup memory; K=2/3 uses device cost scratch.
  Batches are split by `max_scratch_bytes` to bound backpointer/cost scratch.
- `cli/convert.py` / `ponyexl3-convert` — current CLI is the oracle-comparable
  one-tile pilot by default. It accepts `--search-backend cpu|metal`; CPU
  remains the stable default. `--direct-window` runs the M3 direct 128×128
  block quantizer and writes a minimal loadable EXL3 bundle when `--out-dir`
  is provided. `--direct-layer` quantizes the whole selected linear module;
  `--ldlq-layer` runs Hessian/LDLQ. `--layer-modules --only-layer N`
  promotes direct/LDLQ conversion to a bounded module-set driver; routed
  experts are opt-in via `--include-routed-experts`, and `--module-limit`
  keeps smoke runs bounded. `--scale-mode oracle_safe` keeps oracle scales
  but replaces zero entries with `1.0`; `--scale-mode computed` derives fresh
  `suh`/`svh` from source weights. `--calibration-activations` accepts
  pre-captured `.npy`, `.npz`, or `.safetensors` activation rows for layer
  modes. `--skip-g-scale` bypasses computed-scale GSS for fast smoke runs.
  `--model-modules` converts every supported EXL3 module in the oracle
  checkpoint and includes plain tensors for strict-loadable full bundles.
  Module-set/model-set runs write live progress to stderr even when `--json`
  reserves stdout for final machine-readable output.
- `convert/direct.py` — no-LDL direct conversion for one 128×128 Hadamard
  block and for a full linear module: source BF16 public blocks → scaled inner
  blocks → M2 tile quantization → `EXL3Layer`. It now owns the shared
  source/inner/scales basis builder used by both direct and LDLQ conversion,
  including computed-scale regularization and optional GSS when
  `scale_mode="computed"`. It also owns the shared single-shard EXL3 writer
  for one or more converted layers:
  `model.safetensors`, `model.safetensors.index.json`,
  `quantization_config.json`, optional copied model assets, and
  `ponyexl3_convert_manifest.json` for module-set runs.
  Identity/no-scale mode now leaves public weights in the public basis instead
  of applying unused Hadamards; Qwen oracle-scale paths are unchanged.
  Full source linears are read in one safetensors slice, K∈{1,2,4,8}
  trellis pack/unpack is vectorized, and metric paths reuse Metal-returned
  reconstructed inner weights instead of decoding the packed trellis on CPU.
  Computed-scale GSS sample scoring also uses the no-state quantizer path.
- `convert/hessian.py` — M4b Hessian/LDLQ primitives: activation Hessian
  capture, upstream-style diagonal damping, NumPy/Accelerate block-LDL,
  reverse 16-row LDLQ over inner-domain weights, Hessian proxy metrics, and a
  fixture-backed one-linear `ldlq_quantize_layer` path that emits the same
  minimal loadable bundle as direct conversion. LDLQ now consumes the same
  computed/oracle/identity basis as direct conversion, so source scales and
  activation-space Hessians stay aligned. M4b adds
  `public_matrix_to_inner` and oracle comparison weights, so the same Hessian
  reports converted-vs-source, oracle-vs-source, and converted/oracle proxy
  and output ratios. Production Metal LDLQ also has `ldlq_quantize_group`,
  which batches sibling modules at the trellis-search call boundary while
  preserving each module's independent scales/Hessian/LDL state.
- `convert/regularize.py` — post-M4 regularization port: blockwise RMS,
  deterministic random sign flips, upstream `CODEBOOK_SCALE`, output/input
  channel scales, 128-block Hadamards, wrapped-diagonal tile sampling, and
  golden-section global scale search.
- `convert/calibration.py` — pre-captured calibration activation loader for
  `.npy`, `.npz`, and `.safetensors`; fixtures validate shape/finite values
  against the selected module's input dimension before Hessian capture.
- `convert/discovery.py` — source-only quantization planning: discovers EXL3
  linears and plain tensors from BF16 safetensors, runs M5a bit allocation,
  and writes `quantization_config.json` via `--init-quant-config`. Plan-only
  dirs (config without trellis weights) work as `--oracle-dir` with
  `--scale-mode computed`.
- `convert/driver.py` — M4 module-set driver: discovers EXL3 modules for a
  layer from oracle `quantization_config.json`, filters to source adapters,
  optionally includes routed experts, applies `direct` or `ldlq`, emits one
  multi-layer bundle, records completed/skipped modules in the manifest, and
  resumes requested modules already present in `--out-dir`. Module-set runs
  now carry calibration row counts, `skip_g_scale`, and regularization seed
  into the manifest. Model-wide discovery handles MiniCPM/Llama-style
  `model.layers.*` keys and carries non-EXL3 plain tensors into emitted
  full-model bundles.
- `convert/allocation.py` — M5a scaffold: deterministic priority allocation
  from a target bpw to per-module integer `K` values. It is now
  parameter-weighted, supports fixed-cost overrides (`lm_head`/regex), and
  reports weighted average bits plus target delta. This is the cheap
  allocation tier; M5b will replace static priorities with measured proxy-loss
  deltas.
- `tests/test_convert.py` — transition invariant + bit round-trip +
  MSE bounds, k∈{2,3}, BF16 reader gate, Qwen source adapter gate, CPU
  one-tile oracle gate, guarded Metal one-tile oracle gate, and expected
  zero-scale rejection for the MoE gate block.
- `tests/test_convert_metal.py` — Metal-only gates: random-tile quality
  within `1.10×` CPU reference for K∈{2,3,4}, all codebooks at K=4, exact
  recovery of ideal tail-biting tiles for K∈{2,3,4,5,8}, pack/unpack
  round-trip, forced chunked-batch scratch coverage, and explicit K=1
  rejection.
- `tests/test_convert_hessian.py` — M4b gates: Hessian capture/regularize +
  block-LDL identity-block checks, identity-Hessian LDLQ equals direct
  quantization exactly, correlated-Hessian proxy stats stay bounded against
  direct quantization, public→inner identity mode is a no-op, and a guarded
  Metal synthetic 128×128 LDLQ layer emits/reloads while beating its synthetic
  oracle on Hessian proxy and output ratios.
- `tests/test_convert_driver.py` — M4 emit/driver gates: multi-layer bundle
  writes and reloads all layers, copies model assets, records manifest tensor
  and layer counts, and layer-module discovery excludes routed experts unless
  explicitly requested. Resume reloads an existing emitted layer without
  touching the source/oracle checkpoint.
- `tests/test_convert_regularize.py` — post-M4 gates for block RMS,
  regularize/public→inner inverse parity, tile sampling, and GSS.
- `tests/test_convert_calibration.py` — activation loader gates for `.npy`,
  `.npz`, `.safetensors`, and shape validation.
- `tests/test_convert_allocation.py` — M5a priority allocation gates.
- `tests/test_minicpm5_model.py` — MiniCPM5 oracle load gate through the
  `llama` architecture mapping.
- Verification as of 2026-06-18:
  `python -m pytest tests/test_convert.py tests/test_convert_metal.py
  tests/test_convert_hessian.py tests/test_convert_driver.py
  tests/test_convert_regularize.py tests/test_convert_calibration.py
  tests/test_convert_allocation.py tests/test_minicpm5_model.py -q` →
  59 passed. `python -m pyright ponyexl3/convert ponyexl3/cli/convert.py
  ponyexl3/mlx/model.py
  tests/test_convert.py tests/test_convert_metal.py tests/test_convert_hessian.py
  tests/test_convert_driver.py tests/test_convert_regularize.py
  tests/test_convert_calibration.py tests/test_convert_allocation.py
  tests/test_minicpm5_model.py` → clean.

---

## M2 — Metal trellis-search kernel

Status: **functionally complete**. The working primitive is
`quantize_tiles_mlx(_np)` in `convert/metal_search.py`.

Implemented:

- One 16×16 tile per threadgroup through `mx.fast.metal_kernel`.
- K∈[2,8]: K≥4 uses threadgroup-resident `half costs[2][edges]`; K=2/3
  uses device cost scratch because Apple threadgroup memory caps at 32 KB.
- Inline decode for DEFAULT/MCG/MUL1 using the existing codebook formulas.
- Backpointers in device scratch (`temp_edges`) and in-kernel backtracking.
- CUDA-style `roll=128` warmup, then `roll=0` pinned tail-biting solve.
- Returns reconstructed tiles plus full 16-bit states; existing
  `pack_trellis_tile(states & ((1<<K)-1), K)` remains the only pack path.
- Batch chunking via `max_scratch_bytes`, so large tile batches are split
  instead of allocating unbounded `temp_edges` scratch.

M2 notes:

- Decide the CUDA parity contract: bit-identical indices where no ties occur,
  otherwise equal-MSE/tail-biting/pack-roundtrip. Current tests gate quality
  rather than exact random-tile state identity because CPU and Metal use
  different precision/tie behavior.
- Keep optional A/B of a 128 KB device LUT of all 65536 decoded values vs
  inline ALU decode for throughput only; correctness already comes from the
  inline path.
- Measured K=4 MCG random-tile throughput on this machine: about
  `1.2 Mtiles/min` after warm compile at the default 256 MB scratch cap
  (`2048` tiles in `0.102 s`). The old `2 Mtiles/min` note remains an
  aspirational optimization target, not a correctness blocker. A 1024-thread
  per-tile variant was tested and was slower on M5 Max; the 256-thread shape
  has better occupancy.

Original kernel constraints still apply:

- Threadgroup ≤1024 threads (CUDA uses `MIN(1024, 65536>>K)`); cost arrays
  `temp_costs[2][edges]`, edges = 65536>>K. In this Metal version, K≥4 stays
  in threadgroup memory; K=2/3 uses device scratch; K=1 is unsupported.
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
- Backtrack: CUDA does it in-kernel; the Metal port now does the same.
- **Gates**: (a) random-tile reconstruction MSE ≤ `1.10×` CPU reference for
  K∈{2,3,4}; (b) exact recovery on ideal tail-biting tiles; (c) pack→unpack
  round-trip; (d) forced chunked-batch scratch path; (e) one checkpoint-backed
  Qwen tile through the oracle fixture.
- **Metal pitfalls already hit in this repo**: `//` comments inside
  multi-line `#define`s get line-spliced (strip via `decode_bs`);
  register cliffs near 32 simdgroup accumulators; `_decode_expr` declares
  `dq_val` in-scope (wrap call sites in braces).

## M3 — Regularize + direct one-linear conversion

Status: **M3b complete for direct no-LDL full-layer conversion** and
post-M4 direct-scale work is in place. Direct conversion can use identity
scales for synthetic tests, `oracle_safe` for oracle diagnostics, or
`computed` for freshly derived `suh`/`svh` from the source weight. The
computed path ports upstream block RMS, random sign flips, 128-block
Hadamards, codebook scale, and global-scale GSS.

Remaining direct-path work is now quality tuning, not plumbing: compare
`computed` against the current `oracle_safe` Qwen baseline over the fast
module and decide whether MCG/default codebook selection needs an explicit
policy before full-model conversion.

## M4 — Hessian/LDLQ, Driver + Emit

Status: **complete for selected-module/layer-set conversion**. The current
driver is still checkpoint-fixture based rather than a full calibration-data
streamer, but it now covers the complete M4 storage and bounded layer-driver
surface:

- `capture_hessian`: `X.T @ X` accumulation with optional normalization.
- `prepare_hessian_for_ldl`: dead-channel handling and `sigma_reg` diagonal
  damping.
- `block_ldl`: NumPy Cholesky with the upstream retry ladder; diagonal 16×16
  blocks are normalized to identity.
- `ldlq_inner_matrix`: reverse 16-row LDLQ with tensor-core tile order handled
  by the M2 quantizer and packed with the existing trellis packer only.
- `ldlq_quantize_layer`: one full selected linear module through
  source→inner assembly, fixture activation Hessian, LDLQ, `EXL3Layer`, output
  metrics, and optional minimal bundle reload through the existing writer.
- `oracle_comparison_weights`: dequantize the oracle layer, transform it back
  into the same comparable inner basis as the source/converted layer, and
  report oracle proxy/output error plus converted/oracle ratios.
- `write_exl3_layers_bundle`: emits one or more converted layers with
  `quantization_config.json`, `model.safetensors.index.json`, copied
  tokenizer/config assets, and a conversion manifest.
- `convert/driver.py`: layer module discovery, supported-adapter filtering,
  routed-expert opt-in, bounded module limits, direct/LDLQ dispatch, and
  module-set summary/resume.
- CLI: `--ldlq-layer`, `--sigma-reg`, `--buf-size-rows`,
  `--layer-modules`, `--include-routed-experts`, and `--module-limit`.

Post-M4 work is complete within the current checkpoint-backed fixture scope:

1. Regularized/GSS scale selection landed as `scale_mode="computed"`, so
   direct and LDLQ no longer require oracle scale metadata.
2. Fixture random activations can be replaced with pre-captured calibration
   rows via `--calibration-activations`; full text/model streaming remains
   the M5b/M6 layer-sequential driver work below.
3. Module-set conversion has the manifest and resume hooks needed for the
   full block-sequential driver; quantized re-forwarding between layers is the
   next architectural step, not a storage blocker.

- Future `hessian.py`: layer-sequential driver. Load the HF model with mlx_lm's
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
  128-block Hadamard, block RMS, codebook scale, and GSS are in place. JSD
  accept heuristics remain future quality tuning. fp32.
- `block_ldl` port (quantize.py:292): Cholesky on H with the `sigma_reg`
  damping retry ladder. numpy/CPU is fine — microseconds per layer.
- `ldlq` port (quantize.py:365): per 16-row block in kernel order
  (`ref/perm.py` has tensor_core_perm), quantize tiles via M2 with
  error feedback through L, two passes like upstream (call site
  quantize.py:1065). `g_scale_gss` and `block_rms` are now ported.
- **Gate**: quantize ONE Qwen3.5-2B layer; proxy error
  tr(EHEᵀ)/tr(WHWᵀ) within ~10% of the corresponding layer in the
  upstream-made 4.00bpw checkpoint (dequantize theirs and recompute).

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
- **M5a priority allocation** (`convert/allocation.py` now wired):
  integer base K = floor(bpw); spend the remaining weighted budget one bit at
  a time by module priority. Cheap, no extra compute — produces
  "4.15bpw"-style mixes when the fixed-cost overrides make the requested
  budget feasible. CLI/driver support is in place:
  `--allocation-dry-run`, `--use-bit-allocation`, `--bits`, `--head-bits`,
  and repeated `--layer-bits REGEX:K`. Default conversion remains unchanged
  unless allocation/override flags are supplied. MiniCPM dry-run example:
  `--bits 4.60 --head-bits 6` yields weighted average `4.599643`, with
  `63` modules at K=5, `1` module (`lm_head`) at K=6, and `105` at K=4.
  `--bits 4.15 --head-bits 6` is over budget at `4.455764` before any
  upgrades because the fixed `lm_head=6` cost alone exceeds the target.
  Real smoke: forcing one MiniCPM `model.layers.0.mlp.down_proj` to K=5
  emitted and reloaded a `288 x 96 x 80` trellis in `8.3 s`; module output
  rel-RMS was `0.034803` vs the prior K=4 smoke's `~0.068`.
- **M5b measured allocation** (`measure_model.py` + `optimize_model.py`):
  quantize a sample of each module at candidate K, record proxy error,
  optimize the allocation under the budget — the full "calibrate bits per
  layer". Same-K LDLQ already shows a large MiniCPM module-level improvement
  (`down_proj` K=4 output rel-RMS `0.003690` vs direct `~0.068`), so M5b
  should first measure direct-vs-LDLQ and candidate K deltas on bounded
  row/layer subsets before attempting a full KLD sweep. Mitigate the
  K-candidate multiplier by measuring on row subsets like upstream. CLI:
  `--hq`, `--layer-bits regex:K` overrides.

## M6 — Qwen3.6-27B 4.15bpw gatekeeper

Goal: produce
`/Users/beam/llm/models/Exl3/Qwen3.6-27B-PonyExl3-4.15bpw` and show its
KLD-vs-original is on par with the official
`/Users/beam/llm/models/Exl3/Qwen3.6-27B-exl3-4.15bpw` oracle.

First capture real BF16 calibration activations:

```bash
cd /Users/beam/llm/PonyExl3
mkdir -p /Users/beam/llm/PonyExl3/.work/logs
set -o pipefail
/usr/bin/time -p .venv/bin/ponyexl3-convert \
  --in-dir /Users/beam/llm/models/Qwen/Qwen3.6-27B \
  --oracle-dir /Users/beam/llm/models/Exl3/Qwen3.6-27B-exl3-4.15bpw \
  --capture-calibration-map /Users/beam/llm/PonyExl3/.work/qwen3.6-27b-calib-r250.safetensors \
  --calibration-text /Users/beam/llm/kld-eval/mlx_eval/prompt.txt \
  --calibration-rows 250 \
  --calibration-seq-len 2048 \
  --model-modules \
  2>&1 | tee /Users/beam/llm/PonyExl3/.work/logs/qwen3.6-27b-calib-r250.capture.log
```

For a source-only exact 4.00bpw run without consulting the oracle config, omit
`--oracle-dir` and set the head to K4 as well:

```bash
cd /Users/beam/llm/PonyExl3
mkdir -p /Users/beam/llm/PonyExl3/.work/logs
CALIB=/Users/beam/llm/PonyExl3/.work/qwen3.6-27b-calib-r250.safetensors
test -f "$CALIB" || { echo "missing calibration map: $CALIB" >&2; exit 2; }
set -o pipefail
/usr/bin/time -p .venv/bin/ponyexl3-convert \
  --in-dir /Users/beam/llm/models/Qwen/Qwen3.6-27B \
  --out-dir /Users/beam/llm/models/Exl3/Qwen3.6-27B-PonyExl3-4.00bpw \
  --work-dir /Users/beam/llm/PonyExl3/.work/qwen3.6-27b-source-4.00bpw \
  --bits 4.00 \
  --head-bits 4 \
  --ldlq-layer \
  --model-modules \
  --search-backend metal \
  --scale-mode computed \
  --calibration-activations-map "$CALIB" \
  --resume \
  2>&1 | tee /Users/beam/llm/PonyExl3/.work/logs/qwen3.6-27b-ponyexl3-4.00bpw.source.convert.log
```

Then run this conversion command when the machine is free:

```bash
cd /Users/beam/llm/PonyExl3
mkdir -p /Users/beam/llm/PonyExl3/.work/logs
CALIB=/Users/beam/llm/PonyExl3/.work/qwen3.6-27b-calib-r250.safetensors
test -f "$CALIB" || { echo "missing calibration map: $CALIB" >&2; exit 2; }
set -o pipefail
/usr/bin/time -p .venv/bin/ponyexl3-convert \
  --in-dir /Users/beam/llm/models/Qwen/Qwen3.6-27B \
  --oracle-dir /Users/beam/llm/models/Exl3/Qwen3.6-27B-exl3-4.15bpw \
  --out-dir /Users/beam/llm/models/Exl3/Qwen3.6-27B-PonyExl3-4.15bpw \
  --bits 4.15 \
  --head-bits 6 \
  --ldlq-layer \
  --model-modules \
  --search-backend metal \
  --scale-mode oracle_safe \
  --oracle-metrics \
  --full-layer-metrics \
  --calibration-activations-map "$CALIB" \
  --resume \
  2>&1 | tee /Users/beam/llm/PonyExl3/.work/logs/qwen3.6-27b-ponyexl3-4.15bpw.convert.log
```

Important command notes:

- The `CALIB` map must be real per-module calibration activations keyed by
  module name. Do not use the fixture/random fallback for this acceptance run.
- `--bits` and `--head-bits` record the intended budget in the manifest. Since
  `--use-bit-allocation` is deliberately omitted, the driver preserves the
  oracle's exact K plan: `285` K4 linears, `115` K5 linears, and K6 `lm_head`.
- `--oracle-metrics --full-layer-metrics` is now viable on M6 because oracle
  public reconstruction uses the fast Metal trellis decode path. Production
  runs without diagnostics should omit both flags and keep default
  `--fast-layer-metrics`.

Acceptance after conversion:

```bash
cd /Users/beam/llm/kld-eval
mkdir -p /Users/beam/llm/kld-eval/results
set -o pipefail
uv run mlx_eval.compare \
  /Users/beam/llm/models/Exl3/Qwen3.6-27B-exl3-4.15bpw \
  16 /Users/beam/llm/kld-eval/outputs/Qwen3.6-27B \
  2>&1 | tee /Users/beam/llm/kld-eval/results/Qwen3.6-27B-exl3-4.15bpw-oracle-compare.log
uv run mlx_eval.compare \
  /Users/beam/llm/models/Exl3/Qwen3.6-27B-PonyExl3-4.15bpw \
  16 /Users/beam/llm/kld-eval/outputs/Qwen3.6-27B \
  2>&1 | tee /Users/beam/llm/kld-eval/results/Qwen3.6-27B-PonyExl3-4.15bpw-compare.log
```

Remaining polish after the gate: tile batching across MoE experts per launch;
progress UI; `--dry-run` size report; `fallback_quant` port
(quantize.py:484, no-LDL variant for degenerate H / odd layers); MCG/MUL1
codebook flags for non-oracle conversions; README for converted models
(recommend the Phase-26 runtime env, e.g. EXL3_WCACHE default).

## Runtime Expectations

Measured on the current M5 Max workflow unless noted. These are wall-clock
times for the converter, not KLD scoring. They should be treated as planning
bounds until M5b measured allocation and the M6 layer-sequential streamer land.

| scope | mode | observed / expected time | notes |
|---|---|---:|---|
| MiniCPM5 `model.layers.0.self_attn.q_proj` | direct, Metal, oracle-safe scales | `~0.95 s` | Fast iteration gate after removing CPU trellis decode from metrics. |
| MiniCPM5 `model.layers.0.mlp.down_proj` | direct, Metal, oracle-safe scales | `~1.8 s` | Representative small smoke module with live progress output. |
| MiniCPM5 `model.layers.0.mlp.down_proj` | exact LDLQ, Metal, full diagnostics | `2.8 s` | Output rel-RMS `0.003690`; prior `56.8 s` included diagnostic oracle CPU dequantization. |
| MiniCPM5 layer 0 (`7` modules) | exact LDLQ, Metal, no oracle metrics | `8.5 s` | Recommended heavy-job path; keep `--ldlq-feedback-rows 16`. |
| MiniCPM5 `lm_head` | exact LDLQ, Metal, K6, `--full-layer-metrics` | `43.24 s` | Full local diagnostics after vectorized general-K trellis packing; previous full-run head phase was `~2m17s`. |
| MiniCPM5 `lm_head` | exact LDLQ, Metal, K6, default production metrics | `40.30 s` | Skips public/output/proxy diagnostics but emits identical EXL3 tensors. |
| MiniCPM5-1B full model | direct, Metal, oracle-safe scales | `428 s` (`7.1 min`) | `169` EXL3 linears + `50` plain tensors; strict-loadable output. |
| MiniCPM5 `gate_proj` oracle public reconstruct | reference vs fast Metal oracle path | `53.6 s` -> `0.014 s` | `~3900x` faster; bit-identical (`max abs diff = 0`) to the reference oracle decode path. |
| MiniCPM5 full `--oracle-metrics` layer | exact LDLQ, full diagnostics | `56 s` -> `1.78 s` | `~31x` faster after swapping only the hot oracle trellis decode; metrics unchanged. |
| MiniCPM5 `model.layers.0.mlp.down_proj` | exact LDLQ, Metal, GPU trellis pack, no state materialization | `3.55 s` | Bounded smoke after adding `pack_trellis_mlx`; wrote/reloaded K4 layer. |
| MiniCPM5 `model.layers.0.mlp.down_proj` | exact LDLQ, MLX-resident compensation/prod-cache, no state materialization | `3.69 s` | Bounded smoke after adding `mlx_ldlq`; similar wall time on one module, much lower CPU user time (`0.65 s`). |
| Qwen3.6-27B full model | exact LDLQ, Metal, oracle K plan, full oracle diagnostics | long/overnight expected | `401` supported EXL3 linears: `285` K4, `115` K5, K6 `lm_head`; use real per-module calibration and `--resume`. |
| Qwen3.6-35B-A3B `in_proj_qkv` | direct, Metal, oracle-safe scales | `147 s` | Single large pilot linear, shape `(2048, 8192)`. |
| Qwen3.6-35B-A3B layer 0 | direct/LDLQ fixture driver | tens of minutes expected | Depends on routed expert inclusion and activation rows. Use module limits while tuning. |
| Qwen3.6-35B-A3B full model | current fixture-style path | overnight expected | M5b/M6 should add measured allocation, block-sequential calibration, resume, and better progress before relying on this path. |

KLD scoring is much faster for MiniCPM-sized windows: original reference
generation for `4 x 512` tokens took `1.49 s`, oracle-vs-original compare took
`1.39 s`, and converted-vs-original compare took `1.40 s`; saved reference
windows are large (`~510 MB` for `4 x 512`).

## Open questions for the implementer

- Chase bit-identical encodings vs CUDA argmin tie-order, or accept
  equal-MSE? (Recommend equal-MSE; document the tie-break.)
- fp16 cost overflow at large K — CUDA clamps via H_INF; watch for
  inf-poisoning interacting with pinned passes.
- mx.linalg Cholesky stability — numpy fallback is free.
