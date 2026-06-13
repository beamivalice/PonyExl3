# EXL3 CUDA ↔ MLX Parity Audit

**Verdict: PASS — no defect.** The PonyExl3 Metal forward reproduces the
reference CUDA (exllamav3) forward to within the expected cross-platform
floating-point floor. The residual logit difference is fully attributed to
fp16 kernel-rounding (amplified by mixture-of-experts routing near-ties) and
one benign, characterized dtype convention difference — not to any error in
the trellis codec, the linear algebra, attention, normalization, or routing.

| | |
|---|---|
| **Model** | `Qwen3.6-35B-A3B-exl3-4.00bpw` (40 layers, MoE, GatedDeltaNet + full-attn) |
| **Reference** | exllamav3 on CUDA, `attn_mode=flash_attn_nc`, fixed seed, seq_len 512 |
| **Candidate** | PonyExl3 MLX, `engine="exl3"` (exact trellis decode, Metal) |
| **Tooling** | `ponyexl3/reference/compare_trace.py` (module replay, scale-aware) |
| **Result** | top-1 argmax matches; `max\|Δ\|` logits ≈ 0.83 (rel rms 4.6%); per-module rel drift ≤ 7.9%, cos ≥ 0.997 |

A CUDA reference bundle (end-to-end logits + per-module trace + per-layer
linear I/O + per-layer MoE routing) was replayed module-by-module through
the MLX engine and compared against every artifact. All checks pass.

---

## 1. Drift anatomy — examined and attributed

The residual stream's magnitude grows ~45× through the stack (rms 0.021 at
layer 0 → 0.86 at layer 39; max|h| 0.45 → 32). Absolute difference therefore
**must** grow even at a constant relative accuracy. Read in scale-aware terms
(`rel = rms(Δ)/rms(h)`), there is no anomaly:

| Module | rms(h) | rel rms(Δ)/rms(h) | cos | Reading |
|--------|--------|-------------------|-----|---------|
| `embed_tokens` | 0.013 | **0** | 1.000000 | bit-exact |
| layer 0 | 0.021 | 0.20% | 0.999998 | first composed block, fp16-noise class |
| layers 1–12 | 0.03–0.06 | 1.0 → 1.9% | ≥ 0.9998 | smooth cross-impl compounding |
| layer 13 | 0.065 | 5.3% | 0.9986 | routing-flip inflection (§4) |
| layers 13–39 | 0.07–0.86 | **4–7% plateau** | ≥ 0.998 | chaos saturates; does not grow |
| layer 39 (full-attn) | 0.86 | 4.9% | 0.9989 | in-band |
| `norm` | 1.56 | 7.9% | 0.9969 | RMSNorm, ×1.6 relative |
| `lm_head` logits | 2.15 | 4.6% | 0.9991 | inherited, not added |

The largest **absolute** single-layer step (layer 39, full attention) and the
norm "amplification" are scale effects: layer 39 is squarely in the relative
plateau, and the norm scales the difference by ~1.6× relative (not the ~3×
the raw numbers suggest). No layer type shows a systematic relative excess.

---

## 2. MLX is schedule-deterministic (internal noise floor = 0)

Replaying the same 512 tokens as **one chunk vs two 256-token chunks** through
the cache (different attention spans and recurrent-scan boundaries — identical
math, different kernel schedules) produces **bit-identical output at all 42
modules**. The MLX engine's result is independent of how a prompt is chunked.

Consequence: none of the CUDA↔MLX gap is candidate-side scheduling wobble. All
of it is cross-implementation — flash-attention vs Metal SDPA, the fused vs
chunked GatedDeltaNet scan, and EXL3 GEMM vs Metal trellis-GEMV reduction
orders — i.e. the irreducible fp16 floor between two distinct exact engines.

---

## 3. Isolated components match on identical inputs

Feeding the **CUDA-captured activations** through the MLX path (no accumulated
context — pure component test) confirms each subsystem in isolation:

| Component | max \|Δ\| | Note |
|-----------|-----------|------|
| `lm_head` (EXL3 GEMV on CUDA `x`) | 0.0156 | ≈ 1 fp16 ulp at this scale |
| `shared_expert.gate_proj` | 0.0020 | fp16-noise |
| `self_attn.q_proj` | 0.0078 | fp16-noise |
| CUDA fast-kernel vs CUDA `reconstruct` (same `x`) | ≤ 0.008 | the two CUDA paths agree |
| RMSNorm on identical layer-39 hidden | rel 1.8e-3 | ≈ 2 fp16 ulp — pure rounding |
| Router gate = same function | exact | semantics verified in source (§4) |

Every isolated EXL3 linear is orders of magnitude below the end-to-end logit
difference. The trellis decode and GEMV/GEMM are confirmed correct.

---

## 4. MoE routing near-ties are the amplifier (expected, not a bug)

Router semantics were checked against the reference source and **match**:
exllamav3's `routing_std` (top-k on raw logits, then softmax restricted to the
selected k) is numerically equivalent to the MLX path (full softmax → top-k →
renormalize over top-k, `norm_topk_prob=True`). Reference routing weights sum
to 1.0, consistent with the renormalized form.

Given each side's own (already slightly different) hidden states, the **top-8
expert set agrees on 28/40 layers**; the 12 that differ are near-ties (max
routing-weight difference on agreeing layers is 1.2e-2, fp16-class). The first
flip is at layer 9; the relative-drift step at layer 13 and the subsequent
plateau are exactly this regime — expert flips re-randomize the trajectory
rather than accumulate, which is why drift **saturates** instead of compounding.

This was confirmed by a control: replaying the expert weighted-sum in fp32
(strictly more accurate locally) made layer 9–12 agreement **2–2.7× worse**,
washing back to parity by layer 39. A system whose agreement degrades under a
more-accurate local change is at its numerics floor — there is no systematic
error to remove. Routing divergence is a downstream symptom of the fp16 floor,
not a routing defect.

---

## 5. One characterized dtype convention difference (benign)

The reference carries its **residual stream in fp32** (the exported layer
hiddens are off the fp16 grid; `x += y` accumulates on fp32 `x`, and the MoE
expert sum joins without an intermediate half-round), whereas the MLX engine
follows the mlx-lm convention of rounding the stream to fp16 each layer.

Mirroring the reference contract (`compare_trace.py --fp32-residual`) improves
late-stack agreement by 1.5–3× exactly where the stream magnitude is largest:

| Region | fp16-residual rel | fp32-residual rel |
|--------|-------------------|-------------------|
| layers 0–12 | 0.2–1.9% | ~same |
| layers 13–30 | 4.5–6.3% | ~same |
| **layers 31–38** | 3.9–6.7% | **1.9–4.1%** |
| logits `max\|Δ\|` | 0.826 | 0.820 |

End-to-end logits barely move because the final full-attention layer re-injects
~2× and the tail amplifies whatever remains (a 2-ulp norm difference grows ~×13
through the wide `lm_head`). This is a known, optional convention choice, not a
correctness issue; fp32-residual mode is available as a future engine option.

---

## 6. The argmax tie is one ulp deep

On this fixed-seed **random-token** probe, the reference top-2 gap is
`4.5898 − 4.5742 = 0.0156` — exactly one fp16 ulp at that magnitude. Random
tokens place the model far out of distribution and maximize such near-ties.
"Top-1 argmax matches" is therefore reported but treated as weak evidence; the
correct cross-engine quality metric is **behavioral KLD on real text**, which
sits inside the 4-bit-vs-8-bit quantization-noise band (harness:
`tools/dev/drift_eval.py`).

---

## Acceptance criteria

Bit-exact CUDA↔Metal logits is **not** an attainable goal for two
independently-scheduled exact engines (different reduction orders by design)
and is not used as a gate. The parity gates, all currently green, are:

1. **Same-input component parity** — embeddings bit-exact; isolated EXL3
   linears ≤ 2e-2 on real activations; norm ≤ fp16-noise; router = same
   function in source.
2. **Scale-aware trace parity** — `rel rms(Δ)/rms(h) ≤ 10%` and `cos ≥ 0.99`
   at every module (observed: ≤ 7.9%, ≥ 0.9969).
3. **Behavioral parity** — KLD(reference ‖ candidate) on real text within the
   quantization-noise band; near-tie argmax flips (gap < 2 ulp) not asserted.

`tests/test_reference_parity.py` enforces gate (2) on end-to-end logits; the
bit-exact check is retained as a non-strict `xfail` aspiration.

---

## Reproduction

The reference scripts ship in `ponyexl3/reference/`; the binary reference
bundle (logits/trace/linear-io/moe `.npz`) is generated on a CUDA host and is
not bundled in the repository. With a bundle present:

```bash
uv run python ponyexl3/reference/compare_trace.py \
  <stem>_trace.npz \
  -m /path/to/Qwen3.6-35B-A3B-exl3-4.00bpw \
  --reference <stem>.npz \
  --moe <stem>_moe.npz \
  --noise-floor 2 --tail-check --fp32-residual
```

Flags: `--noise-floor N` (schedule-determinism check), `--tail-check`
(reference last-layer hidden → MLX norm+lm_head), `--fp32-residual` (mirror
the reference residual-stream dtype), `--moe` (top-k expert-set agreement),
`--save` (export the MLX trace for the reference host to verify in reverse).
See `ponyexl3/reference/README.md` for producing a bundle on the CUDA side.
