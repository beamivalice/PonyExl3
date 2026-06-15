#!/usr/bin/env python3
"""Replay a CUDA per-module trace (.npz from export_trace.py) through MLX.

Mirrors ``Qwen3_5TextModel.__call__`` module-by-module (embed_tokens, each
``layers.N``, ``norm``, then ``lm_head``), capturing the last-row hidden after
every module, and compares against the CUDA export. Reports absolute drift
(max|d|, rms) AND scale-aware drift (rms(d)/rms(h)) — late layers have much
larger hidden magnitudes, so absolute drift alone overstates them.

Discriminating extras:

--noise-floor N   second MLX replay with the same tokens prefilled in N chunks
                  through the cache machinery. Mathematically identical,
                  different kernel schedules/accumulation order — its drift vs
                  the single-chunk replay is the platform's intrinsic
                  reordering noise. CUDA-vs-MLX drift within a small multiple
                  of this floor is numerics, not a bug.
--tail-check      feed the CUDA layer-39 hidden through MLX norm + lm_head and
                  compare to the CUDA logits — isolates everything after the
                  layer stack.
--moe NPZ         compare MLX top-k expert sets/weights per MoE layer against
                  export_moe_routing.py output (FYI: sets diverge once hidden
                  states drift; only same-hidden disagreement implies a bug).
--save OUT.npz    write the MLX trace in export_trace.py's key convention for
                  the CUDA team.

Example:
    uv run python ponyexl3/reference/compare_trace.py \
      ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win_trace.npz \
      -m /path/to/Qwen3.6-35B-A3B-exl3-4.00bpw \
      --reference ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win.npz \
      --moe ponyexl3/reference/qwen3.6-35B-A3B-exl3-4.00bpw_win_moe.npz \
      --noise-floor 2 --tail-check
"""

from __future__ import annotations

from typing import Any

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections.abc import Callable

from ponyexl3.types import MlxLmModel

import numpy as np


def _stats(ref: np.ndarray, cand: np.ndarray) -> dict[str, Any]:
    ref64: np.ndarray = np.reshape(ref, (-1,)).astype(np.float64)
    cand64: np.ndarray = np.reshape(cand, (-1,)).astype(np.float64)
    d = cand64 - ref64
    rms_ref = float(np.sqrt((ref64**2).mean()) + 1e-30)
    denom = float(np.linalg.norm(ref64) * np.linalg.norm(cand64) + 1e-30)
    ref_u32: np.ndarray = ref64.astype(np.float32).view(np.uint32)
    cand_u32: np.ndarray = cand64.astype(np.float32).view(np.uint32)
    return {
        "max_abs": float(np.abs(d).max()),
        "rms": float(np.sqrt((d**2).mean())),
        "rms_ref": rms_ref,
        "max_ref": float(np.abs(ref64).max()),
        "rel_rms": float(np.sqrt((d**2).mean()) / rms_ref),
        "cos": float(np.dot(ref64, cand64) / denom),
        "bit_exact": bool(ref_u32.tobytes() == cand_u32.tobytes()),
    }


def _moe_key(layer_idx: int) -> str:
    return f"model__language_model__layers__{layer_idx}__mlp"


class _RouterTap:
    """Records the last (inds, scores) each EXL3MoEBlock router emits."""

    def __init__(self) -> None:
        self.records: dict[int, tuple[Any, Any]] = {}

    def install(self, model: MlxLmModel) -> int:
        from ponyexl3.mlx.exl3_moe import EXL3Gemma4MoEBlock, EXL3MoEBlock

        n = 0
        for li, layer in enumerate(model.language_model.model.layers):
            block = getattr(layer, "mlp", None)
            if isinstance(block, EXL3MoEBlock):
                orig = block._router()  # pyright: ignore[reportPrivateUsage]
                block._router_fn = self._wrap(li, orig)  # pyright: ignore[reportPrivateUsage]
                n += 1
                continue
            block = getattr(layer, "router", None)
            if isinstance(block, EXL3Gemma4MoEBlock):
                orig = block._route
                block._route = self._wrap_route(li, orig)  # type: ignore[method-assign]
                n += 1
        return n

    def _wrap_route(
        self, li: int, orig: Callable[[Any], tuple[Any, Any]]
    ) -> Callable[[Any], tuple[Any, Any]]:
        def _fn(h: Any) -> tuple[Any, Any]:
            inds, scores = orig(h)
            self.records[li] = (inds, scores)
            return inds, scores

        return _fn

    def _wrap(self, li: int, orig: Callable[..., tuple[Any, Any]]) -> Callable[..., tuple[Any, Any]]:
        def _fn(*args: Any) -> tuple[Any, Any]:
            inds, scores = orig(*args)
            self.records[li] = (inds, scores)
            return inds, scores

        return _fn

    def last_row(self, li: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        inds, scores = self.records[li]
        return (
            np.array(inds[0, -1, :top_k]).astype(np.int64),
            np.array(scores[0, -1, :top_k]).astype(np.float32),
        )


def replay(
    model: MlxLmModel,
    input_ids: np.ndarray,
    *,
    chunks: int = 1,
    tap: _RouterTap | None = None,
    fp32_residual: bool = False,
) -> tuple[dict[str, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Forward input_ids capturing last-row fp32 hidden after each module.

    chunks > 1 splits the prompt through the cache machinery (the generation
    prefill path) — identical math, different accumulation order.

    fp32_residual=True mirrors exllamav3's dtype contract instead of
    mlx_lm's: the residual stream accumulates in fp32 (norms consume fp32,
    emit fp16 for the modules; the MoE expert weighted-sum joins the stream
    in fp32 without a half round). Measured 2026-06-13: improves late-layer
    (L31-38) trace agreement 1.5-3x; end-to-end logits barely move.
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    if fp32_residual and chunks != 1:
        raise ValueError("fp32_residual replay supports chunks=1 only")

    lm = model.language_model
    inner = lm.model
    cache = lm.make_cache()

    ids = mx.array(input_ids.astype(np.int64))
    n = ids.shape[1]
    bounds = [round(i * n / chunks) for i in range(chunks)] + [n]

    captures: dict[str, np.ndarray] = {}
    logits: Any = None
    for start, stop in zip(bounds[:-1], bounds[1:]):
        h = inner.embed_tokens(ids[:, start:stop])
        last = stop == n
        if last:
            captures["model__language_model__embed_tokens"] = np.array(
                h[:, -1:, :].astype(mx.float32)
            )
        fa_mask = create_attention_mask(h, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
        if fp32_residual:
            h = h.astype(mx.float32)
        for li, (layer, c) in enumerate(zip(inner.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            if fp32_residual:
                y = layer.input_layernorm(h).astype(mx.float16)
                attn = layer.linear_attn if layer.is_linear else layer.self_attn
                h = h + attn(y, mask, c).astype(mx.float32)
                y2 = layer.post_attention_layernorm(h).astype(mx.float16)
                h = h + _moe_fp32(layer.mlp, y2)
            else:
                h = layer(h, mask=mask, cache=c)
            if last:
                cap = h[:, -1:, :].astype(mx.float32)
                mx.eval(cap)
                captures[f"model__language_model__layers__{li}"] = np.array(cap)
        h = inner.norm(h)
        if fp32_residual:
            h = h.astype(mx.float16)
        if last:
            captures["model__language_model__norm"] = np.array(
                h[:, -1:, :].astype(mx.float32)
            )
        logits = lm.lm_head(h[:, -1:, :]).astype(mx.float32)
        mx.eval(logits)
    captures["logits"] = np.array(logits[0, 0, :])  # pyright: ignore[reportOptionalSubscript]

    moe = {}
    if tap is not None:
        for li in sorted(tap.records):
            moe[li] = tap.last_row(li, top_k=8)
    return captures, moe


def _moe_fp32(block: Any, x16: Any) -> Any:
    """EXL3MoEBlock forward with the weighted expert sum in fp32 — the
    exllamav3 BlockSparseMLP accumulation contract (final_hidden_states is
    torch.float; it joins the residual without rounding to half)."""
    import mlx.core as mx

    inds, scores = block._router()(
        x16, block.gate.weight, block.shared_expert_gate.weight
    )
    y = block.switch_mlp(x16, inds)
    return (y.astype(mx.float32) * scores[..., None].astype(mx.float32)).sum(axis=-2)


def _print_table(title: str, rows: list[tuple[str, dict[str, Any]]]) -> None:
    print(f"\n## {title}")
    print(
        f"{'module':<14} {'max|d|':>10} {'rms(d)':>10} {'rms(h)':>9} "
        f"{'max|h|':>9} {'rel_rms':>9} {'cos':>10}"
    )
    for name, s in rows:
        print(
            f"{name:<14} {s['max_abs']:>10.3e} {s['rms']:>10.3e} {s['rms_ref']:>9.3g} "
            f"{s['max_ref']:>9.3g} {s['rel_rms']:>9.3e} {s['cos']:>10.7f}"
        )


def _short(key: str) -> str:
    if key.endswith("embed_tokens"):
        return "embed"
    if key.endswith("norm"):
        return "norm"
    if key == "logits":
        return "lm_head"
    return "L" + key.rsplit("__", 1)[-1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("trace", type=Path, help=".npz from export_trace.py")
    ap.add_argument("-m", "--model-dir", required=True)
    ap.add_argument("--reference", type=Path, default=None,
                    help="end-to-end .npz (export_reference.py) for the logits row")
    ap.add_argument("--moe", type=Path, default=None,
                    help=".npz from export_moe_routing.py")
    ap.add_argument("--noise-floor", type=int, default=0, metavar="N",
                    help="also replay prefill in N chunks; report MLX-vs-MLX drift")
    ap.add_argument("--tail-check", action="store_true",
                    help="CUDA L39 hidden -> MLX norm+lm_head vs CUDA logits")
    ap.add_argument("--fp32-residual", action="store_true",
                    help="also replay with exllamav3's fp32-residual dtype "
                         "contract and print its CUDA-vs-MLX table")
    ap.add_argument("--engine", default="exl3")
    ap.add_argument("--save", type=Path, default=None,
                    help="write the MLX trace npz (export_trace.py key convention)")
    args = ap.parse_args()

    trace = np.load(args.trace, allow_pickle=True)
    input_ids = trace["input_ids"]
    layer_ids = [
        int(k.rsplit("__", 1)[-1])
        for k in trace.files
        if k.startswith("model__language_model__layers__")
        and k.rsplit("__", 1)[-1].isdigit()
    ]
    n_layers = max(layer_ids) + 1 if layer_ids else 0
    print(f"trace:   {args.trace}  ({n_layers} layers, row_slice={trace['row_slice']})")
    print(f"model:   {args.model_dir}")

    ref_logits = None
    if args.reference is not None:
        ref = np.load(args.reference, allow_pickle=True)
        ref_logits = ref["logits"].astype(np.float32)
        if ref_logits.ndim == 2:
            ref_logits = ref_logits[-1]

    from ponyexl3.mlx.model import load_model

    model, _ = load_model(args.model_dir, engine=args.engine, warm=False, verbose=False)

    tap = _RouterTap()
    n_moe = tap.install(model)
    cap, moe = replay(model, input_ids, tap=tap)

    # --- CUDA vs MLX, per module -------------------------------------------
    rows = []
    for key in trace.files:
        if not key.startswith("model__language_model"):
            continue
        if key not in cap:
            continue
        rows.append((_short(key), _stats(trace[key], cap[key])))
    if ref_logits is not None:
        rows.append(("lm_head", _stats(ref_logits, cap["logits"])))
    _print_table("CUDA vs MLX (single-chunk replay)", rows)

    # --- MoE routing agreement ---------------------------------------------
    if args.moe is not None and moe:
        moe_ref = np.load(args.moe, allow_pickle=True)
        agree, diverged = 0, []
        weight_dmax = 0.0
        for li in sorted(moe):
            k_e = f"{_moe_key(li)}__experts"
            if k_e not in moe_ref.files:
                continue
            ref_e = set(int(x) for x in moe_ref[k_e].reshape(-1))
            ref_w = moe_ref[f"{_moe_key(li)}__weights"].reshape(-1)
            mlx_e, mlx_w = moe[li]
            if set(int(x) for x in mlx_e) == ref_e:
                agree += 1
                by_id = dict(zip((int(x) for x in mlx_e), mlx_w))
                ref_by_id = dict(
                    zip((int(x) for x in moe_ref[k_e].reshape(-1)), ref_w)
                )
                weight_dmax = max(
                    weight_dmax,
                    max(abs(by_id[e] - ref_by_id[e]) for e in ref_by_id),
                )
            else:
                diverged.append(li)
        print(f"\n## MoE routing (last token, top-8 sets, {n_moe} layers tapped)")
        print(f"same expert set: {agree}/{len(moe)} layers; "
              f"max weight |d| on agreeing layers: {weight_dmax:.3e}")
        if diverged:
            print(f"diverged layers: {diverged}")
            print("(expected once hiddens drift — see drifts_investigation.md)")

    # --- tail check ----------------------------------------------------------
    if args.tail_check:
        import mlx.core as mx

        last_layer = f"model__language_model__layers__{n_layers - 1}"
        h39 = mx.array(trace[last_layer]).astype(mx.float16)
        lm = model.language_model
        logits_tail = lm.lm_head(lm.model.norm(h39)).astype(mx.float32)
        mx.eval(logits_tail)
        tail = np.array(logits_tail[0, 0, :])
        rows = []
        if f"model__language_model__norm" in trace.files:
            norm_mlx = np.array(
                lm.model.norm(h39).astype(mx.float32)
            )
            rows.append(("norm", _stats(trace["model__language_model__norm"], norm_mlx)))
        if ref_logits is not None:
            rows.append(("lm_head", _stats(ref_logits, tail)))
            print(f"\ntail top-1: cuda={int(ref_logits.argmax())} "
                  f"mlx-tail={int(tail.argmax())}")
        _print_table(
            f"tail check: CUDA {_short(last_layer)} hidden -> MLX norm+lm_head", rows
        )

    # --- fp32-residual mirror -------------------------------------------------
    if args.fp32_residual:
        cap32, _ = replay(model, input_ids, fp32_residual=True)
        rows = []
        for key in trace.files:
            if key not in cap or key not in cap32:
                continue
            rows.append((_short(key), _stats(trace[key], cap32[key])))
        if ref_logits is not None:
            rows.append(("lm_head", _stats(ref_logits, cap32["logits"])))
        _print_table(
            "CUDA vs MLX (fp32-residual replay — exllamav3 dtype contract)", rows
        )

    # --- noise floor ----------------------------------------------------------
    if args.noise_floor and args.noise_floor > 1:
        cap2, _ = replay(model, input_ids, chunks=args.noise_floor)
        rows_nf, rows_cuda2 = [], []
        for key in trace.files:
            if key not in cap or key not in cap2:
                continue
            rows_nf.append((_short(key), _stats(cap[key], cap2[key])))
            rows_cuda2.append((_short(key), _stats(trace[key], cap2[key])))
        rows_nf.append(("lm_head", _stats(cap["logits"], cap2["logits"])))
        if ref_logits is not None:
            rows_cuda2.append(("lm_head", _stats(ref_logits, cap2["logits"])))
        _print_table(
            f"MLX self noise floor (1 chunk vs {args.noise_floor} chunks — "
            "identical math, reordered accumulation)",
            rows_nf,
        )
        _print_table(
            f"CUDA vs MLX ({args.noise_floor}-chunk replay)", rows_cuda2
        )

    if args.save:
        out = {
            "input_ids": input_ids,
            "row_slice": np.array("last:1"),
            "platform": np.array("mac"),
            "engine": np.array(args.engine),
        }
        out.update({k: v for k, v in cap.items() if k != "logits"})
        out["logits"] = cap["logits"][np.newaxis, :]
        np.savez(args.save, allow_pickle=False, **out)
        print(f"\nwrote MLX trace to {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
