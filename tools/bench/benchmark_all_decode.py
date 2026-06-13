#!/usr/bin/env python3
"""Run all 27B decode optimization variants in isolated subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

_MODEL = os.environ.get("PONYEXL3_MODEL_27B") or os.environ.get("MODEL")
_MTP = os.environ.get("PONYEXL3_MTP_DIR") or os.environ.get("MTP")
_STEPS = os.environ.get("STEPS", "32")
_WARMUP = os.environ.get("WARMUP", "4")

# id, description, extra env
VARIANTS: list[tuple[str, str, dict[str, str]]] = [
    ("baseline", "exl3 default (sibling fusion only)", {}),
    ("fuse_post", "EXL3_FUSE_POST=1", {"EXL3_FUSE_POST": "1"}),
    ("fuse_had", "EXL3_FUSE_POST + EXL3_FUSE_HAD", {
        "EXL3_FUSE_POST": "1", "EXL3_FUSE_HAD": "1",
    }),
    ("gemv_simd_off", "FUSE_POST + EXL3_GEMV_SIMD=0 (v10 fallback)", {
        "EXL3_FUSE_POST": "1", "EXL3_GEMV_SIMD": "0",
    }),
    ("gemv_lut", "FUSE_POST + EXL3_GEMV_LUT=1", {
        "EXL3_FUSE_POST": "1", "EXL3_GEMV_LUT": "1",
    }),
    ("mlp_mono_fast", "FUSE_POST + MLP monolith (fast kernel)", {
        "EXL3_FUSE_POST": "1", "EXL3_MLP_MONO": "1", "EXL3_MLP_KERNEL": "fast",
    }),
    ("mlp_mono_moe", "FUSE_POST + MLP monolith (moe kernel)", {
        "EXL3_FUSE_POST": "1", "EXL3_MLP_MONO": "1", "EXL3_MLP_KERNEL": "moe",
    }),
    ("mtp_k3", "FUSE_POST + MTP speculative k=3", {
        "EXL3_FUSE_POST": "1", "USE_MTP": "1", "NUM_DRAFT": "3",
    }),
    ("mtp_k2", "FUSE_POST + MTP speculative k=2", {
        "EXL3_FUSE_POST": "1", "USE_MTP": "1", "NUM_DRAFT": "2",
    }),
]

_WORKER = textwrap.dedent(
    '''
    import json, os, sys, time

    import mlx.core as mx
    from mlx_lm.utils import load_tokenizer
    from ponyexl3.mlx.generate import stream_generate, speculative_stream_generate
    from ponyexl3.mlx.layer_state import clear_layer_caches
    from ponyexl3.mlx.model import load_model
    from ponyexl3.mlx.mtp import load_mtp

    model_dir = sys.argv[1]
    steps = int(sys.argv[2])
    warmup = int(sys.argv[3])
    mtp_dir = sys.argv[4]

    for k, v in json.loads(sys.argv[5]).items():
        os.environ[k] = v

    model, config = load_model(model_dir, engine="exl3", warm=True, verbose=False)
    tok = load_tokenizer(model_dir)
    prompt = "Write a short Python function to compute fibonacci."
    if getattr(tok, "chat_template", None):
        prompt_ids = list(tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True))
    else:
        prompt_ids = list(tok.encode(prompt))

    use_mtp = os.environ.get("USE_MTP") == "1"
    mtp = None
    num_draft = int(os.environ.get("NUM_DRAFT", "3"))
    if use_mtp:
        mtp = load_mtp(model_dir, config, mtp_dir)

    def run_gen():
        if mtp is not None:
            return speculative_stream_generate(
                model, mtp, prompt_ids, max_tokens=steps, num_draft=num_draft)
        return stream_generate(model, prompt_ids, max_tokens=steps)

    for _ in range(warmup):
        list(run_gen())
        clear_layer_caches()

    t0 = time.perf_counter()
    n = 0
    for _ in run_gen():
        n += 1
    mx.synchronize()
    dt = time.perf_counter() - t0
    print(json.dumps({"tok_s": n / dt if dt else 0.0, "steps": n, "seconds": dt}))
    '''
)


def run_variant(model_dir: str, mtp_dir: str | None, vid: str, desc: str, env: dict[str, str]) -> dict:
    worker = Path(__file__).resolve().parent / "_bench_decode_worker.py"
    worker.write_text(_WORKER)
    full_env = os.environ.copy()
    for k in (
        "EXL3_FUSE_POST", "EXL3_FUSE_HAD", "EXL3_GEMV_SIMD", "EXL3_GEMV_LUT",
        "EXL3_MLP_MONO", "EXL3_MLP_KERNEL", "USE_MTP", "NUM_DRAFT",
    ):
        full_env.pop(k, None)
    full_env.update(env)
    cmd = [
        sys.executable,
        str(worker),
        model_dir,
        _STEPS,
        _WARMUP,
        mtp_dir or "",
        json.dumps(env),
    ]
    print(f"  [{vid}] {desc}...", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(_ROOT),
        env=full_env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-800:]
        return {"id": vid, "description": desc, "env": env, "error": err}
    line = proc.stdout.strip().splitlines()[-1]
    payload = json.loads(line)
    payload.update({"id": vid, "description": desc, "env": env})
    print(f"    {payload['tok_s']:.2f} tok/s", flush=True)
    return payload


def main() -> int:
    model_dir = os.environ.get("PONYEXL3_MODEL_27B") or os.environ.get("MODEL")
    if not model_dir or not Path(model_dir).is_dir():
        print(
            "set PONYEXL3_MODEL_27B or MODEL to an EXL3 checkpoint directory",
            file=sys.stderr,
        )
        return 1
    mtp_dir = _MTP if _MTP and Path(_MTP).is_dir() else None
    print(f"Model: {model_dir}")
    print(f"Decode steps: {_STEPS}  warmup: {_WARMUP}")
    if mtp_dir:
        print(f"MTP: {mtp_dir}")
    elif any(env.get("USE_MTP") == "1" for _, _, env in VARIANTS):
        print("MTP: (unset — MTP variants will fail without PONYEXL3_MTP_DIR)", file=sys.stderr)

    results = [
        run_variant(model_dir, mtp_dir, vid, desc, env) for vid, desc, env in VARIANTS
    ]
    ok = [r for r in results if "error" not in r and r.get("tok_s", 0) > 0]
    base = next((r for r in ok if r["id"] == "baseline"), ok[0] if ok else None)
    base_tps = base["tok_s"] if base else 0.0

    print()
    print("| variant | tok/s | vs baseline | notes |")
    print("|---------|-------|-------------|-------|")
    for r in results:
        if "error" in r:
            print(f"| `{r['id']}` | ERROR | — | {r['error'][:60]}... |")
            continue
        vs = f"{r['tok_s'] / base_tps:.3f}x" if base_tps else "—"
        print(f"| `{r['id']}` | **{r['tok_s']:.2f}** | {vs} | {r['description']} |")

    if ok:
        best = max(ok, key=lambda r: r["tok_s"])
        print()
        print(f"Best: `{best['id']}` at {best['tok_s']:.2f} tok/s")

    out = _ROOT / "bench" / "results" / f"decode_sweep_{Path(model_dir).name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model_dir, "results": results}, indent=2) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
