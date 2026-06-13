#!/usr/bin/env python3
"""Benchmark EXL3_MLP_MONO on/off for decode throughput."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mlx.core as mx

from ponyexl3.mlx.exl3_mlp_monolith import EXL3MLPMonolith
from ponyexl3.mlx.generate import stream_generate
from ponyexl3.mlx.layer_state import clear_layer_caches
from ponyexl3.mlx.model import load_model
from mlx_lm.utils import load_tokenizer


def bench(model_dir: str, *, mono: bool, steps: int = 32, warmup: int = 4) -> dict:
    os.environ["EXL3_FUSE_POST"] = "1"
    os.environ["EXL3_MLP_MONO"] = "1" if mono else "0"
    if mono and "EXL3_MLP_KERNEL" not in os.environ:
        os.environ["EXL3_MLP_KERNEL"] = "fast"
    model, _ = load_model(model_dir, engine="exl3", warm=True, verbose=False)
    n_mono = sum(
        1 for layer in model.layers if isinstance(getattr(layer, "mlp", None), EXL3MLPMonolith)
    )
    tok = load_tokenizer(model_dir)
    prompt = "Write a short Python function to compute fibonacci."
    if getattr(tok, "chat_template", None):
        prompt_ids = list(
            tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
        )
    else:
        prompt_ids = list(tok.encode(prompt))

    for _ in range(warmup):
        list(stream_generate(model, prompt_ids, max_tokens=1))
        clear_layer_caches()

    t0 = time.perf_counter()
    n = 0
    for _ in stream_generate(model, prompt_ids, max_tokens=steps):
        n += 1
    mx.synchronize()
    dt = time.perf_counter() - t0
    del model
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    clear_layer_caches()
    return {
        "mono": mono,
        "mlp_monolith_layers": n_mono,
        "tok_s": n / dt if dt else 0.0,
        "steps": n,
        "seconds": dt,
    }


def main() -> int:
    model_dir = os.environ.get("PONYEXL3_MODEL_27B") or os.environ.get("MODEL")
    if not model_dir or not Path(model_dir).is_dir():
        print(
            "set PONYEXL3_MODEL_27B or MODEL to an EXL3 checkpoint directory",
            file=sys.stderr,
        )
        return 1
    steps = int(os.environ.get("STEPS", "32"))
    base = bench(model_dir, mono=False, steps=steps)
    os.environ["EXL3_MLP_KERNEL"] = "fast"
    fast = bench(model_dir, mono=True, steps=steps)
    os.environ["EXL3_MLP_KERNEL"] = "moe"
    moe = bench(model_dir, mono=True, steps=steps)
    print(f"Model: {model_dir}")
    print(f"Decode steps: {steps}")
    print()
    print(f"  baseline (EXL3_MLP_MONO=0):     {base['tok_s']:.2f} tok/s")
    print(f"  monolith fast kernel:           {fast['tok_s']:.2f} tok/s  ({fast['mlp_monolith_layers']} layers)")
    print(f"  monolith moe kernel (experimental): {moe['tok_s']:.2f} tok/s")
    if base["tok_s"]:
        print(f"  fast vs baseline:  {fast['tok_s']/base['tok_s']:.3f}x")
        print(f"  moe vs baseline:   {moe['tok_s']/base['tok_s']:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
