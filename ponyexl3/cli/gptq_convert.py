"""GPTQ-convert a 27B EXL3 target to mlx affine w4 (the opt-in speed lever).

Calibrates per-site Hessians (X^T X) on wikitext through the EXACT exl3
model (fusion disabled so member EXL3Linears are hookable), then GPTQ-solves
every EXL3 linear in the decoder stack (lm_head stays exact trellis;
fp16 b/a untouched) and writes mlx-QuantizedLinear shards to
``<model>/.pony_cache/gptq_w{bits}g{group}/``.

Pilot numbers (down_proj L30, output-space rel-RMS on held-out x):
RTN 0.082 / GPTQ g64 0.073 / GPTQ g32 0.064 (damp 0.10, act-order +1% only).

Layers process in chunks (Hessians for 16 blocks ≈ 24 GB fp32); one
calibration pass per chunk. Expect ~1.5-2 h total for 64 blocks.
"""
from __future__ import annotations

from typing import Any
import mlx.nn as nn
import argparse
import os
import time
from pathlib import Path
os.environ['EXL3_FUSE_MIN_MB'] = '999999'
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
import numpy as np
import mlx.core as mx

def block_sites(blk: Any, i: int) -> list[tuple[str, int, list[tuple[str, Any]]]]:
    """(site_key, in_features, [(module_key, module), ...]) per unique input."""
    sites = []
    if getattr(blk, 'is_linear', False):
        a = blk.linear_attn
        sites.append((f'L{i}.in', a.in_proj_qkv.in_features, [(f'model.layers.{i}.linear_attn.in_proj_qkv', a.in_proj_qkv), (f'model.layers.{i}.linear_attn.in_proj_z', a.in_proj_z)]))
        sites.append((f'L{i}.attn_out', a.out_proj.in_features, [(f'model.layers.{i}.linear_attn.out_proj', a.out_proj)]))
    else:
        a = blk.self_attn
        sites.append((f'L{i}.in', a.q_proj.in_features, [(f'model.layers.{i}.self_attn.q_proj', a.q_proj), (f'model.layers.{i}.self_attn.k_proj', a.k_proj), (f'model.layers.{i}.self_attn.v_proj', a.v_proj)]))
        sites.append((f'L{i}.attn_out', a.o_proj.in_features, [(f'model.layers.{i}.self_attn.o_proj', a.o_proj)]))
    m = blk.mlp
    sites.append((f'L{i}.mlp_in', m.gate_proj.in_features, [(f'model.layers.{i}.mlp.gate_proj', m.gate_proj), (f'model.layers.{i}.mlp.up_proj', m.up_proj)]))
    sites.append((f'L{i}.mlp_down', m.down_proj.in_features, [(f'model.layers.{i}.mlp.down_proj', m.down_proj)]))
    return sites

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model')
    ap.add_argument('--bits', type=int, default=4)
    ap.add_argument('--group', type=int, default=64)
    ap.add_argument('--damp', type=float, default=0.1)
    ap.add_argument('--seqs', type=int, default=96)
    ap.add_argument('--seqlen', type=int, default=1024)
    ap.add_argument('--chunk', type=int, default=16, help='layers per pass')
    args = ap.parse_args()
    from datasets import load_dataset
    from mlx_lm.utils import load_tokenizer
    from ponyexl3.mlx.exl3_linear import EXL3Linear
    from ponyexl3.mlx.gptq import gptq_quantize, pack_q, prepare_hinv_u
    from ponyexl3.mlx.model import load_model
    from ponyexl3.mlx.native import public_weight_chunks
    out_dir = os.path.join(args.model, '.pony_cache', f'gptq_w{args.bits}g{args.group}')
    os.makedirs(out_dir, exist_ok=True)
    model, _ = load_model(args.model, engine='exl3', warm=False, verbose=False)
    tokenizer = load_tokenizer(Path(args.model))
    lm = model.language_model
    layers = lm.model.layers
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join((t for t in ds['text'] if t.strip()))  # pyright: ignore[reportArgumentType]
    ids = tokenizer.encode(text)[:args.seqs * args.seqlen]
    seqs = [ids[i:i + args.seqlen] for i in range(0, len(ids) - args.seqlen, args.seqlen)][:args.seqs]
    print(f'calibration: {len(seqs)} x {args.seqlen}; chunks of {args.chunk} layers', flush=True)
    orig = EXL3Linear.__call__
    for c0 in range(0, len(layers), args.chunk):
        chunk = list(range(c0, min(c0 + args.chunk, len(layers))))
        shard_file = os.path.join(out_dir, f'chunk_{c0:03d}.safetensors')
        if os.path.exists(shard_file):
            print(f'chunk {c0}: shard exists, skipping', flush=True)
            continue
        sites = []
        for i in chunk:
            sites.extend(block_sites(layers[i], i))
        accs = {key: mx.zeros((inf, inf), dtype=mx.float32) for key, inf, _ in sites}
        rep = {}
        for key, _, members in sites:
            rep[id(members[0][1])] = key

        def wrapped(mod: nn.Module, x: mx.array) -> mx.array:
            key = rep.get(id(mod))
            if key is not None:
                x2 = x.reshape(-1, x.shape[-1]).astype(mx.float32)
                accs[key] = accs[key] + x2.T @ x2
            return orig(mod, x)  # pyright: ignore[reportArgumentType]
        tic = time.perf_counter()
        EXL3Linear.__call__ = wrapped  # type: ignore[method-assign]
        try:
            for s in seqs:
                cache = lm.make_cache()
                mx.eval(lm.model(mx.array([s]), cache=cache))
                mx.eval(*accs.values())
                del cache
        finally:
            EXL3Linear.__call__ = orig  # type: ignore[method-assign]
        print(f'chunk {c0}: calibration {time.perf_counter() - tic:.0f}s', flush=True)
        shard = {}
        for key, _in_f, members in sites:
            H = np.array(accs[key], dtype=np.float32)
            del accs[key]
            tic = time.perf_counter()
            U, dead = prepare_hinv_u(H, args.damp)
            del H
            for mkey, mod in members:
                W = np.concatenate([np.array(ch, dtype=np.float32) for ch in public_weight_chunks(mod._exl3)], axis=1).T
                W[:, dead] = 0.0
                Q, sc, bi = gptq_quantize(W, hinv_u=U, bits=args.bits, group_size=args.group)
                shard[f'{mkey}.weight'] = mx.array(pack_q(Q, args.bits))
                shard[f'{mkey}.scales'] = mx.array(sc.astype(np.float16))
                shard[f'{mkey}.biases'] = mx.array(bi.astype(np.float16))
                del W, Q
            del U
            print(f'  {key}: solved {len(members)} in {time.perf_counter() - tic:.0f}s', flush=True)
        mx.save_safetensors(shard_file, shard)
        del shard, accs
        mx.clear_cache()
        print(f'chunk {c0}: saved {shard_file}', flush=True)
    print('done', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())