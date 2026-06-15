"""EAGLE-3 draft head (SpecForge/sglang export) as an MLX drafter.

Same drafting contract as the MTP head (``mtp.py``): per accepted position
the drafter is primed with (feature at pos i, token at pos i+1) pairs and
then chains speculative steps on its own output hiddens. Differences:

- Features are NOT the final hidden: ``fuse()`` concatenates the target's
  residual stream after layers ``aux_ids`` (here 3/31/59 — all
  full-attention layers), each RMS-normed (``fcs.{0,1,2}``), projected by
  ``fc`` (15360 -> 5120).
- The single decoder layer attends over cat(norm(emb), norm(hidden)) — a
  2*hidden (10240) input — llama-style GQA (24q/4kv, head_dim 256, rope
  theta 1e7, no qk norms).
- Its OWN lm_head over a 32k draft vocab; ``d2t`` holds OFFSETS
  (target_id = draft_id + d2t[draft_id], verified against t2d).

Drafts are verify-gated, so this module never needs to be exact — it runs
fp16 (and may be further quantized like ``--draft-w4``).
"""

from __future__ import annotations

from typing import Any

from ponyexl3.mlx.weights import load_safetensors
from ponyexl3.types import KvCache, MlxLmModel

import json
import os

import mlx.core as mx
import mlx.nn as nn


class Eagle3Draft(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        cfg = json.load(open(os.path.join(path, "config.json")))
        w = load_safetensors(os.path.join(path, "model.safetensors"))
        w = {k: (v.astype(mx.float16) if v.dtype == mx.bfloat16 else v) for k, v in w.items()}

        self.aux_ids = tuple(cfg["eagle_config"]["eagle_aux_hidden_state_layer_ids"])
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.rope_base = float(cfg["rope_parameters"]["rope_theta"])

        def lin(name: str) -> nn.Linear:
            m = nn.Linear(1, 1, bias=False)
            m.weight = w[name]
            return m

        self.fc = lin("fc.weight")
        self._w_fcs = [w[f"fcs.{i}.weight"] for i in range(3)]
        self.q_proj = lin("layers.0.self_attn.q_proj.weight")
        self.k_proj = lin("layers.0.self_attn.k_proj.weight")
        self.v_proj = lin("layers.0.self_attn.v_proj.weight")
        self.o_proj = lin("layers.0.self_attn.o_proj.weight")
        self.gate_proj = lin("layers.0.mlp.gate_proj.weight")
        self.up_proj = lin("layers.0.mlp.up_proj.weight")
        self.down_proj = lin("layers.0.mlp.down_proj.weight")
        self.lm_head = lin("lm_head.weight")
        self._w_input_ln = w["layers.0.input_layernorm.weight"]
        self._w_hidden_ln = w["layers.0.hidden_norm.weight"]
        self._w_post_ln = w["layers.0.post_attention_layernorm.weight"]
        self._w_norm = w["norm.weight"]
        # d2t holds offsets: target_id = draft_id + d2t[draft_id]
        self._d2t = (w["d2t"] + mx.arange(w["d2t"].shape[0])).astype(mx.int32)
        mx.eval(self.parameters(), self._d2t, *self._w_fcs)

    def fuse(self, aux: list[mx.array]) -> mx.array:
        """(B, S, H) x3 target residual streams -> (B, S, H) draft features."""
        parts = [
            mx.fast.rms_norm(a.astype(mx.float16), self._w_fcs[i], self.eps)
            for i, a in enumerate(aux)
        ]
        return self.fc(mx.concatenate(parts, axis=-1))

    def __call__(self, emb: mx.array, prev_hidden: mx.array, cache: KvCache) -> mx.array:
        """(B, S, H) token embeddings + features/hiddens -> next hidden."""
        from mlx_lm.models.base import create_attention_mask

        B, S, _ = emb.shape
        residual = prev_hidden.astype(mx.float16)
        x = mx.concatenate(
            [
                mx.fast.rms_norm(emb.astype(mx.float16), self._w_input_ln, self.eps),
                mx.fast.rms_norm(residual, self._w_hidden_ln, self.eps),
            ],
            axis=-1,
        )
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        off = cache.offset if cache is not None else 0
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=off)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=off)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        mask = create_attention_mask(x, cache) if S > 1 else None
        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        o = o.transpose(0, 2, 1, 3).reshape(B, S, -1)
        h = residual + self.o_proj(o)
        r2 = h
        hn = mx.fast.rms_norm(h, self._w_post_ln, self.eps)
        h = r2 + self.down_proj(nn.silu(self.gate_proj(hn)) * self.up_proj(hn))
        return h

    def head_input(self, h: mx.array) -> mx.array:
        return mx.fast.rms_norm(h, self._w_norm, self.eps)

    def draft_token(self, h_last: mx.array) -> mx.array:
        """(B, 1, H) hidden -> (1,) TARGET-vocab token id (greedy)."""
        logits = self.lm_head(self.head_input(h_last))
        return self._d2t[mx.argmax(logits[0, -1])].reshape(1)

    @property
    def d2t(self) -> mx.array:
        """Draft-vocab index -> target token id (offsets pre-applied)."""
        return self._d2t

    def draft_logits(self, h_last: mx.array) -> mx.array:
        """(B, 1, H) -> (V_draft,) raw draft-vocab (32k) logits. ``d2t`` maps the
        index to a target id; used for temperature-correct sampling, where the
        draft ``q`` is scattered into the target vocab via ``d2t``."""
        return self.lm_head(self.head_input(h_last))[0, -1]

    def quantize_draft(self, *, bits: int = 4, group_size: int = 64) -> None:
        """Lossy-quantize the whole drafter (verify-gated, output-exact)."""
        for name in (
            "fc", "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head",
        ):
            mod = getattr(self, name)
            setattr(
                self,
                name,
                nn.QuantizedLinear.from_linear(mod, group_size=group_size, bits=bits),
            )
        mx.eval(self.parameters())


class AuxTrace:
    """Capture the residual stream after the target's aux layers.

    Wraps the DecoderLayer class __call__ with an instance filter (dunder
    lookup is type-level, so per-instance wrapping won't fire)."""

    def __init__(self, model: MlxLmModel, aux_ids: list[int]) -> None:
        self._layers = [model.layers[i] for i in aux_ids]
        self.outputs: dict[int, list[mx.array]] = {id(l): [] for l in self._layers}

    def __enter__(self) -> AuxTrace:
        from mlx_lm.models import qwen3_5 as _q5

        self._cls = _q5.DecoderLayer
        self._orig = self._cls.__call__
        trace = self

        def wrapped(mod: Any, *a: Any, **kw: Any) -> mx.array:
            assert trace._orig is not None
            out = trace._orig(mod, *a, **kw)
            sink = trace.outputs.get(id(mod))
            if sink is not None:
                sink.append(out)
            return out

        self._cls.__call__ = wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> bool:
        assert self._cls is not None and self._orig is not None
        self._cls.__call__ = self._orig  # type: ignore[method-assign]
        return False

    def take(self) -> list[mx.array]:
        """One forward's aux streams, concatenated over chunked calls."""
        outs = []
        for l in self._layers:
            chunks = self.outputs[id(l)]
            outs.append(chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1))
            self.outputs[id(l)] = []
        return outs
