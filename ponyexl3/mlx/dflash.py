"""DFlash block-diffusion drafter (z-lab) — MLX runtime, bf16 or EXL3.

Semantics ported from exllamav3 ``architecture/dflash.py`` +
``modules/arch_specific/dflash.py``:

- The drafter NEVER forwards over the context. Its per-layer KV cache is
  built by projecting fc-fused target features straight through each
  layer's k/v_proj: ``feats = hidden_norm(fc(cat(aux x5)))`` then
  ``k = rope(k_norm(k_proj(feats)))`` / ``v = v_proj(feats)`` at the
  context positions (no input_layernorm, no q side on the feature path).
  Aux features are the target's residual streams after layers
  ``target_layer_ids`` (exllamav3 applies +1 against a states array that
  starts at the embedding — i.e., exactly the OUTPUT of those layers).
- Per cycle: ONE forward over a 16-token block ``[pending, mask x15]``
  (target's embed_tokens; mask_token_id from config) through 5 qwen3-style
  layers. SWA layers attend causally (window 2048 — enforced only as
  plain causal here; affects acceptance, not correctness, beyond 2048);
  the final full-attention layer is BIDIRECTIONAL over cache+block
  (exllamav3 runs flash-attn with causal=False) — the diffusion step.
  Block K/V are never committed to the cache.
- Drafts: final norm -> target lm_head -> argmax at the mask positions
  (1..15). ``EXL3_DFLASH_AR=1`` flips to the AR alignment (0..14).
  Drafts are verify-gated: output is token-identical to plain greedy.

Numerics learned the hard way (Phase 28): the drafter was TRAINED IN BF16
and its residual stream legitimately reaches ~1e4 — fp16 OVERFLOWS at the
last layer (inf -> NaN logits -> all drafts = token 0). The drafter
therefore runs bf16 activations end-to-end and its norm weights load RAW
(the target family's zero-centered +1 heuristic does not apply and was
corrupting one norm). Format auto-detect: ``fc.weight`` (bf16 original) vs
``fc.trellis`` (exl3 quant; note the available 4.00bpw quant measures cos
0.83-0.94 against the CURRENT bf16 export — z-lab is still training — so
bf16 is the reference drafter for now).
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ponyexl3.mlx.exl3_linear import EXL3Linear
from ponyexl3.mlx.weights import load_safetensors
from ponyexl3.mlx.mtp import _infer_exl3_layer  # pyright: ignore[reportPrivateUsage]

WeightsDict = dict[str, mx.array]
LinearMod = EXL3Linear | nn.Linear


def _raw_norm(weights: WeightsDict, key: str) -> mx.array:
    return weights[key].astype(mx.bfloat16)


def _make_linear(weights: WeightsDict, key: str) -> LinearMod:
    """bf16 nn.Linear when ``key.weight`` exists, else EXL3."""
    w = weights.get(f"{key}.weight")
    if w is not None:
        m = nn.Linear(1, 1, bias=False)
        m.weight = w.astype(mx.bfloat16)
        return m
    return EXL3Linear(_infer_exl3_layer(key, weights))


class _DFlashLayer(nn.Module):
    q_proj: LinearMod
    k_proj: LinearMod
    v_proj: LinearMod
    o_proj: LinearMod
    gate_proj: LinearMod
    up_proj: LinearMod
    down_proj: LinearMod
    _w_q_norm: mx.array
    _w_k_norm: mx.array
    _w_input_ln: mx.array
    _w_post_ln: mx.array
    causal: bool

    def __init__(self, idx: int, weights: WeightsDict, causal: bool) -> None:
        super().__init__()
        pre = f"layers.{idx}."
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, _make_linear(weights, f"{pre}self_attn.{name}"))
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, _make_linear(weights, f"{pre}mlp.{name}"))
        self._w_q_norm = _raw_norm(weights, f"{pre}self_attn.q_norm.weight")
        self._w_k_norm = _raw_norm(weights, f"{pre}self_attn.k_norm.weight")
        self._w_input_ln = _raw_norm(weights, f"{pre}input_layernorm.weight")
        self._w_post_ln = _raw_norm(weights, f"{pre}post_attention_layernorm.weight")
        self.causal = causal


class DFlashDraft(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.aux_ids = tuple(cfg["dflash_config"]["target_layer_ids"])
        self.mask_token_id = int(cfg["dflash_config"]["mask_token_id"])
        self.block_size = int(cfg["block_size"])
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.n_heads = int(cfg["num_attention_heads"])
        self.n_kv = int(cfg["num_key_value_heads"])
        self.head_dim = int(cfg["head_dim"])
        self.rope_base = float(cfg["rope_theta"])
        layer_types = cfg["layer_types"]

        weights: WeightsDict = load_safetensors(os.path.join(path, "model.safetensors"))
        self.fc = _make_linear(weights, "fc")
        self._w_hidden_norm = _raw_norm(weights, "hidden_norm.weight")
        self._w_norm = _raw_norm(weights, "norm.weight")
        self.layers = [
            _DFlashLayer(i, weights, causal=(layer_types[i] == "sliding_attention"))
            for i in range(int(cfg["num_hidden_layers"]))
        ]
        self.caches: list[Any] = []
        self._draft_head = None
        mx.eval(self.parameters())

    def make_caches(self) -> None:
        from mlx_lm.models.cache import KVCache

        self.caches = [KVCache() for _ in self.layers]

    def fuse(self, aux: list[mx.array]) -> mx.array:
        x = mx.concatenate([a.astype(mx.bfloat16) for a in aux], axis=-1)
        return mx.fast.rms_norm(
            self.fc(x).astype(mx.bfloat16), self._w_hidden_norm, self.eps
        )

    def _heads(self, t: mx.array, n: int) -> mx.array:
        B, S, _ = t.shape
        return t.reshape(B, S, n, self.head_dim).transpose(0, 2, 1, 3)

    def update_kv(self, feats: mx.array) -> None:
        """Append context positions to the drafter KV from fused features."""
        for layer, cache in zip(self.layers, self.caches):
            k = self._heads(layer.k_proj(feats).astype(mx.bfloat16), self.n_kv)
            k = mx.fast.rms_norm(k, layer._w_k_norm, self.eps)  # pyright: ignore[reportPrivateUsage]
            k = mx.fast.rope(
                k, self.head_dim, traditional=False,
                base=self.rope_base, scale=1.0, offset=cache.offset,
            )
            v = self._heads(layer.v_proj(feats).astype(mx.bfloat16), self.n_kv)
            cache.update_and_fetch(k, v)

    def draft_block(
        self,
        pending: mx.array,
        embed: Any,
        head: Any,
        num_draft: int,
    ) -> mx.array:
        """One 16-token block forward -> (num_draft,) target-vocab drafts."""
        ids = mx.concatenate(
            [
                pending.reshape(1).astype(mx.int32),
                mx.full((self.block_size - 1,), self.mask_token_id, dtype=mx.int32),
            ]
        )
        x = embed(ids[None]).astype(mx.bfloat16)
        L = self.caches[0].offset
        for layer, cache in zip(self.layers, self.caches):
            xa = mx.fast.rms_norm(x, layer._w_input_ln, self.eps)  # pyright: ignore[reportPrivateUsage]
            q = self._heads(layer.q_proj(xa).astype(mx.bfloat16), self.n_heads)
            q = mx.fast.rms_norm(q, layer._w_q_norm, self.eps)  # pyright: ignore[reportPrivateUsage]
            q = mx.fast.rope(
                q, self.head_dim, traditional=False,
                base=self.rope_base, scale=1.0, offset=L,
            )
            kb = self._heads(layer.k_proj(xa).astype(mx.bfloat16), self.n_kv)
            kb = mx.fast.rms_norm(kb, layer._w_k_norm, self.eps)  # pyright: ignore[reportPrivateUsage]
            kb = mx.fast.rope(
                kb, self.head_dim, traditional=False,
                base=self.rope_base, scale=1.0, offset=L,
            )
            vb = self._heads(layer.v_proj(xa).astype(mx.bfloat16), self.n_kv)
            kc, vc = cache.state
            K = mx.concatenate([kc, kb], axis=2)
            V = mx.concatenate([vc, vb], axis=2)
            # SWA layers: causal (mlx aligns "causal" bottom-right = exactly
            # cache+block alignment). Full layer: bidirectional over cache
            # AND block — the diffusion step.
            o = mx.fast.scaled_dot_product_attention(
                q, K, V,
                scale=self.head_dim**-0.5,
                mask="causal" if layer.causal else None,
            )
            o = o.transpose(0, 2, 1, 3).reshape(1, self.block_size, -1)
            x = x + layer.o_proj(o).astype(mx.bfloat16)
            xn = mx.fast.rms_norm(x, layer._w_post_ln, self.eps)  # pyright: ignore[reportPrivateUsage]
            x = x + layer.down_proj(
                nn.silu(layer.gate_proj(xn).astype(mx.bfloat16))
                * layer.up_proj(xn).astype(mx.bfloat16)
            ).astype(mx.bfloat16)
        h = mx.fast.rms_norm(x, self._w_norm, self.eps)
        if os.environ.get("EXL3_DFLASH_AR", "0") == "1":
            h = h[:, 0:num_draft]
        else:
            h = h[:, 1 : 1 + num_draft]
        logits = (self._draft_head or head)(h.astype(mx.float16))
        return mx.argmax(logits[0], axis=-1).astype(mx.int32)

    def quantize_draft(
        self,
        lm_head: Any,
        *,
        bits: int = 4,
        group_size: int = 64,
        cache_dir: str | None = None,
    ) -> None:
        """w4 copy of the target lm_head for the draft argmax (verify-gated).

        With ``cache_dir`` (the TARGET model dir — the head is target-
        specific, drafter-agnostic), the fold+quantize result is cached as a
        safetensors sidecar (~1.7 s -> ~0.1 s per launch; shared with MTP)."""
        if getattr(lm_head, "_exl3", None) is None:
            return
        if cache_dir is not None:
            from ponyexl3.mlx.native import quantized_linear_cached

            self._draft_head = quantized_linear_cached(
                lm_head._exl3,
                os.path.join(
                    cache_dir, ".pony_cache", f"draft_head_w{bits}g{group_size}.safetensors"
                ),
                bits=bits,
                group_size=group_size,
            )
        else:
            from ponyexl3.mlx.native import quantized_linear_from_exl3

            self._draft_head = quantized_linear_from_exl3(
                lm_head._exl3, bits=bits, group_size=group_size
            )

    def _body_targets(self):
        return [("fc", self, "fc")] + [
            (f"layers.{i}.{n}", layer, n)
            for i, layer in enumerate(self.layers)
            for n in ("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj")
        ]

    def quantize_body(
        self, *, bits: int = 4, group_size: int = 64,
        cache_path: str | None = None,
    ) -> None:
        """Affine-quantize the drafter's own linears (bf16 mode only).

        Outputs stay token-identical regardless (verify-gated); the risk is
        ACCEPTANCE — A/B'd in tools/dflash_ab.py: w8 drafts bit-identical to
        bf16, w4 jitters ±8% by prompt. With ``cache_path`` the quantized
        tensors round-trip a safetensors sidecar."""
        from ponyexl3.mlx.native import _ql_from_parts  # pyright: ignore[reportPrivateUsage]

        if cache_path is not None and os.path.exists(cache_path):
            t: WeightsDict = load_safetensors(cache_path)
            for key, owner, attr in self._body_targets():
                if f"{key}.scales" in t:
                    setattr(
                        owner, attr,
                        _ql_from_parts(
                            t[f"{key}.weight"], t[f"{key}.scales"], t[f"{key}.biases"],
                            group_size=group_size, bits=bits,
                        ),
                    )
            mx.eval(self.parameters())
            return
        out = {}
        for key, owner, attr in self._body_targets():
            mod = getattr(owner, attr)
            if isinstance(mod, nn.Linear):
                ql = nn.QuantizedLinear.from_linear(mod, group_size=group_size, bits=bits)
                setattr(owner, attr, ql)
                out[f"{key}.weight"] = ql.weight
                out[f"{key}.scales"] = ql.scales
                out[f"{key}.biases"] = ql.biases
        mx.eval(self.parameters())
        if cache_path is not None and out:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            mx.save_safetensors(cache_path, out)
