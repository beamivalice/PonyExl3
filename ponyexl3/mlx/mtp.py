"""Qwen3.5/3.6 MTP (multi-token prediction) draft module — EXL3 weights.

Ports exllamav3's MTP drafting (v0.0.41+) to MLX. The MTP head is the
DeepSeek-V3 / Qwen3-Next design:

    h   = fc(concat(rmsnorm_emb(embed(tok)), rmsnorm_hid(prev_hidden)))
    h   = full_attention_decoder_layer(h, mtp_kv_cache)
    out = lm_head(rmsnorm_final(h))          # main model's lm_head

It shares the target model's ``embed_tokens`` and ``lm_head``
(``mtp_use_dedicated_embeddings: false``); only ``fc`` and one decoder layer
are extra, EXL3-quantized (~0.2 GB at 4 bpw).
"""

from __future__ import annotations

import os
from glob import glob
from typing import Any, cast

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ponyexl3.mlx.exl3_linear import EXL3Linear
from ponyexl3.mlx.weights import load_safetensors
from ponyexl3.ref.codebook import MCG_MULT, MUL1_MULT
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.types import KvCache

_EPS_DEFAULT = 1e-6


def _infer_exl3_layer(key: str, weights: dict[str, mx.array]) -> EXL3Layer:
    """Build an EXL3Layer from raw tensors (no quantization_config present)."""
    trellis = np.array(weights[f"{key}.trellis"]).astype(np.uint16, copy=False)
    in_tiles, out_tiles, packed = trellis.shape
    k = packed * 16 // 256

    def _flag(name: str, expected: int) -> bool:
        t = weights.get(f"{key}.{name}")
        if t is None:
            return False
        got = int(np.array(t).astype(np.int64)) & 0xFFFFFFFF
        if got != int(expected):
            raise ValueError(f"{key}: {name} multiplier {got:#x} unsupported")
        return True

    def _np16(name: str):
        t = weights.get(f"{key}.{name}")
        return None if t is None else np.array(t.astype(mx.float16))

    return EXL3Layer(
        key=key,
        in_features=in_tiles * 16,
        out_features=out_tiles * 16,
        k=k,
        trellis=trellis,
        suh=_np16("suh"),
        svh=_np16("svh"),
        bias=None,
        mcg=_flag("mcg", int(MCG_MULT)),
        mul1=_flag("mul1", int(MUL1_MULT)),
    )


def _norm_weight(weights: dict[str, mx.array], key: str) -> mx.array:
    """Load an RMSNorm weight; HF-side zero-centered norms get the +1 shift
    (same convention the main loader applies via mlx_lm sanitize)."""
    w = weights[key].astype(mx.float32)
    if float(mx.abs(mx.mean(w))) < 0.5:  # zero-centered storage
        w = w + 1.0
    return w.astype(mx.float16)


class Qwen35MTP(nn.Module):
    """One-layer MTP draft head over the target model's hidden states."""

    _draft_head: nn.Module | None

    def __init__(self, args: Any, weights: dict[str, mx.array]) -> None:
        super().__init__()
        from mlx_lm.models.qwen3_next import Qwen3NextAttention, Qwen3NextMLP

        self.eps = getattr(args, "rms_norm_eps", _EPS_DEFAULT)

        self.fc = EXL3Linear(_infer_exl3_layer("mtp.fc", weights))
        self.self_attn = Qwen3NextAttention(args)
        self.mlp = Qwen3NextMLP(args.hidden_size, args.intermediate_size)

        pre = "mtp.layers.0."
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(
                self.self_attn,
                name,
                EXL3Linear(_infer_exl3_layer(f"{pre}self_attn.{name}", weights)),
            )
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(
                self.mlp, name, EXL3Linear(_infer_exl3_layer(f"{pre}mlp.{name}", weights))
            )
        self.self_attn.q_norm.weight = _norm_weight(weights, f"{pre}self_attn.q_norm.weight")
        self.self_attn.k_norm.weight = _norm_weight(weights, f"{pre}self_attn.k_norm.weight")

        self._w_input_ln = _norm_weight(weights, f"{pre}input_layernorm.weight")
        self._w_post_ln = _norm_weight(weights, f"{pre}post_attention_layernorm.weight")
        self._w_norm = _norm_weight(weights, "mtp.norm.weight")
        self._w_pre_emb = _norm_weight(weights, "mtp.pre_fc_norm_embedding.weight")
        self._w_pre_hid = _norm_weight(weights, "mtp.pre_fc_norm_hidden.weight")
        self._draft_head = None
        mx.eval(self.parameters())

    def __call__(self, emb: mx.array, prev_hidden: mx.array, cache: KvCache) -> mx.array:
        """(B, S, H) token embeddings + previous hidden states -> MTP hidden
        (pre-final-norm, used for chained drafting). RoPE position comes from
        the cache offset."""
        from mlx_lm.models.base import create_attention_mask

        x = mx.concatenate(
            [
                mx.fast.rms_norm(emb, self._w_pre_emb, self.eps),
                mx.fast.rms_norm(prev_hidden.astype(emb.dtype), self._w_pre_hid, self.eps),
            ],
            axis=-1,
        )
        h = self.fc(x)
        mask: mx.array | None = (
            cast(mx.array | None, create_attention_mask(h, cache)) if h.shape[1] > 1 else None
        )
        r = self.self_attn(
            mx.fast.rms_norm(h, self._w_input_ln, self.eps), mask=mask, cache=cache
        )
        h = h + r
        h = h + self.mlp(mx.fast.rms_norm(h, self._w_post_ln, self.eps))
        return h

    def head_input(self, h: mx.array) -> mx.array:
        """Final MTP norm — feed the result to the target model's lm_head."""
        return mx.fast.rms_norm(h, self._w_norm, self.eps)


def quantize_draft(
    mtp: Qwen35MTP,
    lm_head: Any,
    *,
    bits: int = 4,
    group_size: int = 64,
    cache_dir: str | None = None,
) -> None:
    """Affine-quantize the DRAFT side only (output bits unchanged).

    Drafts never reach the output stream directly — the verify forward
    (which keeps the exact lm_head and the exact target weights) gates
    every emitted token, so a lossy draft head can only shift acceptance
    rates, never correctness. Converts the MTP's EXL3 layers in place and
    attaches a quantized copy of the target's lm_head (``mtp._draft_head``,
    ~0.45 GB at w4g64) for the draft chain's argmax.
    """
    from ponyexl3.mlx.exl3_linear import EXL3Linear
    from ponyexl3.mlx.native import quantized_linear_from_exl3

    if getattr(lm_head, "_exl3", None) is not None:
        if cache_dir is not None:
            import os

            from ponyexl3.mlx.native import quantized_linear_cached

            setattr(
                mtp,
                "_draft_head",
                quantized_linear_cached(
                lm_head._exl3,  # pyright: ignore[reportPrivateUsage]
                os.path.join(cache_dir, ".pony_cache", f"draft_head_w{bits}g{group_size}.safetensors"),
                bits=bits,
                group_size=group_size,
                ),
            )
        else:
            setattr(
                mtp,
                "_draft_head",
                quantized_linear_from_exl3(
                    lm_head._exl3, bits=bits, group_size=group_size  # pyright: ignore[reportPrivateUsage]
                ),
            )
    setattr(
        mtp,
        "fc",
        quantized_linear_from_exl3(mtp.fc._exl3, bits=bits, group_size=group_size),  # pyright: ignore[reportPrivateUsage]
    )
    for owner, names in (
        (mtp.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
        (mtp.mlp, ("gate_proj", "up_proj", "down_proj")),
    ):
        for name in names:
            mod = getattr(owner, name)
            if isinstance(mod, EXL3Linear):
                setattr(
                    owner,
                    name,
                    quantized_linear_from_exl3(
                        mod._exl3, bits=bits, group_size=group_size  # pyright: ignore[reportPrivateUsage]
                    ),
                )
    mx.eval(mtp.parameters())


def find_mtp_weights(model_dir: str, mtp_path: str | None = None) -> dict[str, mx.array] | None:
    """Locate ``mtp.*`` tensors: explicit path (file or dir) or the model dir."""
    candidates: list[str] = []
    if mtp_path:
        if os.path.isdir(mtp_path):
            candidates += sorted(glob(os.path.join(mtp_path, "*.safetensors")))
        else:
            candidates.append(mtp_path)
    candidates += sorted(glob(os.path.join(model_dir, "*.safetensors")))
    for p in candidates:
        loaded = load_safetensors(p)
        mtp_w = {k: v for k, v in loaded.items() if k.startswith("mtp.")}
        if "mtp.fc.trellis" in mtp_w:
            return mtp_w
    return None


def load_mtp(model_dir: str, config: dict[str, Any], mtp_path: str | None = None) -> Qwen35MTP | None:
    weights = find_mtp_weights(model_dir, mtp_path)
    if weights is None:
        return None
    from mlx_lm.models import qwen3_5 as arch

    args = arch.TextModelArgs.from_dict(config.get("text_config", config))
    return Qwen35MTP(args, weights)
