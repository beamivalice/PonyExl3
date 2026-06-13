"""Native-MLX fast engines for EXL3 layers.

The EXL3 forward is ``y = had(x*suh) @ W_inner`` then ``had(y)*svh`` — all
linear, so the sign vectors and both Hadamard transforms fold into a public
weight matrix computed once at load:

    W_pub = diag(suh) . H_in . W_inner . H_out . diag(svh)

``W_pub`` is obtained by pushing an fp16 identity through the *same* prefill
forward the exact engine uses (identical semantics, chunked over output
columns so lm_head never materializes more than one stripe of fp32).

Two conversion targets:

- **fold16** (`folded_linear_from_exl3`): plain ``nn.Linear`` holding ``W_pub``
  in fp16. NO requantization — the only deviation from the exact trellis path
  is the fp16 rounding of the folded weight (~1 ulp, ~50x below the 4-bpw
  trellis reconstruction noise). This is the accuracy-preserving fast engine.
- **w8a16 / w4a16** (`quantized_linear_from_exl3`): MLX affine group
  quantization of ``W_pub``. This RE-QUANTIZES an already-quantized model and
  measurably degrades EXL3 accuracy (especially at 4 bits). Opt-in only;
  always check the returned error report and end-to-end token agreement
  before trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ponyexl3.mlx.weights import load_safetensors

import mlx.core as mx
import mlx.nn as nn

from ponyexl3.mlx.layer_state import (
    inner_weight_mlx,
    layer_runtime_mlx,
    stripe_weight_mlx,
)
from ponyexl3.mlx.ops import prefill_matmul_mlx
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.types import MlxLmModel

# One output-column stripe of fp32 intermediate at a time (multiple of 128).
PUBLIC_CHUNK_COLS = 8192
HUGE_WEIGHT_BYTES = 64 * 1024 * 1024


def public_weight_chunks(layer: EXL3Layer, *, chunk_cols: int = PUBLIC_CHUNK_COLS):
    """Yield ``W_pub[:, n0:n1]`` fp16 chunks with all transforms folded in.

    Uses the same ``prefill_matmul_mlx`` block as the exact runtime, fed with an
    identity activation, so the folded weights match runtime numerics exactly.
    """
    if chunk_cols % 128 != 0:
        raise ValueError("chunk_cols must be a multiple of 128")
    rt = layer_runtime_mlx(layer)
    eye = mx.eye(layer.in_features, dtype=mx.float16)
    huge = layer.in_features * layer.out_features * 2 > HUGE_WEIGHT_BYTES

    for n0 in range(0, layer.out_features, chunk_cols):
        n = min(chunk_cols, layer.out_features - n0)
        if huge:
            w_inner = stripe_weight_mlx(layer, n0, n, use_cache=False)
        else:
            w_inner = inner_weight_mlx(layer)[:, n0 : n0 + n]
        svh = None if rt.svh is None else rt.svh[n0 : n0 + n]
        chunk = prefill_matmul_mlx(eye, w_inner, rt.suh, svh, use_compile=False)
        mx.eval(chunk)
        yield chunk


def public_weight_mlx(layer: EXL3Layer) -> mx.array:
    """Full ``(in, out)`` fp16 public weight (prefer chunks for huge layers)."""
    return mx.concatenate(list(public_weight_chunks(layer)), axis=1)


def folded_linear_from_exl3(layer: EXL3Layer) -> nn.Linear:
    """Exact fold: ``nn.Linear`` with the fp16 public weight, no requantization."""
    lin = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
    lin.weight = public_weight_mlx(layer).T.astype(mx.float16)
    if layer.bias is not None:
        lin.bias = mx.array(layer.bias).astype(mx.float16)
    mx.eval(lin.parameters())
    return lin


def quantized_linear_from_exl3(
    layer: EXL3Layer,
    *,
    bits: int,
    group_size: int = 64,
    chunk_cols: int = PUBLIC_CHUNK_COLS,
) -> nn.QuantizedLinear:
    """Affine-requantize ``W_pub`` (LOSSY vs EXL3 — opt-in, validate output).

    Quantization groups run along the input dim and chunking runs along the
    output dim, so chunk-wise quantization is exact w.r.t. whole-matrix
    quantization (no group straddles a chunk boundary).
    """
    ql = nn.QuantizedLinear(
        layer.in_features,
        layer.out_features,
        bias=layer.bias is not None,
        group_size=group_size,
        bits=bits,
    )
    wq_parts, sc_parts, bi_parts = [], [], []
    for chunk in public_weight_chunks(layer, chunk_cols=chunk_cols):
        # QuantizedLinear weight layout is (out, in)
        wq, scales, biases = mx.quantize(
            chunk.T.astype(mx.float16), group_size=group_size, bits=bits
        )
        mx.eval(wq, scales, biases)
        wq_parts.append(wq)
        sc_parts.append(scales)
        bi_parts.append(biases)
    ql.weight = mx.concatenate(wq_parts, axis=0)
    ql.scales = mx.concatenate(sc_parts, axis=0)
    ql.biases = mx.concatenate(bi_parts, axis=0)
    if layer.bias is not None:
        ql.bias = mx.array(layer.bias).astype(mx.float16)
    mx.eval(ql.parameters())
    return ql


class FusedSwiGLU(nn.Module):
    """gate_proj + up_proj fused into one matmul. Bitwise-identical math to the
    unfused MLP (the fused weight is the row-concatenation of the two folded
    weights); halves the projection kernel launches on the decode hot path."""

    def __init__(self, gate_up: nn.Module, down: nn.Module) -> None:
        super().__init__()
        self.gate_up = gate_up
        self.down = down

    def __call__(self, x: mx.array) -> mx.array:
        g, u = mx.split(self.gate_up(x), 2, axis=-1)
        return self.down(nn.silu(g) * u)


def _fuse_two_linears(a: nn.Linear, b: nn.Linear) -> nn.Linear:
    out = nn.Linear(a.weight.shape[1], a.weight.shape[0] + b.weight.shape[0], bias=False)
    out.weight = mx.concatenate([a.weight, b.weight], axis=0)
    mx.eval(out.weight)
    return out


def _fuse_two_quantized(a: nn.QuantizedLinear, b: nn.QuantizedLinear) -> nn.QuantizedLinear:
    if (a.group_size, a.bits) != (b.group_size, b.bits):
        raise ValueError("cannot fuse quantized linears with different quant params")
    out = nn.QuantizedLinear(
        a.weight.shape[1] * 32 // a.bits,
        a.weight.shape[0] + b.weight.shape[0],
        bias=False,
        group_size=a.group_size,
        bits=a.bits,
    )
    out.weight = mx.concatenate([a.weight, b.weight], axis=0)
    out.scales = mx.concatenate([a.scales, b.scales], axis=0)
    out.biases = mx.concatenate([a.biases, b.biases], axis=0)
    mx.eval(out.parameters())
    return out


def fuse_mlps(model: MlxLmModel) -> int:
    """Replace converted gate/up/down MLPs with :class:`FusedSwiGLU`. Exact."""
    n = 0
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate_proj", None)
        up = getattr(mlp, "up_proj", None)
        down = getattr(mlp, "down_proj", None)
        if gate is None or up is None or down is None:
            continue
        if isinstance(gate, nn.QuantizedLinear) and isinstance(up, nn.QuantizedLinear):
            fused = _fuse_two_quantized(gate, up)
        elif type(gate) is nn.Linear and type(up) is nn.Linear:
            fused = _fuse_two_linears(gate, up)
        else:
            continue
        layer.mlp = FusedSwiGLU(fused, down)
        n += 1
    return n


@dataclass
class LayerError:
    """Output-space error of a converted layer vs the exact EXL3 path."""

    key: str
    engine: str
    rms_ref: float
    rms_err: float

    @property
    def rel(self) -> float:
        return self.rms_err / self.rms_ref if self.rms_ref else 0.0

    def __str__(self) -> str:
        return f"{self.key}: rel RMS err {self.rel:.2e} ({self.engine})"


def layer_error(layer: EXL3Layer, module: nn.Module, engine: str, *, rows: int = 8) -> LayerError:
    from ponyexl3.mlx.exl3_linear import EXL3Linear

    x = (mx.random.normal((rows, layer.in_features), key=mx.random.key(0)) * 0.5).astype(
        mx.float16
    )
    y_ref = EXL3Linear(layer)(x).astype(mx.float32)
    y_new = module(x).astype(mx.float32)
    rms_ref = float(mx.sqrt(mx.mean(y_ref * y_ref)))
    rms_err = float(mx.sqrt(mx.mean((y_new - y_ref) ** 2)))
    return LayerError(key=layer.key, engine=engine, rms_ref=rms_ref, rms_err=rms_err)

def _ql_from_parts(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int,
    bits: int,
) -> nn.QuantizedLinear:
    """Build a QuantizedLinear from stored tensors without the init-time
    random-weight quantization pass (matters at lm_head scale)."""
    ql = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(ql)
    ql.weight = weight
    ql.scales = scales
    ql.biases = biases
    ql.group_size = group_size
    ql.bits = bits
    ql.mode = "affine"
    return ql


def quantized_linear_cached(
    layer: EXL3Layer,
    cache_path: str,
    *,
    bits: int,
    group_size: int = 64,
) -> nn.QuantizedLinear:
    """``quantized_linear_from_exl3`` with a safetensors sidecar cache.

    The fold+quantize of an lm_head-scale layer costs ~1.7 s per launch
    (worse on 8 bpw targets); the cache loads in ~0.1 s. Bias-less layers
    only (the draft-head use case)."""
    import os

    if os.path.exists(cache_path):
        t: dict[str, mx.array] = load_safetensors(cache_path)
        ql = _ql_from_parts(
            t["weight"], t["scales"], t["biases"], group_size=group_size, bits=bits
        )
        mx.eval(ql.parameters())
        return ql
    ql = quantized_linear_from_exl3(layer, bits=bits, group_size=group_size)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    mx.save_safetensors(
        cache_path,
        {"weight": ql.weight, "scales": ql.scales, "biases": ql.biases},
    )
    return ql
