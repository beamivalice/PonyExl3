"""Small checkpoint-backed fixtures for HF -> EXL3 converter bring-up.

The first converter loop works on one 16x16 trellis tile.  It reads the
matching BF16 source block directly from safetensors, maps it into EXL3's
regularized inner domain using the oracle layer's scales, then runs the
existing CPU Viterbi search and compares against the oracle tile.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
from typing import Any, Literal

import numpy as np

from ponyexl3.convert.calibration import validate_activation_matrix
from ponyexl3.convert.reference_search import quantize_tile_reference
from ponyexl3.ref.codebook import CodebookMode, codebook_mode_from_flags
from ponyexl3.ref.decode import decode_packed_tile
from ponyexl3.ref.hadamard import HAD_DIM, preapply_had_left, preapply_had_right
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.loader import load_exl3_layer
from ponyexl3.ref.perm import kernel_order_to_row_major, tensor_core_perm
from ponyexl3.ref.signs import unpack_signs_or_pass
from ponyexl3.ref.trellis import pack_trellis_tile, unpack_trellis_tile


LayoutKind = Literal["linear_t", "qwen_gate", "qwen_up", "qwen_down"]
SearchBackend = Literal["cpu", "metal"]

_QWEN_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.mlp)\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)$"
)


@dataclass(frozen=True)
class TensorInfo:
    """Safetensors metadata for one tensor."""

    key: str
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class SourceLinear:
    """A source HF linear exposed in EXL3 quantizer layout."""

    key: str
    source_tensor_key: str
    layout: LayoutKind
    in_features: int
    out_features: int
    expert: int | None = None
    gate_up_mid: int | None = None


@dataclass(frozen=True)
class OracleLinear:
    """An oracle EXL3 layer and its resolved codebook mode."""

    key: str
    layer: EXL3Layer
    cb: CodebookMode


@dataclass(frozen=True)
class QuantizedLinearTensors:
    """Minimal converted EXL3 tensors for a linear layer."""

    trellis: np.ndarray
    suh: np.ndarray | None
    svh: np.ndarray | None
    mcg: np.ndarray | None = None
    mul1: np.ndarray | None = None
    bias: np.ndarray | None = None
    metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class LayerFixture:
    """Source/oracle pair used by fast converter pilots."""

    source: SourceLinear
    oracle: OracleLinear
    activations: np.ndarray


@dataclass(frozen=True)
class TilePilotResult:
    """Result of one source -> reference-search -> oracle tile comparison."""

    module_key: str
    search_backend: SearchBackend
    tile_k: int
    tile_n: int
    k: int
    cb: CodebookMode
    packed: np.ndarray
    states: np.ndarray
    converted_tile: np.ndarray
    oracle_tile: np.ndarray
    target_tile: np.ndarray
    stats: dict[str, float | bool]


def bf16_to_float32(values: bytes | np.ndarray) -> np.ndarray:
    """Convert little-endian BF16 payload/words to float32."""

    if isinstance(values, bytes):
        words = np.frombuffer(values, dtype="<u2")
    else:
        words = np.asarray(values, dtype=np.uint16)
    bits = words.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _dtype_nbytes(dtype: str) -> int:
    if dtype in ("BF16", "F16", "I16", "U16"):
        return 2
    if dtype in ("F32", "I32", "U32"):
        return 4
    raise ValueError(f"unsupported safetensors dtype {dtype!r}")


def _decode_raw(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return bf16_to_float32(raw)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").copy()
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").copy()
    if dtype == "I32":
        return np.frombuffer(raw, dtype="<i4").copy()
    if dtype == "U32":
        return np.frombuffer(raw, dtype="<u4").copy()
    if dtype == "I16":
        return np.frombuffer(raw, dtype="<i2").copy()
    if dtype == "U16":
        return np.frombuffer(raw, dtype="<u2").copy()
    raise ValueError(f"unsupported safetensors dtype {dtype!r}")


class SafetensorIndex:
    """Read small slices from a sharded safetensors checkpoint."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        with index_path.open(encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        self.weight_map = {str(k): str(v) for k, v in weight_map.items()}
        self._headers: dict[Path, tuple[int, dict[str, Any]]] = {}

    def _header(self, shard: Path) -> tuple[int, dict[str, Any]]:
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        with shard.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        out = int(header_len), header
        self._headers[shard] = out
        return out

    def tensor_info(self, key: str) -> TensorInfo:
        shard_name = self.weight_map.get(key)
        if shard_name is None:
            raise KeyError(f"{key!r} not found in {self.model_dir}")
        shard = self.model_dir / shard_name
        _header_len, header = self._header(shard)
        meta = header[key]
        start, end = meta["data_offsets"]
        return TensorInfo(
            key=key,
            shard=shard,
            dtype=str(meta["dtype"]),
            shape=tuple(int(x) for x in meta["shape"]),
            data_offsets=(int(start), int(end)),
        )

    def read_slice(
        self,
        key: str,
        starts: tuple[int, ...],
        sizes: tuple[int, ...],
    ) -> np.ndarray:
        """Read a rectangular slice.  The last dimension must be contiguous."""

        info = self.tensor_info(key)
        if len(starts) != len(info.shape) or len(sizes) != len(info.shape):
            raise ValueError(f"slice rank mismatch for {key}: shape={info.shape}")
        for axis, (start, size, dim) in enumerate(zip(starts, sizes, info.shape)):
            if start < 0 or size < 0 or start + size > dim:
                raise ValueError(
                    f"slice out of bounds for {key} axis {axis}: "
                    f"start={start}, size={size}, dim={dim}"
                )

        bpe = _dtype_nbytes(info.dtype)
        header_len, _header = self._header(info.shard)
        data_base = 8 + header_len + info.data_offsets[0]
        strides = [1] * len(info.shape)
        for i in range(len(info.shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * info.shape[i + 1]

        prefix_shape = sizes[:-1]
        rows: list[np.ndarray] = []
        with info.shard.open("rb") as f:
            for prefix in np.ndindex(prefix_shape):
                elem_offset = 0
                for axis, idx in enumerate(prefix):
                    elem_offset += (starts[axis] + idx) * strides[axis]
                elem_offset += starts[-1] * strides[-1]
                f.seek(data_base + elem_offset * bpe)
                rows.append(_decode_raw(f.read(sizes[-1] * bpe), info.dtype))

        arr = np.stack(rows, axis=0).reshape(sizes)
        return arr

    def read_tensor(self, key: str) -> np.ndarray:
        """Read a complete tensor from its safetensors shard."""

        info = self.tensor_info(key)
        bpe = _dtype_nbytes(info.dtype)
        header_len, _header = self._header(info.shard)
        data_base = 8 + header_len + info.data_offsets[0]
        n_elem = int(np.prod(info.shape, dtype=np.int64))
        with info.shard.open("rb") as f:
            f.seek(data_base)
            raw = f.read(n_elem * bpe)
        return _decode_raw(raw, info.dtype).reshape(info.shape)


def resolve_source_linear(model_dir: str | Path, module_key: str) -> SourceLinear:
    """Resolve a supported HF source tensor for an EXL3 module key."""

    index = SafetensorIndex(model_dir)
    expert_match = _QWEN_EXPERT_RE.match(module_key)
    if expert_match is not None:
        prefix = expert_match.group("prefix")
        expert = int(expert_match.group("expert"))
        proj = expert_match.group("proj")
        if proj in ("gate_proj", "up_proj"):
            source_key = f"{prefix}.experts.gate_up_proj"
            info = index.tensor_info(source_key)
            if len(info.shape) != 3:
                raise ValueError(f"{source_key} must be 3D, got {info.shape}")
            mid = info.shape[1] // 2
            return SourceLinear(
                key=module_key,
                source_tensor_key=source_key,
                layout="qwen_gate" if proj == "gate_proj" else "qwen_up",
                in_features=info.shape[2],
                out_features=mid,
                expert=expert,
                gate_up_mid=mid,
            )
        source_key = f"{prefix}.experts.down_proj"
        info = index.tensor_info(source_key)
        if len(info.shape) != 3:
            raise ValueError(f"{source_key} must be 3D, got {info.shape}")
        return SourceLinear(
            key=module_key,
            source_tensor_key=source_key,
            layout="qwen_down",
            in_features=info.shape[2],
            out_features=info.shape[1],
            expert=expert,
        )

    source_key = f"{module_key}.weight"
    info = index.tensor_info(source_key)
    if len(info.shape) != 2:
        raise ValueError(f"{source_key} must be 2D, got {info.shape}")
    return SourceLinear(
        key=module_key,
        source_tensor_key=source_key,
        layout="linear_t",
        in_features=info.shape[1],
        out_features=info.shape[0],
    )


def read_source_public_block(
    model_dir: str | Path,
    source: SourceLinear,
    *,
    in_start: int,
    out_start: int,
    rows: int = HAD_DIM,
    cols: int = HAD_DIM,
) -> np.ndarray:
    """Read a source public-weight block in EXL3 layout `(in, out)`."""

    index = SafetensorIndex(model_dir)
    if source.layout == "linear_t":
        block = index.read_slice(
            source.source_tensor_key,
            (out_start, in_start),
            (cols, rows),
        )
        return block.T.astype(np.float32, copy=False)

    if source.expert is None:
        raise ValueError(f"{source.key}: expert index missing")
    if source.layout in ("qwen_gate", "qwen_up"):
        if source.gate_up_mid is None:
            raise ValueError(f"{source.key}: gate/up midpoint missing")
        source_out_start = out_start
        if source.layout == "qwen_up":
            source_out_start += source.gate_up_mid
        block = index.read_slice(
            source.source_tensor_key,
            (source.expert, source_out_start, in_start),
            (1, cols, rows),
        )[0]
        return block.T.astype(np.float32, copy=False)

    block = index.read_slice(
        source.source_tensor_key,
        (source.expert, out_start, in_start),
        (1, cols, rows),
    )[0]
    return block.T.astype(np.float32, copy=False)


def load_oracle_linear(model_dir: str | Path, module_key: str) -> OracleLinear:
    layer = load_exl3_layer(str(model_dir), module_key)
    cb = codebook_mode_from_flags(mcg=layer.mcg, mul1=layer.mul1)
    return OracleLinear(key=module_key, layer=layer, cb=cb)


def build_layer_fixture(
    source_dir: str | Path,
    oracle_dir: str | Path,
    module_key: str,
    *,
    activation_rows: int = 4,
    seed: int = 0,
    activations: np.ndarray | None = None,
) -> LayerFixture:
    source = resolve_source_linear(source_dir, module_key)
    oracle = load_oracle_linear(oracle_dir, module_key)
    if source.in_features != oracle.layer.in_features:
        raise ValueError(
            f"{module_key}: source in_features={source.in_features} "
            f"!= oracle {oracle.layer.in_features}"
        )
    if source.out_features != oracle.layer.out_features:
        raise ValueError(
            f"{module_key}: source out_features={source.out_features} "
            f"!= oracle {oracle.layer.out_features}"
        )
    if activations is None:
        rng = np.random.default_rng(seed)
        fixture_activations = rng.standard_normal(
            (activation_rows, source.in_features),
            dtype=np.float32,
        ).astype(np.float16)
    else:
        fixture_activations = validate_activation_matrix(
            activations,
            expected_features=source.in_features,
        )
    return LayerFixture(source=source, oracle=oracle, activations=fixture_activations)


def _scale_slice(scale: np.ndarray | None, start: int, size: int) -> np.ndarray | None:
    unpacked = unpack_signs_or_pass(scale)
    if unpacked is None:
        return None
    return unpacked[start : start + size].astype(np.float32)


def public_block_to_inner_block(
    public_block: np.ndarray,
    *,
    suh: np.ndarray | None,
    svh: np.ndarray | None,
    in_start: int,
    out_start: int,
) -> np.ndarray:
    """Invert EXL3 public reconstruction for one 128x128 Hadamard block."""

    if public_block.shape != (HAD_DIM, HAD_DIM):
        raise ValueError(f"expected {(HAD_DIM, HAD_DIM)} block, got {public_block.shape}")
    block = public_block.astype(np.float32)
    sv = _scale_slice(svh, out_start, HAD_DIM)
    if sv is not None:
        if np.any(np.abs(sv) < 1e-30):
            raise ValueError(
                "cannot invert oracle svh for this 128-column block; "
                f"block starting at output {out_start} contains zero scales"
            )
        block = block / sv.reshape(1, HAD_DIM)
    block = preapply_had_right(block.astype(np.float32)).astype(np.float32)
    su = _scale_slice(suh, in_start, HAD_DIM)
    if su is not None:
        if np.any(np.abs(su) < 1e-30):
            raise ValueError(
                "cannot invert oracle suh for this 128-row block; "
                f"block starting at input {in_start} contains zero scales"
            )
        block = block / su.reshape(HAD_DIM, 1)
    block = preapply_had_left(block.astype(np.float32)).astype(np.float32)
    return block


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(d * d))


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    denom = float(np.sqrt(np.mean(b.astype(np.float32) ** 2))) + 1e-20
    return float(np.sqrt(np.mean(d * d)) / denom)


def run_tile_pilot(
    source_dir: str | Path,
    oracle_dir: str | Path,
    module_key: str,
    *,
    tile_k: int = 0,
    tile_n: int = 0,
    search_backend: SearchBackend = "cpu",
    max_pins: int = 4,
) -> TilePilotResult:
    """Quantize one oracle-comparable tile from the source checkpoint."""

    fixture = build_layer_fixture(source_dir, oracle_dir, module_key)
    source = fixture.source
    oracle = fixture.oracle
    layer = oracle.layer
    if tile_k < 0 or tile_k >= layer.in_features // 16:
        raise ValueError(f"tile_k out of range: {tile_k}")
    if tile_n < 0 or tile_n >= layer.out_features // 16:
        raise ValueError(f"tile_n out of range: {tile_n}")

    in_block = (tile_k * 16 // HAD_DIM) * HAD_DIM
    out_block = (tile_n * 16 // HAD_DIM) * HAD_DIM
    public_block = read_source_public_block(
        source_dir,
        source,
        in_start=in_block,
        out_start=out_block,
        rows=HAD_DIM,
        cols=HAD_DIM,
    )
    inner_block = public_block_to_inner_block(
        public_block,
        suh=layer.suh,
        svh=layer.svh,
        in_start=in_block,
        out_start=out_block,
    )
    r0 = tile_k * 16 - in_block
    c0 = tile_n * 16 - out_block
    target_tile = inner_block[r0 : r0 + 16, c0 : c0 + 16]
    target_kernel = target_tile.reshape(256)[tensor_core_perm()]

    if search_backend == "cpu":
        states, decoded_kernel = quantize_tile_reference(
            target_kernel.astype(np.float32),
            k=layer.k,
            cb=oracle.cb,
            max_pins=max_pins,
        )
    elif search_backend == "metal":
        from ponyexl3.convert.metal_search import quantize_tiles_mlx_np

        decoded_tiles, state_tiles = quantize_tiles_mlx_np(
            target_kernel.reshape(1, 256).astype(np.float32),
            k=layer.k,
            cb=oracle.cb,
        )
        decoded_kernel = decoded_tiles[0].astype(np.float32, copy=False)
        states = state_tiles[0].astype(np.uint16, copy=False)
    else:
        raise ValueError(f"unknown search backend: {search_backend}")
    packed = pack_trellis_tile((states & ((1 << layer.k) - 1)).astype(np.uint16), layer.k)
    converted_tile = decode_packed_tile(packed, layer.k, oracle.cb).astype(np.float32)
    converted_direct = kernel_order_to_row_major(decoded_kernel).astype(np.float32)
    if not np.array_equal(converted_tile, converted_direct):
        raise AssertionError("packed decode disagrees with direct decoded states")

    oracle_tile = decode_packed_tile(layer.trellis[tile_k, tile_n], layer.k, oracle.cb).astype(
        np.float32
    )
    oracle_unpacked = unpack_trellis_tile(layer.trellis[tile_k, tile_n], layer.k)
    oracle_repacked = pack_trellis_tile(
        (oracle_unpacked & ((1 << layer.k) - 1)).astype(np.uint16),
        layer.k,
    )
    stats: dict[str, float | bool] = {
        "converted_target_mse": _mse(converted_tile, target_tile),
        "oracle_target_mse": _mse(oracle_tile, target_tile),
        "converted_vs_oracle_mse": _mse(converted_tile, oracle_tile),
        "converted_target_rel_rms": _rel_rms(converted_tile, target_tile),
        "oracle_target_rel_rms": _rel_rms(oracle_tile, target_tile),
        "converted_vs_oracle_rel_rms": _rel_rms(converted_tile, oracle_tile),
        "oracle_pack_roundtrip": bool(np.array_equal(oracle_repacked, layer.trellis[tile_k, tile_n])),
        "converted_pack_roundtrip": bool(
            np.array_equal(unpack_trellis_tile(packed, layer.k).astype(np.uint16), states)
        ),
    }
    return TilePilotResult(
        module_key=module_key,
        search_backend=search_backend,
        tile_k=tile_k,
        tile_n=tile_n,
        k=layer.k,
        cb=oracle.cb,
        packed=packed,
        states=states,
        converted_tile=converted_tile,
        oracle_tile=oracle_tile,
        target_tile=target_tile.astype(np.float32),
        stats=stats,
    )


def tile_pilot_summary(result: TilePilotResult) -> dict[str, Any]:
    """JSON-friendly summary for CLI/tests."""

    return {
        "module": result.module_key,
        "search_backend": result.search_backend,
        "tile": [result.tile_k, result.tile_n],
        "k": result.k,
        "codebook": result.cb.name.lower(),
        "stats": result.stats,
    }
