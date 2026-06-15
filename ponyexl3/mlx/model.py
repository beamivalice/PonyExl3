"""Build a full MLX model from an EXL3-quantized checkpoint (Phase 4).

Reuses the mlx_lm architecture skeleton for the model graph and swaps every
EXL3-quantized linear for :class:`EXL3Linear`. Non-quantized tensors (norms,
embeddings, DeltaNet params) load through the architecture's ``sanitize`` so
HF-side conventions (conv1d layout, zero-centered norms) are handled by mlx_lm.

Currently mapped architectures: ``qwen3_5`` (Qwen3.5 hybrid linear/full attention).
"""

from __future__ import annotations

import importlib
import json
import os
import re
from glob import glob
from typing import Any

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from ponyexl3.mlx.exl3_linear import EXL3Linear
from ponyexl3.mlx.weights import load_safetensors
from ponyexl3.ref.codebook import MCG_MULT, MUL1_MULT
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.types import JsonDict, MlxLmModel

_EXL3_SUFFIXES = ("trellis", "suh", "su", "svh", "sv", "mcg", "mul1", "bias")

# checkpoint model_type → mlx_lm architecture module
_ARCHITECTURES = {
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen3_5_moe_text": "qwen3_5_moe",
}

_expert_re: re.Pattern[str] | None = None


def _expert_match(key: str) -> re.Match[str] | None:
    global _expert_re
    if _expert_re is None:
        _expert_re = re.compile(
            r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
            r"(gate_proj|up_proj|down_proj)$"
        )
    return _expert_re.match(key)


def _build_moe_experts(
    model: MlxLmModel,
    storage: dict[str, JsonDict],
    weights: dict[str, mx.array],
    verbose: bool,
) -> set[str]:
    """Stack each layer's experts (plus the shared expert as the LAST expert)
    into EXL3SwitchGLU modules wrapped in EXL3MoEBlock. Returns the set of
    storage keys consumed here (skipped by the generic EXL3Linear swap)."""
    from ponyexl3.mlx.exl3_moe import EXL3MoEBlock, EXL3SwitchGLU
    from ponyexl3.ref.codebook import codebook_mode_from_flags

    layers: dict[int, int] = {}
    consumed: set[str] = set()
    for key, info in storage.items():
        m = _expert_match(key)
        if m:
            li, e = int(m.group(1)), int(m.group(2))
            layers[li] = max(layers.get(li, 0), e + 1)
    if not layers:
        return consumed

    for li, E in sorted(layers.items()):
        pre = f"model.language_model.layers.{li}.mlp.experts"
        info = storage[f"{pre}.0.gate_proj"]
        sh = info["stored_tensors"][f"{pre}.0.gate_proj.trellis"]["shape"]
        k = sh[2] * 16 // 256
        cb = codebook_mode_from_flags(
            mcg=bool(info.get("mcg_multiplier")),
            mul1=bool(info.get("mul1_multiplier")),
        )
        gu_parts, gu_suh, gu_svh = [], [], []
        dn_parts, dn_suh, dn_svh = [], [], []
        shared_pre = f"model.language_model.layers.{li}.mlp.shared_expert"
        expert_pres = [f"{pre}.{e}" for e in range(E)] + [shared_pre]
        up_parts = []
        for ep in expert_pres:
            g, u, d = f"{ep}.gate_proj", f"{ep}.up_proj", f"{ep}.down_proj"
            gu_parts.append(weights[f"{g}.trellis"])
            up_parts.append(weights[f"{u}.trellis"])
            gu_suh.append(mx.stack([weights[f"{g}.suh"], weights[f"{u}.suh"]]))
            gu_svh.append(mx.concatenate([weights[f"{g}.svh"], weights[f"{u}.svh"]]))
            dn_parts.append(weights[f"{d}.trellis"])
            dn_suh.append(weights[f"{d}.suh"])
            dn_svh.append(weights[f"{d}.svh"])
            for proj in (g, u, d):
                if proj in storage:
                    consumed.add(proj)
        mod = EXL3SwitchGLU(
            gu_trellis=mx.concatenate(gu_parts + up_parts, axis=1).view(mx.uint16),
            gu_suh=mx.stack(gu_suh).astype(mx.float16),
            gu_svh=mx.stack(gu_svh).astype(mx.float16),
            dn_trellis=mx.concatenate(dn_parts, axis=1).view(mx.uint16),
            dn_suh=mx.stack(dn_suh).astype(mx.float16),
            dn_svh=mx.stack(dn_svh).astype(mx.float16),
            k=k,
            cb=cb,
        )
        mx.eval(
            mod._gu_trellis, mod._gu_suh, mod._gu_svh,  # pyright: ignore[reportPrivateUsage]
            mod._dn_trellis, mod._dn_suh, mod._dn_svh,  # pyright: ignore[reportPrivateUsage]
        )
        old = model.layers[li].mlp
        block = EXL3MoEBlock(
            old.gate, old.shared_expert_gate, mod, old.top_k, bool(old.norm_topk_prob)
        )
        _set_module(model, f"language_model.model.layers.{li}.mlp", block)
        if verbose:
            print(f"  moe layer {li}: {E}+shared experts, k={k}")
    return consumed


def _read_json(model_dir: str, name: str) -> dict[str, Any]:
    with open(os.path.join(model_dir, name), encoding="utf-8") as f:
        return json.load(f)


def _load_all_tensors(model_dir: str) -> dict[str, mx.array]:
    paths = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"no safetensors under {model_dir}")
    weights: dict[str, mx.array] = {}
    for p in paths:
        weights.update(load_safetensors(p))
    return weights


def _exl3_storage(qcfg: dict[str, Any]) -> dict[str, JsonDict]:
    storage = qcfg.get("tensor_storage", {})
    return {k: v for k, v in storage.items() if v.get("quant_format") == "exl3"}


def _check_multiplier(weights: dict[str, mx.array], key: str, name: str, expected: int) -> None:
    t = weights.get(f"{key}.{name}")
    if t is None:
        return
    got = int(np.array(t).astype(np.int64)) & 0xFFFFFFFF
    if got != int(expected):
        raise ValueError(
            f"{key}: stored {name} multiplier {got:#x} != supported {int(expected):#x}"
        )


def _build_exl3_layer(key: str, info: JsonDict, weights: dict[str, mx.array]) -> EXL3Layer:
    trellis = np.array(weights[f"{key}.trellis"]).astype(np.uint16, copy=False)
    in_tiles, out_tiles, packed = trellis.shape
    k = packed * 16 // 256
    mcg = bool(info.get("mcg_multiplier"))
    mul1 = bool(info.get("mul1_multiplier"))
    # The ref codebook hardcodes the exllamav3 default multipliers; refuse to
    # load checkpoints quantized with different ones rather than emit garbage.
    if mcg:
        _check_multiplier(weights, key, "mcg", int(MCG_MULT))
    if mul1:
        _check_multiplier(weights, key, "mul1", int(MUL1_MULT))

    def _np16(name: str) -> np.ndarray | None:
        t = weights.get(f"{key}.{name}")
        return None if t is None else np.array(t.astype(mx.float16))

    suh = _np16("suh")
    if suh is None:
        suh = _np16("su")
    svh = _np16("svh")
    if svh is None:
        svh = _np16("sv")

    return EXL3Layer(
        key=key,
        in_features=in_tiles * 16,
        out_features=out_tiles * 16,
        k=k,
        trellis=trellis,
        suh=suh,
        svh=svh,
        bias=_np16("bias"),
        mcg=mcg,
        mul1=mul1,
    )


def _module_path(checkpoint_key: str) -> str:
    """Translate a checkpoint module key to the mlx_lm attribute path."""
    if checkpoint_key == "lm_head":
        return "language_model.lm_head"
    for prefix in ("model.language_model.", "model."):
        if checkpoint_key.startswith(prefix):
            return "language_model.model." + checkpoint_key[len(prefix):]
    return "language_model." + checkpoint_key


def _set_module(root: MlxLmModel, path: str, module: Any) -> None:
    obj = root
    parts = path.split(".")
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = module
    else:
        setattr(obj, last, module)


def exl3_linears(model: MlxLmModel) -> list[tuple[str, EXL3Linear]]:
    return [
        (path, m)
        for path, m in model.named_modules()
        if isinstance(m, EXL3Linear)
    ]


ENGINES = ("exl3", "fold16", "w8a16", "w4a16", "w4gptq")

# Sibling projections called on the same input within one mlx_lm module —
# candidates for one-launch fused groups (must share in_features and K).
_FUSE_SIBLINGS = (
    ("mlp", ("gate_proj", "up_proj")),
    ("self_attn", ("q_proj", "k_proj", "v_proj")),
    ("linear_attn", ("in_proj_qkv", "in_proj_z")),
)


def fuse_exl3_siblings(model: MlxLmModel) -> int:
    """Replace same-K EXL3Linear sibling groups with FusedEXL3Group views.

    Only groups above ~64 MB (fp16-equivalent) are fused: the fused chain
    (prep -> one kernel -> finish -> split) trades the latency overlap of
    independent sibling GEMVs for throughput, which only pays off when the
    kernels are big (measured: 27B +prefill +decode, 2B decode regressed 2x).
    """
    from ponyexl3.mlx.exl3_linear import HUGE_WEIGHT_BYTES
    from ponyexl3.mlx.exl3_fused import FusedEXL3Group, fusable

    # EXL3_FUSE_MIN_MB overrides the 64 MB gate: 35B-A3B's dense projections
    # (qkv+z 50 MB, attn qkv 38 MB) sit just under it, leaving 131 unfused
    # GEMV chains per decode token (Phase 23 census).
    min_bytes = int(
        float(os.environ.get("EXL3_FUSE_MIN_MB", HUGE_WEIGHT_BYTES / 1048576))
        * 1048576
    )
    # EXL3_FUSE_ONLY / EXL3_FUSE_SKIP restrict fusion by group type
    # (consumer-topology experiments, Phase 23c: fusing members whose
    # outputs are NOT consumed together — DeltaNet's z with qkv — extends
    # the serial critical path and regresses decode even when the fused
    # kernel itself is faster in isolation).
    only = os.environ.get("EXL3_FUSE_ONLY", "")
    skip = os.environ.get("EXL3_FUSE_SKIP", "")

    # The fused group owns an independent concatenated trellis, so each
    # member's per-layer runtime becomes dead once it is replaced. Release
    # those buffers as we go (every few groups) so they don't pile up
    # alongside the accumulating fused buffers — that coexistence is the
    # dominant load-time memory spike on this path (measured ~21 GB device
    # peak collapsing to ~15 GB resident once the members are freed).
    from ponyexl3.mlx.layer_state import clear_layer_caches

    n = 0
    for layer in model.layers:
        for owner_name, names in _FUSE_SIBLINGS:
            if only and owner_name != only:
                continue
            if skip and owner_name == skip:
                continue
            owner = getattr(layer, owner_name, None)
            if owner is None:
                continue
            mods = [getattr(owner, nm, None) for nm in names]
            exl3_mods = [m for m in mods if isinstance(m, EXL3Linear)]
            if len(exl3_mods) != len(names):
                continue
            members = [m._exl3 for m in exl3_mods]  # pyright: ignore[reportPrivateUsage]
            group_bytes = sum(l.in_features * l.out_features * 2 for l in members)
            if group_bytes <= min_bytes or not fusable(members):
                continue
            group = FusedEXL3Group(members)
            for i, nm in enumerate(names):
                setattr(owner, nm, group.sibling(i))
            n += 1
            # drop the just-replaced members' cached runtimes immediately so
            # they never accumulate alongside the growing fused buffers
            clear_layer_caches()
            mx.clear_cache()
    return n


def convert_engine(
    model: MlxLmModel,
    engine: str,
    *,
    group_size: int = 64,
    report_errors: bool = False,
    verbose: bool = False,
    sidecar_dir: str | None = None,
) -> list[Any]:
    """Convert EXL3Linear modules in-place to the requested engine.

    - ``exl3``   — exact trellis decode everywhere (no-op).
    - ``fold16`` — fp16 public-weight fold into plain ``nn.Linear`` (EXACT up to
      ~1 ulp fp16 rounding of the folded weight; preserves EXL3 accuracy).
      Huge layers (lm_head) keep the exact trellis GEMV.
    - ``w8a16`` / ``w4a16`` — affine requantization. **Lossy vs EXL3**; opt-in
      only, returns per-layer error reports — validate before trusting.
    """
    from ponyexl3.mlx.layer_state import clear_layer_caches
    from ponyexl3.mlx.native import (
        folded_linear_from_exl3,
        layer_error,
        quantized_linear_from_exl3,
    )

    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}, expected one of {ENGINES}")
    if engine == "exl3":
        return []

    if engine == "w4gptq":
        # GPTQ sidecar produced by tools/gptq_convert.py: activation-aware
        # affine w4 of every decoder linear; lm_head (and anything not in
        # the sidecar) keeps the exact trellis GEMV. Lossy vs EXL3 — the
        # opt-in speed/accuracy lever; drift measured in tools/drift_eval.py.
        import glob as _glob

        from ponyexl3.mlx.native import _ql_from_parts  # pyright: ignore[reportPrivateUsage]

        if sidecar_dir is None or not os.path.isdir(sidecar_dir):
            raise ValueError(
                f"w4gptq needs the sidecar from tools/gptq_convert.py ({sidecar_dir})"
            )
        side = {}
        for f in sorted(_glob.glob(os.path.join(sidecar_dir, "chunk_*.safetensors"))):
            side.update(load_safetensors(f))
        n = 0
        for path, mod in exl3_linears(model):
            key = path
            if key.startswith("language_model."):
                key = key[len("language_model."):]
            if f"{key}.weight" not in side:
                continue
            new_mod = _ql_from_parts(
                side[f"{key}.weight"],
                side[f"{key}.scales"],
                side[f"{key}.biases"],
                group_size=group_size,
                bits=4,
            )
            new_mod._exl3_engine = engine
            _set_module(model, path, new_mod)
            n += 1
        from ponyexl3.mlx.layer_state import clear_layer_caches

        clear_layer_caches()
        from ponyexl3.mlx.native import fuse_mlps

        fused = fuse_mlps(model)
        if verbose:
            print(f"  w4gptq: {n} linears from sidecar, {fused} MLPs fused")
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        return []

    errors = []
    for path, mod in exl3_linears(model):
        layer = mod._exl3  # pyright: ignore[reportPrivateUsage]
        if engine == "fold16":
            new = folded_linear_from_exl3(layer)
        else:
            bits = 8 if engine == "w8a16" else 4
            if mod._huge:  # pyright: ignore[reportPrivateUsage]
                bits = max(bits, 8)
            new = quantized_linear_from_exl3(layer, bits=bits, group_size=group_size)
        new._exl3_engine = engine
        if report_errors:
            err = layer_error(layer, new, engine)
            errors.append(err)
            if verbose:
                print(f"  {err}")
        _set_module(model, path, new)
        # Decoded-W / stripe / runtime caches for this layer are now dead weight.
        clear_layer_caches()

    from ponyexl3.mlx.native import fuse_mlps

    fused = fuse_mlps(model)
    if verbose and fused:
        print(f"  fused gate+up in {fused} MLPs")
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return errors


def _apply_device_memory_limit() -> None:
    """Cap MLX memory to the GPU's recommended working set so the process
    never wires more than the OS allows. A 32 GB Mac kills the process at
    ~26.5 GB *wired*, and MLX's buffer cache will otherwise grow well past
    that during a heavy load. Override with ``PONYEXL3_MEM_LIMIT_GB=<n>``;
    disable the cap with ``PONYEXL3_MEM_LIMIT_GB=0``."""
    env = os.environ.get("PONYEXL3_MEM_LIMIT_GB", "").strip()
    try:
        if env:
            gb = float(env)
            if gb <= 0:
                return
            lim = int(gb * 1024**3)
        else:
            info = mx.device_info()
            ws = int(info.get("max_recommended_working_set_size", 0))
            if ws <= 0:
                return
            lim = int(ws * 0.92)  # leave headroom below the wired ceiling
        mx.set_memory_limit(lim)
    except Exception:
        pass


def load_model(
    model_dir: str,
    *,
    engine: str = "exl3",
    warm: bool = True,
    verbose: bool = False,
    group_size: int = 64,
    report_errors: bool = False,
) -> tuple[MlxLmModel, dict[str, Any]]:
    """Load an EXL3 checkpoint into an mlx_lm skeleton.

    Returns ``(model, config)``. ``engine`` selects the linear implementation
    (see :func:`convert_engine`); ``fold16`` is the default fast engine and
    preserves EXL3 accuracy. ``warm=True`` pre-decodes the cached fp16 ``W``
    for the exact engine (no-op for converted engines).
    """
    _apply_device_memory_limit()
    config = _read_json(model_dir, "config.json")
    qcfg = _read_json(model_dir, "quantization_config.json")
    if qcfg.get("quant_method", config.get("quantization_config", {}).get("quant_method")) != "exl3":
        raise ValueError(f"{model_dir} is not an EXL3 checkpoint")

    model_type = config["model_type"]
    arch_name = _ARCHITECTURES.get(model_type)
    if arch_name is None:
        raise ValueError(f"no architecture mapping for model_type={model_type!r}")
    arch = importlib.import_module(f"mlx_lm.models.{arch_name}")

    cfg = dict(config)
    if "text_config" in cfg:
        text_cfg = dict(cfg["text_config"])
        # The checkpoint carries a separately quantized lm_head (head_bits);
        # untie so the skeleton exposes an lm_head slot we can replace.
        text_cfg["tie_word_embeddings"] = False
        cfg["text_config"] = text_cfg
    else:
        cfg["tie_word_embeddings"] = False
    model = arch.Model(arch.ModelArgs.from_dict(cfg))

    weights = _load_all_tensors(model_dir)
    storage = _exl3_storage(qcfg)

    # 1) Swap each quantized linear for EXL3Linear (MoE experts are stacked
    #    into EXL3SwitchGLU modules instead).
    #
    #    Free each layer's source tensors from ``weights`` as it is converted,
    #    and release the buffer cache periodically. Without this, the whole
    #    mmap'd source stays resident through the build loop *alongside* the
    #    accumulating device runtime, so the load transient peaks at ~2.7x the
    #    resident footprint (measured 42 GB RSS for a 15 GB model) — enough to
    #    OOM-kill a 32 GB Mac during load even though the model runs in ~15 GB
    #    once loaded.
    moe_consumed = _build_moe_experts(model, storage, weights, verbose)
    if moe_consumed:
        mx.clear_cache()
    for n, (key, info) in enumerate(sorted(storage.items())):
        if _expert_match(key) or key in moe_consumed:
            continue
        layer = _build_exl3_layer(key, info, weights)
        _set_module(model, _module_path(key), EXL3Linear(layer))
        for sfx in _EXL3_SUFFIXES:
            weights.pop(f"{key}.{sfx}", None)
        # release the just-consumed source buffer every layer so it never
        # accumulates alongside the growing runtime
        mx.clear_cache()
        if verbose:
            print(f"  exl3  {key}  {layer.in_features}x{layer.out_features} k={layer.k}")
    mx.clear_cache()

    # 2) Everything that is not an EXL3 tensor loads as a plain weight.
    #    (EXL3 source tensors were popped above; this filter is a backstop.)
    exl3_tensor_keys = {
        f"{key}.{sfx}" for key in storage for sfx in _EXL3_SUFFIXES
    }
    plain = {k: v for k, v in weights.items() if k not in exl3_tensor_keys}
    plain = model.sanitize(plain)
    plain = {
        k: v.astype(mx.float32 if k.endswith("A_log") else mx.float16)
        for k, v in plain.items()
    }
    model.load_weights(list(plain.items()), strict=True)

    errors = convert_engine(
        model,
        engine,
        group_size=group_size,
        report_errors=report_errors,
        verbose=verbose,
        sidecar_dir=os.path.join(model_dir, ".pony_cache", f"gptq_w4g{group_size}")
        if engine == "w4gptq"
        else None,
    )
    if errors:
        worst = max(errors, key=lambda e: e.rel)
        print(
            f"[{engine}] requantized {len(errors)} layers — "
            f"worst rel RMS err {worst.rel:.2e} ({worst.key}). "
            f"This engine is lossy vs EXL3; validate end-to-end output."
        )

    if engine in ("exl3", "w4gptq"):
        from ponyexl3.mlx.deltanet_patch import install_deltanet_glue

        install_deltanet_glue()
        fused = fuse_exl3_siblings(model)
        if fused:
            # Per-member runtime uploads are now served by the fused groups.
            from ponyexl3.mlx.layer_state import clear_layer_caches

            clear_layer_caches()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            if verbose:
                print(f"  fused {fused} sibling groups")

        from ponyexl3.mlx.exl3_mlp_monolith import install_mlp_monoliths

        mono = install_mlp_monoliths(model)
        if mono and verbose:
            print(f"  mlp monolith {mono} layers")

        if engine == "exl3":
            # Release the host-side numpy trellis now that the device runtime,
            # fused groups, and monoliths own the weights. On unified memory it
            # was dead weight competing with the KV cache (~4.6 GB / ~225k
            # tokens of context on a 27B). No-op under EXL3_WCACHE.
            released = 0
            for _, m in exl3_linears(model):
                released += 1 if getattr(m._exl3, "trellis", None) is not None else 0
                m.release_source()
            mx.clear_cache()
            if verbose:
                print(f"  released host trellis for {released} layers")

    if warm and engine == "exl3":
        for _, m in exl3_linears(model):
            m.warm()
    mx.eval(model.parameters())
    # Best-effort release of the build-transient buffer cache. The hard guard
    # against the OS wired-memory kill is the working-set cap applied at entry
    # (_apply_device_memory_limit): MLX reclaims its cache to stay under that
    # cap, so even though the freed build buffers (~16 GB on the MoE) settle
    # into the pool asynchronously, total wired never exceeds the GPU's
    # recommended working set (a 32 GB Mac kills at ~26.5 GB wired).
    mx.clear_cache()
    model.eval()
    return model, config


def describe(model: MlxLmModel) -> str:
    """One-line summary of the loaded engine mix."""
    exl3 = exl3_linears(model)
    converted = [
        (p, m)
        for p, m in model.named_modules()
        if getattr(m, "_exl3_engine", None) is not None
    ]
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    parts = []
    if exl3:
        huge = sum(1 for _, m in exl3 if m._huge)  # pyright: ignore[reportPrivateUsage]
        parts.append(f"{len(exl3)} exact EXL3 linears ({huge} huge/GEMV)")
    if converted:
        engines = {getattr(m, "_exl3_engine") for _, m in converted}
        parts.append(f"{len(converted)} converted linears ({'/'.join(sorted(engines))})")
    parts.append(f"{n_params / 1e6:.0f}M params in tree")
    return ", ".join(parts)
