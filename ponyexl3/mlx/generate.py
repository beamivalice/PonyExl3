"""Minimal token generation loop for EXL3 MLX models (Phase 4).

Deliberately separate from mlx_lm's generate: prefill runs through the inner
text model only, so the (huge, GEMV-routed) lm_head is applied to exactly one
position per step instead of every prompt position.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx

from ponyexl3.types import DraftModule, KvCache, MlxLmModel, Tokenizer


def _lm_parts(model: Any) -> tuple[Any, Any, Any]:
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    return lm, inner, lm.lm_head


def max_position_embeddings(config: dict[str, Any]) -> int | None:
    text_cfg = config.get("text_config", config)
    limit = text_cfg.get("max_position_embeddings")
    if limit is None:
        return None
    return int(limit)


def validate_generation_params(
    prompt_ids: list[int],
    *,
    max_tokens: int,
    prefill_chunk: int,
    num_draft: int = 1,
    using_spec: bool = False,
    max_context: int | None = None,
) -> None:
    if len(prompt_ids) == 0:
        raise ValueError("prompt is empty after encoding")
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
    if prefill_chunk <= 0:
        raise ValueError(f"prefill_chunk must be positive, got {prefill_chunk}")
    if using_spec and num_draft <= 0:
        raise ValueError(
            f"num_draft must be positive when using speculation, got {num_draft}"
        )
    if max_context is not None and max_context > 0:
        total = len(prompt_ids) + max_tokens
        if total > max_context:
            raise ValueError(
                f"context {total} tokens (prefill {len(prompt_ids)} + gen {max_tokens}) "
                f"exceeds max_position_embeddings={max_context}"
            )


def _prefill_hidden(
    model: Any,
    toks: mx.array,
    cache: KvCache,
    *,
    chunk: int,
) -> mx.array:
    """Run chunked prefill; ``toks`` must have sequence length >= 1."""
    end = min(chunk, toks.shape[1])
    h = model(toks[:, :end], cache=cache)
    for s0 in range(chunk, toks.shape[1], chunk):
        h = model(toks[:, s0 : s0 + chunk], cache=cache)
    return h


@dataclass
class GenStats:
    prompt_tokens: int = 0
    prefill_s: float = 0.0
    gen_tokens: int = 0
    decode_s: float = 0.0
    finish_reason: str = "length"
    spec_cycles: int = 0
    spec_drafted: int = 0
    spec_accepted: int = 0

    def summary(self) -> str:
        pf = self.prompt_tokens / self.prefill_s if self.prefill_s else 0.0
        dc = self.gen_tokens / self.decode_s if self.decode_s else 0.0
        s = (
            f"prompt {self.prompt_tokens} tok in {self.prefill_s:.2f}s ({pf:.1f} tok/s) | "
            f"gen {self.gen_tokens} tok in {self.decode_s:.2f}s ({dc:.1f} tok/s) | "
            f"stop: {self.finish_reason}"
        )
        if self.spec_cycles:
            s += (
                f" | spec: {self.spec_accepted}/{self.spec_drafted} drafts accepted, "
                f"{self.gen_tokens / self.spec_cycles:.2f} tok/cycle"
            )
        return s


def _sample(logits: mx.array, temp: float) -> mx.array:
    logits = logits.astype(mx.float32)
    if temp <= 0.0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits * (1.0 / temp))


def stream_generate(
    model: MlxLmModel,
    prompt_ids: list[int],
    *,
    max_tokens: int = 256,
    temp: float = 0.0,
    prefill_chunk: int = 2048,
    eos_ids: set[int] | frozenset[int] = frozenset(),
    stats: GenStats | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time from wrapped or bare mlx_lm models."""
    validate_generation_params(
        prompt_ids,
        max_tokens=max_tokens,
        prefill_chunk=prefill_chunk,
    )
    lm, inner, lm_head = _lm_parts(model)
    cache = lm.make_cache()
    stats = stats if stats is not None else GenStats()
    stats.prompt_tokens = len(prompt_ids)

    tic = time.perf_counter()
    toks = mx.array([prompt_ids])
    h = _prefill_hidden(inner, toks, cache, chunk=prefill_chunk)
    mx.eval(h)
    logits = lm_head(h[:, -1:, :])
    mx.eval(logits)
    stats.prefill_s = time.perf_counter() - tic

    def _step(prev_y: mx.array) -> mx.array:
        h = inner(prev_y[:, None], cache=cache)
        return lm_head(h)

    # Pipelined decode: while the host syncs on token t (item() + detokenize),
    # the GPU is already running step t+1.
    tic = time.perf_counter()
    y = _sample(logits[:, -1, :], temp)
    mx.async_eval(y)
    for _ in range(max_tokens):
        next_logits = _step(y)
        next_y = _sample(next_logits[:, -1, :], temp)
        mx.async_eval(next_y)
        tok = int(y.item())
        if tok in eos_ids:
            stats.finish_reason = "stop"
            break
        stats.gen_tokens += 1
        yield tok
        y = next_y
    stats.decode_s = time.perf_counter() - tic


def _snapshot_recurrent(cache: list[Any]) -> list[Any]:
    """Hold references to recurrent (ArraysCache) states. mlx_lm's DeltaNet
    REPLACES cache entries with new arrays each step (functional updates), so
    keeping the old references is a complete snapshot."""
    from mlx_lm.models.cache import ArraysCache

    return [list(c.cache) if isinstance(c, ArraysCache) else None for c in cache]


def _restore_caches(cache: list[Any], snap: list[Any], fed_tokens: int) -> None:  # pyright: ignore[reportUnusedFunction]
    """Roll back after a partially-accepted verify forward: recurrent states
    are restored from the snapshot, KV caches trim their offset."""
    for c, s in zip(cache, snap):
        if s is not None:
            c.cache = list(s)
        else:
            c.trim(fed_tokens)


class _DeltaNetTrace:
    """Capture each GatedDeltaNet's scan inputs during a verify forward.

    Hiddens at accepted positions of a verify forward are already true (by
    causality) — only the DeltaNet cache states are wrong after a partial
    acceptance. The post-conv q/k/v and the b/a projections at accepted
    positions are also already true, so with the glue patch installed
    (deltanet_patch) repair re-runs ONLY ``gated_delta_update`` on truncated
    slices and rebuilds the conv state by slicing — no EXL3 projections, no
    conv, no out_proj (measured: ~21 -> ~2 ms per partial-accept cycle).

    Without the glue patch, falls back to capturing layer inputs and
    re-running the whole module (the previous behavior)."""

    def __init__(self) -> None:
        self.records: list[Any] = []
        self._cls: type[Any] | None = None
        self._glue: bool = False
        self._sink_mod: Any = None
        self._orig: Callable[..., Any] | None = None

    def __enter__(self) -> _DeltaNetTrace:
        import os

        # qwen3_5 defines its own GatedDeltaNet; patch that class
        from mlx_lm.models import qwen3_5 as _q5

        self.records = []
        self._cls = _q5.GatedDeltaNet
        # EXL3_SPEC_REPAIR=module forces the legacy full-module replay
        self._glue = getattr(self._cls, "_exl3_glue", False) and (
            os.environ.get("EXL3_SPEC_REPAIR", "scan") != "module"
        )
        if self._glue:
            from ponyexl3.mlx import deltanet_patch

            self._sink_mod = deltanet_patch
            deltanet_patch.set_trace_sink(self.records)
        else:
            trace = self
            self._orig = self._cls.__call__

            def wrapped(
                mod: Any,
                x: mx.array,
                mask: mx.array | None = None,
                cache: KvCache | None = None,
            ) -> Any:
                trace.records.append(("module", mod, trace._orig, x, cache))
                assert trace._orig is not None
                return trace._orig(mod, x, mask=mask, cache=cache)

            self._cls.__call__ = wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> bool:
        if self._glue:
            self._sink_mod.set_trace_sink(None)
        else:
            assert self._orig is not None
            self._cls.__call__ = self._orig  # type: ignore[method-assign]
        return False

    def repair(self, keep_tokens: int) -> None:
        """Advance the (already restored) pre-verify caches by the accepted
        tokens — bit-identical to the legacy full-module replay.

        keep >= 2: slice the verify's post-conv q/k/v and b/a (rows <= 8 all
        ride the same simd GEMM kernel, so the slices ARE the replay's bits;
        verified 96/96 arrays bit-equal) and re-run only the scan.

        keep == 1: a 1-row replay takes the mt=1 GEMV kernels — different
        reduction order than the verify's GEMM, and the one plain greedy
        steps with. Slicing here was measured to drift near-tie argmaxes
        (text diverged from plain greedy by cycle ~40), so recompute the
        single row through the same mt=1 path the module replay used,
        skipping the ops that don't touch the cache (z-gate, out_proj)."""
        from mlx_lm.models.gated_delta import gated_delta_update

        for rec in self.records:
            if rec[0] == "module":
                _, mod, orig, x, cache = rec
                orig(mod, x[:, :keep_tokens], mask=None, cache=cache)
                continue
            _, mod, cache, x, conv_state, qkv, q, k, v, a, b, state = rec
            if keep_tokens == 1:
                from ponyexl3.mlx.deltanet_patch import _pre_fn  # pyright: ignore[reportPrivateUsage]

                x1 = x[:, :1]
                qkv1 = mod.in_proj_qkv(x1)
                b1 = mod.in_proj_b(x1)
                a1 = mod.in_proj_a(x1)
                pre = _pre_fn(
                    mod.key_dim,
                    mod.num_k_heads,
                    mod.num_v_heads,
                    mod.head_k_dim,
                    mod.head_v_dim,
                    mod.conv_dim,
                    mod.conv_kernel_size - 1,
                )
                q1, k1, v1, new_conv = pre(qkv1, conv_state, mod.conv1d.weight)
                cache[0] = mx.contiguous(new_conv)
            else:
                n_keep = conv_state.shape[1]
                conv_input = mx.concatenate(
                    [conv_state, qkv[:, :keep_tokens]], axis=1
                )
                cache[0] = mx.contiguous(conv_input[:, -n_keep:])
                q1 = q[:, :keep_tokens]
                k1 = k[:, :keep_tokens]
                v1 = v[:, :keep_tokens]
                a1 = a[:, :keep_tokens]
                b1 = b[:, :keep_tokens]
            _, st = gated_delta_update(
                q1,
                k1,
                v1,
                a1,
                b1,
                mod.A_log,
                mod.dt_bias,
                state,
                None,
                use_kernel=not mod.training,
            )
            cache[1] = st
            cache.advance(keep_tokens)


def speculative_stream_generate(
    model: MlxLmModel,
    mtp: DraftModule,
    prompt_ids: list[int],
    *,
    max_tokens: int = 256,
    num_draft: int = 3,
    prefill_chunk: int = 2048,
    eos_ids: set[int] | frozenset[int] = frozenset(),
    stats: GenStats | None = None,
    temp: float = 0.0,
    top_p: float = 1.0,
):
    """Speculative decoding with the MTP draft head (exllamav3 port).

    Per cycle: chain ``num_draft`` MTP drafts, verify them in ONE batched
    target forward (which reads each weight once for all rows — that is the
    speedup), accept the longest matching prefix plus the bonus token.

    At ``temp == 0`` the accept is greedy (draft == target argmax) and output
    is identical to plain greedy. At ``temp > 0`` it switches to Leviathan-Chen
    rejection sampling (``spec_sampling.accept_drafts``): drafts are sampled
    from their own distribution and accepted so the emitted stream matches plain
    sampling from the target **exactly** — same distribution, just faster.

    Hybrid-cache rollback: hiddens at accepted positions from the verify
    forward are already true (causality); only the cache needs repair. KV
    caches trim; DeltaNet recurrent states restore from a snapshot and the
    accepted tokens replay (only on partial acceptance).
    """
    validate_generation_params(
        prompt_ids,
        max_tokens=max_tokens,
        prefill_chunk=prefill_chunk,
        num_draft=num_draft,
        using_spec=True,
    )
    import numpy as np

    from mlx_lm.models.cache import KVCache

    from ponyexl3.mlx.spec_sampling import accept_drafts, sample_token

    rng = np.random.default_rng()

    lm = model.language_model
    embed = lm.model.embed_tokens
    cache = lm.make_cache()
    mtp_cache = KVCache()
    stats = stats if stats is not None else GenStats()
    stats.prompt_tokens = len(prompt_ids)

    tic = time.perf_counter()
    toks = mx.array([prompt_ids])
    hs = []
    for s0 in range(0, toks.shape[1], prefill_chunk):
        h = lm.model(toks[:, s0 : s0 + prefill_chunk], cache=cache)
        mx.eval(h)
        hs.append(h)
    h_all = mx.concatenate(hs, axis=1) if len(hs) > 1 else hs[0]
    logits = lm.lm_head(h_all[:, -1:, :])
    if temp > 0.0:
        mx.eval(logits)
        pending = mx.array([sample_token(np.asarray(logits[0, -1]), temp, top_p, rng)], dtype=mx.int32)
    else:
        pending = mx.argmax(logits[0, -1]).reshape(1).astype(mx.int32)  # not yet fed
    mx.eval(pending)
    stats.prefill_s = time.perf_counter() - tic

    # MTP catch-up pairs: (hidden at pos i, token at pos i+1). The prompt's
    # pairs plus the pending token prime the MTP cache; afterwards each cycle
    # contributes its accepted pairs.
    catch_h = h_all
    catch_t = mx.concatenate([toks[0, 1:].astype(mx.int32), pending])[None]

    # Drafts are gated by the exact verify, so the draft chain may use a
    # quantized lm_head copy (mtp.quantize_draft) — output bits unchanged.
    draft_head = getattr(mtp, "_draft_head", None) or lm.lm_head

    def draft_phase(catch_t: mx.array, catch_h: mx.array):
        # First MTP call also extends the cache with the true catch-up pairs;
        # its last output starts the chain.
        C = catch_t.shape[1]
        end = min(prefill_chunk, C)
        h_mtp = mtp(
            embed(catch_t[:, :end]),
            catch_h[:, :end],
            mtp_cache,
        )
        for s0 in range(prefill_chunk, C, prefill_chunk):
            h_mtp = mtp(
                embed(catch_t[:, s0 : s0 + prefill_chunk]),
                catch_h[:, s0 : s0 + prefill_chunk],
                mtp_cache,
            )
        h_chain = h_mtp[:, -1:, :]
        drafts: list[mx.array] = []
        dlogits: list[np.ndarray] = []  # draft q-logits, only when temp > 0
        for j in range(num_draft):
            d_logits = draft_head(mtp.head_input(h_chain))
            dl = d_logits[0, -1]
            if temp > 0.0:
                mx.eval(dl)
                dlogits.append(np.asarray(dl))
                dj = mx.array([sample_token(dlogits[-1], temp, top_p, rng)], dtype=mx.int32)
            else:
                dj = mx.argmax(dl).reshape(1).astype(mx.int32)
            drafts.append(dj)
            if j < num_draft - 1:
                h_chain = mtp(embed(dj[None]), h_chain, mtp_cache)  # speculative
        return drafts, dlogits

    tic = time.perf_counter()
    emitted = 0
    drafts, dlogits = draft_phase(catch_t, catch_h)
    n_spec_mtp = num_draft - 1
    while emitted < max_tokens:
        # ---- verify in one batched target forward (tracing DeltaNet inputs)
        verify_tokens = mx.concatenate([pending] + drafts)  # (k+1,)
        snap = _snapshot_recurrent(cache)

        def _verify_mtp():
            with _DeltaNetTrace() as trace:
                return trace, lm.model(verify_tokens[None], cache=cache)

        trace, h_ver = cast(tuple[_DeltaNetTrace, mx.array], _verify_mtp())
        target_logits = lm.lm_head(h_ver)[0]  # (k+1, V)
        verify_np = np.array(verify_tokens)
        if temp > 0.0:
            # Leviathan-Chen: draft q-dists vs target p-dists -> exact accept
            m, bonus = accept_drafts(
                np.array(target_logits), np.stack(dlogits), verify_np[1:],
                temp=temp, top_p=top_p, rng=rng,
            )
        else:
            preds_np = np.array(mx.argmax(target_logits, axis=-1))  # one host sync
            m = 0
            while m < num_draft and preds_np[m] == verify_np[m + 1]:
                m += 1
            bonus = int(preds_np[m])
        accepted = [int(v) for v in verify_np[: m + 1]]

        stats.spec_cycles += 1
        stats.spec_drafted += num_draft
        stats.spec_accepted += m

        # ---- cache repair: verify hiddens AND the KV entries at accepted
        # positions are already true (causality) — KV just trims the rejected
        # suffix. Only DeltaNet states are sequentially wrong: restore the
        # pre-verify snapshot and re-run JUST those modules on the truncated
        # traced inputs (~1/6 the cost of a full replay).
        h_acc = h_ver
        if m < num_draft:
            discard = int(verify_tokens.shape[0]) - (m + 1)
            for c, s in zip(cache, snap):
                if s is not None:
                    c.cache = list(s)  # DeltaNet: back to pre-verify
                else:
                    c.trim(discard)  # KV: keep the accepted entries
            trace.repair(m + 1)
        mtp_cache.trim(n_spec_mtp)

        # ---- prepare and LAUNCH the next cycle's draft graph before
        # yielding, so the GPU drafts while the caller detokenizes/prints.
        pending = mx.array([bonus], dtype=mx.int32)
        catch_h = h_acc[:, : m + 1, :]
        catch_t = mx.concatenate(
            [mx.array(accepted[1:], dtype=mx.int32), pending]
        )[None]
        next_drafts, next_dlogits = draft_phase(catch_t, catch_h)
        mx.async_eval(next_drafts[-1])

        # ---- emit (host work overlaps the drafting on the GPU)
        stop = False
        for t in accepted:
            if t in eos_ids:
                stats.finish_reason = "stop"
                stop = True
                break
            stats.gen_tokens += 1
            emitted += 1
            yield t
            if emitted >= max_tokens:
                stop = True
                break
        drafts, dlogits = next_drafts, next_dlogits
        if stop:
            break
    stats.decode_s = time.perf_counter() - tic


def eagle3_stream_generate(
    model: MlxLmModel,
    drafter: DraftModule,
    prompt_ids: list[int],
    *,
    max_tokens: int = 256,
    num_draft: int = 3,
    prefill_chunk: int = 2048,
    eos_ids: set[int] | frozenset[int] = frozenset(),
    stats: GenStats | None = None,
    temp: float = 0.0,
    top_p: float = 1.0,
):
    """Speculative decoding with an EAGLE-3 draft head.

    Clone of ``speculative_stream_generate`` (MTP) with two structural
    swaps: drafter features come from the target's aux residual streams
    (``AuxTrace`` + ``drafter.fuse``) instead of the final hidden, and the
    draft chain runs the drafter's own 32k head with d2t mapping. The verify
    forward keeps the exact target weights and lm_head, so output matches plain
    decoding — greedy at ``temp == 0``, and temp-correct at ``temp > 0`` (the
    draft ``q`` is scattered from the 32k head into the target vocab via d2t)."""
    validate_generation_params(
        prompt_ids,
        max_tokens=max_tokens,
        prefill_chunk=prefill_chunk,
        num_draft=num_draft,
        using_spec=True,
    )
    import numpy as np

    from mlx_lm.models.cache import KVCache

    from ponyexl3.mlx.eagle3 import AuxTrace
    from ponyexl3.mlx.spec_sampling import accept_drafts, sample_token, temp_topp_probs

    rng = np.random.default_rng()

    lm = model.language_model
    embed = lm.model.embed_tokens
    cache = lm.make_cache()
    draft_cache = KVCache()
    stats = stats if stats is not None else GenStats()
    stats.prompt_tokens = len(prompt_ids)

    tic = time.perf_counter()
    toks = mx.array([prompt_ids])
    hs: list[mx.array] = []
    g_all: mx.array
    with AuxTrace(lm.model, drafter.aux_ids) as aux:
        for s0 in range(0, toks.shape[1], prefill_chunk):
            hs.append(lm.model(toks[:, s0 : s0 + prefill_chunk], cache=cache))
            mx.eval(hs[-1])
        g_all = drafter.fuse(aux.take())
    h_all = mx.concatenate(hs, axis=1) if len(hs) > 1 else hs[0]
    logits = lm.lm_head(h_all[:, -1:, :])
    if temp > 0.0:
        mx.eval(logits)
        pending = mx.array([sample_token(np.asarray(logits[0, -1]), temp, top_p, rng)], dtype=mx.int32)
    else:
        pending = mx.argmax(logits[0, -1]).reshape(1).astype(mx.int32)
    mx.eval(pending)
    stats.prefill_s = time.perf_counter() - tic

    catch_g = g_all  # pyright: ignore[reportPossiblyUnboundVariable]
    catch_t = mx.concatenate([toks[0, 1:].astype(mx.int32), pending])[None]

    # temp-correct draft side: sample the drafter's 32k head, map to a target id
    # via d2t, and scatter the 32k logits into the target vocab so the accept
    # sees ``q`` over the same vocab as the target ``p``.
    _Vtgt = int(lm.lm_head.out_features)
    _d2t_np = np.asarray(drafter.d2t)

    def _draft_token(h_chain: mx.array):
        if temp <= 0.0:
            return drafter.draft_token(h_chain).astype(mx.int32), None
        d32 = np.asarray(drafter.draft_logits(h_chain))  # (V_draft,)
        q32 = temp_topp_probs(d32, temp, top_p)
        did = int(rng.choice(q32.shape[0], p=q32))
        qlog = np.full(_Vtgt, -1e30, np.float32)
        qlog[_d2t_np] = d32
        return mx.array([int(_d2t_np[did])], dtype=mx.int32), qlog

    def draft_phase(catch_t: mx.array, catch_g: mx.array):
        C = catch_t.shape[1]
        end = min(prefill_chunk, C)
        h_d = drafter(
            embed(catch_t[:, :end]),
            catch_g[:, :end],
            draft_cache,
        )
        for s0 in range(prefill_chunk, C, prefill_chunk):
            h_d = drafter(
                embed(catch_t[:, s0 : s0 + prefill_chunk]),
                catch_g[:, s0 : s0 + prefill_chunk],
                draft_cache,
            )
        h_chain = h_d[:, -1:, :]
        drafts: list[mx.array] = []
        dlogits: list[np.ndarray] = []
        for j in range(num_draft):
            dj, qlog = _draft_token(h_chain)
            drafts.append(dj)
            if qlog is not None:
                dlogits.append(qlog)
            if j < num_draft - 1:
                h_chain = drafter(embed(dj[None]), h_chain, draft_cache)
        return drafts, dlogits

    tic = time.perf_counter()
    emitted = 0
    drafts, dlogits = draft_phase(catch_t, catch_g)
    n_spec = num_draft - 1
    while emitted < max_tokens:
        verify_tokens = mx.concatenate([pending] + drafts)
        snap = _snapshot_recurrent(cache)

        def _verify_eagle():
            with AuxTrace(lm.model, drafter.aux_ids) as aux:
                with _DeltaNetTrace() as trace:
                    return trace, lm.model(verify_tokens[None], cache=cache), drafter.fuse(aux.take())

        trace, h_ver, g_ver = cast(
            tuple[_DeltaNetTrace, mx.array, mx.array], _verify_eagle()
        )
        verify_np = np.array(verify_tokens)
        if temp > 0.0:
            m, bonus = accept_drafts(
                np.array(lm.lm_head(h_ver)[0]), np.stack(dlogits), verify_np[1:],
                temp=temp, top_p=top_p, rng=rng,
            )
        else:
            preds_np = np.array(mx.argmax(lm.lm_head(h_ver)[0], axis=-1))
            m = 0
            while m < num_draft and preds_np[m] == verify_np[m + 1]:
                m += 1
            bonus = int(preds_np[m])
        accepted = [int(v) for v in verify_np[: m + 1]]

        stats.spec_cycles += 1
        stats.spec_drafted += num_draft
        stats.spec_accepted += m

        if m < num_draft:
            discard = int(verify_tokens.shape[0]) - (m + 1)
            for c, s in zip(cache, snap):
                if s is not None:
                    c.cache = list(s)
                else:
                    c.trim(discard)
            trace.repair(m + 1)
        draft_cache.trim(n_spec)

        pending = mx.array([bonus], dtype=mx.int32)
        catch_g = g_ver[:, : m + 1, :]
        catch_t = mx.concatenate(
            [mx.array(accepted[1:], dtype=mx.int32), pending]
        )[None]
        next_drafts, next_dlogits = draft_phase(catch_t, catch_g)
        mx.async_eval(next_drafts[-1])

        stop = False
        for t in accepted:
            if t in eos_ids:
                stats.finish_reason = "stop"
                stop = True
                break
            stats.gen_tokens += 1
            emitted += 1
            yield t
            if emitted >= max_tokens:
                stop = True
                break
        drafts, dlogits = next_drafts, next_dlogits
        if stop:
            break
    stats.decode_s = time.perf_counter() - tic


def dflash_stream_generate(
    model: MlxLmModel,
    drafter: DraftModule,
    prompt_ids: list[int],
    *,
    max_tokens: int = 256,
    num_draft: int = 7,
    prefill_chunk: int = 2048,
    eos_ids: set[int] | frozenset[int] = frozenset(),
    stats: GenStats | None = None,
    temp: float = 0.0,
    top_p: float = 1.0,
):
    """Speculative decoding with the DFlash block drafter.

    Like the MTP/EAGLE loops, but the drafter holds no autoregressive
    state: its KV is built from fc-fused target features at fed positions
    (appended only for ACCEPTED tokens — no trim, no rollback), and each
    cycle drafts via one 16-token masked block forward. Greedy at ``temp == 0``
    (output identical to plain greedy); Leviathan-Chen temp-correct sampling at
    ``temp > 0`` (the block logits are over the target vocab, so ``q`` is exact)."""
    validate_generation_params(
        prompt_ids,
        max_tokens=max_tokens,
        prefill_chunk=prefill_chunk,
        num_draft=num_draft,
        using_spec=True,
    )
    import numpy as np

    from ponyexl3.mlx.eagle3 import AuxTrace
    from ponyexl3.mlx.spec_sampling import accept_drafts, sample_token

    rng = np.random.default_rng()

    lm = model.language_model
    embed = lm.model.embed_tokens
    cache = lm.make_cache()
    drafter.make_caches()
    stats = stats if stats is not None else GenStats()
    stats.prompt_tokens = len(prompt_ids)
    num_draft = min(num_draft, drafter.block_size - 1)

    tic = time.perf_counter()
    toks = mx.array([prompt_ids])
    h: mx.array
    with AuxTrace(lm.model, drafter.aux_ids) as aux:
        h = _prefill_hidden(lm.model, toks, cache, chunk=prefill_chunk)
        mx.eval(h)
        drafter.update_kv(drafter.fuse(aux.take()))
    logits = lm.lm_head(h[:, -1:, :])  # pyright: ignore[reportPossiblyUnboundVariable]
    if temp > 0.0:
        mx.eval(logits)
        pending = mx.array([sample_token(np.asarray(logits[0, -1]), temp, top_p, rng)], dtype=mx.int32)
    else:
        pending = mx.argmax(logits[0, -1]).reshape(1).astype(mx.int32)
    mx.eval(pending)
    stats.prefill_s = time.perf_counter() - tic

    def _draft(pend: mx.array):
        # greedy: lazy token array; temp-correct: sample each block position
        # from its target-vocab logits and keep them for the accept.
        if temp > 0.0:
            bl = drafter.draft_block(pend, embed, lm.lm_head, num_draft, return_logits=True)
            mx.eval(bl)
            dl = np.asarray(bl)  # (num_draft, V)
            d = mx.array([sample_token(dl[j], temp, top_p, rng) for j in range(num_draft)], dtype=mx.int32)
            return d, dl
        return drafter.draft_block(pend, embed, lm.lm_head, num_draft), None

    tic = time.perf_counter()
    emitted = 0
    drafts, dlogits = _draft(pending)
    while emitted < max_tokens:
        verify_tokens = mx.concatenate([pending, drafts])
        snap = _snapshot_recurrent(cache)

        def _verify_dflash():
            with AuxTrace(lm.model, drafter.aux_ids) as aux:
                with _DeltaNetTrace() as trace:
                    return trace, lm.model(verify_tokens[None], cache=cache), drafter.fuse(aux.take())

        trace, h_ver, g_ver = cast(
            tuple[_DeltaNetTrace, mx.array, mx.array], _verify_dflash()
        )
        verify_np = np.array(verify_tokens)
        if temp > 0.0:
            m, bonus = accept_drafts(
                np.array(lm.lm_head(h_ver)[0]), dlogits, verify_np[1:],
                temp=temp, top_p=top_p, rng=rng,
            )
        else:
            preds_np = np.array(mx.argmax(lm.lm_head(h_ver)[0], axis=-1))
            m = 0
            while m < num_draft and preds_np[m] == verify_np[m + 1]:
                m += 1
            bonus = int(preds_np[m])
        accepted = [int(v) for v in verify_np[: m + 1]]

        stats.spec_cycles += 1
        stats.spec_drafted += num_draft
        stats.spec_accepted += m

        if m < num_draft:
            discard = int(verify_tokens.shape[0]) - (m + 1)
            for c, s in zip(cache, snap):
                if s is not None:
                    c.cache = list(s)
                else:
                    c.trim(discard)
            trace.repair(m + 1)

        # drafter KV: append features for exactly the kept positions —
        # nothing speculative ever entered it, so there is nothing to trim
        drafter.update_kv(g_ver[:, : m + 1, :])

        pending = mx.array([bonus], dtype=mx.int32)
        drafts, dlogits = _draft(pending)
        mx.async_eval(drafts)

        stop = False
        for t in accepted:
            if t in eos_ids:
                stats.finish_reason = "stop"
                stop = True
                break
            stats.gen_tokens += 1
            emitted += 1
            yield t
            if emitted >= max_tokens:
                stop = True
                break
        if stop:
            break
    stats.decode_s = time.perf_counter() - tic


class _NGramIndex:
    """Incremental suffix-match drafter for lookup decoding.

    Maps each n-gram (n in [ngram_min, ngram_max]) to the position right
    AFTER its most recent occurrence that has at least one continuation
    token — so a hit always proposes something. Pure host work, O(1) per
    token; the most recent occurrence wins (locality beats frequency for
    code/structured text)."""

    def __init__(self, ngram_min: int = 2, ngram_max: int = 5):
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.ids: list[int] = []
        self._index: dict[tuple[int, ...], int] = {}

    def extend(self, tokens: list[int]) -> None:
        for t in tokens:
            self.ids.append(t)
            j = len(self.ids) - 1  # ngrams ending at j-1 now have continuation j
            for n in range(self.ngram_min, self.ngram_max + 1):
                if j >= n:
                    self._index[tuple(self.ids[j - n : j])] = j

    def propose(self, max_draft: int, min_n: int | None = None) -> list[int]:
        """Continuation after the longest indexed n-gram matching the
        current suffix. ``min_n`` overrides the floor — callers that stack a
        bet on an in-flight token need higher match precision."""
        ids = self.ids
        lo = self.ngram_min if min_n is None else min_n
        for n in range(self.ngram_max, lo - 1, -1):
            if len(ids) < n + 1:
                continue
            pos = self._index.get(tuple(ids[-n:]))
            if pos is not None and pos < len(ids):
                return ids[pos : pos + max_draft]
        return []


def lookup_stream_generate(
    model: MlxLmModel,
    prompt_ids: list[int],
    *,
    max_tokens: int = 256,
    num_draft: int = 3,
    ngram_min: int = 2,
    ngram_max: int = 5,
    prefill_chunk: int = 2048,
    eos_ids: set[int] | frozenset[int] = frozenset(),
    stats: GenStats | None = None,
):
    """Greedy decoding with draft-free n-gram lookup speculation.

    Two-track pipeline. Default track is exactly ``stream_generate``'s lazy
    chain: each step is BUILT and launched before the previous token's value
    is synced, so unmatched text decodes at baseline speed. When the n-gram
    index — which trails the pipeline by the one in-flight token — has a
    continuation for the current suffix, a verify forward is injected that
    consumes the in-flight token LAZILY plus the proposal's tail as drafts
    (the proposal's head is an implicit bet on the in-flight token: a wrong
    bet just means no drafts match; the verify-accept rule never assumes it).
    rows<=4 ride the mt4 GEMM tile, so a verify costs ~1.74 plain steps and
    pays for itself at >=1 accepted draft. Output is token-identical to
    plain greedy by construction.

    A first naive version of this loop synced before building each forward
    and lost ~35% end-to-end — the ~10 ms/cycle of host graph-build must
    stay hidden under GPU execution.
    """
    validate_generation_params(
        prompt_ids,
        max_tokens=max_tokens,
        prefill_chunk=prefill_chunk,
        num_draft=num_draft,
        using_spec=True,
    )
    import numpy as np

    lm = model.language_model
    cache = lm.make_cache()
    stats = stats if stats is not None else GenStats()
    stats.prompt_tokens = len(prompt_ids)

    tic = time.perf_counter()
    toks = mx.array([prompt_ids])
    h = _prefill_hidden(lm.model, toks, cache, chunk=prefill_chunk)
    mx.eval(h)
    logits = lm.lm_head(h[:, -1:, :])
    stats.prefill_s = time.perf_counter() - tic

    ngrams = _NGramIndex(ngram_min, ngram_max)
    ngrams.extend(list(prompt_ids))

    # y: newest token, not yet fed. y_val: its host value when known (after
    # a verify cycle), else None (in-flight from a plain step).
    y = mx.argmax(logits[0, -1]).reshape(1).astype(mx.int32)
    y_val: int | None = None
    mx.async_eval(y)

    tic = time.perf_counter()
    emitted = 0

    def emit(tok: int) -> bool:
        """Yield-side bookkeeping; returns False on stop."""
        nonlocal emitted
        if tok in eos_ids:
            stats.finish_reason = "stop"
            return False
        stats.gen_tokens += 1
        emitted += 1
        return True

    # Adaptive precision floor: a verify only pays off at ~1 accepted draft
    # (the rows<=4 GEMM tax plus the exposed graph-build gap ≈ one plain
    # step), so empty verifies tighten the floor and productive ones relax
    # it — prompts with nothing to copy stop paying within a few cycles.
    min_n = ngram_min

    while emitted < max_tokens:
        if y_val is not None:
            ngrams.extend([y_val])
            drafts = ngrams.propose(num_draft, min_n=min_n)
        else:
            # suffix ends one token early; prop[0] is the bet on y. The
            # stacked uncertainty needs a stronger match (min_n + 1).
            prop = ngrams.propose(num_draft + 1, min_n=min_n + 1)
            drafts = prop[1:]

        if drafts:
            # ---- verify cycle: feeds [y (lazy), drafts...]
            verify = mx.concatenate(
                [y.reshape(1, 1), mx.array([drafts], dtype=mx.int32)], axis=1
            )
            snap = _snapshot_recurrent(cache)

            def _verify_ngram():
                with _DeltaNetTrace() as trace:
                    return trace, lm.model(verify, cache=cache)

            trace, h = cast(tuple[_DeltaNetTrace, mx.array], _verify_ngram())
            preds = mx.argmax(lm.lm_head(h), axis=-1)[0].astype(mx.int32)
            mx.async_eval(preds)

            tok = int(y.item()) if y_val is None else y_val
            if not emit(tok):
                break
            yield tok
            if y_val is None:
                ngrams.extend([tok])
            if emitted >= max_tokens:
                break

            preds_np = np.array(preds)  # blocks until the verify lands
            m = 0
            while m < len(drafts) and preds_np[m] == drafts[m]:
                m += 1
            stats.spec_cycles += 1
            stats.spec_drafted += len(drafts)
            stats.spec_accepted += m
            if m == 0:
                min_n = min(min_n + 1, ngrams.ngram_max)
            elif m >= 2:
                min_n = max(min_n - 1, ngram_min)
            if m < len(drafts):
                discard = len(drafts) - m
                for c, s in zip(cache, snap):
                    if s is not None:
                        c.cache = list(s)
                    else:
                        c.trim(discard)
                trace.repair(m + 1)

            stop = False
            for t in drafts[:m]:
                if not emit(t):
                    stop = True
                    break
                yield t
                ngrams.extend([t])
                if emitted >= max_tokens:
                    stop = True
                    break
            if stop:
                break
            y = mx.array([int(preds_np[m])], dtype=mx.int32)
            y_val = int(preds_np[m])
        else:
            # ---- plain step, fully pipelined (build precedes any sync)
            h = lm.model(y[:, None], cache=cache)
            next_y = mx.argmax(lm.lm_head(h)[0, -1]).reshape(1).astype(mx.int32)
            mx.async_eval(next_y)
            tok = int(y.item()) if y_val is None else y_val
            if not emit(tok):
                break
            yield tok
            if y_val is None:
                ngrams.extend([tok])
            y = next_y
            y_val = None
    stats.decode_s = time.perf_counter() - tic


def generate_text(
    model: MlxLmModel,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    prompt_ids: list[int] | None = None,
    max_tokens: int = 256,
    temp: float = 0.0,
    prefill_chunk: int = 2048,
    use_chat_template: bool = True,
    extra_eos: tuple[int, ...] = (),
    on_segment: Callable[[str], None] | None = None,
    mtp: DraftModule | None = None,
    num_draft: int = 3,
    lookup: bool = False,
    eagle3: DraftModule | None = None,
    dflash: DraftModule | None = None,
    max_context: int | None = None,
) -> tuple[str, GenStats]:
    """Encode, generate, and detokenize. ``on_segment`` streams text chunks.
    With an ``mtp`` draft module and greedy sampling, uses speculative
    decoding; with ``lookup`` (and no mtp), draft-free n-gram lookup
    speculation (both verified — output identical to plain greedy).

    Pass ``prompt_ids`` to skip string encoding (used by throughput benches
    that need an exact prefill length)."""
    if prompt_ids is None:
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
        else:
            prompt_ids = tokenizer.encode(prompt)
    else:
        prompt_ids = list(prompt_ids)
    prompt_ids = list(cast(list[int], prompt_ids))

    using_spec = bool(
        dflash is not None
        or eagle3 is not None
        or mtp is not None
        or (lookup and temp <= 0.0)
    )
    validate_generation_params(
        prompt_ids,
        max_tokens=max_tokens,
        prefill_chunk=prefill_chunk,
        num_draft=num_draft,
        using_spec=using_spec,
        max_context=max_context,
    )

    eos_ids = set(extra_eos)
    tok_eos = getattr(tokenizer, "eos_token_ids", None)
    if tok_eos:
        eos_ids |= set(tok_eos)
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(tokenizer.eos_token_id)

    stats = GenStats()
    if dflash is not None:
        gen = dflash_stream_generate(
            model,
            dflash,
            prompt_ids,
            max_tokens=max_tokens,
            num_draft=num_draft,
            prefill_chunk=prefill_chunk,
            eos_ids=eos_ids,
            stats=stats,
            temp=temp,
        )
    elif eagle3 is not None:
        gen = eagle3_stream_generate(
            model,
            eagle3,
            prompt_ids,
            max_tokens=max_tokens,
            num_draft=num_draft,
            prefill_chunk=prefill_chunk,
            eos_ids=eos_ids,
            stats=stats,
            temp=temp,
        )
    elif mtp is not None:
        gen = speculative_stream_generate(
            model,
            mtp,
            prompt_ids,
            max_tokens=max_tokens,
            num_draft=num_draft,
            prefill_chunk=prefill_chunk,
            eos_ids=eos_ids,
            stats=stats,
            temp=temp,
        )
    elif lookup and temp <= 0.0:
        gen = lookup_stream_generate(
            model,
            prompt_ids,
            max_tokens=max_tokens,
            num_draft=num_draft,
            prefill_chunk=prefill_chunk,
            eos_ids=eos_ids,
            stats=stats,
        )
    else:
        gen = stream_generate(
            model,
            prompt_ids,
            max_tokens=max_tokens,
            temp=temp,
            prefill_chunk=prefill_chunk,
            eos_ids=eos_ids,
            stats=stats,
        )
    detok = tokenizer.detokenizer
    detok.reset()
    for tok in gen:
        detok.add_token(tok)
        if on_segment is not None:
            seg = detok.last_segment
            if seg:
                on_segment(seg)
    detok.finalize()
    if on_segment is not None:
        seg = detok.last_segment
        if seg:
            on_segment(seg)
    return detok.text, stats
