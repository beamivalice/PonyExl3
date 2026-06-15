"""Temperature-correct speculative sampling (Leviathan & Chen, 2023).

PonyExl3's drafters (MTP, DFlash, EAGLE-3) verify greedily: a draft is accepted
iff it equals the target's argmax. That is exact only at ``temp == 0``; at real
settings (``temp=0.6``) it quietly changes the output distribution. This module
is the exact alternative — the draft proposes from its own distribution ``q``,
the target accepts each draft token ``x`` with probability ``min(1, p(x)/q(x))``,
and on the first rejection resamples the bonus from the residual ``(p - q)+``
(renormalized). By the Leviathan-Chen theorem the emitted stream is distributed
**exactly** like plain sampling from the target ``p`` — same distribution, faster.

Ported from pony's ``mtp_sampling.py``. The accept math is the risky part, so it
is pure host-side numpy with a Monte-Carlo distribution-parity self-test:

    python -m ponyexl3.mlx.spec_sampling
"""

from __future__ import annotations

import numpy as np


def temp_topp_probs(logits, temp: float, top_p: float = 1.0) -> np.ndarray:
    """logits [V] -> sampling distribution [V] under temperature (+ optional
    nucleus top_p). The SAME transform plain decoding applies, so target ``p``
    and draft ``q`` are produced identically — what makes the accept exact."""
    l = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temp <= 0:
        p = np.zeros_like(l)
        p[int(np.argmax(l))] = 1.0
        return p
    l = l / temp
    l -= l.max()
    p = np.exp(l)
    p /= p.sum()
    if top_p and top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        cutoff = int(np.searchsorted(cum, top_p)) + 1
        keep = order[:cutoff]
        masked = np.zeros_like(p)
        masked[keep] = p[keep]
        s = masked.sum()
        if s > 0:
            p = masked / s
    return p


def sample_token(logits, temp: float, top_p: float, rng: np.random.Generator) -> int:
    """Draw one token from the (temp, top_p) distribution of ``logits``. Drafters
    must propose from this exact ``q`` for the accept rule to stay distribution-
    exact."""
    if temp <= 0:
        return int(np.argmax(np.asarray(logits).reshape(-1)))
    q = temp_topp_probs(logits, temp, top_p)
    return int(rng.choice(q.shape[0], p=q))


def speculative_accept(p_probs: np.ndarray, q_probs: np.ndarray,
                       draft_tokens, rng: np.random.Generator) -> tuple[int, int]:
    """Leviathan-Chen rejection accept over a drafted chain.

    p_probs [k+1, V] : target sampling dists at positions 0..k (verify forward).
    q_probs [k, V]   : draft sampling dists at positions 0..k-1.
    draft_tokens [k] : the sampled draft tokens.

    Returns ``(n_accepted, bonus_token)`` — the first ``n_accepted`` drafts are
    emitted, then ``bonus_token``; the emitted sequence is distributed as ``p``.
    """
    k = len(draft_tokens)
    for j in range(k):
        x = int(draft_tokens[j])
        qx = float(q_probs[j, x])
        px = float(p_probs[j, x])
        accept_prob = 1.0 if qx <= 0.0 else min(1.0, px / qx)
        if rng.random() < accept_prob:
            continue
        resid = np.clip(p_probs[j] - q_probs[j], 0.0, None)
        s = resid.sum()
        bonus = (int(rng.choice(resid.shape[0], p=resid / s)) if s > 0
                 else int(np.argmax(p_probs[j])))
        return j, bonus
    return k, int(rng.choice(p_probs.shape[1], p=p_probs[k]))


def accept_drafts(target_logits, draft_logits, draft_tokens, *,
                  temp: float, top_p: float = 1.0,
                  rng: np.random.Generator | None = None) -> tuple[int, int]:
    """Unified accept for the drafters: greedy at ``temp == 0`` (the existing
    fast argmax-match), Leviathan-Chen rejection sampling otherwise.

    target_logits [k+1, V] : lm_head over [pending, draft_0 .. draft_{k-1}].
    draft_logits  [k, V]   : the drafter's logits that produced each draft token.
    draft_tokens  [k]      : the sampled drafts.
    """
    tgt = np.asarray(target_logits)
    drafts = [int(t) for t in np.asarray(draft_tokens).reshape(-1)]
    k = len(drafts)
    if temp <= 0:
        preds = np.argmax(tgt, axis=-1)
        m = 0
        while m < k and int(preds[m]) == drafts[m]:
            m += 1
        return m, int(preds[m])
    if rng is None:
        rng = np.random.default_rng()
    drf = np.asarray(draft_logits)
    p = np.stack([temp_topp_probs(tgt[j], temp, top_p) for j in range(k + 1)])
    q = np.stack([temp_topp_probs(drf[j], temp, top_p) for j in range(k)])
    return speculative_accept(p, q, drafts, rng)


# --------------------------------------------------------------------------- #
# self-test: the emitted distribution must equal plain sampling from p (KL ~ 0)
# --------------------------------------------------------------------------- #
def _kl(emp: np.ndarray, p: np.ndarray) -> float:
    m = (emp > 0) & (p > 0)
    return float(np.sum(emp[m] * np.log(emp[m] / p[m])))


def _selftest() -> None:
    rng = np.random.default_rng(0)
    V, N = 48, 120_000
    for temp, top_p in [(1.0, 1.0), (0.8, 1.0), (0.6, 0.95), (1.0, 0.9)]:
        tl = rng.standard_normal(V) * 2.0
        dl = tl + rng.standard_normal(V) * 1.5
        p = temp_topp_probs(tl, temp, top_p)
        q = temp_topp_probs(dl, temp, top_p)
        counts = np.zeros(V)
        p2 = np.stack([p, p])
        for _ in range(N):
            x = int(rng.choice(V, p=q))
            n_acc, bonus = speculative_accept(p2, q[None], [x], rng)
            counts[x if n_acc >= 1 else bonus] += 1
        emp = counts / counts.sum()
        kl = _kl(emp, p)
        tv = 0.5 * float(np.abs(emp - p).sum())
        print(f"temp={temp} top_p={top_p}: KL(emp||p)={kl:.5f}  TV={tv:.4f}")
        assert kl < 2e-3, f"distribution parity broken: KL={kl}"
    tl = rng.standard_normal(V) * 2.0
    p = temp_topp_probs(tl, 0.8, 1.0)
    greedy = np.zeros(V); greedy[int(np.argmax(tl))] = 1.0
    print(f"greedy-accept TV vs true p (temp=0.8): {0.5*np.abs(greedy-p).sum():.3f}  "
          f"(the bug temp-correct fixes)")
    print("spec_sampling self-test: PASS")


if __name__ == "__main__":
    _selftest()
