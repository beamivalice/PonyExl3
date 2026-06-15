"""Gate the temperature-correct speculative accept (distribution parity)."""

import numpy as np

from ponyexl3.mlx.spec_sampling import (
    accept_drafts,
    speculative_accept,
    temp_topp_probs,
)


def test_speculative_accept_distribution_parity():
    """Leviathan-Chen: the emitted token's distribution must equal plain
    sampling from the target ``p`` (KL ~ 0), at several temp/top_p settings."""
    rng = np.random.default_rng(0)
    V, N = 48, 60_000
    for temp, top_p in [(1.0, 1.0), (0.8, 1.0), (0.6, 0.95)]:
        tl = rng.standard_normal(V) * 2.0
        dl = tl + rng.standard_normal(V) * 1.5  # draft: correlated but off
        p = temp_topp_probs(tl, temp, top_p)
        q = temp_topp_probs(dl, temp, top_p)
        counts = np.zeros(V)
        p2 = np.stack([p, p])
        for _ in range(N):
            x = int(rng.choice(V, p=q))
            n_acc, bonus = speculative_accept(p2, q[None], [x], rng)
            counts[x if n_acc >= 1 else bonus] += 1
        emp = counts / counts.sum()
        m = (emp > 0) & (p > 0)
        kl = float(np.sum(emp[m] * np.log(emp[m] / p[m])))
        assert kl < 5e-3, f"temp={temp} top_p={top_p}: KL={kl}"


def test_greedy_accept_is_argmax_match():
    """At temp 0, ``accept_drafts`` reduces to the greedy argmax-prefix match."""
    rng = np.random.default_rng(1)
    V, k = 32, 3
    tgt = rng.standard_normal((k + 1, V))
    preds = tgt.argmax(-1)
    drafts = [int(preds[0]), int(preds[1]), (int(preds[2]) + 1) % V]  # 2 match, 1 not
    m, bonus = accept_drafts(tgt, None, drafts, temp=0.0)
    assert m == 2
    assert bonus == int(preds[2])
