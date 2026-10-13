"""Offline A/B test simulation with honest statistics.

Users are hashed into control and treatment, each arm ranks with its own policy,
and the outcome is hit@3: did the user click something placed in the top 3 slots
(a proxy for a 3 card feed). This is replay on logged data, not a live test, so it
can only estimate lift on items the user was actually shown.

The randomisation unit is the user, so impressions are first averaged per user.
Treating impressions as independent would understate the variance and overstate
significance, a common mistake in A/B analysis.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import stats as st


def bucket(user_id: str, salt: str = "exp1") -> int:
    h = hashlib.md5(f"{salt}:{user_id}".encode()).hexdigest()
    return int(h, 16) % 2


def per_user_means(user_ids, outcomes) -> dict[str, float]:
    sums, counts = {}, {}
    for u, y in zip(user_ids, outcomes):
        sums[u] = sums.get(u, 0.0) + y
        counts[u] = counts.get(u, 0) + 1
    return {u: sums[u] / counts[u] for u in sums}


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    diffs, chunk = [], 100  # chunked so memory stays small with many users
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        ia = rng.integers(0, len(a), (m, len(a)))
        ib = rng.integers(0, len(b), (m, len(b)))
        diffs.append(b[ib].mean(1) - a[ia].mean(1))
    diffs = np.concatenate(diffs)
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def analyse(control: np.ndarray, treatment: np.ndarray, seed=0) -> dict:
    diff = float(treatment.mean() - control.mean())
    lo, hi = bootstrap_diff_ci(control, treatment, seed=seed)
    t = st.ttest_ind(treatment, control, equal_var=False)
    return {
        "n_users_control": int(len(control)), "n_users_treatment": int(len(treatment)),
        "control_mean": float(control.mean()), "treatment_mean": float(treatment.mean()),
        "abs_lift": diff, "rel_lift": diff / control.mean() if control.mean() else float("nan"),
        "ci95_low": lo, "ci95_high": hi, "p_value": float(t.pvalue),
        "significant": bool(lo > 0 or hi < 0),
    }


def simulate(user_ids, hit_control, hit_treatment, salt="exp1", seed=0) -> dict:
    """hit_control / hit_treatment: per impression outcome under each policy."""
    arms = np.array([bucket(u, salt) for u in user_ids])
    users = np.asarray(user_ids)
    c = per_user_means(users[arms == 0], np.asarray(hit_control)[arms == 0])
    t = per_user_means(users[arms == 1], np.asarray(hit_treatment)[arms == 1])
    return analyse(np.array(list(c.values())), np.array(list(t.values())), seed)
