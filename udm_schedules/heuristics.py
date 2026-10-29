"""CosMap and Logit-Normal timestep schedules (Esser et al., 2024,
https://arxiv.org/abs/2403.03206), used as training-free baselines on the
binomial benchmark.

Both reparametrize an equally spaced quantile grid u in [eps, 1-eps] through
a closed-form inverse CDF:
    cosmap(u)      = 1 - 1/(tan(pi*u/2) + 1)
    logitnormal(u) = sigmoid(Phi^-1(u) * s + m),  m=0, s=1
"""
import math

import numpy as np
import scipy.stats


def cosmap(u):
    return 1.0 - 1.0 / (np.tan(math.pi * u / 2.0) + 1.0)


def logitnormal(u, m=0.0, s=1.0):
    z = scipy.stats.norm.ppf(u) * s + m
    return 1.0 / (1.0 + np.exp(-z))


HEURISTICS = {
    "cosmap": cosmap,
    "logitnormal": logitnormal,
}


def heuristic_schedules(fn, eps, tgt_nfe):
    """One ascending list of NFE+1 times per NFE = 1, 2, 4, ..., tgt_nfe."""
    out = []
    for log_nfe in range(int(np.log2(tgt_nfe)) + 1):
        n_points = 2 ** log_nfe + 1
        u = np.linspace(eps, 1 - eps, n_points)
        t = fn(u)
        t = np.clip(t, 0.0, 1.0)
        t[0] = eps
        t[-1] = 1.0
        assert np.all(np.diff(t) >= 0), f"non-monotonic heuristic schedule at NFE={2**log_nfe}"
        out.append(t.tolist())
    return out
