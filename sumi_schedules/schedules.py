"""Timestep schedule -> per-step ``tokens_per_step`` budget.

The one idea, stated once
-------------------------
Under the uniform forward process the expected number of still-corrupted tokens in
an ``L``-token window at time ``t`` is ``L * t`` (each position is resampled with
probability exactly ``t``). Stepping ``t -> s`` therefore cleans ``L * (t - s)`` of
them, so a schedule ``t_0 < ... < t_N`` traversed downward prescribes::

    k_j = L * (t_{N-j} - t_{N-j-1})        # tokens committed at step j

A uniform grid gives the constant ``k = L / N`` the stock sampler already uses;
EDS / WDS give a non-uniform ``k_j``. Nothing else about the confidence sampler
changes.

Schedule files have the same format as the rest of the repo: a
``list[list[float]]`` saved with ``torch.save``, whose element ``k`` has length
``2**k + 1`` and is **ascending** in ``t``. Selection is by exact length, so only
powers of two exist.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

__all__ = [
    "SCHEDULE_NAMES",
    "uniform_schedule",
    "load_schedule",
    "token_budgets",
    "resolve_token_budgets",
]

SCHEDULE_NAMES = ("uniform", "eds", "wds")


def uniform_schedule(nfe: int, eps: float = 1e-3) -> List[float]:
    """The uniform baseline grid, ascending in ``t`` like a stored schedule, so
    that every schedule goes through one code path."""
    return np.linspace(eps, 1.0, nfe + 1, endpoint=True).tolist()


def load_schedule(path: str, nfe: int) -> List[float]:
    """Pick the ``nfe``-step entry (the one of length ``nfe + 1``) out of a
    schedule file."""
    import torch

    schedules = torch.load(path, map_location="cpu", weights_only=False)
    matches = [s for s in schedules if len(s) == nfe + 1]
    if not matches:
        available = sorted({len(s) - 1 for s in schedules})
        raise ValueError(
            f"no schedule with {nfe} steps in {path}; available NFE: {available}"
        )
    return [float(t) for t in matches[0]]


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Apportion ``total`` integer units over ``weights`` (Hamilton's method)."""
    exact = weights * (total / weights.sum())
    base = np.floor(exact).astype(np.int64)
    leftover = total - int(base.sum())
    if leftover > 0:
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:leftover]] += 1
    return base


def token_budgets(t_grid: Sequence[float], num_tokens: int,
                  min_per_step: int = 1) -> List[int]:
    """``k_j`` for each denoising step, from an **ascending** timestep grid.

    Guarantees ``sum(k) == num_tokens`` and ``min(k) >= min_per_step``, so the whole
    content window is covered exactly once and no denoising step is a wasted NFE.

    The floor is applied as a *repair*: budgets are first apportioned proportionally
    to the schedule's own gaps, and only the steps that fall below the floor are
    lifted, with the deficit taken from the largest budgets (which never drop below
    the floor themselves). Where the floor does not bind the schedule's shape is
    preserved exactly -- in particular a uniform grid with ``num_tokens % nfe == 0``
    returns a constant budget.
    """
    t = np.asarray(t_grid, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError(f"t_grid must be a 1-D grid of >= 2 timesteps, got {t.shape}")
    num_steps = t.size - 1
    if min_per_step * num_steps > num_tokens:
        raise ValueError(
            f"{num_steps} steps at >= {min_per_step} token(s) each cannot fit in "
            f"{num_tokens} tokens; lower min_per_step or the NFE"
        )

    # Descending traversal (noisy -> clean): step j spans t[N-j] -> t[N-j-1].
    deltas = np.diff(t)[::-1]
    if not np.all(deltas > 0):
        raise ValueError("t_grid must be strictly increasing")

    k = _largest_remainder(deltas, int(num_tokens))

    # Repair the floor by lifting the starved steps and reclaiming the cost from the
    # others *in proportion to their surplus*, so the schedule's shape is squashed
    # uniformly rather than having its largest step gutted.
    if k.min() < min_per_step:
        k = np.maximum(k, min_per_step)
        deficit = int(k.sum()) - int(num_tokens)
        surplus = k - min_per_step
        while deficit > 0:
            take = _largest_remainder(surplus.astype(np.float64), deficit)
            take = np.minimum(take, surplus)
            if take.sum() == 0:  # unreachable given the feasibility check above
                raise RuntimeError("could not satisfy min_per_step")
            k -= take
            deficit -= int(take.sum())
            surplus = k - min_per_step

    return [int(v) for v in k]


def resolve_token_budgets(schedule_name: str, schedule_path: str, nfe: int,
                          num_tokens: int, eps: float = 1e-3,
                          min_per_step: int = 1) -> List[int]:
    """``(schedule name, file, NFE, L) -> [k_0, ..., k_{NFE-1}]``."""
    if schedule_name not in SCHEDULE_NAMES:
        raise ValueError(
            f"sampling_schedule_name must be one of {SCHEDULE_NAMES}, got {schedule_name!r}"
        )
    if schedule_name == "uniform":
        t_grid = uniform_schedule(nfe, eps)
    else:
        if not schedule_path:
            raise ValueError(
                f"sampling_schedule_name={schedule_name!r} requires sampling_schedule_path"
            )
        t_grid = load_schedule(schedule_path, nfe)
    return token_budgets(t_grid, num_tokens, min_per_step=min_per_step)
