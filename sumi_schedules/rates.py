"""Entropy-production / Wasserstein rate estimation for a mean-parameterized
uniform-diffusion LM (Sumi).

Sumi predicts ``x_hat`` (clean-token logits) instead of the SEDD concrete score,
and takes **no timestep input**.

Forward process
---------------
Sumi's forward process (``sumi_eval/model.py:_sample_noisy_tokens``) resamples each
position uniformly over ``[0, V)`` with probability ``t``::

    q(z_t = v | x) = alpha_t * 1[v = x] + (1 - alpha_t) / V,    alpha_t = 1 - t

which is the uniform-graph SEDD kernel with the ``loglinear`` schedule
``sigma_bar(t) = -log(1 - (1 - eps) t)`` (``udm_schedules/noise.py``) as ``eps -> 0``.

The bridge
----------
The concrete score is the ratio of marginals, and under the model's own prediction
the per-position marginal is the forward marginal with ``x_hat`` in place of the
data::

    p_t(v) = alpha * x_hat_v + (1 - alpha) / V
    s_theta(y | z_t) = p_t(y) / p_t(z_t)

so ``log s`` is a difference of two ``logaddexp``-stabilised terms, and it is
identically ``0`` at ``y = z_t``.

Collapsing the rate
-------------------
Eq. (3) of the paper is ``EPR = -sum_y sigma(t) Q_tok(z_t, y) log s(y | z_t)``. For
the uniform graph ``Q_tok`` is ``1/V`` off-diagonal; since ``log s`` vanishes on the
diagonal the sum over ``y != z_t`` may be extended to all ``y``::

    EPR(t) = -sigma(t)/V * sum_d [ sum_y log p_t(y)^d - V * log p_t(z_t^d)^d ]

The dataset-averaged EPR is non-negative (Chebyshev's sum inequality applied to the
similarly-ordered ``p`` and ``log p``).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

__all__ = [
    "sigma",
    "sigma_bar",
    "perturb_prob",
    "add_noise",
    "log_marginal",
    "log_score_from_xhat",
    "entropy_production_rate_rows",
    "dynamical_activity",
    "wasserstein_rate",
]


# --------------------------------------------------------------- noise schedule
# The loglinear schedule of udm_schedules/noise.py, the one Sumi was trained with:
# perturb_prob = (1 - eps) * t  ==>  alpha_t = 1 - t.
def sigma_bar(t, eps: float = 1e-3):
    """Total noise ``sigma_bar(t) = -log1p(-(1 - eps) t)``."""
    t = torch.as_tensor(t, dtype=torch.float64)
    return -torch.log1p(-(1.0 - eps) * t)


def sigma(t, eps: float = 1e-3):
    """Instantaneous noise ``sigma(t) = d sigma_bar / dt``."""
    t = torch.as_tensor(t, dtype=torch.float64)
    return (1.0 - eps) / (1.0 - (1.0 - eps) * t)


def perturb_prob(t, eps: float = 1e-3):
    """``1 - exp(-sigma_bar(t))`` -- the probability a token was resampled.

    Equals ``(1 - eps) * t``; kept in this form to mirror ``udm_schedules/noise.py``.
    """
    return 1.0 - torch.exp(-sigma_bar(t, eps))


def add_noise(x: torch.Tensor, t: float, vocab_size: int, eps: float = 1e-3,
              generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """``x_0 -> z_t`` for the uniform graph.

    Behaviourally identical to ``sumi_eval/model.py:_sample_noisy_tokens`` once
    ``eps -> 0``.
    """
    p = float(perturb_prob(t, eps))
    move = torch.rand(x.shape, device=x.device, generator=generator) < p
    # Drawing the replacement tokens from the generator too (unlike randint_like)
    # keeps the whole estimation reproducible from one seed.
    fresh = torch.randint(0, vocab_size, x.shape, device=x.device,
                          dtype=x.dtype, generator=generator)
    return torch.where(move, fresh, x)


# ------------------------------------------------------------ x_hat -> score
def log_marginal(logits: torch.Tensor, t: float, vocab_size: int) -> torch.Tensor:
    """``log p_t(v) = log(alpha * x_hat_v + (1 - alpha) / V)`` for every ``v``.

    ``logits`` is ``[..., V]`` (raw, already sliced to the real vocab). Computed via
    ``logaddexp`` on ``log_softmax`` so it is stable for tiny ``x_hat`` and for
    ``t -> 0`` (where the uniform floor underflows).
    """
    alpha = 1.0 - float(t)
    log_alpha = math.log(max(alpha, 1e-38))
    log_floor = math.log(max(1.0 - alpha, 1e-38)) - math.log(float(vocab_size))
    log_x_hat = torch.log_softmax(logits.float(), dim=-1)
    return torch.logaddexp(
        log_x_hat + log_alpha,
        torch.full_like(log_x_hat, log_floor),
    )


def log_score_from_xhat(logits: torch.Tensor, z_t: torch.Tensor, t: float,
                        vocab_size: int) -> torch.Tensor:
    """``log s_theta(y | z_t) = log p_t(y) - log p_t(z_t)``, shape ``[..., V]``.

    The SEDD-parameterization output, reconstructed from Sumi's ``x_hat``.
    Exactly ``0`` at ``y == z_t``.
    """
    log_p = log_marginal(logits, t, vocab_size)
    log_p_at_z = log_p.gather(-1, z_t.unsqueeze(-1))
    return log_p - log_p_at_z


# ----------------------------------------------------------- entropy production
@torch.no_grad()
def entropy_production_rate_rows(
    logits: torch.Tensor,
    z_t: torch.Tensor,
    starts: Sequence[int],
    ends: Sequence[int],
    t: float,
    vocab_size: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Per-sample EPR at time ``t``, summed over each row's ``[start, end)`` window.

    ``logits`` ``[B, S, V]`` raw model output (sliced to the real vocab), ``z_t``
    ``[B, S]`` the noised canvas the logits were produced from. Rows are processed
    one at a time and sliced -- never gathered -- so only ``[window, V]`` fp32 is
    materialised at once (the canvas is 2048 wide but the content window is ~256).

    Returns ``[B]`` (float64).
    """
    batch_size = logits.shape[0]
    out = torch.zeros(batch_size, dtype=torch.float64, device=logits.device)
    sig = float(sigma(t, eps))
    for i in range(batch_size):
        a, b = int(starts[i]), int(ends[i])
        if b <= a:
            continue
        log_p = log_marginal(logits[i, a:b], t, vocab_size)          # [w, V]
        log_p_at_z = log_p.gather(-1, z_t[i, a:b].unsqueeze(-1)).squeeze(-1)  # [w]
        # sum_y log s_y  ==  sum_y log p_y - V * log p_{z_t}   (log s vanishes at y=z_t)
        per_pos = log_p.double().sum(dim=-1) - vocab_size * log_p_at_z.double()
        out[i] = -(sig / vocab_size) * per_pos.sum()
    return out


# ------------------------------------------------------- activity & Wasserstein
def dynamical_activity(t, num_tokens: int, eps: float = 1e-3):
    """``A(t) = L * sigma(t)`` for a uniform kernel over ``L`` positions."""
    return num_tokens * sigma(t, eps)


def wasserstein_rate(ep_rates, activity):
    """``rho_WDS(t) = sqrt(A(t)/2 * EPR(t))`` (eq. 6)."""
    ep_rates = torch.as_tensor(ep_rates, dtype=torch.float64)
    activity = torch.as_tensor(activity, dtype=torch.float64)
    return torch.sqrt(ep_rates.clamp(min=0.0) * activity / 2)
