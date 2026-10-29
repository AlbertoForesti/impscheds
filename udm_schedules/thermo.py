"""Thermodynamic quantities of the reverse process of a uniform discrete
diffusion model: entropy production rate H_tot(t) (eq. 3), dynamical
activity A(t) (eq. 4) and the Wasserstein rate sqrt(A(t)/2 * H_tot(t))
(eq. 6).

Two parameterizations are supported:
  "sedd": SEDD-style model returning log-scores (binomial oracle),
  "ctmc": tauLDR uniform-rate CTMC returning x0 probabilities (piano).
"""
import torch


def _check_nan_count(tensor, context=""):
    nan_count = torch.isnan(tensor).sum().item()
    if nan_count > 0:
        raise ValueError(f"NaN values encountered in {context}: {nan_count} NaNs/{tensor.numel()} total values")


def _check_inf_count(tensor, context=""):
    inf_count = torch.isinf(tensor).sum().item()
    if inf_count > 0:
        raise ValueError(f"Inf values encountered in {context}: {inf_count} Infs/{tensor.numel()} total values")


def forward_rate_fn(scheduler, model, parameterization):
    """sigma(t): total forward jump rate per token."""
    if parameterization == "sedd":
        return scheduler.sigma
    if parameterization == "ctmc":
        return lambda t: model.rate_const * torch.ones_like(t)
    raise ValueError(f"Unknown parameterization: {parameterization}")


def dynamical_activity(sigma, num_tokens):
    """A(t) for a uniform kernel factorized over `num_tokens` positions."""
    return num_tokens * sigma


def get_activity_rates(t, num_tokens, sigma_fn):
    t = torch.as_tensor(t, dtype=torch.float64)
    return dynamical_activity(sigma_fn(t), num_tokens)


def get_wasserstein_rates(ep_rates, ep_times, num_tokens, sigma_fn):
    """rho_WDS(t) = sqrt(A(t)/2 * H_tot(t))."""
    ep_rates = torch.as_tensor(ep_rates, dtype=torch.float32)
    ep_times = torch.as_tensor(ep_times, dtype=torch.float32)

    dyn_activity = dynamical_activity(sigma_fn(ep_times), num_tokens)
    assert not torch.isnan(dyn_activity).any(), "NaN values found in dynamical activity."
    assert not torch.isinf(dyn_activity).any(), "Inf values found in dynamical activity."
    assert torch.all(dyn_activity >= 0), "Negative values found in dynamical activity."
    # Tiny negative rates are Monte Carlo noise around a true rate of ~0 (once
    # the forward process has reached stationarity); anything below the
    # tolerance is a real error.
    ep_rate_tol = -1e-6
    assert torch.all(ep_rates >= ep_rate_tol), f"Negative values found in entropy production rates. min: {ep_rates.min().item()}, max: {ep_rates.max().item()}"
    assert not torch.isnan(ep_rates).any(), "NaN values found in entropy production rates."
    assert not torch.isinf(ep_rates).any(), "Inf values found in entropy production rates."
    ep_rates = ep_rates.clamp_min(0.0)
    return torch.sqrt(ep_rates * dyn_activity / 2)


def get_entropy_rate(model, scheduler, t, dl, device="cuda", parameterization="sedd"):
    """Dataset average of the entropy production rate at time t."""
    with torch.no_grad():
        total_entropy_rate = 0.0
        total_samples = 0
        for batch in dl:
            x, _ = batch
            x = x.to(device)
            entropy_rate = entropy_production_rate(x, t, scheduler, model, device, parameterization).mean()
            total_entropy_rate += entropy_rate.item() * x.size(0)
            total_samples += x.size(0)
        return total_entropy_rate / total_samples


def entropy_production_rate(x, t, scheduler, model, device="cuda", parameterization="sedd"):
    """Per-sample entropy production rate at time t (eq. 3), on x_t ~ q(x_t | x):

        H_tot(t) = -sum_{y != x_t} Q_t(y, x_t) log s_theta(x_t)_y

    with the uniform forward rate Q_t(y, x_t) = sigma(t) / S of the SEDD uniform
    graph, or rate_const for the tauLDR CTMC (a constant factor that rescales
    the rate but not the schedules). The Monte Carlo estimate of a sample can
    be slightly negative near t = 1, where the true rate is ~0; it is clamped
    at 0.
    """
    t = t * torch.ones(x.shape[0], device=device)
    xt = scheduler.add_noise(x, t)

    if parameterization == "ctmc":
        rate_const = model.rate_const
        model_output = model(xt, t)
        score = scheduler.get_score(model_output, t, xt)

        assert (score >= 0).all(), "Score contains negative values."
        if torch.any(score == 0):
            score = torch.clamp(score, min=1e-8)
        log_score = score.log()
        assert not torch.isnan(log_score).any(), "NaN values found in log_score computation."
        assert not torch.isinf(log_score).any(), "Inf values found in log_score computation."

        # sum over y != x_t: total minus the self term
        off_diag_sum = log_score.sum(dim=-1) - torch.gather(log_score, -1, xt[..., None]).squeeze(-1)
        rate = -rate_const * off_diag_sum
    elif parameterization == "sedd":
        sigma = scheduler.sigma(t)
        sigma_bar = scheduler.sigma_bar(t)
        log_score = model(xt, sigma_bar)
        _check_nan_count(log_score, context=f"score computation at time t={t[0].item()}")
        _check_inf_count(log_score, context=f"score computation at time t={t[0].item()}")

        num_vocabs = scheduler.num_vocabs
        off_diag_sum = log_score.sum(dim=-1) - torch.gather(log_score, -1, xt[..., None]).squeeze(-1)
        rate = -(sigma[..., None] / num_vocabs) * off_diag_sum
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    ret = rate.sum(dim=-1)
    _check_nan_count(ret, context=f"entropy production rate computation at time t={t[0].item()}")
    _check_inf_count(ret, context=f"entropy production rate computation at time t={t[0].item()}")
    return ret.clamp_min(0.0)
