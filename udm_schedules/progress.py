"""Cumulative-progress schedules (Sec. 3.2, eq. 7).

Given a non-negative rate rho(t) on a fine grid, the cumulative progress
C(t) = int_0^t rho is inverted so that every sampling step covers the same
amount of progress: t_k = C^-1(k/K * C(T)).
  EDS: rho(t) = H_tot(t)
  WDS: rho(t) = sqrt(A(t)/2 * H_tot(t))
"""
import numpy as np
import scipy.integrate


def progress_schedules(rates, ep_times, eps, tgt_nfe):
    """Schedules for NFE = 1, 2, 4, ..., tgt_nfe.

    Returns one ascending list of NFE+1 times per NFE. The rates are only
    defined on [eps, 1-eps], so the top entry is set to exactly 1.0 (every
    sampler starts from the stationary distribution at t = 1).
    """
    progress = scipy.integrate.cumulative_trapezoid(rates, ep_times, initial=0.0)
    total = float(progress[-1])
    schedules = []
    for log_nfe in range(int(np.log2(tgt_nfe).item()) + 1):
        times = np.linspace(eps, 1, 2**log_nfe + 1, endpoint=True)
        mapped = np.interp(times * total, progress, ep_times).tolist()
        mapped[-1] = 1.0
        assert not any(np.isnan(t) for t in mapped), f"NaN in schedule at NFE {2**log_nfe}"
        schedules.append(mapped)
    return schedules
