"""EDS / WDS schedules for the Sumi uniform-diffusion LM (Ye et al., 2026).

An overlay on the upstream ``sumi_eval`` package (``third_party/sumi``); nothing
upstream is modified. It adds

* :mod:`sumi_schedules.rates` -- entropy-production / Wasserstein rate estimation
  for a mean-parameterized (x0-predicting) uniform-diffusion model;
* :mod:`sumi_schedules.build_schedules` -- estimates those rates on MBPP-train and
  turns them into EDS / WDS timestep schedules;
* :mod:`sumi_schedules.model` -- an ``lm-eval`` plugin (``--model sumi_sched``) that
  turns a timestep schedule into a per-step ``tokens_per_step`` budget for Sumi's
  confidence ("adaptive") sampler.
"""

__all__ = ["rates", "sampling", "schedules", "model"]
