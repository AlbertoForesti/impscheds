"""Euler / tau-leaping sampler driven by an arbitrary timestep schedule."""
import torch


def schedule_to_timesteps(sampling_schedule, tgt_nfe, src_nfe, sample_eps, device):
    """Descending timesteps t_0 = 1 > ... > t_tgt_nfe used by the sampler.

    sampling_schedule is None (uniform), a list of indices into a
    src_nfe-step uniform grid (JYS), or an ascending list of times (EDS, WDS,
    CosMap, Logit-Normal).
    """
    if sampling_schedule is None:
        return torch.linspace(1, sample_eps, tgt_nfe+1, device=device)
    is_float = all([isinstance(s, float) for s in sampling_schedule])
    if not is_float:
        timesteps = torch.linspace(1, sample_eps, src_nfe+1)
        return torch.tensor([timesteps[i].item() for i in sampling_schedule] + [timesteps[-1].item()]).to(device)
    sampling_schedule = [sample_eps] + sampling_schedule
    timesteps = torch.tensor(sampling_schedule, device=device)
    return timesteps.flip(dims=(0,))


@torch.no_grad()
def tau_leaping_sampling_fn(
    model, scheduler, device, tgt_nfe, src_nfe, seed=None,
    num_samples=16, sample_eps=1e-3, sampling_schedule=None,
    fix_length=0, max_length=None, x0=None, dtype=torch.float64,
):
    timesteps = schedule_to_timesteps(sampling_schedule, tgt_nfe, src_nfe, sample_eps, device)
    generator = seed if seed is None else torch.Generator(device).manual_seed(seed)

    if fix_length > 0 and x0 is not None:
        xt = scheduler.sample_latent(num_samples).to(device).repeat(x0.size(0), 1)
    else:
        xt = scheduler.sample_latent(num_samples).to(device)

    if max_length is not None:
        xt = xt[:, :max_length]

    for i in range(tgt_nfe):
        if fix_length > 0 and x0 is not None:
            xt[:, :fix_length] = x0[:, :fix_length].repeat(num_samples, 1)

        dt = timesteps[i] - timesteps[i+1]
        t = timesteps[i] * torch.ones(xt.shape[0], device=device, dtype=dtype)

        sigma_bar = scheduler.sigma_bar(t)
        output = model(xt, sigma_bar)
        xt = scheduler.step(output, xt, t, dt, generator=generator).xt

    # Final readout at t = timesteps[tgt_nfe], the time xt is at after the loop.
    t_final = timesteps[tgt_nfe] * torch.ones(xt.shape[0], device=device, dtype=dtype)
    sigma_bar_final = scheduler.sigma_bar(t_final)
    output = model(xt, sigma_bar_final)
    xt = scheduler.step(output, xt, t_final, 0, rev_rate=None, generator=generator, if_last=True).xt
    if fix_length > 0 and x0 is not None:
        xt[:, :fix_length] = x0[:, :fix_length].repeat(num_samples, 1)
    return xt
