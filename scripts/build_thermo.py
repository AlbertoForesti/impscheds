"""Estimate the entropy production rate, dynamical activity and Wasserstein
rate along the reverse process, and build the EDS and WDS schedules from them
(Algorithm 1).

Outputs
  outputs/rates/<task>/rates-<tag>.pt        ep_times, ep_rates, wass_rates, dynamical_activity
  outputs/schedules/<task>/{eds,wds}-<tag>.pt  one ascending time list per NFE = 1, 2, 4, ..., tgt_nfe
  outputs/timing/<task>/thermo-<tag>.json    wall-clock times

The reported time (Table 2) covers rate estimation and the construction of both
schedules, after the model and data are loaded, like the JYS time of
build_jys.py. File writes are not timed.

    python scripts/build_thermo.py --task binomial --binomial_n 31 --noise_schedule linear --max_length 10
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from udm_schedules.progress import progress_schedules
from udm_schedules.tasks import (
    Stopwatch, add_task_args, configure, load_dataset, load_model, load_scheduler, output_path, seed_everything,
)
from udm_schedules.thermo import forward_rate_fn, get_activity_rates, get_entropy_rate, get_wasserstein_rates


def main():
    parser = argparse.ArgumentParser()
    add_task_args(parser)
    parser.add_argument("--num_samples", type=int, default=1024, help="data samples N used to estimate the rates")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--src_nfe", type=int, default=1024, help="number of time points N_t of the rate grid")
    parser.add_argument("--tgt_nfe", type=int, default=256, help="largest NFE to build")
    parser.add_argument("--reuse_rates", action="store_true", help="load the rates from a previous run instead of estimating them")
    args = configure(parser.parse_args())

    seed_everything(args.seed)
    setup_clock = Stopwatch(args.device)

    dtype = torch.float
    model = load_model(args)
    model.to(args.device, dtype)
    ds = load_dataset(args, "train", args.num_samples)

    seed_everything(args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    scheduler = load_scheduler(args, model)
    sigma_fn = forward_rate_fn(scheduler, model, args.parameterization)
    setup_runtime_sec = setup_clock.elapsed()

    # The entropy-rate estimator breaks down numerically at t = 1, so the
    # grid stops at 1 - eps; progress_schedules sets the first step to t = 1.
    ep_times = np.linspace(args.eps, 1 - args.eps, args.src_nfe).tolist()

    rates_path = output_path(args, "rates", "rates")
    reused_rates = args.reuse_rates and os.path.exists(rates_path)
    if reused_rates:
        cache = torch.load(rates_path, map_location="cpu")
        assert np.allclose(cache["ep_times"], ep_times), f"{rates_path} was computed on a different time grid"
        ep_rates, wass_rates, dynamical_activity = cache["ep_rates"], cache["wass_rates"], cache["dynamical_activity"]
        print(f"Loaded rates from {rates_path}")

    schedule_clock = Stopwatch(args.device)
    if not reused_rates:
        ep_rates = []
        for t in tqdm(ep_times, desc="entropy rates"):
            ep_rate = get_entropy_rate(model, scheduler, t, dl, args.device, parameterization=args.parameterization)
            assert not np.isnan(ep_rate), f"NaN entropy rate at t={t}"
            ep_rates.append(float(ep_rate))
        wass_rates = get_wasserstein_rates(ep_rates, ep_times, args.max_length, sigma_fn).tolist()
        dynamical_activity = get_activity_rates(ep_times, args.max_length, sigma_fn).tolist()
    schedules = {
        "eds": progress_schedules(np.array(ep_rates, dtype=np.float64), ep_times, args.eps, args.tgt_nfe),
        "wds": progress_schedules(np.array(wass_rates, dtype=np.float64), ep_times, args.eps, args.tgt_nfe),
    }
    schedule_runtime_sec = schedule_clock.elapsed()

    if not reused_rates:
        torch.save({
            "ep_times": ep_times,
            "ep_rates": ep_rates,
            "wass_rates": wass_rates,
            "dynamical_activity": dynamical_activity,
            "meta": {k: getattr(args, k) for k in (
                "task", "tag", "noise_schedule", "eps", "binomial_n", "binomial_p", "max_length",
                "fix_length", "src_nfe", "num_samples", "seed")},
        }, rates_path)
        print(f"Saved rates to {rates_path}")
    for name, schedule in schedules.items():
        schedule_path = output_path(args, "schedules", name)
        torch.save(schedule, schedule_path)
        print(f"Saved {name.upper()} schedules to {schedule_path}")

    timing_path = output_path(args, "timing", "thermo", ext=".json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": args.task, "tag": args.tag, "src_nfe": args.src_nfe, "num_samples": args.num_samples,
            "seed": args.seed, "device": args.device, "reused_rates": reused_rates,
            "setup_runtime_sec": setup_runtime_sec, "schedule_runtime_sec": schedule_runtime_sec,
        }, f, indent=2)

if __name__ == "__main__":
    main()
