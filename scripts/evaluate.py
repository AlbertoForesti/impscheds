"""Sample with a given schedule and NFE, then evaluate the samples.

binomial: KL divergence and 1-Wasserstein distance to the true pmf (Fig. 1)
piano:    outlier score and Hellinger distance to the test continuations (Fig. 3)

    python scripts/evaluate.py --task binomial --binomial_n 31 --noise_schedule linear --max_length 10 \
        --schedule wds --nfe 8 --seed 42
"""
import argparse
import json
import os

import torch
from tqdm import tqdm

from udm_schedules.sampling import tau_leaping_sampling_fn
from udm_schedules.tasks import (
    SAMPLE_EPS, add_task_args, configure, load_dataset, load_metric_fn, load_model, load_scheduler, output_path,
    seed_everything,
)

SCHEDULES = ("uniform", "jys", "eds", "wds", "cosmap", "logitnormal")


def main():
    parser = argparse.ArgumentParser()
    add_task_args(parser)
    parser.add_argument("--schedule", required=True, choices=SCHEDULES)
    parser.add_argument("--schedule_path", default="", help="defaults to outputs/schedules/<task>/<schedule>-<tag>.pt")
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--src_nfe", type=int, default=1024, help="time grid of the JYS schedule")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="binomial: number of samples (default 16384); piano: number of test prompts (default all)")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--cond_num_samples", type=int, default=1, help="piano: samples per prompt")
    parser.add_argument("--overwrite", action="store_true")
    args = configure(parser.parse_args())
    if args.task == "binomial" and args.num_samples == -1:
        args.num_samples = 16384

    run_name = f"{args.schedule}-nfe{args.nfe}-seed{args.seed}"
    metric_path = output_path(args, "metrics", run_name, ext=".json")
    if os.path.exists(metric_path) and not args.overwrite:
        with open(metric_path) as f:
            print(f"{run_name} / Metric: {json.load(f)} (cached)")
        return

    dtype = torch.float32
    seed_everything(args.seed)
    model = load_model(args)
    model.to(args.device, dtype)
    scheduler = load_scheduler(args, model)

    # Test split: the prompts for piano. For binomial it is not used, but
    # drawing it advances the global RNG, which also draws the initial noise
    # below, exactly as in the runs reported in the paper.
    eval_ds = load_dataset(args, "test", args.num_samples)
    eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    if args.schedule == "uniform":
        sampling_schedule = None
    else:
        schedule_path = args.schedule_path or output_path(args, "schedules", args.schedule)
        sampling_schedule = [s for s in torch.load(schedule_path) if len(s) == args.nfe + 1][0]

    sample_kwargs = dict(
        tgt_nfe=args.nfe, src_nfe=args.src_nfe, sample_eps=SAMPLE_EPS, sampling_schedule=sampling_schedule, dtype=dtype,
    )
    if args.task == "piano":
        xt = []
        for batch_idx, (x0, _) in enumerate(tqdm(eval_dl)):
            xt_i = tau_leaping_sampling_fn(
                model, scheduler, args.device, seed=args.seed * 100000 + batch_idx,
                num_samples=args.cond_num_samples, fix_length=args.fix_length, max_length=args.max_length,
                x0=x0, **sample_kwargs,
            ).cpu()
            assert torch.all(
                xt_i[:, : args.fix_length] == x0[:, : args.fix_length].repeat(args.cond_num_samples, 1).cpu()
            ), "generated samples do not start with the conditioning tokens"
            xt.append(xt_i)
    else:
        xt = []
        for batch_idx in tqdm(range(args.num_samples // args.batch_size)):
            xt.append(tau_leaping_sampling_fn(
                model, scheduler, args.device, seed=args.seed + batch_idx, num_samples=args.batch_size,
                **sample_kwargs,
            ).cpu())
    xt = torch.cat(xt, dim=0)
    torch.save(xt, output_path(args, "samples", run_name))
    del model, scheduler

    metric = load_metric_fn(args)(xt)
    with open(metric_path, "w") as f:
        json.dump(metric, f, indent=2)
    print(f"{run_name} / Metric: {metric}")


if __name__ == "__main__":
    main()
