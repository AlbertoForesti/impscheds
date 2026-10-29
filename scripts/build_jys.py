"""Build the Jump Your Steps (JYS) baseline schedules for NFE = 1, 2, 4, ..., tgt_nfe.

Each refinement inserts one timestep between every pair of consecutive steps,
chosen by golden-section search on the KLUB. Schedules are stored as indices
into a uniform src_nfe-step grid.

The reported time (Table 2) covers the schedule search only, after the model
and data are loaded, like the EDS/WDS time of build_thermo.py.

    python scripts/build_jys.py --task binomial --binomial_n 31 --noise_schedule linear --max_length 10
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from udm_schedules.jys import golden_section_algo, klub
from udm_schedules.tasks import (
    SAMPLE_EPS, Stopwatch, add_task_args, configure, load_dataset, load_model, load_scheduler, output_path,
    seed_everything,
)


def round_list(list_):
    return [round(i, 4) for i in list_]


def main():
    parser = argparse.ArgumentParser()
    add_task_args(parser)
    parser.add_argument("--num_samples", type=int, default=1024, help="data samples used to estimate the KLUB")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--src_nfe", type=int, default=1024, help="resolution of the time grid")
    parser.add_argument("--tgt_nfe", type=int, default=256, help="largest NFE to build")
    parser.add_argument("--max_iter", type=int, default=32, help="golden-section iterations per timestep")
    args = configure(parser.parse_args())

    seed_everything(args.seed)
    setup_clock = Stopwatch(args.device)

    # full precision for the KLUB optimization (half leads to numerical instability)
    dtype = torch.float

    model = load_model(args)
    model.to(args.device, dtype)
    ds = load_dataset(args, "train", args.num_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    scheduler = load_scheduler(args, model)
    setup_runtime_sec = setup_clock.elapsed()

    schedule_clock = Stopwatch(args.device)
    klub_vars = torch.linspace(1, SAMPLE_EPS, 2).tolist()
    klub_vars_space = torch.linspace(1, SAMPLE_EPS, args.src_nfe + 1)
    err = 0.25 / args.src_nfe
    print("start optimizing the sampling schedule:", round_list(klub_vars))

    sampling_schedule_list = [[0, args.src_nfe]]
    for log_nfe in range(int(np.log2(args.tgt_nfe)) + 1):
        # NFE=1 is the trivial [0, src_nfe] schedule
        if 2**log_nfe + 1 <= max(len(s) for s in sampling_schedule_list):
            sampling_schedule = [s for s in sampling_schedule_list if len(s) == 2**log_nfe + 1][0]
            klub_vars = torch.tensor(
                [klub_vars_space[min(i, args.src_nfe)] for i in sampling_schedule]
            ).tolist()
            continue
        print(f"*** current nfe: {2**log_nfe} ***")
        print(f"*** current sampling schedule: {round_list(klub_vars)} ***")

        sampling_schedule = [(klub_vars_space - v).abs().argmin().item() for v in klub_vars]

        # avoid consecutive steps at neighboring grid points
        for i in range(len(sampling_schedule) - 1):
            if sampling_schedule[i + 1] - sampling_schedule[i] < 2:
                sampling_schedule[i + 1] = min(sampling_schedule[i] + 2, args.src_nfe)
        for i in range(len(sampling_schedule) - 1, 0, -1):
            if sampling_schedule[i] - sampling_schedule[i - 1] < 2:
                sampling_schedule[i - 1] = max(sampling_schedule[i] - 2, 0)
        klub_vars = torch.tensor([klub_vars_space[i] for i in sampling_schedule]).tolist()

        # optimization step
        v2_list = []
        for v1, v3 in tqdm(zip(klub_vars[:-1], klub_vars[1:]), desc="optimizing sampling schedule..."):
            f = lambda v2: -klub(
                v1, v2, v3, model, scheduler, dl, args.seed, args.fix_length, args.max_length, args.device, dtype,
            )
            _, _, klub_dict = golden_section_algo(f, v3 + args.eps, v1 - args.eps, err=err, max_iter=args.max_iter)
            best_v2 = min(klub_dict.items(), key=lambda kv: kv[1])[0]
            v2_list.append(best_v2)

        # update sampling schedule
        new_klub_vars = []
        for v, v_new in zip(klub_vars[:-1], v2_list):
            new_klub_vars.append(v)
            new_klub_vars.append(v_new)
        new_klub_vars += [klub_vars[-1]]

        # avoid different sampling steps at the same grid point
        sampling_schedule = [(klub_vars_space - v2).abs().argmin().item() for v2 in new_klub_vars]
        for i in range(len(sampling_schedule) - 1):
            if sampling_schedule[i + 1] - sampling_schedule[i] < 1:
                sampling_schedule[i + 1] = min(sampling_schedule[i] + 1, args.src_nfe)
        klub_vars = torch.tensor([klub_vars_space[i] for i in sampling_schedule]).tolist()

        sampling_schedule_list.append(sampling_schedule)
    schedule_runtime_sec = schedule_clock.elapsed()

    schedule_path = output_path(args, "schedules", "jys")
    torch.save(sampling_schedule_list, schedule_path)

    timing_path = output_path(args, "timing", "jys", ext=".json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": args.task, "tag": args.tag, "src_nfe": args.src_nfe, "tgt_nfe": args.tgt_nfe,
            "num_samples": args.num_samples, "seed": args.seed, "device": args.device,
            "setup_runtime_sec": setup_runtime_sec, "schedule_runtime_sec": schedule_runtime_sec,
        }, f, indent=2)
    print(f"Saved JYS schedules to {schedule_path} (search: {schedule_runtime_sec:.1f}s)")


if __name__ == "__main__":
    main()
