"""Build the CosMap and Logit-Normal baseline schedules (no model or data needed).

    python scripts/build_heuristic.py --task binomial --binomial_n 31 --noise_schedule linear --max_length 10
"""
import argparse

import torch

from udm_schedules.heuristics import HEURISTICS, heuristic_schedules
from udm_schedules.tasks import add_task_args, configure, output_path


def main():
    parser = argparse.ArgumentParser()
    add_task_args(parser)
    parser.add_argument("--tgt_nfe", type=int, default=256, help="largest NFE to build")
    args = configure(parser.parse_args())

    for name, fn in HEURISTICS.items():
        sched = heuristic_schedules(fn, args.eps, args.tgt_nfe)
        path = output_path(args, "schedules", name)
        torch.save(sched, path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
