"""Print (or fingerprint) the per-step token budget a schedule produces.

Used by ``experiments/sumi.sh`` to skip cells whose budget
vector is identical to one already run -- most notably ``NFE == max_new_tokens``,
where the "cover the window once, >= 1 token per step" constraint admits exactly one
budget vector (all ones), so every schedule collapses onto the stock sampler there.

    python -m sumi_schedules.print_budgets --schedule-name wds --path ... --nfe 64 [--hash]
"""

import argparse
import hashlib

from sumi_schedules.schedules import resolve_token_budgets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule-name", required=True)
    p.add_argument("--path", default="")
    p.add_argument("--nfe", type=int, required=True)
    p.add_argument("--num-tokens", type=int, default=256)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--min-per-step", type=int, default=1)
    p.add_argument("--hash", action="store_true", help="print a short fingerprint only")
    args = p.parse_args()

    k = resolve_token_budgets(args.schedule_name, args.path, args.nfe,
                              args.num_tokens, args.eps, args.min_per_step)
    if args.hash:
        print(hashlib.sha1(",".join(map(str, k)).encode()).hexdigest()[:12])
    else:
        print(f"nfe={args.nfe} L={args.num_tokens} sum={sum(k)} "
              f"min={min(k)} max={max(k)}")
        print(k)


if __name__ == "__main__":
    main()
