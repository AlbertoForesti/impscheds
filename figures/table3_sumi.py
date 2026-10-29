"""Table 3: WDS vs Uniform on the Sumi-7B reasoning tasks, as the ratio of
their accuracies at the same NFE.

Reads outputs/metrics/sumi/<task>/<schedule>-nfe<N>-seed42.json written by
experiments/sumi.sh.
"""
import json
import os

METRICS_DIR = os.path.join("outputs", "metrics", "sumi")
TASKS = {
    "GSM8K": ("gsm8k", [4, 8, 16, 32]),
    "MBPP": ("mbpp", [16, 32, 64, 128]),
    "HumanEval": ("humaneval", [16, 32, 64, 128]),
}


def load(task, schedule, nfe, seed=42):
    path = os.path.join(METRICS_DIR, task, f"{schedule}-nfe{nfe}-seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["value"]


def main():
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Task & NFE & Uniform & WDS & Performance Improvement (WDS / Uniform) \\",
        r"\midrule",
    ]
    for name, (task, nfes) in TASKS.items():
        for i, nfe in enumerate(nfes):
            uniform, wds = load(task, "uniform", nfe), load(task, "wds", nfe)
            cell = name if i == 0 else ""
            if uniform is None or wds is None:
                lines.append(f"{cell} & {nfe} & n/a & n/a & n/a \\\\")
                continue
            lines.append(f"{cell} & {nfe} & {uniform:.4f} & {wds:.4f} & {wds / uniform:.2f}$\\times$ \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
