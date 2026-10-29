"""Figure 2: Sumi-7B accuracy vs NFE (MBPP, GSM8K, HumanEval) under the
Uniform, EDS and WDS schedules.

Reads outputs/metrics/sumi/<task>/<schedule>-nfe<N>-seed42.json written by
experiments/sumi.sh. The largest NFE of each sweep (NFE = L) is not shown:
there every schedule reduces to the same one-token-per-step budget.
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from style import SCHEDULES, apply_style, save

METRICS_DIR = os.path.join("outputs", "metrics", "sumi")
TASKS = {
    "mbpp": dict(nfes=[16, 32, 64, 128], ylabel="pass@1"),
    "gsm8k": dict(nfes=[4, 8, 16, 32], ylabel="Exact match (flexible-extract)"),
    "humaneval": dict(nfes=[16, 32, 64, 128], ylabel="pass@1"),
}
SUMI_SCHEDULES = ["uniform", "eds", "wds"]


def load(task, schedule, nfe, seed=42):
    path = os.path.join(METRICS_DIR, task, f"{schedule}-nfe{nfe}-seed{seed}.json")
    if not os.path.exists(path):
        print(f"  missing {path}")
        return None
    with open(path) as f:
        return json.load(f)["value"]


def make_figure(task, nfes, ylabel):
    fig, ax = plt.subplots(figsize=(5, 5))
    legend = []
    for schedule in SUMI_SCHEDULES:
        color, label = SCHEDULES[schedule]
        points = [(nfe, v) for nfe in nfes if (v := load(task, schedule, nfe)) is not None]
        if not points:
            continue
        ax.plot(*zip(*points), linestyle="solid", marker="o", c=color, label=label)
        legend.append(Line2D([0], [0], color=color, marker="o", label=label))
    ax.legend(handles=legend, title="Sampling Schedule", loc="best",
              fontsize="medium", title_fontsize="medium", frameon=True)
    ax.set_xlabel("NFE")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(nfes, nfes)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    fig.tight_layout()
    save(fig, f"fig2-sumi-{task}.pdf")


if __name__ == "__main__":
    apply_style()
    for task, cfg in TASKS.items():
        make_figure(task, cfg["nfes"], cfg["ylabel"])
