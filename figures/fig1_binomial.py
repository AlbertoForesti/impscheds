"""Figure 1: KL divergence vs NFE on the 10-dimensional binomial benchmark,
mean and standard deviation over 5 seeds, one panel per (n, noise schedule).

Reads outputs/metrics/binomial/<schedule>-nfe<N>-seed<S>-n<n>-<schedule>-L10.json
written by experiments/binomial.sh.
"""
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatter, LogLocator

from style import SCHEDULES, apply_style, save

METRICS_DIR = os.path.join("outputs", "metrics", "binomial")
NFES = [2, 4, 8, 16, 32, 64, 128, 256]
SEEDS = [42, 43, 44, 45, 46]
CONFIGS = [(n, ns) for n in (31, 256) for ns in ("linear", "loglinear")]
LENGTH = 10


def mean_std(schedule, tag, metric):
    means, stds = [], []
    for nfe in NFES:
        vals = []
        for seed in SEEDS:
            path = os.path.join(METRICS_DIR, f"{schedule}-nfe{nfe}-seed{seed}-{tag}.json")
            if os.path.exists(path):
                with open(path) as f:
                    vals.append(json.load(f)[metric])
            else:
                print(f"  missing {path}")
        means.append(np.mean(vals) if vals else float("nan"))
        stds.append(np.std(vals) if vals else float("nan"))
    return means, stds


def make_figure(n, noise_schedule, metric="kl", ylabel="KL divergence"):
    tag = f"n{n}-{noise_schedule}-L{LENGTH}"
    fig, ax = plt.subplots(figsize=(5, 5))
    legend = []
    for schedule, (color, label) in SCHEDULES.items():
        means, stds = mean_std(schedule, tag, metric)
        if all(math.isnan(m) for m in means):
            continue
        ax.errorbar(NFES, means, yerr=stds, linestyle="solid", marker="o",
                    markerfacecolor=color, markeredgecolor=color, c=color, capsize=3)
        legend.append(Line2D([0], [0], color=color, marker="o", label=label))
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              fontsize="medium", frameon=True)
    ax.set_xlabel("NFE")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(NFES, NFES)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    fig.tight_layout()
    save(fig, f"fig1-binomial-n{n}-{noise_schedule}-{metric}.pdf")


if __name__ == "__main__":
    apply_style()
    for n, noise_schedule in CONFIGS:
        make_figure(n, noise_schedule)
