"""Figure 3: outlier score and Hellinger distance vs NFE on piano generation.

Reads outputs/metrics/piano/<schedule>-nfe<N>-seed42-piano.json written by
experiments/piano.sh.
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import SCHEDULES, apply_style, save

METRICS_DIR = os.path.join("outputs", "metrics", "piano")
NFES = [2, 4, 8, 16, 32, 64]
PIANO_SCHEDULES = ["uniform", "jys", "eds", "wds"]


def load(schedule, nfe, seed=42):
    path = os.path.join(METRICS_DIR, f"{schedule}-nfe{nfe}-seed{seed}-piano.json")
    if not os.path.exists(path):
        print(f"  missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def make_figure(metric, ylabel):
    fig, ax = plt.subplots(figsize=(5, 5))
    legend = []
    for schedule in PIANO_SCHEDULES:
        color, label = SCHEDULES[schedule]
        points = [(nfe, r[metric]) for nfe in NFES if (r := load(schedule, nfe)) is not None]
        if not points:
            continue
        ax.plot(*zip(*points), marker="o", color=color, label=label)
        legend.append(Line2D([0], [0], color=color, marker="o", label=label))
    ax.set_xlabel("NFE")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NFES, NFES)
    ax.grid(True)
    ax.legend(handles=legend, title="Sampling Schedule", loc="best", frameon=True)
    fig.tight_layout()
    save(fig, f"fig3-piano-{metric}.pdf")


if __name__ == "__main__":
    apply_style()
    plt.rcParams.update({"legend.fontsize": 11})
    make_figure("outlier", "Outlier score")
    make_figure("hellinger", "Hellinger distance")
