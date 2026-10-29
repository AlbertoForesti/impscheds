"""Shared plotting style: one color per schedule across all figures."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.environ.get("FIG_DIR", os.path.join("outputs", "figures"))

C_MAP = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
SCHEDULES = {  # name -> (color, legend label)
    "uniform": (C_MAP[0], "Uniform"),
    "jys": (C_MAP[1], "JYS"),
    "eds": (C_MAP[2], "EDS"),
    "wds": (C_MAP[3], "WDS"),
    "cosmap": (C_MAP[6], "CosMap"),
    "logitnormal": (C_MAP[7], "Logit-Normal"),
}


def apply_style(font_size=12):
    plt.rcParams.update({
        'text.usetex': False,
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman', 'Times New Roman'],
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
