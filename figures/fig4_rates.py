"""Figure 4: entropy production rate H_tot(t) and Wasserstein rate
sqrt(A(t)/2 * H_tot(t)) for the binomial, piano and Sumi-7B experiments.

Reads the rate files written by scripts/build_thermo.py and
sumi_schedules/build_schedules.py. The Wasserstein rate is recomputed from
the stored H_tot(t) and A(t) so that every panel uses the same formula.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from style import save

SURFACE = "#fcfcfb"
INK = "#1a1a19"
GRID = "#e3e2dc"
ACCENT_ENTROPY = "#2ca02c"
ACCENT_WASS = "#d62728"

RATES = {
    "binomial": os.path.join("outputs", "rates", "binomial", "rates-n31-linear-L10.pt"),
    "piano": os.path.join("outputs", "rates", "piano", "rates-piano.pt"),
    "sumi": os.path.join("outputs", "rates", "sumi", "rates-mbpp-L256.pt"),
}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
        "xtick.color": INK, "ytick.color": INK,
        "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "legend.frameon": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.2, "figure.dpi": 130,
    })


def single_plot(x, y, color, ylabel, name):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(x, y, color=color)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    save(fig, name)


def main():
    apply_style()
    for experiment, path in RATES.items():
        if not os.path.exists(path):
            print(f"SKIP {experiment}: missing {path}")
            continue
        rates = torch.load(path, map_location="cpu", weights_only=False)
        t = np.asarray(rates["ep_times"], dtype=np.float64)
        entropy_rate = np.clip(np.asarray(rates["ep_rates"], dtype=np.float64), 0.0, None)
        activity = np.asarray(rates["dynamical_activity"], dtype=np.float64)
        wasserstein_rate = np.sqrt(activity / 2 * entropy_rate)

        single_plot(t, entropy_rate, ACCENT_ENTROPY, r"$\mathcal{H}^{\mathrm{tot}}(t)$",
                    f"fig4-{experiment}-entropy_rate.pdf")
        single_plot(t, wasserstein_rate, ACCENT_WASS, r"$\sqrt{\mathcal{A}(t)/2\cdot\mathcal{H}^{\mathrm{tot}}(t)}$",
                    f"fig4-{experiment}-wasserstein_rate.pdf")


if __name__ == "__main__":
    main()
