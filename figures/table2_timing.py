"""Table 2: schedule computation time of JYS vs. EDS/WDS.

Both times exclude model and data loading and are measured on the same device:
  JYS:     golden-section search on the KLUB (scripts/build_jys.py)
  EDS/WDS: entropy production rate estimate, Wasserstein rate and the
           inversion of both schedules (scripts/build_thermo.py); EDS and
           WDS share the rate estimate, which dominates the cost.

Reads outputs/timing/<task>/{jys,thermo}-<tag>.json.
"""
import json
import os

TIMING_DIR = os.path.join("outputs", "timing")
ROWS = [
    ("binomial", f"n{n}-{ns}-L10", f"Binomial ($n={n}$, {ns})")
    for n in (256, 31) for ns in ("linear", "loglinear")
] + [("piano", "piano", "Piano")]


def load(task, name, tag):
    path = os.path.join(TIMING_DIR, task, f"{name}-{tag}.json")
    if not os.path.exists(path):
        print(f"% missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & JYS (s) & EDS/WDS (s) & JYS / (EDS/WDS) \\",
        r"\midrule",
    ]
    for task, tag, label in ROWS:
        jys, thermo = load(task, "jys", tag), load(task, "thermo", tag)
        if jys is None or thermo is None:
            continue
        if thermo["reused_rates"]:
            print(f"% skipping {label}: thermo-{tag}.json was built from cached rates; rerun without --reuse_rates")
            continue
        if jys["device"] != thermo["device"]:
            print(f"% warning: {label} timed on different devices ({jys['device']} vs {thermo['device']})")
        jys_s, eds_wds_s = jys["schedule_runtime_sec"], thermo["schedule_runtime_sec"]
        lines.append(f"{label} & {jys_s:.2f} & {eds_wds_s:.2f} & {jys_s / eds_wds_s:.2f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
