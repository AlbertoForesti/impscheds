#!/usr/bin/env bash
# Regenerate every figure (outputs/figures/) and table (stdout) of the paper
# from the outputs of experiments/{binomial,piano,sumi}.sh. Missing results
# are reported and skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=$PWD/figures${PYTHONPATH:+:$PYTHONPATH}
for script in fig1_binomial fig2_sumi fig3_piano fig4_rates; do
  python figures/$script.py
done
echo; echo "% Table 2"; python figures/table2_timing.py
echo; echo "% Table 3"; python figures/table3_sumi.py
