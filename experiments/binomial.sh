#!/usr/bin/env bash
#####################################################################
# Binomial benchmark (Fig. 1, Table 2, Fig. 4 binomial panel).
#
# 10-dimensional Binomial(n, 0.5) data with the analytic score, for
# n in {31, 256} x noise schedule in {linear, loglinear}:
#   1. JYS schedule                  (scripts/build_jys.py)
#   2. entropy-production / Wasserstein rates -> EDS, WDS (scripts/build_thermo.py)
#   3. CosMap, Logit-Normal          (scripts/build_heuristic.py)
#   4. KL / 1-Wasserstein of 16384 samples at NFE 2..256, 5 seeds (scripts/evaluate.py)
#
# eps convention of the reported runs: Uniform and JYS use the noise
# schedule with EPS_BASELINE=1e-2, EDS/WDS/CosMap/Logit-Normal with
# EPS=1e-5 (eps sets alpha(1) = eps, i.e. the forward process itself).
#
# Every size below can be overridden from the environment, e.g. for a
# quick run:  SRC_NFE=64 NUM_SAMPLES=256 TGT_NFE=16 NFES="4 16" SEEDS=42 experiments/binomial.sh
#
# Runs on CPU (DEVICE=cpu) in a few hours; a GPU is much faster.
#####################################################################
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

DEVICE=${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
OUT_DIR=${OUT_DIR:-outputs}
LENGTH=${LENGTH:-10}
NS=${NS:-"31 256"}
NOISE_SCHEDULES=${NOISE_SCHEDULES:-"linear loglinear"}
EPS=${EPS:-1e-5}
EPS_BASELINE=${EPS_BASELINE:-1e-2}
SRC_NFE=${SRC_NFE:-1024}          # time grid (N_t in Table 1)
NUM_SAMPLES=${NUM_SAMPLES:-1024}  # data samples used to build the schedules (N in Table 1)
TGT_NFE=${TGT_NFE:-256}
NFES=${NFES:-"2 4 8 16 32 64 128 256"}
SEEDS=${SEEDS:-"42 43 44 45 46"}
EVAL_SAMPLES=${EVAL_SAMPLES:-16384}
EVAL_BATCH=${EVAL_BATCH:-4096}

for n in $NS; do
  for ns in $NOISE_SCHEDULES; do
    common=(--task binomial --binomial_n "$n" --noise_schedule "$ns" --max_length "$LENGTH"
            --device "$DEVICE" --out_dir "$OUT_DIR")
    build=(--num_samples "$NUM_SAMPLES" --batch_size "$NUM_SAMPLES" --src_nfe "$SRC_NFE" --tgt_nfe "$TGT_NFE")
    tag="n$n-$ns-L$LENGTH"
    echo "=== $tag ==="

    [ -f "$OUT_DIR/schedules/binomial/jys-$tag.pt" ] || \
      python scripts/build_jys.py "${common[@]}" "${build[@]}" --eps "$EPS_BASELINE"
    [ -f "$OUT_DIR/schedules/binomial/wds-$tag.pt" ] || \
      python scripts/build_thermo.py "${common[@]}" "${build[@]}" --eps "$EPS"
    python scripts/build_heuristic.py "${common[@]}" --tgt_nfe "$TGT_NFE" --eps "$EPS"

    for schedule in uniform jys eds wds cosmap logitnormal; do
      case $schedule in uniform|jys) eps=$EPS_BASELINE ;; *) eps=$EPS ;; esac
      for nfe in $NFES; do
        for seed in $SEEDS; do
          python scripts/evaluate.py "${common[@]}" --eps "$eps" --schedule "$schedule" --nfe "$nfe" \
            --seed "$seed" --src_nfe "$SRC_NFE" --num_samples "$EVAL_SAMPLES" --batch_size "$EVAL_BATCH"
        done
      done
    done
  done
done
