#!/usr/bin/env bash
#####################################################################
# Piano generation (Fig. 3, Table 2, Fig. 4 piano panel).
#
# Pretrained tauLDR uniform-rate CTMC on Lakh pianoroll, conditioned on
# the first 32 tokens (2 bars), generating 256 tokens:
#   1. JYS schedule                   (scripts/build_jys.py)
#   2. rates -> EDS, WDS               (scripts/build_thermo.py), 64 training samples, 1024 timesteps
#   3. outlier score / Hellinger distance on the test set at NFE 2..64 (scripts/evaluate.py)
#
# Needs the tauLDR checkpoint and data (see README):
#   CKPT_DIR (default weights/piano-ctmc) with ckpt.pt and config.yaml
#   DATA_DIR (default datasets/piano) with train.npy and test.npy
# and one GPU.
#####################################################################
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

DEVICE=${DEVICE:-cuda}
OUT_DIR=${OUT_DIR:-outputs}
CKPT_DIR=${CKPT_DIR:-weights/piano-ctmc}
DATA_DIR=${DATA_DIR:-datasets/piano}
SRC_NFE=${SRC_NFE:-1024}
NUM_SAMPLES=${NUM_SAMPLES:-64}
TGT_NFE=${TGT_NFE:-64}
NFES=${NFES:-"2 4 8 16 32 64"}
SEED=${SEED:-42}

common=(--task piano --ckpt_dir "$CKPT_DIR" --data_dir "$DATA_DIR" --fix_length 32 --max_length 256
        --device "$DEVICE" --out_dir "$OUT_DIR" --seed "$SEED")
build=(--num_samples "$NUM_SAMPLES" --batch_size "$NUM_SAMPLES" --src_nfe "$SRC_NFE" --tgt_nfe "$TGT_NFE")

[ -f "$OUT_DIR/schedules/piano/jys-piano.pt" ] || python scripts/build_jys.py "${common[@]}" "${build[@]}"
[ -f "$OUT_DIR/schedules/piano/wds-piano.pt" ] || python scripts/build_thermo.py "${common[@]}" "${build[@]}"

for nfe in $NFES; do
  for schedule in uniform jys eds wds; do
    python scripts/evaluate.py "${common[@]}" --schedule "$schedule" --nfe "$nfe" --src_nfe "$SRC_NFE" \
      --batch_size 64 --cond_num_samples 1
  done
done
