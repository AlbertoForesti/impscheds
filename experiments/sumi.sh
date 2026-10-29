#!/usr/bin/env bash
#####################################################################
# Sumi-7B (Fig. 2, Table 3): accuracy vs NFE under the Uniform, EDS and
# WDS token-budget schedules.
#
#   experiments/sumi.sh humaneval|mbpp|gsm8k
#
# 1. Builds the EDS/WDS schedules from rates estimated on MBPP-train
#    (N=64 samples, N_t=512 timesteps), once per generation length L
#    (L=256 for HumanEval/MBPP, L=64 for GSM8K).
# 2. Runs lm-eval for every (schedule, NFE) cell, single seed, greedy
#    confidence sampler, full test sets.
#
# Needs the Sumi uv environment (see README) and one GPU; select it with
# CUDA_VISIBLE_DEVICES. Set LIMIT=<n> for a quick smoke test.
#
# NFE == L leaves every schedule with the same all-ones token budget,
# so cells whose budget equals an already-run one reuse its metric.
#####################################################################
set -uo pipefail
TASK=${1:?usage: experiments/sumi.sh humaneval|mbpp|gsm8k}
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=${PY:-${UV_PROJECT_ENVIRONMENT:-third_party/sumi/.venv}/bin/python}
export PYTHONPATH=$PWD
PRETRAINED=${PRETRAINED:-tohoku-nlp/sumi-7b}
CANVAS=${CANVAS:-2048}
GEN_BATCH=${GEN_BATCH:-8}
TEMPERATURE=${TEMPERATURE:-0}
SEED=${SEED:-42}
LIMIT=${LIMIT:-}
SCHEDULES=${SCHEDULES:-"uniform eds wds"}

case "$TASK" in
  humaneval)
    L=256; NFES=${NFES:-"16 32 64 128 256"}
    LM_EVAL_ARGS=(--num_fewshot 0 --confirm_run_unsafe_code)
    export HF_ALLOW_CODE_EVAL=1 ;;  # pass@1 executes the generated code
  mbpp)
    # No --num_fewshot: lm-eval's mbpp task uses its default 3-shot prompt.
    L=256; NFES=${NFES:-"16 32 64 128 256"}
    LM_EVAL_ARGS=(--confirm_run_unsafe_code)
    export HF_ALLOW_CODE_EVAL=1 ;;
  gsm8k)
    L=64; NFES=${NFES:-"4 8 16 32 64"}
    LM_EVAL_ARGS=(--num_fewshot 0) ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

LOG_DIR=outputs/logs/sumi/$TASK
RUN_DIR=outputs/lm_eval/sumi/$TASK
METRIC_DIR=outputs/metrics/sumi/$TASK
mkdir -p "$LOG_DIR" "$RUN_DIR" "$METRIC_DIR"

schedule_path() {
  [ "$1" = uniform ] || echo "outputs/schedules/sumi/$1-mbpp-L$L.pt"
}

##############################
# 1. EDS / WDS schedules
##############################
if [ ! -f "$(schedule_path wds)" ]; then
  echo "[schedules] estimating rates on MBPP-train (L=$L)..."
  $PY -m sumi_schedules.build_schedules --pretrained "$PRETRAINED" \
    --src-nfe 512 --num-samples 64 --max-new-tokens "$L" --canvas-length "$CANVAS" --seed "$SEED" || exit 1
fi

##############################
# 2. NFE sweep
##############################
declare -A SEEN_BUDGET
for nfe in $NFES; do
  for sched in $SCHEDULES; do
    path=$(schedule_path "$sched")
    run="$sched-nfe$nfe-seed$SEED"
    metric_file="$METRIC_DIR/$run.json"
    fp=$($PY -m sumi_schedules.print_budgets --hash --schedule-name "$sched" \
           --path "$path" --nfe "$nfe" --num-tokens "$L")

    if [ -f "$metric_file" ]; then
      echo "SKIP  $TASK $run (done)"
      SEEN_BUDGET[$fp]=$run
      continue
    fi
    twin=${SEEN_BUDGET[$fp]:-}
    if [ -n "$twin" ]; then
      echo "REUSE $TASK $run (same token budget as $twin)"
      cp "$METRIC_DIR/$twin.json" "$metric_file"
      continue
    fi
    SEEN_BUDGET[$fp]=$run
    echo "RUN   $TASK $run"

    margs="pretrained=$PRETRAINED,sampling_schedule_name=$sched"
    [ -n "$path" ] && margs="$margs,sampling_schedule_path=$path"
    margs="$margs,num_denoising_steps=$nfe,max_new_tokens=$L,canvas_length=$CANVAS"
    margs="$margs,gen_batch_size=$GEN_BATCH,gen_temperature=$TEMPERATURE"
    margs="$margs,fill_mode=anchor,sampler=adaptive,seed=$SEED"

    $PY -m sumi_schedules --model sumi_sched --model_args "$margs" \
      --tasks "$TASK" "${LM_EVAL_ARGS[@]}" --output_path "$RUN_DIR/$run" \
      ${LIMIT:+--limit $LIMIT} > "$LOG_DIR/$run.log" 2>&1

    $PY - "$TASK" "$RUN_DIR/$run" "$metric_file" <<'PYEOF'
import json, pathlib, sys
task, run_dir, out = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
files = sorted(run_dir.rglob("results_*.json"))
if not files:
    sys.exit(f"no lm-eval results under {run_dir}")
res = json.loads(files[-1].read_text())["results"][task]
prefixes = {"gsm8k": ("exact_match,flexible-extract",), "humaneval": ("pass@1",), "mbpp": ("pass@1", "pass_at_1")}[task]
key = next(k for p in prefixes for k in res if k.startswith(p) and "stderr" not in k)
json.dump({"metric": key, "value": res[key]}, open(out, "w"), indent=2)
print(f"      -> {key} = {res[key]}")
PYEOF
    [ $? -eq 0 ] || { echo "      -> FAILED; see $LOG_DIR/$run.log" >&2; tail -5 "$LOG_DIR/$run.log" >&2; }
  done
done
