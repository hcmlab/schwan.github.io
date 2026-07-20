#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FT_DIR="${FT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=/dev/null
source "${FT_DIR}/slurm/common.sh"

REPO_ROOT="$(cd "${FT_DIR}/.." && pwd)"
swan_load_env "$REPO_ROOT"

usage() {
cat <<'HELP'
Usage:
  bash slurm/run_pipeline.sh --profile PROFILE --dataset-spec SPEC --model MODEL [options]

Required:
  --profile PROFILE
  --dataset-spec SPEC
  --model MODEL

Options:
  --folds F0 [F1 ...]
  --role-mode joint|infant|caregiver
  --exclude-label LABEL
  --test-fraction FLOAT
  --seed INT
  --n-folds INT
  --prep
  --overwrite
  --dry-run
  --partition NAME
  --account NAME
  --qos NAME
  --gres VALUE
  --cpus-per-task N
  --mem VALUE
  --time VALUE
  --nodelist VALUE
  --constraint VALUE
  --array-parallelism N
  --sbatch-arg ARG
HELP
}

PROFILE=""
DATASET_SPEC=""
MODEL=""
ROLE_MODE=""
OVERWRITE=0
DRY_RUN=0
DO_PREP=0
PARTITION=""
ACCOUNT=""
QOS=""
GRES=""
CPUS_PER_TASK=""
MEMORY=""
TIME_LIMIT=""
NODELIST=""
CONSTRAINT=""
ARRAY_PARALLELISM=""
TEST_FRACTION="0.2"
SPLIT_SEED="42"
N_FOLDS="5"
FOLDS=(0 1 2 3 4)
EXCLUDE_LABELS=()
EXTRA_SBATCH_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --dataset-spec) DATASET_SPEC="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --folds)
      shift; FOLDS=()
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
        if [[ "$1" == *,* ]]; then split_folds=(); swan_split_csv "$1" split_folds; FOLDS+=("${split_folds[@]}"); else FOLDS+=("$1"); fi
        shift
      done ;;
    --role-mode) ROLE_MODE="$2"; shift 2 ;;
    --exclude-label) EXCLUDE_LABELS+=("$2"); shift 2 ;;
    --test-fraction) TEST_FRACTION="$2"; shift 2 ;;
    --seed) SPLIT_SEED="$2"; shift 2 ;;
    --n-folds) N_FOLDS="$2"; shift 2 ;;
    --prep) DO_PREP=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --nodelist) NODELIST="$2"; shift 2 ;;
    --constraint) CONSTRAINT="$2"; shift 2 ;;
    --array-parallelism) ARRAY_PARALLELISM="$2"; shift 2 ;;
    --sbatch-arg) EXTRA_SBATCH_ARGS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[ -z "$PROFILE" ] || [ -z "$DATASET_SPEC" ] || [ -z "$MODEL" ] && { usage >&2; exit 1; }
[ ! -f "${FT_DIR}/configs/profiles/${PROFILE}.json" ] && { echo "Unknown profile: $PROFILE" >&2; exit 1; }
[ ! -f "${FT_DIR}/configs/dataset_specs/${DATASET_SPEC}.json" ] && { echo "Unknown dataset spec: $DATASET_SPEC" >&2; exit 1; }
[ ! -f "${FT_DIR}/configs/model_specs/${MODEL}.json" ] && { echo "Unknown model: $MODEL" >&2; exit 1; }

if [ "${#EXCLUDE_LABELS[@]}" -eq 0 ] && [ "$DATASET_SPEC" = "icep_no_bg_audio" ]; then
  EXCLUDE_LABELS=(bg)
fi

# Load SLURM defaults from profile JSON if not overridden on command line
_profile_json="${FT_DIR}/configs/profiles/${PROFILE}.json"
if [ -f "$_profile_json" ] && command -v python3 >/dev/null 2>&1; then
  eval "$(python3 -c "
import json, sys
p = json.load(open('$_profile_json'))
s = p.get('slurm', {})
print(f'_P_PARTITION={s.get(\"partition\", \"\")}')
print(f'_P_GRES={s.get(\"gres\", \"\")}')
print(f'_P_CPUS={s.get(\"cpus_per_task\", \"\")}')
print(f'_P_MEM={s.get(\"mem\", \"\")}')
print(f'_P_TIME={s.get(\"time\", \"\")}')
print(f'_P_ARRAY_PAR={s.get(\"array_parallelism\", \"\")}')
")"
  [ -z "$PARTITION" ] && PARTITION="${_P_PARTITION}"
  [ -z "$GRES" ] && GRES="${_P_GRES}"
  [ -z "$CPUS_PER_TASK" ] && CPUS_PER_TASK="${_P_CPUS}"
  [ -z "$MEMORY" ] && MEMORY="${_P_MEM}"
  [ -z "$TIME_LIMIT" ] && TIME_LIMIT="${_P_TIME}"
  [ -z "$ARRAY_PARALLELISM" ] && ARRAY_PARALLELISM="${_P_ARRAY_PAR}"
fi

LOGS_ROOT="$(swan_logs_root "$PROFILE" "$FT_DIR")"
mkdir -p "$LOGS_ROOT"

FOLDS_CSV="$(IFS=,; printf '%s' "${FOLDS[*]}")"
EXCLUDE_LABELS_CSV="$(IFS=,; printf '%s' "${EXCLUDE_LABELS[*]}")"
EXPORT_SPEC="ALL,FT_DIR=${FT_DIR},PROFILE=${PROFILE},DATASET_SPEC=${DATASET_SPEC},MODEL=${MODEL},ROLE_MODE=${ROLE_MODE},EXCLUDE_LABELS=${EXCLUDE_LABELS_CSV},FOLDS_CSV=${FOLDS_CSV},OVERWRITE=${OVERWRITE},DRY_RUN=${DRY_RUN},TEST_FRACTION=${TEST_FRACTION},SPLIT_SEED=${SPLIT_SEED},N_FOLDS=${N_FOLDS}"

append_sbatch_args() {
  local -n out="$1"
  local include_gpu="$2"
  out=(sbatch --parsable)
  [ -n "$PARTITION" ] && out+=("--partition=${PARTITION}")
  [ -n "$ACCOUNT" ] && out+=("--account=${ACCOUNT}")
  [ -n "$QOS" ] && out+=("--qos=${QOS}")
  [ "$include_gpu" = "1" ] && [ -n "$GRES" ] && out+=("--gres=${GRES}")
  [ -n "$CPUS_PER_TASK" ] && out+=("--cpus-per-task=${CPUS_PER_TASK}")
  [ -n "$MEMORY" ] && out+=("--mem=${MEMORY}")
  [ -n "$TIME_LIMIT" ] && out+=("--time=${TIME_LIMIT}")
  [ -n "$NODELIST" ] && out+=("--nodelist=${NODELIST}")
  [ -n "$CONSTRAINT" ] && out+=("--constraint=${CONSTRAINT}")
  for arg in "${EXTRA_SBATCH_ARGS[@]}"; do out+=("$arg"); done
}

print_cmd() {
  printf 'DRY-RUN:'
  printf ' %q' "$@"
  echo
}

echo "================================================================================"
echo "Swan generic SLURM pipeline submission"
echo "Profile: ${PROFILE}"
echo "Dataset spec: ${DATASET_SPEC}"
echo "Model: ${MODEL}"
echo "Folds: ${FOLDS_CSV}"
echo "Test fraction: ${TEST_FRACTION}"
echo "Split seed: ${SPLIT_SEED}"
echo "Exclude labels: ${EXCLUDE_LABELS_CSV:-<none>}"
echo "Node list: ${NODELIST:-<scheduler default>}"
echo "================================================================================"

LAST_PREREQ_JOB=""

# Splits: skip if folds.json already exists for this variant
FOLDS_FILE="${FT_DIR}/configs/folds.json"
if [ -f "$FOLDS_FILE" ] && [ "$OVERWRITE" != "1" ]; then
  echo "Splits already exist (${FOLDS_FILE}), skipping create_splits. Use --overwrite to force."
else
  split_cmd=(); append_sbatch_args split_cmd 0
  split_cmd+=(--job-name=swan_split "--output=${LOGS_ROOT}/%x_%j.log" "--export=${EXPORT_SPEC}" "${SCRIPT_DIR}/create_splits.sh")
  if [ "$DRY_RUN" = "1" ]; then print_cmd "${split_cmd[@]}"; else LAST_PREREQ_JOB="$("${split_cmd[@]}")"; fi
fi

# Prep: only if requested
if [ "$DO_PREP" = "1" ]; then
  prep_cmd=(); append_sbatch_args prep_cmd 0
  prep_cmd+=(--job-name=swan_prep "--output=${LOGS_ROOT}/%x_%j.log" "--export=${EXPORT_SPEC}")
  [ -n "$LAST_PREREQ_JOB" ] && prep_cmd+=("--dependency=afterok:${LAST_PREREQ_JOB}")
  prep_cmd+=("${SCRIPT_DIR}/prepare.sh")
  if [ "$DRY_RUN" = "1" ]; then print_cmd "${prep_cmd[@]}"; else LAST_PREREQ_JOB="$("${prep_cmd[@]}")"; fi
fi

train_cmd=(); append_sbatch_args train_cmd 1
train_cmd+=(--job-name=swan_cv_train "--output=${LOGS_ROOT}/%x_fold%a_%j.log" "--export=${EXPORT_SPEC}")
[ -n "$LAST_PREREQ_JOB" ] && train_cmd+=("--dependency=afterok:${LAST_PREREQ_JOB}")
ARRAY_SPEC="${FOLDS_CSV}"
[ -n "$ARRAY_PARALLELISM" ] && ARRAY_SPEC="${ARRAY_SPEC}%${ARRAY_PARALLELISM}"
train_cmd+=("--array=${ARRAY_SPEC}" "${SCRIPT_DIR}/train_fold.sh")
[ "$DRY_RUN" = "1" ] && print_cmd "${train_cmd[@]}" || TRAIN_JOB="$("${train_cmd[@]}")"

cv_eval_cmd=(); append_sbatch_args cv_eval_cmd 1
cv_eval_cmd+=(--job-name=swan_cv_eval "--output=${LOGS_ROOT}/%x_%j.log" "--export=${EXPORT_SPEC}" "--dependency=afterok:${TRAIN_JOB}" "${SCRIPT_DIR}/evaluate.sh")
[ "$DRY_RUN" = "1" ] && print_cmd "${cv_eval_cmd[@]}" || CV_EVAL_JOB="$("${cv_eval_cmd[@]}")"

final_train_cmd=(); append_sbatch_args final_train_cmd 1
final_train_cmd+=(--job-name=swan_final_train "--output=${LOGS_ROOT}/%x_%j.log" "--export=${EXPORT_SPEC}" "--dependency=afterok:${CV_EVAL_JOB}" "${SCRIPT_DIR}/train_final.sh")
[ "$DRY_RUN" = "1" ] && print_cmd "${final_train_cmd[@]}" || FINAL_TRAIN_JOB="$("${final_train_cmd[@]}")"

test_eval_cmd=(); append_sbatch_args test_eval_cmd 1
test_eval_cmd+=(--job-name=swan_test_eval "--output=${LOGS_ROOT}/%x_%j.log" "--export=${EXPORT_SPEC}" "--dependency=afterok:${FINAL_TRAIN_JOB}" "${SCRIPT_DIR}/evaluate_test.sh")
[ "$DRY_RUN" = "1" ] && print_cmd "${test_eval_cmd[@]}" || TEST_EVAL_JOB="$("${test_eval_cmd[@]}")"

echo "All jobs submitted. Monitor with: squeue -u ${USER}"
echo "Logs root: ${LOGS_ROOT}"
