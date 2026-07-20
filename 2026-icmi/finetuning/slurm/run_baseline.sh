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
  bash slurm/run_baseline.sh --profile PROFILE --baseline-spec BSPEC [options]

  Extracts features once from video clips, then trains+evaluates all modality
  combos (video-only, audio-only, video+audio) defined in the baseline spec.
  All combos use the same folds/labels from --dataset-spec.

Required:
  --profile PROFILE
  --baseline-spec BSPEC

Options:
  --dataset-spec SPEC          (default: icep_no_bg_video)
  --folds F0 [F1 ...]         (default: 0 1 2 3 4)
  --role-mode joint|infant|caregiver
  --exclude-label LABEL
  --device DEVICE              (default: cuda)
  --overwrite
  --dry-run
  --partition NAME
  --gres VALUE
  --cpus-per-task N
  --mem VALUE
  --time VALUE
  --nodelist VALUE
  --constraint VALUE
  --sbatch-arg ARG
HELP
}

PROFILE=""
BASELINE_SPEC=""
DATASET_SPEC="icep_no_bg_video"
ROLE_MODE=""
OVERWRITE=0
DRY_RUN=0
DEVICE="cuda"
PARTITION=""
ACCOUNT=""
QOS=""
GRES=""
CPUS_PER_TASK=""
MEMORY=""
TIME_LIMIT=""
NODELIST=""
CONSTRAINT=""
FOLDS=(0 1 2 3 4)
EXCLUDE_LABELS=()
EXTRA_SBATCH_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --baseline-spec) BASELINE_SPEC="$2"; shift 2 ;;
    --dataset-spec) DATASET_SPEC="$2"; shift 2 ;;
    --folds)
      shift; FOLDS=()
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
        if [[ "$1" == *,* ]]; then split_folds=(); swan_split_csv "$1" split_folds; FOLDS+=("${split_folds[@]}"); else FOLDS+=("$1"); fi
        shift
      done ;;
    --role-mode) ROLE_MODE="$2"; shift 2 ;;
    --exclude-label) EXCLUDE_LABELS+=("$2"); shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
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
    --sbatch-arg) EXTRA_SBATCH_ARGS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[ -z "$PROFILE" ] || [ -z "$BASELINE_SPEC" ] && { usage >&2; exit 1; }
[ ! -f "${FT_DIR}/configs/profiles/${PROFILE}.json" ] && { echo "Unknown profile: $PROFILE" >&2; exit 1; }
[ ! -f "${FT_DIR}/configs/baseline_specs/${BASELINE_SPEC}.json" ] && { echo "Unknown baseline spec: $BASELINE_SPEC" >&2; exit 1; }
[ ! -f "${FT_DIR}/configs/dataset_specs/${DATASET_SPEC}.json" ] && { echo "Unknown dataset spec: $DATASET_SPEC" >&2; exit 1; }

# Load SLURM defaults from profile JSON if not overridden
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
")"
  [ -z "$PARTITION" ] && PARTITION="${_P_PARTITION}"
  [ -z "$GRES" ] && GRES="${_P_GRES}"
  [ -z "$CPUS_PER_TASK" ] && CPUS_PER_TASK="${_P_CPUS}"
  [ -z "$MEMORY" ] && MEMORY="${_P_MEM}"
  [ -z "$TIME_LIMIT" ] && TIME_LIMIT="${_P_TIME}"
fi

LOGS_ROOT="$(swan_logs_root "$PROFILE" "$FT_DIR")"
mkdir -p "$LOGS_ROOT"

FOLDS_CSV="$(IFS=,; printf '%s' "${FOLDS[*]}")"
EXCLUDE_LABELS_CSV="$(IFS=,; printf '%s' "${EXCLUDE_LABELS[*]}")"

# Read modality combos from baseline spec
MODALITY_COMBOS_CSV="$(python3 -c "
import json
with open('${FT_DIR}/configs/baseline_specs/${BASELINE_SPEC}.json') as f:
    d = json.load(f)
print(','.join(d.get('modality_combos', ['video', 'audio', 'omni'])))
")"
IFS=',' read -ra COMBOS <<< "${MODALITY_COMBOS_CSV}"
NUM_COMBOS="${#COMBOS[@]}"
NUM_FOLDS="${#FOLDS[@]}"
NUM_TASKS=$(( NUM_COMBOS * NUM_FOLDS ))

# Export vars with commas (FOLDS_CSV, MODALITY_COMBOS_CSV) as shell env vars
# so --export=ALL picks them up correctly. Sbatch --export treats commas as
# separators, which would mangle "0,1,2,3,4" into separate entries.
export FT_DIR PROFILE DATASET_SPEC BASELINE_SPEC ROLE_MODE DEVICE OVERWRITE
export FOLDS_CSV MODALITY_COMBOS_CSV
export EXCLUDE_LABELS="${EXCLUDE_LABELS_CSV}"
EXPORT_SPEC="ALL"

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
echo "Swan baseline SLURM pipeline"
echo "Profile:       ${PROFILE}"
echo "Baseline spec: ${BASELINE_SPEC}"
echo "Dataset spec:  ${DATASET_SPEC}"
echo "Folds:         ${FOLDS_CSV}"
echo "Device:        ${DEVICE}"
echo "Node list:     ${NODELIST:-<scheduler default>}"
echo "Combos:        ${MODALITY_COMBOS_CSV} (${NUM_COMBOS})"
echo "Array tasks:   ${NUM_TASKS} (${NUM_COMBOS} combos × ${NUM_FOLDS} folds)"
echo ""
echo "Runs: extract (1 GPU job) → train-fold (${NUM_TASKS} array jobs) → report-cv (1 job)"
echo "================================================================================"

# Step 1: Extract features (GPU) — extracts both DINOv3 + W2V2 from video clips
extract_cmd=(); append_sbatch_args extract_cmd 1
extract_cmd+=(--job-name=swan_bl_extract "--output=${LOGS_ROOT}/%x_%j.log")
extract_cmd+=("--export=${EXPORT_SPEC},ACTION=extract")
extract_cmd+=("${SCRIPT_DIR}/baseline.sh")
if [ "$DRY_RUN" = "1" ]; then
  print_cmd "${extract_cmd[@]}"
  EXTRACT_JOB="dry"
else
  EXTRACT_JOB="$("${extract_cmd[@]}")"
  echo "Submitted extract job: ${EXTRACT_JOB}"
fi

# Step 2: Train + predict as array job (combo_idx * num_folds + fold_idx)
# Array index: 0..(NUM_TASKS-1), each task handles one (combo, fold) pair
ARRAY_MAX=$(( NUM_TASKS - 1 ))
train_cmd=(); append_sbatch_args train_cmd 0
train_cmd+=(--job-name=swan_bl_train "--output=${LOGS_ROOT}/%x_%a_%j.log")
train_cmd+=("--array=0-${ARRAY_MAX}")
train_cmd+=("--export=${EXPORT_SPEC},ACTION=train-fold")
[ "$EXTRACT_JOB" != "dry" ] && train_cmd+=("--dependency=afterok:${EXTRACT_JOB}")
train_cmd+=("${SCRIPT_DIR}/baseline.sh")
if [ "$DRY_RUN" = "1" ]; then
  print_cmd "${train_cmd[@]}"
  TRAIN_JOB="dry"
  echo "  Array mapping (${NUM_TASKS} tasks):"
  for (( i=0; i<NUM_TASKS; i++ )); do
    combo_idx=$(( i / NUM_FOLDS ))
    fold_idx=$(( i % NUM_FOLDS ))
    echo "    task $i → combo=${COMBOS[$combo_idx]}, fold=${FOLDS[$fold_idx]}"
  done
else
  TRAIN_JOB="$("${train_cmd[@]}")"
  echo "Submitted train array job: ${TRAIN_JOB} (${NUM_TASKS} tasks: ${MODALITY_COMBOS_CSV} × folds ${FOLDS_CSV})"
fi

# Step 3: Report all combos (single job, runs after all array tasks complete)
report_cmd=(); append_sbatch_args report_cmd 0
report_cmd+=(--job-name=swan_bl_report "--output=${LOGS_ROOT}/%x_%j.log")
report_cmd+=("--export=${EXPORT_SPEC},ACTION=report-cv")
[ "$TRAIN_JOB" != "dry" ] && report_cmd+=("--dependency=afterok:${TRAIN_JOB}")
report_cmd+=("${SCRIPT_DIR}/baseline.sh")
if [ "$DRY_RUN" = "1" ]; then
  print_cmd "${report_cmd[@]}"
else
  REPORT_JOB="$("${report_cmd[@]}")"
  echo "Submitted report-cv job: ${REPORT_JOB}"
fi

echo ""
echo "All baseline jobs submitted. Monitor with: squeue -u ${USER}"
echo "Logs root: ${LOGS_ROOT}"
