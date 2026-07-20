#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FT_DIR="${FT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=ft/slurm/common.sh
source "${FT_DIR}/slurm/common.sh"

REPO_ROOT="$(cd "${FT_DIR}/.." && pwd)"
swan_load_env "$REPO_ROOT"

PROFILE="${PROFILE:?PROFILE is required}"
DATASET_SPEC="${DATASET_SPEC:?DATASET_SPEC is required}"
MODEL="${MODEL:?MODEL is required}"
ROLE_MODE="${ROLE_MODE:-}"
EXCLUDE_LABELS="${EXCLUDE_LABELS:-}"
FOLDS_CSV="${FOLDS_CSV:-0,1,2,3,4}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

swan_setup_runtime "$PROFILE" "$FT_DIR"
swan_log_runtime "$PROFILE" "$FT_DIR"
PYTHON_BIN="$(swan_python_bin "$PROFILE")"

echo "================================================================"
echo "Swan evaluation"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Profile: ${PROFILE}"
echo "Dataset spec: ${DATASET_SPEC}"
echo "Model: ${MODEL}"
echo "Folds: ${FOLDS_CSV}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================================================"

cd "${FT_DIR}"

predict_cmd=(
    "${PYTHON_BIN}" -m swan_ft predict cv
    --profile "${PROFILE}"
    --dataset-spec "${DATASET_SPEC}"
    --model "${MODEL}"
    --execute-local
)
swan_add_fold_args predict_cmd "${FOLDS_CSV}"
swan_add_filter_args predict_cmd "${ROLE_MODE}" "${EXCLUDE_LABELS}"
if [ "$OVERWRITE" = "1" ]; then
    predict_cmd+=(--overwrite)
fi
if [ "$DRY_RUN" = "1" ]; then
    predict_cmd+=(--dry-run)
fi

report_cmd=(
    "${PYTHON_BIN}" -m swan_ft report cv
    --profile "${PROFILE}"
    --dataset-spec "${DATASET_SPEC}"
    --model "${MODEL}"
)
swan_add_fold_args report_cmd "${FOLDS_CSV}"
swan_add_filter_args report_cmd "${ROLE_MODE}" "${EXCLUDE_LABELS}"
if [ "$OVERWRITE" = "1" ]; then
    report_cmd+=(--overwrite)
fi
if [ "$DRY_RUN" = "1" ]; then
    report_cmd+=(--dry-run)
fi

printf 'Running:'
printf ' %q' "${predict_cmd[@]}"
printf '\n'
"${predict_cmd[@]}"

printf 'Running:'
printf ' %q' "${report_cmd[@]}"
printf '\n'
"${report_cmd[@]}"

echo ""
echo "================================================================"
echo "Evaluation complete at: $(date)"
echo "================================================================"