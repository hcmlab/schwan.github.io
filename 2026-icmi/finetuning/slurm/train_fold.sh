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
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
FOLD="${SLURM_ARRAY_TASK_ID:-${FOLD:-}}"
if [ -z "$FOLD" ]; then
    echo "ERROR: FOLD is required or must be provided by SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

swan_setup_runtime "$PROFILE" "$FT_DIR"
swan_log_runtime "$PROFILE" "$FT_DIR"
PYTHON_BIN="$(swan_python_bin "$PROFILE")"

echo "================================================================"
echo "Swan fine-tuning"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Profile: ${PROFILE}"
echo "Dataset spec: ${DATASET_SPEC}"
echo "Model: ${MODEL}"
echo "Fold: ${FOLD}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================================================"

cd "${FT_DIR}"
cmd=(
    "${PYTHON_BIN}" -m swan_ft train fold
    --profile "${PROFILE}"
    --dataset-spec "${DATASET_SPEC}"
    --model "${MODEL}"
    --folds "${FOLD}"
    --execute-local
)
swan_add_filter_args cmd "${ROLE_MODE}" "${EXCLUDE_LABELS}"
if [ "$OVERWRITE" = "1" ]; then
    cmd+=(--overwrite)
fi
if [ "$DRY_RUN" = "1" ]; then
    cmd+=(--dry-run)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo ""
echo "================================================================"
echo "Fold ${FOLD} complete at: $(date)"
echo "================================================================"