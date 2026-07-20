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
ROLE_MODE="${ROLE_MODE:-}"
EXCLUDE_LABELS="${EXCLUDE_LABELS:-}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

swan_setup_runtime "$PROFILE" "$FT_DIR"
swan_log_runtime "$PROFILE" "$FT_DIR"
PYTHON_BIN="$(swan_python_bin "$PROFILE")"

echo "================================================================"
echo "Swan dataset variant preparation"
echo "Job started at: $(date)"
echo "Profile: ${PROFILE}"
echo "Dataset spec: ${DATASET_SPEC}"
echo "================================================================"

cd "${FT_DIR}"
cmd=("${PYTHON_BIN}" -m swan_ft dataset build --profile "${PROFILE}" --dataset-spec "${DATASET_SPEC}")
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
echo "Preparation complete at: $(date)"
echo "================================================================"