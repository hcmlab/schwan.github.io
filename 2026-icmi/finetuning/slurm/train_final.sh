#!/bin/bash
set -euo pipefail

FT_DIR="${FT_DIR:?FT_DIR is required}"
# shellcheck source=/dev/null
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

swan_setup_runtime "$PROFILE" "$FT_DIR"
swan_log_runtime "$PROFILE" "$FT_DIR"
PYTHON_BIN="$(swan_python_bin "$PROFILE")"

cd "$FT_DIR"
cmd=(
  "$PYTHON_BIN" -m swan_ft train final
  --profile "$PROFILE"
  --dataset-spec "$DATASET_SPEC"
  --model "$MODEL"
  --execute-local
)
swan_add_filter_args cmd "$ROLE_MODE" "$EXCLUDE_LABELS"
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
