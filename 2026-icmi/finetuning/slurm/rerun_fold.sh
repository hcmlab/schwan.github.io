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
  bash slurm/rerun_fold.sh --profile PROFILE --dataset-spec SPEC --model MODEL --fold FOLD [options]

Required:
  --profile PROFILE
  --dataset-spec SPEC
  --model MODEL
  --fold FOLD

Options:
  --role-mode joint|infant|caregiver
  --exclude-label LABEL
  --overwrite
HELP
}

PROFILE=""
DATASET_SPEC=""
MODEL=""
FOLD=""
ROLE_MODE=""
OVERWRITE=0
EXCLUDE_LABELS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --dataset-spec) DATASET_SPEC="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --role-mode) ROLE_MODE="$2"; shift 2 ;;
    --exclude-label) EXCLUDE_LABELS+=("$2"); shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[ -z "$PROFILE" ] || [ -z "$DATASET_SPEC" ] || [ -z "$MODEL" ] || [ -z "$FOLD" ] && { usage >&2; exit 1; }

cd "$FT_DIR"
cmd=(
  python3 -m swan_ft train fold
  --profile "$PROFILE"
  --dataset-spec "$DATASET_SPEC"
  --model "$MODEL"
  --folds "$FOLD"
  --submit
)
swan_add_filter_args cmd "$ROLE_MODE" "$(IFS=,; printf '%s' "${EXCLUDE_LABELS[*]}")"
if [ "$OVERWRITE" = "1" ]; then
  cmd+=(--overwrite)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '
'
"${cmd[@]}"
