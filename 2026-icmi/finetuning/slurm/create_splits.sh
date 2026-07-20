#!/bin/bash
set -euo pipefail

FT_DIR="${FT_DIR:?FT_DIR is required}"
# shellcheck source=/dev/null
source "${FT_DIR}/slurm/common.sh"

REPO_ROOT="$(cd "${FT_DIR}/.." && pwd)"
swan_load_env "$REPO_ROOT"

PROFILE="${PROFILE:?PROFILE is required}"
DATASET_SPEC="${DATASET_SPEC:?DATASET_SPEC is required}"
N_FOLDS="${N_FOLDS:-5}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
SPLIT_SEED="${SPLIT_SEED:-42}"

swan_setup_runtime "$PROFILE" "$FT_DIR"
swan_log_runtime "$PROFILE" "$FT_DIR"
PYTHON_BIN="$(swan_python_bin "$PROFILE")"

echo "================================================================"
echo "Swan split creation"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Profile: ${PROFILE}"
echo "Dataset spec: ${DATASET_SPEC}"
echo "================================================================"

cd "$FT_DIR"
"$PYTHON_BIN" - <<PY
from swan_ft.config import load_dataset_spec, load_profile

profile = load_profile("${PROFILE}")
dataset_spec = load_dataset_spec("${DATASET_SPEC}")
folds_path = profile.data_root / dataset_spec.base_folds_file

checks = [
    ("session_root", profile.session_root),
    ("data_root", profile.data_root),
    ("output_root", profile.output_root),
    ("logs_root", profile.logs_root),
    ("folds_file", folds_path),
]

print("Resolved path checks:")
for label, value in checks:
    if value is None:
        print(f"  {label}: <unset>")
    else:
        print(f"  {label}: {value} exists={value.exists()}")
PY

cmd=(
  "$PYTHON_BIN" -m swan_ft folds create
  --profile "$PROFILE"
  --dataset-spec "$DATASET_SPEC"
  --n-folds "$N_FOLDS"
  --test-fraction "$TEST_FRACTION"
  --seed "$SPLIT_SEED"
)

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
