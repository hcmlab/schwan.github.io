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
BASELINE_SPEC="${BASELINE_SPEC:?BASELINE_SPEC is required}"
ROLE_MODE="${ROLE_MODE:-}"
EXCLUDE_LABELS="${EXCLUDE_LABELS:-}"
FOLDS_CSV="${FOLDS_CSV:-0,1,2,3,4}"
OVERWRITE="${OVERWRITE:-0}"
DEVICE="${DEVICE:-cuda}"
ACTION="${ACTION:-pipeline}"
# For train-fold array jobs: comma-separated combo list and SLURM_ARRAY_TASK_ID
MODALITY_COMBOS_CSV="${MODALITY_COMBOS_CSV:-video,audio,omni}"

# Lightweight env setup: activate swan2 directly (skip swan_setup_runtime which
# does pip check / torchcodec repair that can fail on cloned envs).
# swan2 already has torch, transformers, torchcodec from the LLM pipeline.
_conda_activate="$(swan_conda_activate "$PROFILE")"
source "$_conda_activate"
conda activate "${SWAN_CONDA_ENV_NAME:-swan2}"

# Temp / cache dirs
_temp_root="$(swan_temp_root "$PROFILE" "$FT_DIR")"
mkdir -p "$_temp_root"
export TMPDIR="$_temp_root"
export PYTHONNOUSERSITE=1

_cache_root="$(swan_cache_root "$PROFILE")"
if [ -n "$_cache_root" ]; then
    export HF_HOME="$_cache_root"
    mkdir -p "$HF_HOME"
fi

# Install baseline deps to node-local /tmp (CIFS mount can't handle pip at all)
_bl_prefix="/tmp/swan_baseline_site_packages"
if ! PYTHONPATH="${_bl_prefix}" python -c "import timm" >/dev/null 2>&1; then
    echo "Installing baseline dependencies to ${_bl_prefix}..."
    TMPDIR=/tmp python -m pip install -q --target "$_bl_prefix" --no-deps timm scikit-learn threadpoolctl joblib
fi
export PYTHONPATH="${_bl_prefix}:${PYTHONPATH:-}"
swan_log_runtime "$PROFILE" "$FT_DIR"
PYTHON_BIN="$(swan_python_bin "$PROFILE")"

echo "================================================================"
echo "Swan baseline: ${ACTION}"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local} (array task: ${SLURM_ARRAY_TASK_ID:-N/A})"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Profile: ${PROFILE}"
echo "Dataset spec: ${DATASET_SPEC}"
echo "Baseline spec: ${BASELINE_SPEC}"
echo "Folds: ${FOLDS_CSV}"
echo "Device: ${DEVICE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================================================"

cd "${FT_DIR}"

build_cmd() {
    local action="$1"
    local -n cmd_out="$2"
    cmd_out=(
        "${PYTHON_BIN}" -m swan_ft baseline "${action}"
        --profile "${PROFILE}"
        --dataset-spec "${DATASET_SPEC}"
        --baseline-spec "${BASELINE_SPEC}"
        --device "${DEVICE}"
        --execute-local
    )
    swan_add_fold_args cmd_out "${FOLDS_CSV}"
    swan_add_filter_args cmd_out "${ROLE_MODE}" "${EXCLUDE_LABELS}"
    if [ "$OVERWRITE" = "1" ]; then
        cmd_out+=(--overwrite)
    fi
}

run_step() {
    local action="$1"
    shift
    local cmd=()
    build_cmd "$action" cmd
    # Append any extra args (e.g. --modality-combo, --folds override)
    cmd+=("$@")
    printf 'Running:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
}

case "${ACTION}" in
    extract)
        run_step extract
        ;;
    train-fold)
        # Array job: SLURM_ARRAY_TASK_ID encodes (combo_idx * num_folds + fold_idx)
        IFS=',' read -ra COMBOS <<< "${MODALITY_COMBOS_CSV}"
        IFS=',' read -ra FOLDS_ARR <<< "${FOLDS_CSV}"
        TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required for train-fold}"
        NUM_FOLDS="${#FOLDS_ARR[@]}"
        COMBO_IDX=$(( TASK_ID / NUM_FOLDS ))
        FOLD_IDX=$(( TASK_ID % NUM_FOLDS ))
        COMBO="${COMBOS[$COMBO_IDX]}"
        FOLD="${FOLDS_ARR[$FOLD_IDX]}"
        echo "Array task ${TASK_ID}: combo=${COMBO}, fold=${FOLD}"
        # Override FOLDS_CSV so build_cmd uses the single fold
        FOLDS_CSV="${FOLD}"
        run_step train-fold --modality-combo "${COMBO}"
        ;;
    train-cv)
        run_step train-cv
        ;;
    report-cv)
        run_step report-cv
        ;;
    pipeline)
        run_step extract
        run_step train-cv
        run_step report-cv
        ;;
    *)
        echo "ERROR: Unknown ACTION: ${ACTION}. Use extract, train-fold, train-cv, report-cv, or pipeline." >&2
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "Baseline ${ACTION} complete at: $(date)"
echo "================================================================"
