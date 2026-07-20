#!/usr/bin/env bash
# 04_build_dataset.sh — Generate the LlamaFactory ShareGPT dataset JSON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# Automatically load .env if it exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

: "${DATA_ROOT:=/mnt/data/Schwan_T3_FineTune}"
: "${OUT_DIR:=gpu_server/data}"
export DATA_ROOT OUT_DIR

echo "=== DEBUG INFO ==="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUT_DIR: $OUT_DIR"
echo "PWD: $(pwd)"
echo "=================="

# Set to 1 to include no_annotation segments in training
# export INCLUDE_NO_ANNOTATION=0

echo "Building LlamaFactory dataset..."
python src/build_labelpredictor_dataset.py
