#!/usr/bin/env bash
# 04_build_dataset.sh — Generate the LlamaFactory ShareGPT dataset JSON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

: "${DATA_ROOT:=/mnt/data/Schwan_T3_FineTune}"
: "${OUT_DIR:=gpu_server/data}"
export DATA_ROOT OUT_DIR

# Set to 1 to include no_annotation segments in training
# export INCLUDE_NO_ANNOTATION=0

echo "Building LlamaFactory dataset..."
python src/build_llamafactory_dataset.py
