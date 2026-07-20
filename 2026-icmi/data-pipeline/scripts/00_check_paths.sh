#!/usr/bin/env bash
# 00_check_paths.sh — Verify that the data mount and key dirs exist.
set -euo pipefail

: "${DATA_ROOT:=/mnt/data/Schwan_T3_FineTune}"
: "${OUT_DIR:=gpu_server/data}"

echo "========================================"
echo " Path Check"
echo "========================================"
echo "DATA_ROOT : $DATA_ROOT"
echo "OUT_DIR   : $OUT_DIR"
echo ""

if [ -d "$DATA_ROOT" ]; then
    echo "✓ DATA_ROOT exists"
    n_dirs=$(find "$DATA_ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l)
    echo "  Session directories: $n_dirs"
else
    echo "✗ DATA_ROOT does not exist!"
    exit 1
fi

echo ""
echo "Checking for .done markers..."
n_done=$(find "$DATA_ROOT" -name ".done" -path "*/chunks/.done" | wc -l)
echo "  Completed sessions (.done): $n_done"

echo ""
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  nvidia-smi not found (no GPU or drivers not installed)"
fi

echo ""
echo "Checking conda env..."
if command -v conda &> /dev/null; then
    conda info --envs | grep icep_omni || echo "  icep_omni env not found"
else
    echo "  conda not found"
fi

echo ""
echo "All checks complete."
