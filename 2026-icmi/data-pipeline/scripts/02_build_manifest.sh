#!/usr/bin/env bash
# 02_build_manifest.sh — Build chunk-level manifest from completed sessions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

: "${DATA_ROOT:=/mnt/data/Schwan_T3_FineTune}"
: "${OUT_DIR:=gpu_server/data}"
export DATA_ROOT OUT_DIR

echo "Building chunk manifest..."
python src/build_manifest.py
