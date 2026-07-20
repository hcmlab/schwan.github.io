#!/usr/bin/env bash
# 01_discover_sessions.sh — Find completed sessions with .done markers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

: "${DATA_ROOT:=/mnt/data/Schwan_T3_FineTune}"
: "${OUT_DIR:=gpu_server/data}"
export DATA_ROOT OUT_DIR

echo "Running session discovery..."
python src/discover_sessions.py
