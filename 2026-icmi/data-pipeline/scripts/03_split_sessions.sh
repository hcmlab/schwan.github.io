#!/usr/bin/env bash
# 03_split_sessions.sh — Create session-disjoint train/val/test splits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

: "${OUT_DIR:=gpu_server/data}"
export OUT_DIR

echo "Splitting sessions..."
python src/split_sessions.py
