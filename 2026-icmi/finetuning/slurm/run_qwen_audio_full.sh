#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[DEPRECATED] This wrapper is kept for compatibility. Use bash slurm/run_pipeline.sh ... instead." >&2
exec bash "${SCRIPT_DIR}/run_pipeline.sh" "$@"
