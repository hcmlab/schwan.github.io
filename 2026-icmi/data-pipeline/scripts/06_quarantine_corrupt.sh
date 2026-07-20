#!/usr/bin/env bash
# 06_quarantine_corrupt.sh — Physically move corrupted videos to a quarantine folder
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Load env variables
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Ensure conda env is activated
if ! command -v conda &> /dev/null; then
    echo "conda not found. Please activate the environment manually."
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate icep_omni
fi

echo "============================================================"
echo "Quarantining Corrupted Videos with PyAV"
echo "============================================================"

# Pass any extra args (like --dry-run) to the python script
python src/quarantine_corrupt_videos.py "$@"

echo "============================================================"
echo "Cleanup finished!"
