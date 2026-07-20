#!/usr/bin/env bash
# 05_verify_videos.sh — Verify all videos in the dataset can be decoded
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Load env variables (for DATA_ROOT and OUT_DIR)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Ensure conda env is activated
if ! command -v conda &> /dev/null; then
    echo "conda not found. Please activate the environment manually."
else
    # Allow conda activation in script
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate icep_omni
fi

echo "============================================================"
echo "Verifying Dataset Videos with PyAV"
echo "============================================================"

python src/verify_videos.py

echo "============================================================"
echo "Verification finished!"
