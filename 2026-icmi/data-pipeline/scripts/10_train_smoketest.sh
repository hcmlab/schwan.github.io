#!/usr/bin/env bash
# 10_train_smoketest.sh — Run a LoRA SFT smoke-test on Qwen2.5-Omni.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# Disable wandb unless explicitly configured
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

echo "========================================"
echo " Qwen2.5-Omni LoRA SFT Smoke-Test"
echo "========================================"

# Sanity check: CUDA available?
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); assert torch.cuda.is_available()"

# Read OUT_DIR and resolve the dataset directory
export OUT_DIR="${OUT_DIR:-gpu_server/data}"
DATASET_DIR="$OUT_DIR/llamafactory"

# Resolve the models output directory (e.g. ../models/qwen2_omni/icep_smoketest)
repo_root="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$repo_root/models/qwen2_omni/icep_smoketest"

echo ""
echo "Starting LlamaFactory training (dataset_dir=$DATASET_DIR)..."
echo "Output will be saved to: $OUTPUT_DIR"

# Generate a temporary config file that merges our dynamic paths
TMP_CONFIG="configs/run_smoketest.yaml"

python -c "
import yaml

with open('configs/llamafactory_qwen2_omni_lora.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Inject our dynamic paths
config['dataset_dir'] = '${DATASET_DIR}'
config['output_dir'] = '${OUTPUT_DIR}'

with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)
"

# Run training against this temporary merged config
llamafactory-cli train $TMP_CONFIG



