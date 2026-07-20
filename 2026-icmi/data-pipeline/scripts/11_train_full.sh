#!/usr/bin/env bash
# 11_train_full.sh — Run a full LoRA SFT on Qwen2.5-Omni.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable wandb for full training
export WANDB_PROJECT="schwan-icep-sft"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"

echo "========================================"
echo " Qwen2.5-Omni LoRA SFT Full Fine-tune"
echo "========================================"

# Sanity check: CUDA available?
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); assert torch.cuda.is_available()"

# Resolve the models output directory
repo_root="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$repo_root/models/qwen2_omni/icep_sft_full"

# Read OUT_DIR and resolve the dataset directory to an ABSOLUTE path
export OUT_DIR="${OUT_DIR:-data}"
if [[ "$OUT_DIR" = /* ]]; then
    DATASET_DIR="$OUT_DIR/llamafactory"
else
    DATASET_DIR="$repo_root/$OUT_DIR/llamafactory"
fi

echo ""
echo "Starting LlamaFactory training (dataset_dir=$DATASET_DIR)..."
echo "Output will be saved to: $OUTPUT_DIR"

# Generate a temporary config file that merges our dynamic paths
TMP_CONFIG="configs/run_full.yaml"

python -c "
import yaml

with open('configs/llamafactory_qwen2_omni_lora_full.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Inject our dynamic paths
config['dataset_dir'] = '${DATASET_DIR}'
config['output_dir'] = '${OUTPUT_DIR}'

with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)
"

# Run training against this temporary merged config
llamafactory-cli train $TMP_CONFIG
