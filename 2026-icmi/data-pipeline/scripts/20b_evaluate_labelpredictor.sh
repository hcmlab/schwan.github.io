#!/usr/bin/env bash
# 20_evaluate_model.sh — Evaluate the trained LoRA adapter on the test set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# Automatically load .env if it exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Disable wandb unless explicitly configured
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

echo "========================================"
echo " Evaluating Qwen2.5-Omni LoRA SFT"
echo "========================================"

# Resolve the models output directory where we saved the LoRA
repo_root="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$repo_root/models/qwen2_omni/icep_labelpredictor_full"
PREDICT_DIR="$OUTPUT_DIR/eval_results"

# Read OUT_DIR and resolve the dataset directory
export OUT_DIR="${OUT_DIR:-data}"
if [[ "$OUT_DIR" = /* ]]; then
    DATASET_DIR="$OUT_DIR/llamafactory"
else
    DATASET_DIR="$(pwd)/$OUT_DIR/llamafactory"
fi

echo ""
echo "Model adapter path: $OUTPUT_DIR"
echo "Dataset dir: $DATASET_DIR"
echo "Writing predictions to: $PREDICT_DIR"

# Generate a temporary eval config file
TMP_CONFIG="configs/run_eval.yaml"

python -c "
import yaml

with open('configs/llamafactory_labelpredictor_eval.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Inject our dynamic paths safely bypassing CLI argument parser bugs
config['dataset_dir'] = '${DATASET_DIR}'
config['adapter_name_or_path'] = '${OUTPUT_DIR}'
config['output_dir'] = '${PREDICT_DIR}'

with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)
"

echo "Starting predictions..."

echo "========================================"
echo "DEBUG: Content of ${DATASET_DIR}/dataset_info.json"
echo "========================================"
cat "${DATASET_DIR}/dataset_info.json"
echo "========================================"

llamafactory-cli train $TMP_CONFIG

echo ""
echo "Evaluation generation complete!"
echo "Predictions saved to $PREDICT_DIR/generated_predictions.jsonl"

echo ""
echo "Calculating F1, Precision, and Recall Metrics..."
python src/evaluate_labelpredictor_metrics.py --predictions "$PREDICT_DIR/generated_predictions.jsonl"
