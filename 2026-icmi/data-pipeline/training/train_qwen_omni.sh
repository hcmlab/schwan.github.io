#!/bin/bash

# Ensure CUDA and conda environment are set up
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate llamafactory

cd d:/GitHub/daksitha.withanage.don/gpu_server/training

# Optional: Disable wandb if not logged in
export WANDB_DISABLED=true

# Launch LlamaFactory using the CLI
echo "Starting LlamaFactory fine-tuning for Qwen-Omni on H100..."
llamafactory-cli train llamafactory_config.yaml
