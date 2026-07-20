# Schwan T3 Fine-Tune: Data Preparation Pipeline

This directory contains the scripts for processing raw Schwan T3 session data into a multimodal dataset for LlamaFactory (Omni) fine-tuning.

## 0. Environmental Setup

### Using Conda (Recommended)

To create a new environment and install dependencies:

```bash
# Create the environment
conda create -n schwan-dataprep python=3.10 -y

# Activate the environment
conda activate schwan-dataprep

# Install dependencies
pip install -r requirements.txt
```

### System Requirements

- **FFmpeg**: Must be installed and available in your PATH.
- **NVIDIA GPU**: Required for hardware-accelerated chunking (`h264_nvenc`).

## 1. Configuration

All paths and script parameters are centralized in `config.yaml`.

### Profiles

The pipeline supports multiple platform profiles:

- **windows**: Uses `X:\` mount points for local processing.
- **ubuntu**: Uses `/mnt/dataset-swan/` mount points for H100 VM processing.

The profile is **automatically detected** based on your OS. To manually override, use the `DATA_PREP_PROFILE` environment variable:

```bash
# Force Ubuntu profile on Windows (or vice versa)
export DATA_PREP_PROFILE=ubuntu  # Linux/macOS
set DATA_PREP_PROFILE=ubuntu     # Windows CMD
$env:DATA_PREP_PROFILE="ubuntu"  # Windows PowerShell
```

## 2. Pipeline Execution Steps

Run these scripts in order to build the dataset:

### Step 0: Prepare Annotations

Reads raw session JSONs and creates enriched annotation files with context buffers and descriptions.

```bash
python 00_prepare_annotations.py
```

### Step 1: Extract Chunks

Uses FFmpeg to extract video and audio clips based on the annotations.

- Includes a progress bar and graceful shutdown (`Ctrl+C`).
- Hardware-accelerated with NVENC.

```bash
python 01_extract_chunks.py
```

### Step 2: Create Dataset

Aggregates all extracted chunks into a single `llamafactory_dataset_full.json`.

```bash
python 02_create_dataset.py
```

### Step 3: Split Data

Splits the full dataset into train, validation, and test sets.

- Generates the `dataset_info.json` required by LlamaFactory.

```bash
python 03_split_data.py
```

### Step 4: Verify Format

Validates the final `dataset_train.json` to ensure it matches the LlamaFactory Omni format.

```bash
python 04_verify_omni_format.py
```
