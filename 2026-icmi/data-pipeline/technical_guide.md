# Technical Guide — ICEP Fine-Tuning Pipeline on H100 VM

Complete copy-paste guide for setting up and running the Qwen2.5-Omni ICEP fine-tuning pipeline on an Ubuntu H100 GPU server.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Getting the Code](#2-getting-the-code)
3. [Environment Installation](#3-environment-installation)
4. [Path Configuration](#4-path-configuration)
5. [Pipeline Step 0 — Check Paths & GPU](#5-pipeline-step-0--check-paths--gpu)
6. [Pipeline Step 1 — Discover Sessions](#6-pipeline-step-1--discover-sessions)
7. [Pipeline Step 2 — Build Chunk Manifest](#7-pipeline-step-2--build-chunk-manifest)
8. [Pipeline Step 3 — Session-Disjoint Split](#8-pipeline-step-3--session-disjoint-split)
9. [Pipeline Step 4 — Build LlamaFactory Dataset](#9-pipeline-step-4--build-llamafactory-dataset)
10. [Pipeline Step 5 — Smoke-Test Training](#10-pipeline-step-5--smoke-test-training)
11. [Troubleshooting](#11-troubleshooting)
12. [Script Reference](#12-script-reference)

---

## 1. Prerequisites

Your VM needs:

- **OS**: Ubuntu 22.04 or 24.04
- **GPU**: NVIDIA H100 (94 GB VRAM) with drivers installed
- **Storage**: The Schwan T3 FineTune dataset mounted (SMB/NFS share)
- **Software**: `git`, `conda` (Miniconda or Anaconda)

### Check GPU Drivers

```bash
nvidia-smi
```

You should see the H100 listed. If not, install NVIDIA drivers first.

### Install Conda (if not installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc
```

---

## 2. Getting the Code

### Option A: Git Clone

```bash
cd ~
git clone <YOUR_REPO_URL> schwan-finetune
cd schwan-finetune
```

### Option B: SCP/Rsync from Windows

From your local Windows machine (PowerShell or WSL):

```bash
# From WSL:
rsync -avz /mnt/d/GitHub/schwan-finetune/ user@<VM_IP>:~/schwan-finetune/

# From PowerShell:
scp -r D:\GitHub\schwan-finetune\ user@<VM_IP>:~/schwan-finetune/
```

Then on the VM:

```bash
cd ~/schwan-finetune
```

---

## 3. Environment Installation

### 3.1 Create Conda Environment

```bash
cd ~/schwan-finetune/gpu_server
conda env create -f env/conda_env.yaml
```

This installs:

- Python 3.11
- PyTorch 2.6.0 (CUDA)
- Transformers ≥ 4.57
- LlamaFactory
- ffmpeg, decord, opencv, deepspeed, and more

### 3.2 Activate Environment

```bash
conda activate icep_omni
```

> **Tip:** Add this to your `~/.bashrc` for auto-activation:
>
> ```bash
> echo "conda activate icep_omni" >> ~/.bashrc
> ```

### 3.3 Verify Installation

Run each line one at a time:

```bash
# Check GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA available:', torch.cuda.is_available())"

# Check LlamaFactory
llamafactory-cli version

# Check ffprobe (needed for manifest building)
ffprobe -version | head -1

# Check key packages
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import peft; print('peft:', peft.__version__)"
python -c "import accelerate; print('accelerate:', accelerate.__version__)"
```

**Expected output:**

```
GPU: NVIDIA H100 NVL
CUDA available: True
transformers: 4.57.x
peft: 0.15.x
accelerate: 1.4.x
```

### 3.4 Optional: Install FlashAttention-2

Recommended for faster training (requires CUDA ≥ 11.6):

```bash
pip install flash-attn --no-build-isolation
```

---

## 4. Path Configuration

### 4.1 Create .env File

```bash
cd ~/schwan-finetune/gpu_server
cp .env.example .env
nano .env
```

Edit the file to match your mount point:

```bash
# Root of the dataset (where session folders like ANSEBE01_HD_T3/ live)
DATA_ROOT=/mnt/data/Schwan_T3_FineTune

# Output directory for pipeline artefacts (relative to repo root or absolute)
OUT_DIR=gpu_server/data
```

### 4.2 Load Environment Variables

```bash
set -a; source .env; set +a
```

> **Tip:** Add this to `~/.bashrc` so it auto-loads:
>
> ```bash
> echo 'set -a; source ~/schwan-finetune/gpu_server/.env; set +a' >> ~/.bashrc
> ```

### 4.3 Verify Configuration

```bash
echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_DIR=$OUT_DIR"
ls "$DATA_ROOT" | head -5
```

You should see session directory names like `ANSEBE01_HD_T3`, `ANOKPE01_HD_T3`, etc.

---

## 5. Pipeline Step 0 — Check Paths & GPU

**What it does:** Verifies that the data mount exists, counts sessions with `.done` markers, and checks GPU/conda availability.

**Script:** `scripts/00_check_paths.sh`

```bash
cd ~/schwan-finetune/gpu_server
bash scripts/00_check_paths.sh
```

**Expected output:**

```
========================================
 Path Check
========================================
DATA_ROOT : /mnt/data/Schwan_T3_FineTune
OUT_DIR   : gpu_server/data

✓ DATA_ROOT exists
  Session directories: 45

Checking for .done markers...
  Completed sessions (.done): 30

Checking CUDA...
NVIDIA H100 NVL, 94208 MiB

Checking conda env...
icep_omni    /home/user/miniconda3/envs/icep_omni

All checks complete.
```

**If DATA_ROOT doesn't exist:** Check that your network share is mounted:

```bash
# Example for SMB mount:
sudo mount -t cifs //server/share /mnt/data -o username=user,password=pass
```

---

## 6. Pipeline Step 1 — Discover Sessions

**What it does:** Scans `DATA_ROOT` for session directories that:

- Match `*_T3` pattern OR contain a `chunks/` subdirectory
- Have a `.done` marker in `chunks/` (created by `data_prep/01_extract_chunks.py`)

**Script:** `scripts/01_discover_sessions.sh`

**Source:** `src/discover_sessions.py`

```bash
cd ~/schwan-finetune/gpu_server
bash scripts/01_discover_sessions.sh
```

**Output file:** `gpu_server/data/sessions_done.jsonl`

**Verify:**

```bash
# Count sessions found
wc -l gpu_server/data/sessions_done.jsonl

# View first 2 entries
head -2 gpu_server/data/sessions_done.jsonl | python -m json.tool
```

**Expected format per line:**

```json
{
    "session_id": "ANSEBE01_HD_T3",
    "session_path": "/mnt/data/Schwan_T3_FineTune/ANSEBE01_HD_T3",
    "chunks_path": "/mnt/data/Schwan_T3_FineTune/ANSEBE01_HD_T3/chunks",
    "has_done": true,
    "mp4_count": 250,
    "wav_count": 50
}
```

---

## 7. Pipeline Step 2 — Build Chunk Manifest

**What it does:** For each completed session:

1. Scans `chunks/` for MP4 and WAV files
2. Groups files by shared stem (WAVs have no camera suffix; MP4s have `_kamera1`, `_kamera2`, etc.)
3. Parses filename fields: `session_id`, `track`, `idx`, `short_code`
4. Probes each file's duration via `ffprobe`
5. Creates one manifest row per (chunk × camera angle)

**Script:** `scripts/02_build_manifest.sh`

**Source:** `src/build_manifest.py`

```bash
cd ~/schwan-finetune/gpu_server
bash scripts/02_build_manifest.sh
```

> ⏱ **This step may take 10–30 minutes** depending on the number of files (runs `ffprobe` on each).

**Output file:** `gpu_server/data/chunk_manifest.jsonl`

**Verify:**

```bash
# Total rows
wc -l gpu_server/data/chunk_manifest.jsonl

# How many are no_annotation
grep -c '"no_annotation"' gpu_server/data/chunk_manifest.jsonl

# Pretty-print first row
head -1 gpu_server/data/chunk_manifest.jsonl | python -m json.tool
```

**Expected format per line:**

```json
{
    "session_id": "ANSEBE01_HD_T3",
    "track": "Caregiver_Engagement",
    "idx": 1,
    "short_code": "Cneu",
    "video_path": "/mnt/data/.../ANSEBE01_HD_T3_Caregiver_Engagement_idx0001_Cneu_kamera1.mp4",
    "audio_path": "/mnt/data/.../ANSEBE01_HD_T3_Caregiver_Engagement_idx0001_Cneu.wav",
    "duration_sec": 8.234
}
```

**Filename parsing logic:**

```
ANSEBE01_HD_T3_Caregiver_Engagement_idx0001_Cneu_kamera1.mp4
└─ session_id ─┘ └──── track ─────┘ └idx┘ └code┘ └camera┘
```

---

## 8. Pipeline Step 3 — Session-Disjoint Split

**What it does:** Assigns each session to train/val/test using a deterministic hash so chunks from the same session **never** appear in different splits (prevents data leakage).

**Split rule:** `MD5(session_id) % 100`

| Range | Split | Proportion |
|-------|-------|------------|
| 0–79  | train | ~80%       |
| 80–89 | val   | ~10%       |
| 90–99 | test  | ~10%       |

**Script:** `scripts/03_split_sessions.sh`

**Source:** `src/split_sessions.py`

```bash
cd ~/schwan-finetune/gpu_server
bash scripts/03_split_sessions.sh
```

**Output files:**

```
gpu_server/data/splits/train_sessions.json
gpu_server/data/splits/val_sessions.json
gpu_server/data/splits/test_sessions.json
```

**Verify:**

```bash
# Count sessions per split
python -c "
import json
for split in ['train', 'val', 'test']:
    ids = json.load(open(f'gpu_server/data/splits/{split}_sessions.json'))
    print(f'{split:5s}: {len(ids)} sessions')
"

# View train session IDs
cat gpu_server/data/splits/train_sessions.json | python -m json.tool
```

---

## 9. Pipeline Step 4 — Build LlamaFactory Dataset

**What it does:**

1. Reads the chunk manifest + train split + enriched annotation JSONs
2. Builds ShareGPT-formatted entries with:
   - **`conversations`**: system prompt + human prompt (with `<video>` tag) + assistant JSON response
   - **`videos`**: list with the MP4 path
3. Uses full ICEP descriptions from annotation JSONs as ground-truth rationale

**By default, `no_annotation` segments are excluded.** Set `INCLUDE_NO_ANNOTATION=1` to include them.

**Script:** `scripts/04_build_dataset.sh`

**Source:** `src/build_llamafactory_dataset.py`

```bash
cd ~/schwan-finetune/gpu_server
bash scripts/04_build_dataset.sh
```

**Output files:**

```
gpu_server/data/llamafactory/schwan_icep_sft.json      (training data)
gpu_server/data/llamafactory/dataset_info.json          (LlamaFactory descriptor)
```

**Verify:**

```bash
# Count examples
python -c "
import json
d = json.load(open('gpu_server/data/llamafactory/schwan_icep_sft.json'))
print(f'Total training examples: {len(d)}')
"

# Pretty-print first entry
python -c "
import json
d = json.load(open('gpu_server/data/llamafactory/schwan_icep_sft.json'))
print(json.dumps(d[0], indent=2, ensure_ascii=False))
"

# Verify dataset_info.json
cat gpu_server/data/llamafactory/dataset_info.json | python -m json.tool
```

**Expected entry format:**

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nPredict the ICEP engagement label for the caregiver's behavior in this clip. Respond with a JSON object containing: track, short_code, label, and rationale."
    },
    {
      "from": "gpt",
      "value": "{\"track\": \"Caregiver_Engagement\", \"short_code\": \"Cneu\", \"label\": \"Social Monitor / No Vocs or Neutral Vocs\", \"caregiver_phase\": \"Cneu\", \"infant_phase\": null, \"rationale\": \"The adult watches...\"}"
    }
  ],
  "system": "You are an expert in the Infant Caregiver Engagement Phases...",
  "videos": ["/mnt/data/.../ANSEBE01_HD_T3_Caregiver_Engagement_idx0001_Cneu_kamera1.mp4"]
}
```

**To include no_annotation segments:**

```bash
INCLUDE_NO_ANNOTATION=1 bash scripts/04_build_dataset.sh
```

---

## 10. Pipeline Step 5 — Smoke-Test Training

**What it does:** Runs a LoRA SFT fine-tune on Qwen2.5-Omni-7B using LlamaFactory with:

- Template: `qwen2_omni`
- LoRA rank 16, alpha 32
- `max_samples: 200` (first 200 examples only)
- 1 epoch, bf16, batch size 1 + gradient accumulation 8
- Cosine LR schedule, lr=1e-5

**Config file:** `configs/llamafactory_qwen2_omni_lora.yaml`

**Script:** `scripts/10_train_smoketest.sh`

```bash
cd ~/schwan-finetune/gpu_server
bash scripts/10_train_smoketest.sh
```

**Model will be downloaded on first run** (~15 GB for Qwen2.5-Omni-7B). Subsequent runs use the HuggingFace cache.

**Expected output:**

```
========================================
 Qwen2.5-Omni LoRA SFT Smoke-Test
========================================
GPU: NVIDIA H100 NVL

Starting LlamaFactory training...
[INFO] Loading model Qwen/Qwen2.5-Omni-7B...
[INFO] Loading dataset schwan_icep_sft...
...
{'loss': 2.345, 'learning_rate': 9.5e-06, 'epoch': 0.08}
{'loss': 1.876, 'learning_rate': 8.2e-06, 'epoch': 0.16}
...
TrainOutput(global_step=25, ...)
```

**Checkpoint output:** `gpu_server/saves/qwen2_omni/icep_smoketest/`

### Adjusting Training Parameters

Edit `configs/llamafactory_qwen2_omni_lora.yaml`:

```yaml
# To train on more data:
max_samples: 1000           # or remove line entirely for all data

# To train longer:
num_train_epochs: 3

# If OOM on long videos:
cutoff_len: 4096            # reduce from 8192

# For full training (not smoke-test):
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 2.0e-5
num_train_epochs: 3
save_steps: 500
```

---

## 11. Troubleshooting

### Environment Issues

| Problem | Solution |
|---|---|
| `conda env create` fails | Try `conda env create -f env/conda_env.yaml --force` |
| `torch.cuda.is_available()` returns `False` | Check NVIDIA drivers: `nvidia-smi`. May need `conda install pytorch-cuda=12.4 -c nvidia` |
| `llamafactory-cli: command not found` | `pip install llamafactory` |
| `ffprobe: command not found` | `conda install -c conda-forge ffmpeg` |

### Data Issues

| Problem | Solution |
|---|---|
| `DATA_ROOT does not exist` | Mount the share: `sudo mount -t cifs //server/share /mnt/data -o username=X` |
| `sessions_done.jsonl` is empty | `data_prep/01_extract_chunks.py` hasn't finished — wait for `.done` markers |
| `chunk_manifest.jsonl` has 0 rows | Check that `chunks/` directories contain `.mp4` and `.wav` files |
| No train examples generated | Check `splits/train_sessions.json` isn't empty; verify annotation JSONs exist |

### Training Issues

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Lower `cutoff_len` to 4096; reduce `max_samples` to 100 |
| `template qwen2_omni not found` | Upgrade: `pip install --upgrade llamafactory` |
| Very slow first step | Model downloading (~15 GB) — check with `htop` / `watch nvidia-smi` |
| `KeyError: 'conversations'` | Verify `dataset_info.json` maps  `"messages": "conversations"` |

---

## 12. Script Reference

### Source Scripts (Python)

| Script | Input | Output | Description |
|---|---|---|---|
| `src/discover_sessions.py` | `DATA_ROOT` env var | `data/sessions_done.jsonl` | Finds sessions with `.done` markers |
| `src/build_manifest.py` | `sessions_done.jsonl` | `data/chunk_manifest.jsonl` | Pairs MP4/WAV, parses filenames, probes duration |
| `src/split_sessions.py` | `sessions_done.jsonl` | `data/splits/*.json` | Hash-based session-disjoint splits |
| `src/build_llamafactory_dataset.py` | manifest + splits + annotations | `data/llamafactory/*.json` | ShareGPT multimodal dataset for LlamaFactory |

### Shell Wrappers

| Script | Purpose |
|---|---|
| `scripts/00_check_paths.sh` | Sanity check: data mount, GPU, conda env |
| `scripts/01_discover_sessions.sh` | Wrapper for `discover_sessions.py` |
| `scripts/02_build_manifest.sh` | Wrapper for `build_manifest.py` |
| `scripts/03_split_sessions.sh` | Wrapper for `split_sessions.py` |
| `scripts/04_build_dataset.sh` | Wrapper for `build_llamafactory_dataset.py` |
| `scripts/10_train_smoketest.sh` | CUDA check + `llamafactory-cli train` |

### Config Files

| File | Purpose |
|---|---|
| `.env.example` | Template for `DATA_ROOT` and `OUT_DIR` |
| `env/conda_env.yaml` | Conda environment (Python 3.11, torch 2.6, LlamaFactory) |
| `configs/llamafactory_qwen2_omni_lora.yaml` | LoRA SFT hyperparameters for Qwen2.5-Omni |
| `configs/dataset_info.json` | LlamaFactory dataset descriptor (ShareGPT columns) |
