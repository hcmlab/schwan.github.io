# GPU Server — Qwen2.5-Omni ICEP Fine-Tuning Pipeline

End-to-end pipeline for fine-tuning **Qwen2.5-Omni-7B** on ICEP (Infant Caregiver Engagement Phases) multimodal video annotations using **LlamaFactory**.

## Architecture

```
gpu_server/
├── .env.example                     # Environment variable template
├── README.md
├── env/
│   └── conda_env.yaml               # Conda environment (Python 3.11, torch 2.6, LlamaFactory)
├── src/
│   ├── discover_sessions.py          # Step B: Find completed sessions via .done
│   ├── build_manifest.py             # Step C: Chunk-level manifest (MP4/WAV pairing)
│   ├── split_sessions.py             # Step D: Session-disjoint train/val/test splits
│   └── build_llamafactory_dataset.py # Step E: LlamaFactory multimodal ShareGPT JSON
├── configs/
│   ├── llamafactory_qwen2_omni_lora.yaml  # LoRA SFT config for smoke-test
│   └── dataset_info.json                   # LlamaFactory dataset descriptor
├── scripts/
│   ├── 00_check_paths.sh             # Verify data mount + CUDA + conda
│   ├── 01_discover_sessions.sh       # → discover_sessions.py
│   ├── 02_build_manifest.sh          # → build_manifest.py
│   ├── 03_split_sessions.sh          # → split_sessions.py
│   ├── 04_build_dataset.sh           # → build_llamafactory_dataset.py
│   └── 10_train_smoketest.sh         # → llamafactory-cli train
└── data/                             # Generated at runtime (gitignored)
    ├── sessions_done.jsonl
    ├── chunk_manifest.jsonl
    ├── splits/
    │   ├── train_sessions.json
    │   ├── val_sessions.json
    │   └── test_sessions.json
    └── llamafactory/
        ├── schwan_icep_sft.json
        └── dataset_info.json
```

## Quick Start

### 1. Environment Setup

```bash
# On the H100 VM:
cd gpu_server
conda env create -f env/conda_env.yaml
conda activate icep_omni

# Sanity check:
python -c "import torch; print(torch.cuda.get_device_name(0))"
llamafactory-cli -h
```

### 2. Configure Paths

```bash
cp .env.example .env
# Edit .env to set DATA_ROOT to your mounted dataset path
# Default: /mnt/data/Schwan_T3_FineTune
```

### 3. Run the Pipeline

```bash
# Step 0: Verify paths and GPU
bash scripts/00_check_paths.sh

# Step 1: Discover completed sessions
bash scripts/01_discover_sessions.sh

# Step 2: Build chunk manifest (requires ffprobe)
bash scripts/02_build_manifest.sh

# Step 3: Session-disjoint train/val/test split
bash scripts/03_split_sessions.sh

# Step 4: Generate LlamaFactory dataset
bash scripts/04_build_dataset.sh

# Step 5: Run smoke-test training
bash scripts/10_train_smoketest.sh
```

## Design Decisions

### Session Discovery

Sessions are identified by the `.done` marker file inside `chunks/`, created by the existing `data_prep/01_extract_chunks.py` pipeline. This ensures only fully extracted sessions are included.

### Data Splitting

Split unit is **session** (not chunk) to prevent data leakage — all caregiver and infant chunks from the same session stay in the same split. Uses `hash(session_id) % 100` for deterministic, seed-free assignment:

- **train**: 0–79 (80%)
- **val**: 80–89 (10%)
- **test**: 90–99 (10%)

### Multimodal Format

- **Video-only inputs**: MP4 contains audio; Qwen2.5-Omni loads it via `use_audio_in_video=True`
- **ShareGPT format**: `conversations` column with `from: human/gpt`
- **Template**: `qwen2_omni` (not `qwen2_vl`)
- **`no_annotation` segments**: Excluded from training by default (set `INCLUDE_NO_ANNOTATION=1` to include)

### Assistant Response Format

Machine-parseable JSON containing:

```json
{
  "track": "Infant_Engagement",
  "short_code": "Ineu",
  "label": "Social Monitor",
  "caregiver_phase": null,
  "infant_phase": "Ineu",
  "rationale": "Full ICEP description + visual/auditory cues..."
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | `/mnt/data/Schwan_T3_FineTune` | Root of the dataset |
| `OUT_DIR` | `gpu_server/data` | Output directory for pipeline artefacts |
| `INCLUDE_NO_ANNOTATION` | `0` | Set to `1` to include unannotated segments |

## Relationship to Existing Code

This pipeline (`src/`) is designed to run **after** the existing `data_prep/` scripts have completed:

1. `data_prep/00_prepare_annotations.py` → enriched annotation JSONs
2. `data_prep/01_extract_chunks.py` → MP4/WAV chunks + `.done` markers

The `src/` scripts then discover, manifest, split, and reformat everything for LlamaFactory training.

## Hardware Notes

- **H100 94GB**: bf16 training, batch size 1 + gradient accumulation 8
- **Long segments**: If OOM occurs, reduce `cutoff_len` or add max-duration filtering
- **FlashAttention-2**: Optional; install after env is stable if desired
