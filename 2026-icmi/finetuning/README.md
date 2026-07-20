# ICEP-R Fine-Tuning Pipeline

## New CLI

The primary entrypoint is now the package CLI:

```bash
python -m swan_ft --help
```

The CLI is profile-driven and replaces the old hard-coded path flow. Key commands:

```bash
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_no_bg_infant_video
python -m swan_ft train cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b
python -m swan_ft predict cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b
python -m swan_ft report cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b
python -m swan_ft run pipeline --profile slurm_a100 --dataset-spec icep_no_bg_joint_omni --model qwen25_omni_7b --submit
```

Named profiles live in `configs/profiles/`, dataset variants in `configs/dataset_specs/`, and model definitions in `configs/model_specs/`.

Legacy scripts such as `prepare_dataset.py` and `evaluate.py` now forward into the CLI for compatibility.

## .env Loading

Runtime paths can be defined in a repo-root `.env` file. The file is loaded automatically when Python imports `swan_ft.config`, so the CLI and compatibility wrappers pick it up without extra flags. The loader checks, in order:

1. `SWAN_ENV_FILE`
2. `<repo>/.env`
3. `ft/.env` (legacy fallback)

Supported variables:

- `SWAN_SESSION_ROOT`
- `SWAN_DATA_ROOT`
- `SWAN_OUTPUT_ROOT`
- `SWAN_CACHE_ROOT`
- `SWAN_TEMP_ROOT`
- `SWAN_LOGS_ROOT`
- `SWAN_VARIANTS_ROOT`
- `SWAN_WORK_ROOT`
- `SWAN_ENV_FILE`

You can also scope variables to a specific named profile by inserting the profile id after `SWAN_`. For example, profile `vm_h100` supports:

- `SWAN_VM_H100_SESSION_ROOT`
- `SWAN_VM_H100_DATA_ROOT`
- `SWAN_VM_H100_OUTPUT_ROOT`
- `SWAN_VM_H100_CACHE_ROOT`
- `SWAN_VM_H100_TEMP_ROOT`
- `SWAN_VM_H100_LOGS_ROOT`
- `SWAN_VM_H100_VARIANTS_ROOT`
- `SWAN_VM_H100_PYTHON_BIN`

Example VM/H100 `.env`:

```bash
SWAN_VM_H100_SESSION_ROOT=/mnt/dataset-swan/data/Schwan_T3_Clean
SWAN_VM_H100_DATA_ROOT=/mnt/dataset-swan/data/Schwan_FT
SWAN_VM_H100_OUTPUT_ROOT=/mnt/dataset-swan/data/Schwan_FT/output
SWAN_VM_H100_CACHE_ROOT=/mnt/dataset-swan/data/Schwan_FT/cache
SWAN_VM_H100_TEMP_ROOT=/mnt/dataset-swan/data/Schwan_FT/tmp
SWAN_VM_H100_LOGS_ROOT=/mnt/dataset-swan/data/Schwan_FT/logs/vm100
SWAN_VM_H100_VARIANTS_ROOT=/mnt/dataset-swan/data/Schwan_FT/variants
```

`output`, `cache`, `tmp`, and `logs` directories are created automatically if they do not already exist.

If you omit some derived paths, the loader will fall back to:

- `output_root = <data_root>/output`
- `cache_root = <data_root>/cache`
- `temp_root = <data_root>/tmp`
- `logs_root = <data_root>/logs/<profile_id>`
- `variants_root = <data_root>/variants` (unless `SWAN_VARIANTS_ROOT` is set)

Local repo defaults (recommended when the dataset share is read-only):

```bash
SWAN_OUTPUT_ROOT=dataset_config/output
SWAN_CACHE_ROOT=dataset_config/cache
SWAN_TEMP_ROOT=dataset_config/tmp
SWAN_LOGS_ROOT=dataset_config/logs/local
SWAN_VARIANTS_ROOT=dataset_config/variants
```

## Role Modes

The pipeline now supports three task modes:

- `joint`: predict both infant and caregiver in one sample
- `infant`: predict only infant labels and descriptions
- `caregiver`: predict only caregiver labels and descriptions

Each mode writes its own dataset variant under `variants/<dataset_spec_id>/`, with separate `dataset.json`, `dataset_info.json`, `fold_*_{train,val}.json`, `labels.json`, and `dataset_manifest.json`.

You can choose a prebuilt dataset spec such as `icep_no_bg_infant_video`, or override the role mode dynamically:

```bash
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_no_bg_video --role-mode caregiver
```

## Label Filtering

Starting labels:

- Infant: `ineg`, `ipro`, `iwit`, `inon`, `ineu`, `ipos`, `islp`, `iusc`, `bg`
- Caregiver: `cneg`, `cwit`, `cint`, `chos`, `cnon`, `cneu`, `cpos`, `cpvc`, `ctch`, `bg`

Common no-background specs exclude `bg` by default. You can also exclude labels on demand:

```bash
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_with_bg_joint_video --exclude-label bg
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_with_bg_infant_video --exclude-labels bg,iusc
```

## Dataset Format

Canonical dataset root:

```text
<data_root>/
  folds.json
  dataset.json
  dataset_info.json
  clips/
    <SessionID>/
      segment_<start>_<end>.mp4
      bg_<start>_<end>.mp4
  videos/
    <SessionID>_merged.mp4
    <SessionID>_normalized.mp4
  variants/
    <dataset_spec_id>/
      dataset.json
      dataset_info.json
      fold_0_train.json
      fold_0_val.json
      ...
      audio_clips/
        <SessionID>/
          segment_<start>_<end>.wav
      labels.json
      dataset_manifest.json
```

LLaMA-Factory ShareGPT examples:

```json
{
  "conversations": [
    {"from": "human", "value": "<video>\n...prompt..."},
    {"from": "gpt", "value": "{\"infant_code\":\"ineu\",\"infant_description\":\"...\"}"}
  ],
  "videos": ["clips/SESSION_A/segment_10.00_20.00.mp4"]
}
```

```json
{
  "conversations": [
    {"from": "human", "value": "<audio>\n...prompt..."},
    {"from": "gpt", "value": "{\"caregiver_code\":\"cneu\",\"caregiver_description\":\"...\"}"}
  ],
  "audios": ["audio_clips/SESSION_A/segment_10.00_20.00.wav"]
}
```

`dataset_info.json` is variant-local and uses `formatting: "sharegpt"` with `messages`, plus `videos` and/or `audios` columns depending on modality.

## How Dataset Creation Works

Dataset creation happens in two layers:

### 1. Base dataset layer

The base dataset already contains the segmented training items derived from the annotated sessions:

- normalized session videos in `videos/`
- per-segment clips in `clips/`
- a base `dataset.json`
- a base `folds.json`

That base dataset is what the variant builder reads from.

### 2. Variant dataset layer

When you run `python -m swan_ft dataset build ...`, the current pipeline does not usually go back to the raw session videos and recreate all segmentation from scratch. Instead, it builds a new variant from the existing base dataset and fold assignments.

What changes during variant creation depends on modality:

- `video` variant:
  - rebuilds the prompt and target JSON
  - filters samples by role mode and excluded labels
  - re-materializes `fold_*_train.json` and `fold_*_val.json`
  - reuses existing clip paths from the base dataset
  - usually no new media files are created
- `audio` variant:
  - rebuilds the prompt and target JSON
  - filters samples by role mode and excluded labels
  - re-materializes fold JSON files
  - first reuses existing `audios` references if they are already present in the base dataset
  - otherwise looks for shared audio files under `<data_root>/audio_clips/`
  - only extracts `.wav` files from the source video clips with `ffmpeg` as a fallback
  - fallback-generated audio assets are written under `variants/<dataset_spec_id>/audio_clips/`
- `omni` variant:
  - rebuilds the prompt and target JSON
  - filters samples by role mode and excluded labels
  - re-materializes fold JSON files
  - reuses the existing video clips
  - first reuses existing shared audio if it already exists
  - only extracts variant-local `.wav` files if no reusable audio asset is found

So the short version is:

- `video` dataset creation is mostly a matter of creating different JSON files over the existing clips
- `audio` dataset creation usually creates new JSON files and reuses existing audio when available
- `omni` dataset creation usually creates new JSON files and reuses existing video/audio when available
- audio extraction only happens as a fallback when reusable audio assets are missing

If you need to regenerate the base clips from raw session videos and raw annotations, that is a separate earlier pipeline step. The variant builder assumes the base dataset already exists.

## Model-Specific Dataset Considerations

### Qwen Vision Model

Use this for models such as `qwen25_vl_7b`.

Recommended dataset specs:

- `icep_no_bg_infant_video`
- `icep_no_bg_caregiver_video`
- `icep_no_bg_joint_video`

What to consider:

- input uses only the video clip
- the dataset builder mostly rewrites JSON and fold files
- no additional audio extraction is required
- this is the lightest variant to create when the base dataset already exists

Example:

```bash
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_no_bg_infant_video
python -m swan_ft train cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b
python -m swan_ft predict cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b
python -m swan_ft report cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b
```

### Qwen Multimodal Omni Model

Use this for models such as `qwen25_omni_7b`.

Recommended dataset specs:

- `icep_no_bg_infant_omni`
- `icep_no_bg_caregiver_omni`
- `icep_no_bg_joint_omni`

What to consider:

- input uses both video and audio from the same segment window
- the builder reuses existing video clips
- the current omni specs are configured to reuse the same clip `.mp4` as both the `videos` and `audios` source
- this assumes the clip `.mp4` already contains an embedded audio stream
- excluding labels is therefore mostly just creating new JSON files for omni as well

Quick validation command:

```bash
python check_mp4_audio.py /mnt/dataset-swan/data/Schwan_FT/clips --limit 20
```

Example:

```bash
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_no_bg_joint_omni
python -m swan_ft train cv --profile vm_h100 --dataset-spec icep_no_bg_joint_omni --model qwen25_omni_7b
python -m swan_ft predict cv --profile vm_h100 --dataset-spec icep_no_bg_joint_omni --model qwen25_omni_7b
python -m swan_ft report cv --profile vm_h100 --dataset-spec icep_no_bg_joint_omni --model qwen25_omni_7b
```

### Qwen Audio Model

Use this for models such as `qwen2_audio_7b`.

Recommended dataset specs:

- `icep_no_bg_infant_audio`
- `icep_no_bg_caregiver_audio`

What to consider:

- input uses only audio derived from the segment clip
- the builder first reuses existing shared audio files if they are already available
- it only extracts `.wav` files from the segment videos with `ffmpeg` as a fallback
- video clips are used as the source for extraction, but the final dataset samples point to `audios`
- this is useful when you want to test how much of the task is recoverable from audio alone

Example:

```bash
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_no_bg_caregiver_audio
python -m swan_ft train cv --profile vm_h100 --dataset-spec icep_no_bg_caregiver_audio --model qwen2_audio_7b
python -m swan_ft predict cv --profile vm_h100 --dataset-spec icep_no_bg_caregiver_audio --model qwen2_audio_7b
python -m swan_ft report cv --profile vm_h100 --dataset-spec icep_no_bg_caregiver_audio --model qwen2_audio_7b
```

## Split Policy

Current behavior:

- folds are created at the session level
- each fold has `train` and `val` sessions
- training uses `fold_k_train`
- early stopping and best-checkpoint selection use `fold_k_val`
- final fold reporting also uses `fold_k_val`

This means the current pipeline is a session-level cross-validation pipeline, but it does not yet implement a separate untouched final test split.

Best-practice target workflow:

- reserve a session-level `test` split that is never used for early stopping or model selection
- run k-fold CV only on the remaining `train_dev` sessions
- choose the baseline recipe from CV results
- train one final model on all `train_dev` sessions
- evaluate once on the untouched `test` sessions

Status in this codebase:

- current CV workflow: implemented
- proper `train_dev` plus untouched `test` workflow: not yet implemented end-to-end

What to run on SLURM today:

```bash
bash slurm/run_qwen_audio_full.sh   --profile slurm_a40   --dataset-spec icep_no_bg_audio   --model qwen2_audio_7b   --exclude-label bg   --prep   --eval
```

This will give you cross-validation results with `bg` excluded, but those reported scores are still based on each fold's validation split rather than on a final unseen test set.

## Early Stopping

All rendered training configs now apply a common validation-driven policy unless overridden:

- `do_eval: true`
- matched `eval_strategy` and `save_strategy`
- `load_best_model_at_end: true`
- `metric_for_best_model: eval_loss`
- `greater_is_better: false`
- `early_stopping_steps`
- `save_total_limit`

You can disable best-model loading for a run if needed:

```bash
python -m swan_ft train cv --profile vm_h100 --dataset-spec icep_no_bg_caregiver_audio --model qwen2_audio_7b --disable-best-model-load
```

## Weights And Biases

LLaMA-Factory logging can be sent to Weights & Biases for both training and prediction runs.

Enable it with environment variables:

```bash
export WANDB_API_KEY=...
export SWAN_WANDB_ENABLED=1
export SWAN_WANDB_PROJECT=swan-ft
export SWAN_WANDB_ENTITY=your_team
```

When enabled, rendered configs include:

- `report_to: wandb`
- a fold-specific `run_name`
- default `logging_steps: 10`

Run grouping defaults to `<model>_<variant>`, but you can override it with:

```bash
export SWAN_WANDB_GROUP=phase1_qwen25_omni
export SWAN_WANDB_TAGS=vm_h100,omni,no_bg
```

## Running On vm_h100 Linux

This is the recommended workflow for a Linux VM with an H100 GPU and dataset mounts under `/mnt/dataset-swan/data/`.

### 1. Prepare the environment

```bash
cd /path/to/swan/ft
conda activate swan
pip install -r requirements.txt
ffmpeg -version
python -m swan_ft --help
```

### 2. Create the repo-root `.env`

Create [`/.env`](D:/GitHub/swan/.env) at the repo root with the VM-specific paths:

```bash
SWAN_VM_H100_SESSION_ROOT=/mnt/dataset-swan/data/Schwan_T3_Clean
SWAN_VM_H100_DATA_ROOT=/mnt/dataset-swan/data/Schwan_FT
SWAN_VM_H100_OUTPUT_ROOT=/mnt/dataset-swan/data/Schwan_FT/output
SWAN_VM_H100_CACHE_ROOT=/mnt/dataset-swan/data/Schwan_FT/cache
SWAN_VM_H100_TEMP_ROOT=/mnt/dataset-swan/data/Schwan_FT/tmp
SWAN_VM_H100_LOGS_ROOT=/mnt/dataset-swan/data/Schwan_FT/logs/vm100
```

When the CLI starts, it will load these values automatically. The `output`, `cache`, `tmp`, and `logs` directories are created if they do not already exist.

### 3. Build the dataset variant you want to train

Examples:

```bash
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_no_bg_infant_video
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_no_bg_caregiver_audio
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_no_bg_joint_omni
```

If you want to override the role mode or labels dynamically:

```bash
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_no_bg_video --role-mode caregiver
python -m swan_ft dataset build --profile vm_h100 --dataset-spec icep_with_bg_joint_video --exclude-label bg
```

### 4. Inspect the generated dataset

```bash
python -m swan_ft dataset inspect --profile vm_h100 --dataset-spec icep_no_bg_infant_video
```

This reports sample counts, label counts, role mode, and modality information for the selected dataset variant.

### 5. Run fine-tuning

Examples for the Phase 1 models:

```bash
python -m swan_ft train cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b
python -m swan_ft train cv --profile vm_h100 --dataset-spec icep_no_bg_caregiver_audio --model qwen2_audio_7b
python -m swan_ft train cv --profile vm_h100 --dataset-spec icep_no_bg_joint_omni --model qwen25_omni_7b
```

Single-fold debugging:

```bash
python -m swan_ft train fold --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b --folds 0
```

Helpful overrides:

```bash
python -m swan_ft train cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b --early-stopping-steps 8
python -m swan_ft train cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b --disable-best-model-load
```

### 6. Run prediction

```bash
python -m swan_ft predict cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b
python -m swan_ft predict cv --profile vm_h100 --dataset-spec icep_no_bg_caregiver_audio --model qwen2_audio_7b
python -m swan_ft predict cv --profile vm_h100 --dataset-spec icep_no_bg_joint_omni --model qwen25_omni_7b
```

### 7. Generate evaluation reports

```bash
python -m swan_ft report cv --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b
python -m swan_ft report cv --profile vm_h100 --dataset-spec icep_no_bg_caregiver_audio --model qwen2_audio_7b
python -m swan_ft report cv --profile vm_h100 --dataset-spec icep_no_bg_joint_omni --model qwen25_omni_7b
```

### 8. Run the full local pipeline in one command

```bash
python -m swan_ft run pipeline --profile vm_h100 --dataset-spec icep_no_bg_infant_video --model qwen25_vl_7b
```

This uses the local launcher for `vm_h100`, so it builds or reuses the dataset variant, runs the training flow, performs prediction, and then writes fold reports plus the cross-validation summary.

### 9. Where outputs go

With the VM `.env` values above, the main outputs land under:

```text
/mnt/dataset-swan/data/Schwan_FT/
  variants/<dataset_spec_id>/
  output/<model_id>/<variant_id>/
  logs/vm100/
  cache/
  tmp/
```

The rendered LLaMA-Factory YAMLs are stored under the corresponding model output directory so each run remains reproducible.

Fine-tune VLMs (Qwen3-VL-8B, Qwen3.5-9B) on the 87-session Schwan dataset to improve ICEP-R behavioral coding accuracy over zero-shot inference.

## What it does

1. Normalizes all session videos to a uniform 2-panel side-by-side format at 8fps
2. Aligns infant + caregiver annotation tracks into unified time segments
3. Extracts video clips per segment and builds ShareGPT-format training data
4. Trains LoRA adapters via LLaMA-Factory with 5-fold stratified cross-validation
5. Evaluates per-fold and aggregate P/R/F1 using the existing `ClassificationStats`

## Data layout

```
/mnt/swan/data/
  Schwan_T3_Clean/           # Source: 87 sessions with videos + annotations (READ ONLY)
  Schwan_FT/                 # Output: all generated data goes here
    videos/                  #   Normalized 8fps videos (MUC merged, HD re-encoded)
    clips/                   #   Extracted per-segment video clips
      {SessionID}/           #     segment_72.60_101.60.mp4, bg_10.00_20.00.mp4, ...
    video_map.json           #   Session ID -> normalized video path
    dataset.json             #   Full ShareGPT dataset (all sessions)
    dataset_info.json        #   LLaMA-Factory dataset registry
    fold_{0-4}_train.json    #   Per-fold training sets
    fold_{0-4}_val.json      #   Per-fold validation sets
    output/                  #   Training outputs (LoRA adapters, eval reports)
      qwen3_vl_8b/
        fold_{0-4}/          #     LoRA adapter checkpoints per fold
        fold_{0-4}_eval_report.json
        qwen3_vl_8b_cv_summary.json
      qwen35_9b/
        ...
```

## Pipeline overview

```
Step 1: create_folds.py       Generate 5-fold CV splits (stratified by location + duration)
Step 2: merge_cameras.py      Normalize videos: MUC Kam2+4 merge, HD re-encode, all at 8fps
Step 3: prepare_dataset.py    Extract clips + build LLaMA-Factory dataset JSONs per fold
Step 4: LLaMA-Factory train   LoRA fine-tuning (local dev test or SLURM cluster)
Step 5: evaluate.py           Inference on val sets, compute metrics, aggregate CV summary
```

## Setup

```bash
conda activate swan  # Python 3.11+
pip install -r scripts/ft/requirements.txt

# ffmpeg required on PATH for video processing
ffmpeg -version
```

## Step-by-step usage

### 1. Generate CV folds

```bash
python scripts/ft/create_folds.py
```

Output: `scripts/ft/configs/folds.json` — 87 sessions split into 5 folds (~17-18 each, balanced MUC/HD).

### 2. Normalize videos

```bash
# Dry run first
python scripts/ft/merge_cameras.py --dry-run

# Full run (47 MUC merges + 40 HD re-encodes -> /mnt/swan/data/Schwan_FT/videos/)
python scripts/ft/merge_cameras.py --workers 4
```

- **MUC**: Merges Kamera 2 (infant face) + Kamera 4 (caregiver face) into 960x1080 side-by-side
- **HD**: Re-encodes existing 2-panel video at target fps
- Output: `Schwan_FT/videos/{SessionID}_merged.mp4` / `{SessionID}_normalized.mp4`
- Writes `Schwan_FT/video_map.json`

Options:
- `--output-dir DIR` — video output directory (default: `/mnt/swan/data/Schwan_FT/videos`)
- `--fps 8` — output framerate (default 8; Qwen VL enforces min 4 frames per clip)
- `--crf 23` — quality (lower = better, default 23)

### 3. Prepare dataset

```bash
# Dry run — shows segment counts and code distribution
python scripts/ft/prepare_dataset.py --dry-run

# Full run — extracts clips and builds dataset
python scripts/ft/prepare_dataset.py
```

- Aligns infant + caregiver tracks by time overlap into unified segments
- Generates background (`bg`) samples from within-annotation gaps and pre/post-annotation regions (60s safety pad from video edges to avoid identity leakage)
- Extracts clips to `Schwan_FT/clips/{SessionID}/`
- Writes `Schwan_FT/dataset.json`, `fold_*_train.json`, `fold_*_val.json`, `dataset_info.json`

Options:
- `--clips-dir DIR` — clip output directory (default: `/mnt/swan/data/Schwan_FT/clips`)
- `--output PATH` — dataset JSON path (default: `/mnt/swan/data/Schwan_FT/dataset.json`)
- `--bg-pad 60` — safety padding in seconds for background samples (default 60)
- `--max-sessions 5` — limit sessions for quick pipeline testing

### 4. Train

#### Dev test (single GPU, 24GB)

The default config uses 4-bit quantization + batch_size=1 to fit on 24GB GPUs.

```bash
# Test with a small subset first
python scripts/ft/prepare_dataset.py --max-sessions 5

# Run one fold
sed 's/fold_X/fold_0/g' scripts/ft/configs/qwen3_vl_8b_lora.yaml > /tmp/test_config.yaml
llamafactory-cli train /tmp/test_config.yaml
```

#### SLURM cluster (3x A40 48GB)

For A40s, remove quantization and increase batch size for faster training:
```bash
# Remove quantization lines and set batch_size=2, grad_accum=8
sed 's/fold_X/fold_0/g' scripts/ft/configs/qwen3_vl_8b_lora.yaml \
  | grep -v 'quantization_' \
  | sed 's/per_device_train_batch_size: 1/per_device_train_batch_size: 2/' \
  | sed 's/gradient_accumulation_steps: 16/gradient_accumulation_steps: 8/' \
  > /tmp/config.yaml
```

```bash
# Submit all 5 folds for one model (array job, 1 GPU per fold)
sbatch --array=0-4 scripts/ft/slurm/train_fold.sh qwen3_vl_8b_lora

# Or both models
bash scripts/ft/slurm/run_cv.sh

# Monitor
squeue -u $USER
```

SLURM runs 3 folds concurrently on 3 GPUs, remaining 2 queue automatically.

### 5. Evaluate

```bash
# Uses defaults: reads val data from Schwan_FT/, adapters from Schwan_FT/output/qwen3_vl_8b/
python scripts/ft/evaluate.py --model qwen3_vl_8b

# Evaluate specific folds only
python scripts/ft/evaluate.py --model qwen3_vl_8b --folds 0 1
```

Output in `Schwan_FT/output/{model}/`:
- `fold_{i}_eval_report.json` — per-fold P/R/F1 + predictions (compatible with annotation viewer)
- `{model}_cv_summary.json` — aggregate metrics across all folds

## Code files

```
scripts/ft/
  create_folds.py            # Step 1: CV split generation
  merge_cameras.py           # Step 2: Video normalization
  prepare_dataset.py         # Step 3: Clip extraction + dataset building
  evaluate.py                # Step 5: Fold evaluation + CV aggregation
  requirements.txt           # Python dependencies
  configs/
    folds.json               # Generated fold assignments
    qwen3_vl_8b_lora.yaml   # LoRA training config for Qwen3-VL-8B
    qwen35_9b_lora.yaml     # LoRA training config for Qwen3.5-9B
    dataset_info.json        # LLaMA-Factory dataset registry (template)
  slurm/
    train_fold.sh            # SLURM array job script
    run_cv.sh                # Convenience launcher for all folds x models
```

## Key design decisions

- **8fps source videos**: Qwen VL's processor defaults to 2fps sampling but enforces `FPS_MIN_FRAMES=4`. At 8fps, even our shortest segments (0.5s) produce 4 frames. Storage is ~100 GB vs ~365 GB at 25fps.
- **2-panel merge (Kam2+4)**: Kam2 shows infant face, Kam4 shows caregiver face — together they match HD's existing side-by-side layout.
- **Background class from pre/post-annotation**: Within-annotation gaps alone yield too few samples (~16). Pre/post-annotation regions with 60s safety padding from video edges yield ~730 samples, avoiding clapperboard/setup frames that could leak participant identity across folds.
- **Segment alignment**: Infant and caregiver tracks are aligned by time overlap. Each training sample includes both codes, matching the prompt format used in zero-shot inference.
- **Output isolation**: All generated data goes to `Schwan_FT/`, keeping `Schwan_T3_Clean/` read-only.

## Excluded sessions

2 of 87 MUC sessions cannot produce a Kam2+4 merge and are excluded from training (85/87 = 98%):

| Session | Issue |
|---------|-------|
| `MEDEAN01_MUC_T3` | Non-standard recording — no Kamera files, only `Job (_Clip ...)` exports |
| `REJACL01_MUC_T3` | Only Kamera1 present, missing Kamera 2/3/4 |
