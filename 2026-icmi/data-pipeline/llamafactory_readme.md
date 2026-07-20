# Qwen-Omni Finetuning with LlamaFactory for ICEP Annotations

This codebase is designed to automatically extract, format, split, and finetune the **Qwen2.5-Omni** model to predict detailed ICEP (Infant Coding) annotations based on multimodal Caregiver-Infant interaction videos.

## Prerequisites

- A server with an NVIDIA H100 GPU (e.g., AMD EPYC + H100 NVL 94GB).
- Conda environment with `LlamaFactory` installed.
- `ffmpeg` installed and available in the system path.

## Workflow

### 1. Data Preparation

Follow these scripts in order to generate the LlamaFactory datasets:

1. **`python data_prep/01_extract_chunks.py`**
   - Parses `X:\data\Schwan_T3_Clean` for `.mp4` and `.wav` files.
   - **Multi-camera Augmentation**: Extracts context-buffered video chunks from *all* available camera angles (Kamera1-4 + Splitscreen), multiplying the dataset size up to 5× per multi-camera session.
   - Uses HW-accelerated ffmpeg (`h264_nvenc`) to extract chunks into `X:\data\Schwan_T3_FineTune\{session}\chunks`.
   - Extracts a single shared audio track (`.wav`) per annotation.
   - Runs concurrently on 5 sessions at a time with `.done` markers for resume capability.

2. **`python data_prep/02_create_dataset.py`**
   - Builds the conversational ShareGPT multimodal format required by LlamaFactory.
   - Maps each video/audio chunk pair to a multimodal entry using `<video><audio>` tags.
   - Uses detailed visual and auditory ICEP descriptions as the ground-truth reasoning + label in the `assistant` answer.

3. **`python data_prep/03_split_data.py`**
   - Essential step to prevent **Data Leakage**.
   - Groups chunks by the `SessionName` and randomly assigns the entire session to either Train (80%), Val (10%), or Test (10%).
   - Generates `dataset_info.json`. **You must copy the contents of this `dataset_info.json` into LlamaFactory's global `dataset_info.json` directory!** (It includes the required `videos` and `audios` columns).

4. **`python data_prep/04_verify_omni_format.py`**
   - A quick dry-run sanity check on the generated training JSON file to ensure the `<video>` and `<audio>` markers and message formats are valid before spinning up the GPU.

### 2. Training

1. Ensure your LlamaFactory `dataset_info.json` has the custom lines created from step 3.
2. Review the hyperparameters in `training/llamafactory_config.yaml`. The current configuration uses `per_device_train_batch_size: 4` and `gradient_accumulation_steps: 4` suited for the H100 GPU.
3. Run the bash launch script:

   ```bash
   bash training/train_qwen_omni.sh
   ```

### Hardware Considerations for H100

- The H100 NVL 94GB provides massive VRAM and is optimized perfectly for BF16/FP8 training.
- We enable `bf16: true` in the YAML configuration to speed up Training while preserving numerical stability.
- Using LoRA (`finetuning_type: lora`) on `all` target modules enables high-quality alignment without catastrophic forgetting.
