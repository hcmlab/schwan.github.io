"""
verify_videos.py
================
Step 4.5: Verify all videos in the generated LlamaFactory datasets.

LLaMA-Factory uses `av` (PyAV) to decode videos during tokenization. If a
video chunk is corrupted (e.g., "moov atom not found"), PyAV will throw an
`av.error.InvalidDataError` and crash the entire training/evaluation run.

This script parses the generated train and test JSON files, attempts to
open every video using `av.open(..., "r")`, and if it fails, removes that
sample from the dataset and appends it to `corrupt_data.json`.
"""

import av
import json
import os
from pathlib import Path
from tqdm import tqdm


def get_env_paths():
    out_dir = Path(os.environ.get("OUT_DIR", "gpu_server/data"))
    return out_dir


def verify_and_clean_dataset(dataset_path: Path, corrupt_path: Path):
    if not dataset_path.exists():
        print(f"[WARN] {dataset_path} does not exist. Skipping.")
        return

    print(f"\nVerifying videos in {dataset_path.name}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Load existing corrupt data to append to
    corrupt_data = []
    if corrupt_path.exists():
        with open(corrupt_path, "r", encoding="utf-8") as f:
            corrupt_data = json.load(f)

    clean_dataset = []
    new_corrupt_count = 0

    for i, item in enumerate(tqdm(dataset)):
        videos = item.get("videos", [])
        is_corrupt = False
        
        for video_path in videos:
            if not os.path.exists(video_path):
                is_corrupt = True
                item["error_reason"] = "File not found"
                break
            
            try:
                # This is exactly what LLaMA-Factory mm_plugin.py does
                with av.open(video_path, "r") as container:
                    # just verify it opens successfully
                    pass
            except Exception as e:
                is_corrupt = True
                item["error_reason"] = f"PyAV Error: {str(e)}"
                break

        if is_corrupt:
            corrupt_data.append(item)
            new_corrupt_count += 1
        else:
            clean_dataset.append(item)

    if new_corrupt_count > 0:
        print(f"  Found {new_corrupt_count} corrupted videos! Removing them...")
        
        # Save cleaned dataset
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(clean_dataset, f, indent=2, ensure_ascii=False)
            
        # Update corrupt tracking file
        with open(corrupt_path, "w", encoding="utf-8") as f:
            json.dump(corrupt_data, f, indent=2, ensure_ascii=False)
    else:
        print("  All videos validated successfully. Zero corrupted chunks.")


def main():
    out_dir = get_env_paths()
    llama_dir = out_dir / "llamafactory"
    
    if not llama_dir.exists():
        print(f"Error: {llama_dir} not found. Run 04_build_dataset.sh first.")
        return

    train_file = llama_dir / "schwan_icep_sft_train.json"
    test_file = llama_dir / "schwan_icep_sft_test.json"
    corrupt_file = llama_dir / "corrupt_data.json"

    verify_and_clean_dataset(train_file, corrupt_file)
    verify_and_clean_dataset(test_file, corrupt_file)

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
