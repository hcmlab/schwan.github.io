import os
import json
import random
from pathlib import Path
from collections import defaultdict
from load_config import cfg, FINETUNE_DIR

# Directories (derived from config)
DATASET_FULL = FINETUNE_DIR / "llamafactory_dataset_full.json"
TRAIN_OUT    = FINETUNE_DIR / "dataset_train.json"
VAL_OUT      = FINETUNE_DIR / "dataset_val.json"
TEST_OUT     = FINETUNE_DIR / "dataset_test.json"
DATASET_INFO = FINETUNE_DIR / "dataset_info.json"

# Splitting Ratios (from config.yaml)
TRAIN_RATIO = cfg["train_ratio"]
VAL_RATIO   = cfg["val_ratio"]
SEED        = cfg["seed"]
# TEST_RATIO implicitly is 1.0 - (TRAIN_RATIO + VAL_RATIO)

def main():
    if not DATASET_FULL.exists():
        print(f"Error: {DATASET_FULL} not found. Run 02_create_dataset.py first.")
        return
        
    with open(DATASET_FULL, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} total samples.")
    
    # Group by session to prevent data leakage
    # Path example: X:/data/Schwan_T3_FineTune/chunks/ANOKPE01_HD_T3/ANOKPE01_HD_T3_Infant_Engagement_idx0001_Inon.mp4
    session_data = defaultdict(list)
    
    unparsable_count = 0
    for entry in dataset:
        video_paths = entry.get("videos", [])
        if not video_paths:
            unparsable_count += 1
            continue
            
        vid_path = video_paths[0]
        # the session name should be the parent folder name of the chunk
        path_obj = Path(vid_path)
        session_name = path_obj.parent.name
        
        session_data[session_name].append(entry)
        
    keys = list(session_data.keys())
    print(f"Found {len(keys)} unique sessions.")
    
    # Deterministic shuffle for reproducibility
    random.seed(SEED)
    random.shuffle(keys)
    
    train_split_idx = int(len(keys) * TRAIN_RATIO)
    val_split_idx = int(len(keys) * (TRAIN_RATIO + VAL_RATIO))
    
    train_sessions = keys[:train_split_idx]
    val_sessions = keys[train_split_idx:val_split_idx]
    test_sessions = keys[val_split_idx:]
    
    train_data = []
    for s in train_sessions:
        train_data.extend(session_data[s])
        
    val_data = []
    for s in val_sessions:
        val_data.extend(session_data[s])
        
    test_data = []
    for s in test_sessions:
        test_data.extend(session_data[s])
        
    print(f"Split results (Session-level):")
    print(f"Train: {len(train_sessions)} sessions, {len(train_data)} samples")
    print(f"Val:   {len(val_sessions)} sessions, {len(val_data)} samples")
    print(f"Test:  {len(test_sessions)} sessions, {len(test_data)} samples")
    
    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(VAL_OUT, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    with open(TEST_OUT, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
        
    # Generate dataset_info for LlamaFactory
    dataset_info = {
        "icep_omni_train": {
            "file_name": "dataset_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "videos": "videos",
                "audios": "audios"
            }
        },
        "icep_omni_val": {
            "file_name": "dataset_val.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "videos": "videos",
                "audios": "audios"
            }
        }
    }
    
    with open(DATASET_INFO, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
        
    print(f"Saved LlamaFactory dataset definitions to {DATASET_INFO}")

if __name__ == "__main__":
    main()
