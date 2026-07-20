import os
import json
from pathlib import Path
from load_config import FINETUNE_DIR

# Directories (derived from config)
DATASET_TRAIN = FINETUNE_DIR / "dataset_train.json"

def verify_format(dataset_path: Path):
    print(f"--- Verifying LlamaFactory Omni Format for {dataset_path.name} ---")
    if not dataset_path.exists():
        print(f"Error: {dataset_path} does not exist.")
        return False
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) == 0:
        return False
        
    # Check first sample
    sample = dataset[0]
    
    errors = []
    if "messages" not in sample:
        errors.append("Missing 'messages' key")
    else:
        messages = sample["messages"]
        if len(messages) < 3:
            errors.append("Messages list should have at least system, user, and assistant")
        else:
            if messages[0].get("role") != "system":
                errors.append("First message should be from system")
            
            if messages[1].get("role") != "user":
                errors.append("Second message should be from user")
                
            user_content = messages[1].get("content", "")
            if "<video>" not in user_content:
                errors.append("User message missing <video> tag")
            if "<audio>" not in user_content:
                errors.append("User message missing <audio> tag")
                
            if messages[2].get("role") != "assistant":
                errors.append("Third message should be from assistant")
                
    if "videos" not in sample:
        errors.append("Missing 'videos' key")
    else:
        videos = sample["videos"]
        if not isinstance(videos, list) or len(videos) == 0:
            errors.append("'videos' should be a non-empty list of paths")
        else:
            vid_path = Path(videos[0])
            if not vid_path.exists():
                print(f"Warning: Video path {vid_path} does not physically exist during validation (might be okay if moving to another server, but double check paths).")
                
    if "audios" not in sample:
        errors.append("Missing 'audios' key")
    else:
        audios = sample["audios"]
        if not isinstance(audios, list) or len(audios) == 0:
            errors.append("'audios' should be a non-empty list of paths")
        else:
            aud_path = Path(audios[0])
            if not aud_path.exists():
                print(f"Warning: Audio path {aud_path} does not physically exist during validation.")
                
    if errors:
        print("Format Validation FAILED with errors:")
        for e in errors:
            print(f" - {e}")
        return False
        
    print("Format Validation PASSED!")
    print("\nSample Output:")
    print(json.dumps(sample, indent=2))
    return True

def main():
    verify_format(DATASET_TRAIN)

if __name__ == "__main__":
    main()
