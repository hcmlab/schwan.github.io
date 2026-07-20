"""
quarantine_corrupt_videos.py
============================
Physically move corrupted video files to a quarantine folder to prevent
LlamaFactory and other tools from attempting to open them.

Usage:
    python src/quarantine_corrupt_videos.py [--dry-run]
"""

import os
import shutil
import av
from pathlib import Path
from tqdm import tqdm
import argparse

def get_env_paths():
    data_root = Path(os.environ.get("DATA_ROOT", "/mnt/dataset-swan/data/Schwan_T3_FineTune"))
    return data_root

def verify_video(file_path: Path) -> bool:
    """Check if video is valid using PyAV."""
    try:
        with av.open(str(file_path), "r") as container:
            pass
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't move files, just log")
    args = parser.parse_args()

    data_root = get_env_paths()
    if not data_root.exists():
        print(f"Error: DATA_ROOT {data_root} not found.")
        return

    # Use a local path for quarantine to avoid permission errors on the Mount
    out_dir = Path(os.environ.get("OUT_DIR", "gpu_server/data"))
    quarantine_dir = out_dir / "_quarantine_corrupt"
    
    if not args.dry_run:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning DATA_ROOT: {data_root}")
    
    # Find all mp4 files
    mp4_files = list(data_root.glob("**/chunks/*.mp4"))
    print(f"Found {len(mp4_files)} videos. Verifying integrity...")

    corrupt_paths = []
    for mp4 in tqdm(mp4_files):
        if not verify_video(mp4):
            corrupt_paths.append(str(mp4))
            print(f"\n[CORRUPT] {mp4}")

    # Write the list to a file
    out_list_path = out_dir / "corrupt_videos_list.txt"
    with open(out_list_path, "w", encoding="utf-8") as f:
        for path in corrupt_paths:
            f.write(path + "\n")

    print(f"\n{'='*60}")
    print(f"Detection complete. Found {len(corrupt_paths)} corrupted files.")
    print(f"The full list has been saved to: {out_list_path}")
    print(f"Please use this list to manually address or delete the files if possible.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
