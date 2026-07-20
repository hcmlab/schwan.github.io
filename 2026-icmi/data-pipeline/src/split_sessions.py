"""
split_sessions.py
=================
Step D: Split sessions into train / val / test using a deterministic
hash-based assignment at the session level (no data leakage).

Split rule (hash(session_id) % 100):
    train :  0–79   (80 %)
    val   : 80–89   (10 %)
    test  : 90–99   (10 %)

Input:
    {OUT_DIR}/sessions_done.jsonl  (from discover_sessions.py)

Output:
    {OUT_DIR}/splits/train_sessions.json
    {OUT_DIR}/splits/val_sessions.json
    {OUT_DIR}/splits/test_sessions.json
"""

import hashlib
import json
import os
from pathlib import Path


def get_env_paths():
    out_dir = Path(os.environ.get("OUT_DIR", "gpu_server/data"))
    return out_dir


def session_split_bucket(session_id: str) -> str:
    """
    Deterministic split assignment via MD5 hash.

    Returns 'train', 'val', or 'test'.
    """
    digest = hashlib.md5(session_id.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 100

    if bucket < 80:
        return "train"
    elif bucket < 90:
        return "val"
    else:
        return "test"


def main():
    out_dir = get_env_paths()
    sessions_file = out_dir / "sessions_done.jsonl"

    if not sessions_file.exists():
        print(f"Error: {sessions_file} not found — run discover_sessions.py first.")
        return

    # Load session IDs
    sessions = []
    with open(sessions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))

    # Assign splits
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for s in sessions:
        sid = s["session_id"]
        bucket = session_split_bucket(sid)
        splits[bucket].append(sid)

    # Sort for reproducibility
    for k in splits:
        splits[k].sort()

    # Write split files
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, session_ids in splits.items():
        out_path = splits_dir / f"{split_name}_sessions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(session_ids, f, indent=2)

    print(f"Session-disjoint split results:")
    print(f"  train : {len(splits['train'])} sessions")
    print(f"  val   : {len(splits['val'])} sessions")
    print(f"  test  : {len(splits['test'])} sessions")
    print(f"  total : {sum(len(v) for v in splits.values())} sessions")
    print(f"\nOutput directory: {splits_dir}")


if __name__ == "__main__":
    main()
