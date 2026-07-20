"""
discover_sessions.py
====================
Step B: Discover completed sessions by scanning for .done markers.

Input env vars:
    DATA_ROOT  – root of the Schwan_T3_FineTune dataset
                 (default: /mnt/data/Schwan_T3_FineTune)
    OUT_DIR    – working output directory (default: gpu_server/data)

Output:
    {OUT_DIR}/sessions_done.jsonl
"""

import json
import os
from pathlib import Path


def get_env_paths():
    """Resolve DATA_ROOT and OUT_DIR from environment."""
    data_root = Path(os.environ.get("DATA_ROOT", "/mnt/data/Schwan_T3_FineTune"))
    out_dir = Path(os.environ.get("OUT_DIR", "gpu_server/data"))
    return data_root, out_dir


def discover_sessions(data_root: Path) -> list[dict]:
    """
    Walk DATA_ROOT for session directories.

    A directory is a session candidate if:
      - Its name matches *_T3  OR  it contains a chunks/ subdirectory.

    A session is "completed" when <session_dir>/chunks/.done exists
    (matching the marker written by 01_extract_chunks.py).
    """
    sessions = []

    if not data_root.is_dir():
        print(f"[WARN] DATA_ROOT does not exist: {data_root}")
        return sessions

    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue

        chunks_dir = child / "chunks"

        # Must be a *_T3 dir  OR  contain a chunks/ subdirectory
        is_t3 = child.name.endswith("_T3")
        has_chunks = chunks_dir.is_dir()

        if not (is_t3 or has_chunks):
            continue

        # Check for .done marker (inside chunks/)
        done_marker = chunks_dir / ".done"
        has_done = done_marker.exists()

        # Count media files in chunks/
        mp4_count = 0
        wav_count = 0
        if has_chunks:
            mp4_count = len(list(chunks_dir.glob("*.mp4")))
            wav_count = len(list(chunks_dir.glob("*.wav")))

        sessions.append({
            "session_id": child.name,
            "session_path": str(child),
            "chunks_path": str(chunks_dir) if has_chunks else None,
            "has_done": has_done,
            "mp4_count": mp4_count,
            "wav_count": wav_count,
        })

    return sessions


def main():
    data_root, out_dir = get_env_paths()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning DATA_ROOT: {data_root}")
    sessions = discover_sessions(data_root)

    # Filter to only completed sessions
    done_sessions = [s for s in sessions if s["has_done"]]

    out_path = out_dir / "sessions_done.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in done_sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Total session dirs scanned : {len(sessions)}")
    print(f"Completed (.done)          : {len(done_sessions)}")
    print(f"Output                     : {out_path}")

    if done_sessions:
        total_mp4 = sum(s["mp4_count"] for s in done_sessions)
        total_wav = sum(s["wav_count"] for s in done_sessions)
        print(f"Total MP4 chunks           : {total_mp4}")
        print(f"Total WAV chunks           : {total_wav}")


if __name__ == "__main__":
    main()
