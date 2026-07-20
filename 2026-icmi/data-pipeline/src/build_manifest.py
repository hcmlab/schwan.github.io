"""
build_manifest.py
=================
Step C: Build a chunk-level manifest from completed sessions.

For each done session, reads the chunks/ directory and pairs MP4/WAV files
by their shared stem.  Parses filename fields and probes duration via ffprobe.

Filename convention (from 01_extract_chunks.py):
    {session_id}_{track}_idx{NNNN}_{short_code}[_{suffix}].{ext}

    HD sessions (single camera):
        MP4: NAAPCA01_HD_T3_..._Cneu_14032024.mp4   (date suffix)
        WAV: NAAPCA01_HD_T3_..._Cneu.wav             (no suffix)

    MUC sessions (multi-camera):
        MP4: Jujulu01_MUC_T3_..._Cneu_kamera1.mp4    (camera suffix)
        MP4: Jujulu01_MUC_T3_..._Cneu_splitscreen.mp4 (combined grid)
        WAV: Jujulu01_MUC_T3_..._Cneu.wav             (no suffix)

Each manifest row includes a `camera_view` field:
    - splitscreen_hd : single-camera HD session (date suffix)
    - splitscreen    : combined grid from multi-camera MUC session
    - kamera1..4     : individual camera angle
    - unknown        : could not determine

Input env vars:
    DATA_ROOT, OUT_DIR (same as discover_sessions.py)

Output:
    {OUT_DIR}/chunk_manifest.jsonl
"""

import json
import os
import re
import subprocess
from pathlib import Path


def get_env_paths():
    data_root = Path(os.environ.get("DATA_ROOT", "/mnt/dataset-swan/data/Schwan_T3_FineTune"))
    out_dir = Path(os.environ.get("OUT_DIR", "gpu_server/data"))
    return data_root, out_dir


# ── Filename parsing ────────────────────────────────────────────

# Pattern matches:  {session}_{Track_Name}_idx{NNNN}_{code}
#   group(1) = session_id  (everything before _Infant_ or _Caregiver_)
#   group(2) = track       (Infant_Engagement or Caregiver_Engagement)
#   group(3) = idx         (integer)
#   group(4) = short_code  (e.g. Cneu, no_annotation)
_STEM_RE = re.compile(
    r"^(.+?)_((?:Infant|Caregiver)_Engagement)_idx(\d+)_(.+)$"
)


def parse_chunk_stem(stem: str) -> dict | None:
    """
    Parse a chunk filename stem (without camera/date suffix or extension).

    Returns dict with session_id, track, idx, short_code  or  None.
    """
    m = _STEM_RE.match(stem)
    if not m:
        return None
    
    short_code = m.group(4)
    # Strip trailing date suffix (_DDMMYYYY) from short_code.
    # This handles filenames like:
    #   ..._no_annotation_15092023.mp4  → no_annotation
    #   ..._Cneu_14032024.mp4           → Cneu
    # WAV files never have a date suffix so this is a no-op for them.
    short_code = re.sub(r"_\d{6,8}$", "", short_code)

    return {
        "session_id": m.group(1),
        "track": m.group(2),
        "idx": int(m.group(3)),
        "short_code": short_code,
    }



def wav_stem_to_base(wav_path: Path) -> str | None:
    """
    WAV files have no camera suffix:  {base}.wav
    Return the base stem for grouping.
    """
    stem = wav_path.stem
    if parse_chunk_stem(stem) is not None:
        return stem
    return None


def mp4_stem_to_base(mp4_path: Path) -> str | None:
    """
    MP4 files have a camera/date suffix:  {base}_{suffix}.mp4
    Strip the last _component to get the base stem for grouping.
    """
    stem = mp4_path.stem
    
    # ALWAYS try stripping the camera/date suffix first, 
    # as 01_extract_chunks.py appends one (like _kamera1, _splitscreen, or _15092023).
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        base = parts[0]
        if parse_chunk_stem(base) is not None:
            return base
            
    # Fallback to full stem (though usually MP4s have a suffix)
    if parse_chunk_stem(stem) is not None:
        return stem
    return None


def extract_camera_view(mp4_path: Path, base_stem: str) -> str:
    """
    Determine the camera view type from the MP4 filename suffix.

    Returns one of:
        'splitscreen_hd'  — single-camera HD session (suffix is a date)
        'splitscreen'     — combined grid view from multi-camera session
        'kamera1'...'kamera4' — individual camera angle
        'unknown'         — could not determine
    """
    stem = mp4_path.stem
    if stem == base_stem:
        # No suffix at all — unusual, treat as unknown
        return "unknown"

    suffix = stem[len(base_stem) + 1:]  # strip base + underscore
    suffix_lower = suffix.lower()

    if suffix_lower == "splitscreen":
        return "splitscreen"
    elif suffix_lower.startswith("kamera"):
        return suffix_lower  # kamera1, kamera2, etc.
    elif re.match(r"^\d{6,8}$", suffix):
        # Date suffix (e.g., 14032024) → single-camera HD session
        return "splitscreen_hd"
    else:
        return "unknown"


import av
import subprocess

# ── Duration and Integrity probing ────────────────────────────────────────────

def probe_duration(file_path: Path) -> float | None:
    """Probe media duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return round(float(result.stdout.strip()), 3)
    except Exception:
        pass
    return None

def verify_video(file_path: Path) -> bool:
    """Strictly verify via PyAV to prevent LLaMA-Factory crash later."""
    try:
        with av.open(str(file_path), "r") as container:
            pass
        return True
    except Exception:
        return False


# ── Main ────────────────────────────────────────────────────────

def load_done_sessions(out_dir: Path) -> list[dict]:
    """Load sessions_done.jsonl written by discover_sessions.py."""
    jsonl = out_dir / "sessions_done.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(
            f"{jsonl} not found — run discover_sessions.py first."
        )
    sessions = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions


def build_manifest_for_session(session: dict) -> list[dict]:
    """Build chunk-level manifest rows for one session."""
    chunks_path = session.get("chunks_path")
    if not chunks_path:
        return []

    chunks_dir = Path(chunks_path)
    if not chunks_dir.is_dir():
        return []

    # Group WAVs by base stem
    wav_by_base: dict[str, Path] = {}
    for wav in chunks_dir.glob("*.wav"):
        base = wav_stem_to_base(wav)
        if base:
            wav_by_base[base] = wav

    # Group MP4s by base stem
    mp4_by_base: dict[str, list[Path]] = {}
    for mp4 in chunks_dir.glob("*.mp4"):
        base = mp4_stem_to_base(mp4)
        if base:
            mp4_by_base.setdefault(base, []).append(mp4)

    # All known base stems
    all_bases = sorted(set(wav_by_base.keys()) | set(mp4_by_base.keys()))

    rows = []
    for base in all_bases:
        parsed = parse_chunk_stem(base)
        if parsed is None:
            continue

        wav_path = wav_by_base.get(base)
        mp4_paths = sorted(mp4_by_base.get(base, []))

        # Probe duration for the base chunk
        probe_target = mp4_paths[0] if mp4_paths else wav_path
        duration_sec = probe_duration(probe_target) if probe_target else None

        # One row per MP4 (each camera angle is its own training sample)
        if mp4_paths:
            for mp4 in mp4_paths:
                camera_view = extract_camera_view(mp4, base)
                is_valid = verify_video(mp4)
                
                row = {
                    **parsed,
                    "video_path": str(mp4),
                    "audio_path": str(wav_path) if wav_path else None,
                    "duration_sec": duration_sec,
                    "camera_view": camera_view,
                    "corrupted": not is_valid,
                }
                rows.append(row)
        elif wav_path:
            # Audio-only chunk (no video) — include for completeness
            row = {
                **parsed,
                "video_path": None,
                "audio_path": str(wav_path),
                "duration_sec": duration_sec,
                "camera_view": None,
                "corrupted": False,
            }
            rows.append(row)

    return rows


def main():
    data_root, out_dir = get_env_paths()
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = load_done_sessions(out_dir)
    print(f"Loaded {len(sessions)} completed sessions from sessions_done.jsonl")

    out_path = out_dir / "chunk_manifest.jsonl"
    total_rows = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for session in sessions:
            sid = session["session_id"]
            rows = build_manifest_for_session(session)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_rows += len(rows)
            if rows:
                print(f"  {sid}: {len(rows)} chunk rows")
            else:
                print(f"  {sid}: (no chunks found)")

    # Summary stats
    no_ann_count = 0
    corrupt_count = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if '"no_annotation"' in line:
                no_ann_count += 1
            if '"corrupted": true' in line:
                corrupt_count += 1

    print(f"\n{'='*60}")
    print(f"Total chunk rows     : {total_rows}")
    print(f"  no_annotation rows : {no_ann_count}")
    if corrupt_count > 0:
        print(f"  [!] Corrupted files (PyAV failed) : {corrupt_count}")
        print(f"      -> These corrupted files are strictly flagged 'True'")
        print(f"      -> They will NOT be included in the Train, Test, or Eval datasets.")
    print(f"Output               : {out_path}")


if __name__ == "__main__":
    main()
