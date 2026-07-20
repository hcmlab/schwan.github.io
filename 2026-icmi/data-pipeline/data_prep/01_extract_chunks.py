"""
01_extract_chunks.py
====================
Step 1: Read the enriched annotation JSONs created by 00_prepare_annotations.py
and use ffmpeg to extract context-buffered multimodal chunks (audio + video).

Reads from:  X:\\data\\Schwan_T3_FineTune\\{session}\\{session}_finetune_annotations.json
Source video: X:\\data\\Schwan_T3_Clean\\{session}\\*.mp4
Source audio: X:\\data\\Schwan_T3_Clean\\{session}\\*.wav
Output to:   X:\\data\\Schwan_T3_FineTune\\{session}\\chunks\\

Features:
 - Multi-camera augmentation (extracts from all available MP4s)
 - Hardware-accelerated video encoding (h264_nvenc)
 - Audio extraction (pcm_s16le)
 - Concurrent processing (batch of 5) with resume capability (.done markers)
"""

import os
import json
import subprocess
import signal
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from load_config import cfg, RAW_DATA_DIR, FINETUNE_DIR

# ──────────────────────────────────────────────────────────────
# Configuration (from config.yaml)
# ──────────────────────────────────────────────────────────────
MAX_WORKERS   = cfg["max_workers"]
HW_ACCEL      = cfg["hw_accel"]
VIDEO_CODEC   = cfg["video_codec"]
VIDEO_PRESET  = cfg["video_preset"]

# Global shutdown event
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    """Handles Ctrl+C to trigger a graceful shutdown."""
    if not shutdown_event.is_set():
        print("\n[INFO] Termination signal received. Shutting down gracefully after current chunks finish...")
        shutdown_event.set()
    else:
        print("\n[WARNING] Forced termination! Exiting immediately.")
        os._exit(1)

signal.signal(signal.SIGINT, signal_handler)


def get_video_files(session_dir: Path) -> list:
    """Finds all MP4 video files in the raw session directory."""
    mp4_files = list(session_dir.glob("*.mp4"))
    return mp4_files


def get_audio_file(session_dir: Path, session_name: str) -> Path | None:
    """Finds the best audio file in the raw session directory."""
    # Priority 1: {session}_audio.wav
    audio_path = session_dir / f"{session_name}_audio.wav"
    if audio_path.exists():
        return audio_path
        
    # Priority 2: Splitscreen.wav
    audio_path = session_dir / "Splitscreen.wav"
    if audio_path.exists():
        return audio_path
        
    # Priority 3: Any .wav file
    wav_files = list(session_dir.glob("*.wav"))
    if wav_files:
        wav_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        return wav_files[0]
        
    return None


def extract_video_chunk(input_path: Path, start_time: float, duration: float, output_path: Path):
    """Extracts a chunk of video using HW-accelerated ffmpeg (NVENC), no audio."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return True

    command = [
        "ffmpeg",
        "-y",
        "-hwaccel", HW_ACCEL,
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
        "-c:v", VIDEO_CODEC,
        "-preset", VIDEO_PRESET,
        "-an",              # No audio — extracted separately as WAV
        "-loglevel", "fatal",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        # Clean up 0-byte failed files so they retry on next run
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        print(f"  [ERROR] {output_path.name}: exit code {e.returncode}")
        return False


def extract_audio_chunk(input_path: Path, start_time: float, duration: float, output_path: Path):
    """Extracts a chunk of audio using ffmpeg."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return True

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
        "-c:a", "pcm_s16le",
        "-loglevel", "fatal",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        # Clean up 0-byte failed files so they retry on next run
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        print(f"  [ERROR] {output_path.name}: exit code {e.returncode}")
        return False


def process_session(session_out_dir: Path) -> dict:
    """Process a single session (extract all audio/video chunks)."""
    if shutdown_event.is_set():
        return {"session": session_out_dir.name, "status": "aborted", "chunks": 0}

    session_name = session_out_dir.name
    chunks_dir = session_out_dir / "chunks"
    done_marker = chunks_dir / ".done"
    
    if done_marker.exists():
        return {"session": session_name, "status": "skipped", "chunks": 0}

    ann_file = session_out_dir / f"{session_name}_finetune_annotations.json"
    if not ann_file.exists():
        return {"session": session_name, "status": "no_annotations", "chunks": 0}

    with open(ann_file, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    raw_session_dir = RAW_DATA_DIR / session_name
    if not raw_session_dir.exists():
        return {"session": session_name, "status": "no_raw_data", "chunks": 0}

    video_files = get_video_files(raw_session_dir)
    audio_file = get_audio_file(raw_session_dir, session_name)
    
    if not video_files:
        return {"session": session_name, "status": "no_video", "chunks": 0}
    if not audio_file:
        return {"session": session_name, "status": "no_audio", "chunks": 0}

    chunks_dir.mkdir(parents=True, exist_ok=True)
    annotations = ann_data.get("annotations", [])
    
    extracted_count = 0
    # print(f"Started [{session_name}]: {len(annotations)} annotations...")

    for idx, ann in enumerate(annotations):
        if shutdown_event.is_set():
            return {"session": session_name, "status": "aborted", "chunks": extracted_count}

        buf_start = ann["buffered_start"]
        duration = ann["duration"]
        track = ann["track"]
        code = ann["short_code"]
        safe_code = code.replace(" ", "_").replace("/", "_")
        
        base_name = f"{session_name}_{track}_idx{idx:04d}_{safe_code}"
        
        # 1. Extract audio (once per annotation)
        audio_out = chunks_dir / f"{base_name}.wav"
        if extract_audio_chunk(audio_file, buf_start, duration, audio_out):
            extracted_count += 1
            
        # 2. Extract video for ALL available camera angles
        for vid_file in video_files:
            if shutdown_event.is_set():
                return {"session": session_name, "status": "aborted", "chunks": extracted_count}
                
            # Create a suffix from the original filename (e.g. 'Kamera1')
            vid_suffix = vid_file.stem.split("_")[-1].lower() 
            video_out = chunks_dir / f"{base_name}_{vid_suffix}.mp4"
            if extract_video_chunk(vid_file, buf_start, duration, video_out):
                extracted_count += 1

    # Mark as completed only if not aborted
    if not shutdown_event.is_set():
        done_marker.touch()
        return {"session": session_name, "status": "success", "chunks": extracted_count}
    else:
        return {"session": session_name, "status": "aborted", "chunks": extracted_count}


def main():
    session_dirs = sorted([
        d for d in FINETUNE_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}_finetune_annotations.json").exists()
    ])

    print(f"Found {len(session_dirs)} sessions with prepared annotations")
    total_chunks = 0
    sessions_completed = 0
    sessions_skipped = 0
    sessions_aborted = 0

    # Process in batches using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_session = {executor.submit(process_session, d): d for d in session_dirs}
        
        # Use tqdm for a progress bar
        with tqdm(total=len(session_dirs), desc="Processing sessions", unit="session") as pbar:
            for future in as_completed(future_to_session):
                result = future.result()
                s_name = result["session"]
                s_status = result["status"]
                s_chunks = result["chunks"]
                
                if s_status == "skipped":
                    sessions_skipped += 1
                elif s_status == "success":
                    sessions_completed += 1
                    total_chunks += s_chunks
                elif s_status == "aborted":
                    sessions_aborted += 1
                    total_chunks += s_chunks
                else:
                    print(f"  ✗ Failed {s_name}: {s_status}")
                
                pbar.update(1)
                if shutdown_event.is_set():
                    # We don't break immediately to allow as_completed to finish or executor to shutdown
                    pass

    print(f"\n{'='*60}")
    if shutdown_event.is_set():
        print("SHUTDOWN COMPLETE (Graceful)")
    else:
        print("PROCESS COMPLETE")
        
    print(f"Skipped: {sessions_skipped}, Completed: {sessions_completed}, Aborted/Partial: {sessions_aborted}")
    print(f"Total new files extracted: {total_chunks}")
    print(f"Output: {FINETUNE_DIR}")


if __name__ == "__main__":
    main()
