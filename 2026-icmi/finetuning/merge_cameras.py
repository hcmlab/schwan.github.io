#!/usr/bin/env python3
"""
Merge MUC Kamera 2 + Kamera 4 into a side-by-side 2-panel composite video.

MUC sessions have 4 individual cameras:
  - Kamera 2: behind caregiver (infant face visible) -> left panel
  - Kamera 4: behind infant (caregiver face visible) -> right panel

HD sessions already have a 2-panel side-by-side view -> used as-is.

Output: {SessionID}_merged.mp4 in the session directory.
"""

import os
import json
import glob
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from swan_ft.config import load_profile


def find_merge_inputs(session_path):
    """Find Kamera2 and Kamera4 video files for a MUC session."""
    kam2 = glob.glob(os.path.join(session_path, "*Kamera2*.mp4"))
    kam4 = glob.glob(os.path.join(session_path, "*Kamera4*.mp4"))

    if not kam2 or not kam4:
        return None, None
    return kam2[0], kam4[0]


def find_audio_source(session_path):
    """Find an audio source: prefer Splitscreen.wav, then Splitscreen.mp4, then any .wav."""
    # Splitscreen wav
    wavs = glob.glob(os.path.join(session_path, "*Splitscreen*.wav"))
    if wavs:
        return wavs[0], "wav"

    # Splitscreen mp4 (extract audio from it)
    splits = glob.glob(os.path.join(session_path, "*Splitscreen*.mp4"))
    if splits:
        return splits[0], "video"

    # Session audio wav
    wavs = glob.glob(os.path.join(session_path, "*_audio.wav"))
    if wavs:
        return wavs[0], "wav"

    # Kamera2 itself (has audio track)
    kam2 = glob.glob(os.path.join(session_path, "*Kamera2*.mp4"))
    if kam2:
        return kam2[0], "video"

    return None, None


def merge_session(session_path, output_dir=None, crf=18, fps=None, overwrite=False):
    """
    Merge Kamera 2 + 4 into a side-by-side video for one MUC session.
    Returns (session_id, output_path, success, message).
    """
    session_id = os.path.basename(session_path)

    if output_dir:
        out_path = os.path.join(output_dir, f"{session_id}_merged.mp4")
    else:
        out_path = os.path.join(session_path, f"{session_id}_merged.mp4")

    if os.path.exists(out_path) and not overwrite:
        return session_id, out_path, True, "already exists"

    kam2, kam4 = find_merge_inputs(session_path)
    if not kam2 or not kam4:
        return session_id, None, False, "missing Kamera2 or Kamera4"

    audio_src, audio_type = find_audio_source(session_path)

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]

    # Video inputs
    cmd += ["-i", kam2, "-i", kam4]

    # Audio input (if separate from video inputs)
    audio_input_idx = None
    if audio_src and audio_type == "wav":
        cmd += ["-i", audio_src]
        audio_input_idx = 2
    elif audio_src and audio_type == "video" and audio_src not in (kam2, kam4):
        cmd += ["-i", audio_src]
        audio_input_idx = 2

    # Filter: scale each to 960x1080, stack horizontally
    cmd += [
        "-filter_complex",
        "[0:v]scale=960:1080[l];[1:v]scale=960:1080[r];[l][r]hstack=inputs=2"
    ]

    # Video codec
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-c:v", "libx264", "-crf", str(crf)]

    # Audio mapping
    if audio_input_idx is not None:
        cmd += ["-map", "2:a", "-c:a", "aac"]
    elif audio_src and audio_type == "video" and audio_src == kam2:
        # Audio from Kamera2 (input 0)
        cmd += ["-map", "0:a?", "-c:a", "aac"]
    else:
        cmd += ["-an"]

    # Shortest stream determines duration
    cmd += ["-shortest"]

    cmd += [out_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return session_id, out_path, False, f"ffmpeg error: {result.stderr[-500:]}"
        return session_id, out_path, True, "merged"
    except subprocess.TimeoutExpired:
        return session_id, out_path, False, "ffmpeg timeout"
    except Exception as e:
        return session_id, out_path, False, str(e)


def get_hd_video(session_path):
    """Get the single video file for an HD session (already 2-panel)."""
    videos = glob.glob(os.path.join(session_path, "*.mp4"))
    # Filter out any normalized/merged files
    videos = [v for v in videos if "_normalized" not in os.path.basename(v)
              and "_merged" not in os.path.basename(v)]
    return videos[0] if videos else None


def normalize_hd_session(session_path, output_dir=None, crf=23, fps=None, overwrite=False):
    """
    Re-encode an HD session video at target fps.
    HD videos are already 2-panel but at 25fps — re-encode to match MUC output.
    Returns (session_id, output_path, success, message).
    """
    session_id = os.path.basename(session_path)
    src_video = get_hd_video(session_path)
    if not src_video:
        return session_id, None, False, "no source video"

    if output_dir:
        out_path = os.path.join(output_dir, f"{session_id}_normalized.mp4")
    else:
        out_path = os.path.join(session_path, f"{session_id}_normalized.mp4")

    if os.path.exists(out_path) and not overwrite:
        return session_id, out_path, True, "already exists"

    cmd = ["ffmpeg", "-y", "-i", src_video]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-c:a", "aac", out_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return session_id, out_path, False, f"ffmpeg error: {result.stderr[-500:]}"
        return session_id, out_path, True, "normalized"
    except subprocess.TimeoutExpired:
        return session_id, out_path, False, "ffmpeg timeout"
    except Exception as e:
        return session_id, out_path, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Normalize all session videos to 2-panel @ target fps")
    parser.add_argument("--profile", type=str, default=None,
                        help="Named runtime profile from configs/profiles/")
    parser.add_argument("--root", type=str, default=None,
                        help="Root session directory. Required unless --profile provides session_root.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for processed videos. Required unless --profile is provided.")
    parser.add_argument("--crf", type=int, default=23, help="CRF quality (lower=better, default 23)")
    parser.add_argument("--fps", type=float, default=8.0,
                        help="Output framerate (default: 8). Qwen VL processor defaults to 2fps "
                             "but enforces FPS_MIN_FRAMES=4. At 8fps even 0.5s clips have 4 frames.")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel ffmpeg workers (default: 6, saturates 1GbE NAS link)")
    parser.add_argument("--overwrite", action="store_true", help="Re-process existing files")
    parser.add_argument("--sessions", nargs="+", help="Process specific sessions only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    if args.profile:
        profile = load_profile(args.profile)
        if args.root is None and profile.session_root is not None:
            args.root = str(profile.session_root)
        if args.output_dir is None:
            args.output_dir = str(profile.data_root / "videos")

    if args.root is None:
        parser.error("--root is required unless provided by --profile")
    if args.output_dir is None:
        parser.error("--output-dir is required unless provided by --profile")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    muc_sessions = []
    hd_sessions = []

    for name in sorted(os.listdir(args.root)):
        if args.sessions and name not in args.sessions:
            continue
        session_path = os.path.join(args.root, name)
        if not os.path.isdir(session_path):
            continue
        if "_MUC_" in name:
            muc_sessions.append(session_path)
        elif "_HD_" in name:
            hd_sessions.append(session_path)

    print(f"MUC sessions to merge Kam2+4: {len(muc_sessions)}")
    print(f"HD sessions to re-encode:     {len(hd_sessions)}")
    print(f"Target: {args.fps}fps, CRF {args.crf}")

    if args.dry_run:
        for sp in muc_sessions:
            sid = os.path.basename(sp)
            kam2, kam4 = find_merge_inputs(sp)
            audio_src, audio_type = find_audio_source(sp)
            status = "OK" if (kam2 and kam4) else "MISSING INPUTS"
            print(f"  [MUC] {sid}: {status}")
            if kam2:
                print(f"    Kam2: {os.path.basename(kam2)}")
            if kam4:
                print(f"    Kam4: {os.path.basename(kam4)}")
            if audio_src:
                print(f"    Audio: {os.path.basename(audio_src)} ({audio_type})")
        for sp in hd_sessions:
            vid = get_hd_video(sp)
            print(f"  [HD]  {os.path.basename(sp)}: {os.path.basename(vid) if vid else 'NO VIDEO'}")
        return

    # Process all sessions in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for sp in muc_sessions:
            futures[executor.submit(
                merge_session, sp, args.output_dir, args.crf, args.fps, args.overwrite
            )] = ("MUC", sp)
        for sp in hd_sessions:
            futures[executor.submit(
                normalize_hd_session, sp, args.output_dir, args.crf, args.fps, args.overwrite
            )] = ("HD", sp)

        for future in as_completed(futures):
            loc, sp = futures[future]
            session_id, out_path, success, msg = future.result()
            status = "OK" if success else "FAIL"
            print(f"  [{status}] [{loc}] {session_id}: {msg}")
            results.append({
                "session_id": session_id,
                "location": loc,
                "output": out_path,
                "success": success,
                "message": msg,
            })

    ok = sum(1 for r in results if r["success"])
    fail = sum(1 for r in results if not r["success"])
    print(f"\nProcessing complete: {ok} succeeded, {fail} failed")

    # Write video map (session_id -> normalized video path)
    video_map = {}
    for r in results:
        if r["success"] and r["output"]:
            video_map[r["session_id"]] = r["output"]

    map_path = os.path.join(args.output_dir, "..", "video_map.json")
    map_path = os.path.normpath(map_path)
    with open(map_path, "w") as f:
        json.dump(video_map, f, indent=2)
    print(f"Video map saved to {map_path} ({len(video_map)} sessions)")


if __name__ == "__main__":
    main()
