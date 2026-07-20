"""
00_prepare_annotations.py
=========================
Step 0: Create the session folder structure under X:\\data\\Schwan_T3_FineTune
and generate enriched annotation JSONs for each session.

Each enriched annotation contains:
  - short_code:         The original ICEP label (e.g. "Inon", "Cpos")
  - label:              Human-readable label name (e.g. "Object/Environment Engagement")
  - full_description:   The complete, unshortened ICEP manual description
  - video_description:  Description focused on visual cues for video models
  - audio_description:  Description focused on auditory cues for audio models
  - track:              Which track (Infant_Engagement / Caregiver_Engagement)
  - original_start:     Original human-annotated start time (seconds)
  - original_end:       Original human-annotated end time (seconds)
  - buffered_start:     Start time with context buffer, clamped to 0
  - buffered_end:       End time with context buffer, clamped to video duration
  - duration:           Duration of the buffered segment
  - buffer_seconds:     How much buffer was applied

Only Infant_Engagement and Caregiver_Engagement tracks are processed.
Paradigm_Phases and Other tracks are skipped.
"""

import os
import json
from pathlib import Path
from load_config import cfg, RAW_DATA_DIR, FINETUNE_DIR

# ──────────────────────────────────────────────────────────────
# Configuration (from config.yaml)
# ──────────────────────────────────────────────────────────────
CONTEXT_BUFFER_SEC = cfg["context_buffer_sec"]
MIN_GAP_DURATION   = cfg["min_gap_duration"]
TARGET_TRACKS      = cfg["target_tracks"]

# The comprehensive ICEP codes file with full descriptions
# Located in the same directory as this script
SCRIPT_DIR = Path(__file__).resolve().parent
ICEP_CODES_FILE = SCRIPT_DIR / "icep_codes.json"


def load_icep_codes(icep_file: Path) -> dict:
    """
    Load the ICEP codes JSON and build a flat lookup dict:
      code -> { label, full_description, video_description, audio_description }
    """
    with open(icep_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    lookup = {}

    for section_key in ("infant_codes", "caregiver_codes"):
        section = raw.get(section_key, {})
        for code, info in section.items():
            lookup[code] = {
                "label": info.get("label", code),
                "full_description": info.get("full_description", ""),
                "video_description": info.get("video_description", ""),
                "audio_description": info.get("audio_description", ""),
            }

    return lookup


# No-annotation description used for gap segments
NO_ANNOTATION_INFO = {
    "label": "No Annotation",
    "full_description": (
        "This segment has no ICEP engagement annotation. The interaction during this "
        "period was not coded by the human annotator. This may indicate a transition, "
        "an ambiguous period, or a segment outside the primary coding window. No specific "
        "infant or caregiver engagement phase is assigned."
    ),
    "video_description": (
        "No specific visual engagement behavior was coded during this segment. "
        "The visual content may show a transition, an ambiguous interaction, or "
        "a period that falls outside the annotated coding window."
    ),
    "audio_description": (
        "No specific auditory engagement behavior was coded during this segment. "
        "The audio may contain ambient sounds, silence, transitions, or interactions "
        "that were not assigned a specific ICEP code."
    ),
}



def get_video_duration(session_dir: Path) -> float:
    """Read video duration from metadata.json, fallback to a large number."""
    meta_file = session_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tech = meta.get("technical_metadata", {})
        dur = tech.get("duration")
        if dur:
            return float(dur)
    return 99999.0  # fallback


def find_gaps(events: list, video_duration: float) -> list:
    """
    Find unannotated time gaps between sorted annotation events for a track.
    Returns a list of (gap_start, gap_end) tuples.
    """
    if not events:
        return []

    # Sort events by start time
    sorted_events = sorted(events, key=lambda e: float(e.get("start", 0)))

    gaps = []

    # Check for gap before the first annotation (from video start)
    first_start = float(sorted_events[0].get("start", 0))
    if first_start > MIN_GAP_DURATION:
        gaps.append((0.0, first_start))

    # Check gaps between consecutive annotations
    for i in range(len(sorted_events) - 1):
        current_end = float(sorted_events[i].get("end", 0))
        next_start = float(sorted_events[i + 1].get("start", 0))
        gap_duration = next_start - current_end

        if gap_duration >= MIN_GAP_DURATION:
            gaps.append((current_end, next_start))

    # Check for gap after the last annotation (to video end)
    last_end = float(sorted_events[-1].get("end", 0))
    if video_duration - last_end >= MIN_GAP_DURATION:
        gaps.append((last_end, video_duration))

    return gaps


def main():
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load the ICEP codes with full descriptions ──
    if not ICEP_CODES_FILE.exists():
        print(f"ERROR: ICEP codes file not found at {ICEP_CODES_FILE}")
        print("Make sure icep_codes.json is in the same directory as this script.")
        return

    icep_lookup = load_icep_codes(ICEP_CODES_FILE)
    print(f"Loaded {len(icep_lookup)} ICEP codes from {ICEP_CODES_FILE.name}")

    # Also save a copy of the ICEP codes to the finetune directory for reference
    icep_out = FINETUNE_DIR / "icep_codes.json"
    with open(ICEP_CODES_FILE, "r", encoding="utf-8") as f:
        icep_raw = json.load(f)
    with open(icep_out, "w", encoding="utf-8") as f:
        json.dump(icep_raw, f, indent=2, ensure_ascii=False)
    print(f"Copied ICEP codes mapping → {icep_out}")

    # ── Process all sessions ──
    session_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(session_dirs)} session directories in {RAW_DATA_DIR}")

    total_annotations = 0
    sessions_processed = 0
    unknown_codes = set()

    for session_dir in session_dirs:
        session_name = session_dir.name

        # Read the original human annotation file
        annotation_file = session_dir / f"{session_name}.json"
        if not annotation_file.exists():
            print(f"  SKIP {session_name}: No {session_name}.json")
            continue

        with open(annotation_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        video_duration = get_video_duration(session_dir)

        # Get video source name from the original annotation metadata
        metadata = session_data.get("metadata", {})
        video_source = metadata.get("videoSource", "")

        tracks = session_data.get("tracks", [])

        enriched_annotations = []

        for track_obj in tracks:
            track_name = track_obj.get("name", "Unknown")

            # Only process Infant and Caregiver tracks
            if track_name not in TARGET_TRACKS:
                continue

            events = track_obj.get("events", [])
            for event in events:
                code = event.get("code")
                orig_start = float(event.get("start", 0))
                orig_end = float(event.get("end", 0))

                if orig_end <= orig_start:
                    continue

                # Apply context buffer, clamped to valid range
                buf_start = max(0.0, orig_start - CONTEXT_BUFFER_SEC)
                buf_end = min(video_duration, orig_end + CONTEXT_BUFFER_SEC)
                buf_duration = round(buf_end - buf_start, 3)

                # Look up the full descriptions
                code_info = icep_lookup.get(code)
                if code_info is None:
                    unknown_codes.add(code)
                    code_info = {
                        "label": code,
                        "full_description": f"Unknown ICEP code: {code}",
                        "video_description": f"Unknown ICEP code: {code}",
                        "audio_description": f"Unknown ICEP code: {code}",
                    }

                enriched_annotations.append({
                    "short_code": code,
                    "label": code_info["label"],
                    "full_description": code_info["full_description"],
                    "video_description": code_info["video_description"],
                    "audio_description": code_info["audio_description"],
                    "track": track_name,
                    "original_start": round(orig_start, 3),
                    "original_end": round(orig_end, 3),
                    "buffered_start": round(buf_start, 3),
                    "buffered_end": round(buf_end, 3),
                    "duration": buf_duration,
                    "buffer_seconds": CONTEXT_BUFFER_SEC,
                })

            # ── Find unannotated gaps for this track ──
            gaps = find_gaps(events, video_duration)
            for gap_start, gap_end in gaps:
                buf_start = max(0.0, gap_start - CONTEXT_BUFFER_SEC)
                buf_end = min(video_duration, gap_end + CONTEXT_BUFFER_SEC)
                buf_duration = round(buf_end - buf_start, 3)

                enriched_annotations.append({
                    "short_code": "no_annotation",
                    "label": NO_ANNOTATION_INFO["label"],
                    "full_description": NO_ANNOTATION_INFO["full_description"],
                    "video_description": NO_ANNOTATION_INFO["video_description"],
                    "audio_description": NO_ANNOTATION_INFO["audio_description"],
                    "track": track_name,
                    "original_start": round(gap_start, 3),
                    "original_end": round(gap_end, 3),
                    "buffered_start": round(buf_start, 3),
                    "buffered_end": round(buf_end, 3),
                    "duration": buf_duration,
                    "buffer_seconds": CONTEXT_BUFFER_SEC,
                })

        # Sort all annotations (labeled + gaps) by original_start for clean ordering
        enriched_annotations.sort(key=lambda a: (a["track"], a["original_start"]))

        if not enriched_annotations:
            print(f"  SKIP {session_name}: No Infant/Caregiver events")
            continue

        # Create session output folder
        session_out_dir = FINETUNE_DIR / session_name
        session_out_dir.mkdir(parents=True, exist_ok=True)

        # Save enriched annotation JSON
        out_json = {
            "session": session_name,
            "video_source": video_source,
            "video_duration": video_duration,
            "buffer_seconds": CONTEXT_BUFFER_SEC,
            "total_annotations": len(enriched_annotations),
            "annotations": enriched_annotations,
        }

        out_path = session_out_dir / f"{session_name}_finetune_annotations.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2, ensure_ascii=False)

        total_annotations += len(enriched_annotations)
        sessions_processed += 1
        print(f"  ✓ {session_name}: {len(enriched_annotations)} annotations → {out_path.name}")

    print(f"\n{'='*60}")
    print(f"Done! Processed {sessions_processed} sessions, {total_annotations} total annotations")
    print(f"Output directory: {FINETUNE_DIR}")
    print(f"Buffer: ±{CONTEXT_BUFFER_SEC}s around each annotation")

    if unknown_codes:
        print(f"\n⚠ Unknown ICEP codes encountered (not in icep_codes.json):")
        for c in sorted(unknown_codes):
            print(f"  - {c}")


if __name__ == "__main__":
    main()
