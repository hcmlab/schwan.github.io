"""
load_config.py
==============
Loads config.yaml and resolves the active profile based on the OS.

Usage in any pipeline script:
    from load_config import cfg, RAW_DATA_DIR, FINETUNE_DIR

Override the auto-detected profile with env var:
    DATA_PREP_PROFILE=ubuntu python 01_extract_chunks.py
"""

import os
import platform
from pathlib import Path

import yaml

# ── Locate config.yaml next to this file ──
_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load() -> dict:
    """Load the YAML config and resolve the active profile paths."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Determine profile
    profile = os.environ.get("DATA_PREP_PROFILE", "").lower()
    if not profile:
        profile = "windows" if platform.system() == "Windows" else "ubuntu"

    profiles = raw.get("profiles", {})
    if profile not in profiles:
        raise ValueError(
            f"Unknown profile '{profile}'. Available: {list(profiles.keys())}"
        )

    paths = profiles[profile]

    return {
        "profile": profile,
        # Paths
        "raw_data_dir": Path(paths["raw_data_dir"]),
        "finetune_dir": Path(paths["finetune_dir"]),
        # Extraction
        "max_workers":  raw.get("extraction", {}).get("max_workers", 5),
        "hw_accel":     raw.get("extraction", {}).get("hw_accel", "cuda"),
        "video_codec":  raw.get("extraction", {}).get("video_codec", "h264_nvenc"),
        "video_preset": raw.get("extraction", {}).get("video_preset", "p4"),
        # Annotations
        "context_buffer_sec": raw.get("annotations", {}).get("context_buffer_sec", 3.0),
        "min_gap_duration":   raw.get("annotations", {}).get("min_gap_duration", 1.0),
        "target_tracks":      set(raw.get("annotations", {}).get("target_tracks", [])),
        # Splitting
        "train_ratio": raw.get("splitting", {}).get("train_ratio", 0.80),
        "val_ratio":   raw.get("splitting", {}).get("val_ratio", 0.10),
        "seed":        raw.get("splitting", {}).get("seed", 42),
    }


# ── Module-level singleton ──
cfg = _load()

# Convenience shortcuts used by every script
RAW_DATA_DIR = cfg["raw_data_dir"]
FINETUNE_DIR = cfg["finetune_dir"]
