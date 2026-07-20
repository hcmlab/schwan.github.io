"""Cache extracted features as .npz files on disk."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _feature_path(cache_root: Path, encoder_name: str, clip_path: Path) -> Path:
    """Deterministic cache path for a clip's features."""
    # encoder_name e.g. "dinov3_vitl16" or "wav2vec2_large"
    # clip_path e.g. .../clips/SESSION/segment_10.00_20.00.mp4
    session_id = clip_path.parent.name
    clip_stem = clip_path.stem
    return cache_root / encoder_name / session_id / f"{clip_stem}.npz"


def _encoder_shortname(model_name: str) -> str:
    """Convert HF model name to a filesystem-friendly short name."""
    # "facebook/dinov3-vitl16" -> "dinov3_vitl16"
    # "facebook/wav2vec2-large-960h" -> "wav2vec2_large_960h"
    short = model_name.split("/")[-1]
    return short.replace("-", "_")


def has_features(cache_root: Path, encoder_name: str, clip_path: Path) -> bool:
    return _feature_path(cache_root, encoder_name, clip_path).exists()


def save_features(cache_root: Path, encoder_name: str, clip_path: Path, **arrays: np.ndarray) -> Path:
    """Save feature arrays to a .npz file. Returns the saved path."""
    out_path = _feature_path(cache_root, encoder_name, clip_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **arrays)
    return out_path


def load_features(cache_root: Path, encoder_name: str, clip_path: Path) -> dict[str, np.ndarray]:
    """Load cached feature arrays from a .npz file."""
    feat_path = _feature_path(cache_root, encoder_name, clip_path)
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not cached: {feat_path}")
    data = np.load(str(feat_path))
    return dict(data)
