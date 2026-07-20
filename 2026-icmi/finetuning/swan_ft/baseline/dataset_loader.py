"""Load samples from existing variant fold JSONs for baseline feature extraction and training."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default clips root — used to derive video path for audio-only variants
_DEFAULT_CLIPS_ROOT = Path("/mnt/swan/data/Schwan_FT/clips")


@dataclass
class BaselineSample:
    clip_id: str  # e.g. "SESSION/segment_10.00_20.00"
    video_path: Path | None  # path to .mp4 clip (always set if video variant or derivable)
    audio_path: Path | None  # path to .wav (audio variants) or None
    feature_key: Path  # canonical path for feature cache lookup (always the video .mp4 path)
    labels: dict[str, str]  # e.g. {"infant_code": "ineu", "caregiver_code": "cpos"}


def _parse_gt_response(response_text: str) -> dict[str, str]:
    """Parse the GT JSON from conversations[1]['value']."""
    try:
        payload = json.loads(response_text)
        return {k: v for k, v in payload.items() if k.endswith("_code")}
    except (json.JSONDecodeError, AttributeError):
        return {}


def _clip_id_from_path(media_path: str) -> str:
    """Extract session/clip_stem from an absolute media path."""
    p = Path(media_path)
    return f"{p.parent.name}/{p.stem}"


def _video_path_from_clip_id(clip_id: str, clips_root: Path = _DEFAULT_CLIPS_ROOT) -> Path:
    """Derive the canonical video .mp4 path from a clip_id."""
    # clip_id is "SESSION/segment_stem"
    return clips_root / f"{clip_id}.mp4"


def load_fold_samples(fold_json_path: Path, clips_root: Path = _DEFAULT_CLIPS_ROOT) -> list[BaselineSample]:
    """Load all samples from a fold JSON file."""
    with fold_json_path.open("r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    samples = []
    for entry in raw_samples:
        conversations = entry.get("conversations", [])
        if len(conversations) < 2:
            continue

        gt_text = conversations[1].get("value", "")
        labels = _parse_gt_response(gt_text)
        if not labels:
            logger.warning("Could not parse GT labels from sample, skipping")
            continue

        videos = entry.get("videos") or []
        audios = entry.get("audios") or []

        video_path = Path(videos[0]) if videos else None
        audio_path = Path(audios[0]) if audios else None

        # Derive clip_id from whichever media path is available
        media_path = videos[0] if videos else (audios[0] if audios else None)
        if media_path is None:
            continue

        clip_id = _clip_id_from_path(media_path)

        # Feature cache key is always the video .mp4 path (even for audio-only variants)
        feature_key = video_path if video_path is not None else _video_path_from_clip_id(clip_id, clips_root)

        samples.append(BaselineSample(
            clip_id=clip_id,
            video_path=video_path,
            audio_path=audio_path,
            feature_key=feature_key,
            labels=labels,
        ))

    return samples


def collect_unique_clips(variant_root: Path, folds: tuple[int, ...] = (0, 1, 2, 3, 4), clips_root: Path = _DEFAULT_CLIPS_ROOT) -> list[BaselineSample]:
    """Collect all unique clips across all folds (for feature extraction).

    Returns deduplicated samples by clip_id.
    """
    seen: dict[str, BaselineSample] = {}
    for fold_id in folds:
        for split in ("train", "val"):
            fold_path = variant_root / f"fold_{fold_id}_{split}.json"
            if not fold_path.exists():
                continue
            for sample in load_fold_samples(fold_path, clips_root):
                if sample.clip_id not in seen:
                    seen[sample.clip_id] = sample
    logger.info("Collected %d unique clips from %d folds", len(seen), len(folds))
    return list(seen.values())
