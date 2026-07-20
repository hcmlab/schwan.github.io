"""Feature extraction orchestration: iterate clips, extract, cache."""

from __future__ import annotations

import logging
from typing import Any

from ..config import RunSpec
from ..paths import resolver_for
from .config import BaselineSpec
from .dataset_loader import collect_unique_clips
from .extractors import DINOv3Extractor, Wav2Vec2Extractor
from .feature_store import _encoder_shortname, has_features, save_features

logger = logging.getLogger(__name__)


def extract_features(
    run_spec: RunSpec,
    baseline_spec: BaselineSpec,
    device: str = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract and cache DINOv3 + Wav2Vec2 features for all clips in a variant.

    Always extracts both vision and audio from video clips (W2V2 reads audio
    from .mp4 via ffmpeg), so a single extract run on a video variant covers
    all modality combinations.
    """
    resolver = resolver_for(run_spec)
    variant_root = resolver.variant_root(run_spec.dataset_spec, run_spec.options)
    cache_root = run_spec.profile.data_root / "baseline_features"

    samples = collect_unique_clips(variant_root, run_spec.options.folds)
    video_samples = [s for s in samples if s.video_path is not None]

    stats = {"visual_extracted": 0, "visual_cached": 0, "audio_extracted": 0, "audio_cached": 0, "errors": 0}

    # Visual feature extraction (DINOv3 from video frames)
    if baseline_spec.visual_encoder and video_samples:
        vis_name = _encoder_shortname(baseline_spec.visual_encoder)
        vis_extractor = DINOv3Extractor(baseline_spec.visual_encoder, device=device)
        logger.info("Extracting visual features for %d clips with %s", len(video_samples), baseline_spec.visual_encoder)

        for i, sample in enumerate(video_samples):
            if not overwrite and has_features(cache_root, vis_name, sample.feature_key):
                stats["visual_cached"] += 1
                continue
            try:
                frame_feats = vis_extractor.extract_clip(sample.video_path, batch_size=baseline_spec.feature_batch_size)
                save_features(cache_root, vis_name, sample.feature_key, frames=frame_feats)
                stats["visual_extracted"] += 1
            except Exception as e:
                logger.error("Failed to extract visual features for %s: %s", sample.video_path, e)
                stats["errors"] += 1

            if (i + 1) % 100 == 0:
                logger.info("Visual: %d/%d done", i + 1, len(video_samples))

        del vis_extractor
        _free_gpu_memory()

    # Audio feature extraction (W2V2 from .mp4 or .wav — ffmpeg handles both)
    if baseline_spec.audio_encoder and video_samples:
        aud_name = _encoder_shortname(baseline_spec.audio_encoder)
        aud_extractor = Wav2Vec2Extractor(baseline_spec.audio_encoder, device=device)
        # Use audio_path if available (e.g. from audio/omni variants), else video_path (mp4 has audio track)
        logger.info("Extracting audio features for %d clips with %s", len(video_samples), baseline_spec.audio_encoder)

        for i, sample in enumerate(video_samples):
            if not overwrite and has_features(cache_root, aud_name, sample.feature_key):
                stats["audio_cached"] += 1
                continue
            # Use video_path as audio source — ffmpeg extracts audio track from .mp4
            audio_source = sample.video_path
            try:
                pooled = aud_extractor.extract_clip(audio_source)
                save_features(cache_root, aud_name, sample.feature_key, pooled=pooled)
                stats["audio_extracted"] += 1
            except Exception as e:
                logger.error("Failed to extract audio features for %s: %s", audio_source, e)
                stats["errors"] += 1

            if (i + 1) % 100 == 0:
                logger.info("Audio: %d/%d done", i + 1, len(video_samples))

        del aud_extractor
        _free_gpu_memory()

    logger.info("Extraction complete: %s", stats)
    return stats


def _free_gpu_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
