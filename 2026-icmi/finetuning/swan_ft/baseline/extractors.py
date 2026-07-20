"""Frozen feature extractors for DINOv3 (vision via timm) and Wav2Vec2 (audio)."""

from __future__ import annotations

import io
import logging
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model

logger = logging.getLogger(__name__)


def _decode_video_frames(video_path: Path) -> np.ndarray:
    """Decode all frames from a video clip via ffmpeg. Returns (N, H, W, 3) uint8 array."""
    # Use ffmpeg directly — guaranteed to work, no library compat issues
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}")
    w, h = [int(x) for x in result.stdout.strip().split(",")]

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "quiet", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed on {video_path}")

    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    frame_size = h * w * 3
    n_frames = len(raw) // frame_size
    return raw[:n_frames * frame_size].reshape(n_frames, h, w, 3)


def _load_audio_ffmpeg(audio_path: Path, target_sr: int = 16000) -> torch.Tensor:
    """Load audio from any format (wav, mp4, etc) via ffmpeg. Returns (samples,) float32 tensor."""
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "wav", "-ac", "1", "-ar", str(target_sr),
        "-v", "quiet", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {audio_path}: {result.stderr.decode()[:200]}")
    buf = io.BytesIO(result.stdout)
    with wave.open(buf) as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.tensor(audio)


class DINOv3Extractor:
    """Extract CLS token features from DINOv3 via timm (no gated license needed)."""

    def __init__(self, model_name: str = "vit_large_patch16_dinov3", device: str = "cuda"):
        import timm
        import timm.data

        logger.info("Loading DINOv3 model via timm: %s", model_name)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(self.device)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        self.model_name = model_name

    @torch.no_grad()
    def extract_clip(self, video_path: Path, batch_size: int = 32) -> np.ndarray:
        """Extract per-frame CLS features. Returns (N_frames, D) float32 array."""
        from PIL import Image

        frames = _decode_video_frames(video_path)  # (N, H, W, 3)
        n_frames = len(frames)
        if n_frames == 0:
            raise ValueError(f"No frames decoded from {video_path}")

        all_features = []
        for start in range(0, n_frames, batch_size):
            batch_frames = frames[start:start + batch_size]
            # timm transforms expect PIL images
            tensors = [self.transform(Image.fromarray(f)) for f in batch_frames]
            batch = torch.stack(tensors).to(self.device)
            features = self.model(batch)  # (B, D)
            all_features.append(features.cpu().float().numpy())

        return np.concatenate(all_features, axis=0)  # (N_frames, D)


class Wav2Vec2Extractor:
    """Extract audio features from Wav2Vec2-Large. Accepts any audio/video format via ffmpeg."""

    TARGET_SAMPLE_RATE = 16000

    def __init__(self, model_name: str = "facebook/wav2vec2-large-960h", device: str = "cuda"):
        logger.info("Loading Wav2Vec2 model: %s", model_name)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2Model.from_pretrained(model_name).eval().to(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model_name = model_name

    @torch.no_grad()
    def extract_clip(self, audio_path: Path) -> np.ndarray:
        """Extract mean-pooled audio features. Returns (D,) float32 array.

        Accepts .wav, .mp4, or any format ffmpeg can decode.
        """
        waveform = _load_audio_ffmpeg(audio_path, self.TARGET_SAMPLE_RATE)

        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state  # (1, T, D)
        pooled = hidden_states.mean(dim=1).squeeze(0)  # (D,)
        return pooled.cpu().float().numpy()
