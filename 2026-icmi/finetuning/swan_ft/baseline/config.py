from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ModelSpec


BASELINE_SPEC_DIR = Path(__file__).resolve().parents[2] / "configs" / "baseline_specs"


MODALITY_COMBOS = {
    "video": ("video",),
    "audio": ("audio",),
    "omni": ("video", "audio"),
}


@dataclass(frozen=True)
class BaselineSpec:
    id: str
    visual_encoder: str | None
    audio_encoder: str | None
    visual_dim: int
    audio_dim: int
    temporal_pooling: str  # "mean" or "attention"
    classifier: str  # "logreg" or "mlp"
    logreg_C: float
    logreg_class_weight: str
    mlp_hidden_dim: int
    mlp_dropout: float
    mlp_epochs: int
    mlp_lr: float
    feature_batch_size: int
    output_name: str
    modality_combos: tuple[str, ...]  # which combos to run: ("video", "audio", "omni")

    @classmethod
    def from_dict(cls, spec_id: str, data: dict[str, Any]) -> BaselineSpec:
        combos = data.get("modality_combos", ["video", "audio", "omni"])
        return cls(
            id=spec_id,
            visual_encoder=data.get("visual_encoder"),
            audio_encoder=data.get("audio_encoder"),
            visual_dim=int(data.get("visual_dim", 1024)),
            audio_dim=int(data.get("audio_dim", 1024)),
            temporal_pooling=data.get("temporal_pooling", "mean"),
            classifier=data.get("classifier", "logreg"),
            logreg_C=float(data.get("logreg_C", 1.0)),
            logreg_class_weight=data.get("logreg_class_weight", "balanced"),
            mlp_hidden_dim=int(data.get("mlp_hidden_dim", 512)),
            mlp_dropout=float(data.get("mlp_dropout", 0.3)),
            mlp_epochs=int(data.get("mlp_epochs", 50)),
            mlp_lr=float(data.get("mlp_lr", 1e-3)),
            feature_batch_size=int(data.get("feature_batch_size", 32)),
            output_name=data.get("output_name", f"baseline_{spec_id}"),
            modality_combos=tuple(combos),
        )

    def to_model_spec(self, modality_combo: str | None = None) -> ModelSpec:
        """Create a synthetic ModelSpec for path resolution and evaluation reuse.

        If modality_combo is given (e.g. "video", "audio", "omni"), it is
        appended to the output_name so each combo gets its own output dir.
        """
        suffix = f"_{modality_combo}" if modality_combo else ""
        name = f"{self.output_name}{suffix}"
        return ModelSpec(
            id=name,
            base_model=f"{self.visual_encoder or ''}+{self.audio_encoder or ''}".strip("+"),
            template="baseline",
            modality=modality_combo or "omni",
            family="baseline",
            output_name=name,
            trust_remote_code=False,
            allow_quantization=False,
            h100_full_precision_default=True,
            train_defaults={},
            predict_defaults={},
            extra_config={},
        )


def load_baseline_spec(spec_id: str) -> BaselineSpec:
    path = BASELINE_SPEC_DIR / f"{spec_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown baseline spec: {spec_id}")
    with path.open("r", encoding="utf-8") as f:
        return BaselineSpec.from_dict(spec_id, json.load(f))


def list_baseline_specs() -> list[str]:
    return sorted(p.stem for p in BASELINE_SPEC_DIR.glob("*.json"))
