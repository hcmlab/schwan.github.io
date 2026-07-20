"""Training and prediction for the feature-extraction baseline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DatasetSpec, RunOptions, RunSpec
from ..paths import resolver_for
from ..prompts import get_prompt
from .classifier import LogRegClassifier, MLPClassifier, pool_features
from .config import MODALITY_COMBOS, BaselineSpec
from .dataset_loader import BaselineSample, load_fold_samples
from .feature_store import _encoder_shortname, load_features

logger = logging.getLogger(__name__)


def _active_roles(run_spec: RunSpec) -> tuple[str, ...]:
    role_mode = run_spec.dataset_spec.effective_role_mode(run_spec.options)
    if role_mode == "joint":
        return ("infant", "caregiver")
    return (role_mode,)


def _feature_cache_root(run_spec: RunSpec) -> Path:
    return run_spec.profile.data_root / "baseline_features"


def _assemble_features(
    samples: list[BaselineSample],
    cache_root: Path,
    baseline_spec: BaselineSpec,
    modalities: tuple[str, ...],
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Load cached features for all samples, pool, and concatenate.

    Returns:
        X: (N, D_total) feature matrix
        labels: list of label dicts per sample
    """
    feature_vectors = []
    labels = []
    skipped = 0

    vis_name = _encoder_shortname(baseline_spec.visual_encoder) if baseline_spec.visual_encoder else None
    aud_name = _encoder_shortname(baseline_spec.audio_encoder) if baseline_spec.audio_encoder else None

    for sample in samples:
        parts = []

        if "video" in modalities and vis_name:
            try:
                vis_data = load_features(cache_root, vis_name, sample.feature_key)
                frame_feats = vis_data["frames"]  # (N_frames, D)
                pooled = pool_features(frame_feats, baseline_spec.temporal_pooling)
                parts.append(pooled)
            except FileNotFoundError:
                skipped += 1
                continue

        if "audio" in modalities and aud_name:
            try:
                aud_data = load_features(cache_root, aud_name, sample.feature_key)
                pooled_audio = aud_data["pooled"]  # (D,)
                if baseline_spec.temporal_pooling == "mean_std":
                    # Audio is already mean-pooled; pad with zeros for std to match dim
                    pooled_audio = np.concatenate([pooled_audio, np.zeros_like(pooled_audio)])
                parts.append(pooled_audio)
            except FileNotFoundError:
                skipped += 1
                continue

        if not parts:
            skipped += 1
            continue

        feature_vectors.append(np.concatenate(parts))
        labels.append(sample.labels)

    if skipped > 0:
        logger.warning("Skipped %d samples due to missing features", skipped)

    return np.stack(feature_vectors), labels


def _make_classifier(baseline_spec: BaselineSpec, input_dim: int, num_classes: int):
    if baseline_spec.classifier == "logreg":
        return LogRegClassifier(C=baseline_spec.logreg_C, class_weight=baseline_spec.logreg_class_weight)
    elif baseline_spec.classifier == "mlp":
        return MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=baseline_spec.mlp_hidden_dim,
            dropout=baseline_spec.mlp_dropout,
            epochs=baseline_spec.mlp_epochs,
            lr=baseline_spec.mlp_lr,
        )
    raise ValueError(f"Unknown classifier: {baseline_spec.classifier}")


def _train_fold_for_combo(
    run_spec: RunSpec,
    baseline_spec: BaselineSpec,
    fold_id: int,
    modality_combo: str,
    train_samples: list[BaselineSample],
    val_samples: list[BaselineSample],
) -> dict[str, Any]:
    """Train classifiers for one modality combo on one fold."""
    resolver = resolver_for(run_spec)
    model_spec = baseline_spec.to_model_spec(modality_combo)
    dataset_spec = run_spec.dataset_spec
    options = run_spec.options
    cache_root = _feature_cache_root(run_spec)
    modalities = MODALITY_COMBOS[modality_combo]

    X_train, train_labels = _assemble_features(train_samples, cache_root, baseline_spec, modalities)
    X_val, val_labels = _assemble_features(val_samples, cache_root, baseline_spec, modalities)

    logger.info("[%s] Fold %d: feature matrix train=%s, val=%s", modality_combo, fold_id, X_train.shape, X_val.shape)

    roles = _active_roles(run_spec)
    predictions_per_role: dict[str, list[str]] = {}

    for role in roles:
        code_key = f"{role}_code"
        y_train = [lab.get(code_key, "(MISSING)") for lab in train_labels]

        num_classes = len(set(y_train))
        clf = _make_classifier(baseline_spec, X_train.shape[1], num_classes)

        logger.info("[%s] Training %s classifier for %s (classes=%d)", modality_combo, baseline_spec.classifier, role, num_classes)

        if baseline_spec.classifier == "mlp":
            y_val_labels = [lab.get(code_key, "(MISSING)") for lab in val_labels]
            clf.fit(X_train, y_train, X_val=X_val, y_val=y_val_labels)
        else:
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        predictions_per_role[role] = y_pred

        output_dir = resolver.fold_output_dir(model_spec, dataset_spec, fold_id, options)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".pkl" if baseline_spec.classifier == "logreg" else ".pt"
        clf.save(output_dir / f"{role}_classifier{suffix}")

    # Write generated_predictions.jsonl
    pred_dir = resolver.fold_prediction_dir(model_spec, dataset_spec, fold_id, options)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "generated_predictions.jsonl"

    with pred_path.open("w", encoding="utf-8") as f:
        for i, label_dict in enumerate(val_labels):
            gt_payload = {}
            pred_payload = {}
            for role in roles:
                code_key = f"{role}_code"
                gt_payload[code_key] = label_dict.get(code_key, "")
                gt_payload[f"{role}_description"] = ""
                pred_payload[code_key] = predictions_per_role[role][i]
                pred_payload[f"{role}_description"] = ""
            entry = {"label": json.dumps(gt_payload), "predict": json.dumps(pred_payload)}
            f.write(json.dumps(entry) + "\n")

    logger.info("[%s] Wrote %d predictions to %s", modality_combo, len(val_labels), pred_path)
    return {"fold": fold_id, "modality": modality_combo, "train_samples": len(train_labels), "val_samples": len(val_labels)}


def train_and_predict_cv(
    run_spec: RunSpec,
    baseline_spec: BaselineSpec,
    modality_combo: str | None = None,
) -> dict[str, Any]:
    """Train and predict on all CV folds for modality combos.

    Args:
        modality_combo: If set, run only this combo. Otherwise run all from baseline_spec.
    """
    resolver = resolver_for(run_spec)
    dataset_spec = run_spec.dataset_spec
    options = run_spec.options
    variant_root = resolver.variant_root(dataset_spec, options)
    combos = (modality_combo,) if modality_combo else baseline_spec.modality_combos

    results = {}
    for fold_id in run_spec.options.folds:
        train_samples = load_fold_samples(variant_root / f"fold_{fold_id}_train.json")
        val_samples = load_fold_samples(variant_root / f"fold_{fold_id}_val.json")
        logger.info("Fold %d: %d train, %d val samples", fold_id, len(train_samples), len(val_samples))

        for combo in combos:
            key = f"fold_{fold_id}_{combo}"
            results[key] = _train_fold_for_combo(run_spec, baseline_spec, fold_id, combo, train_samples, val_samples)

    return results
