from __future__ import annotations

import json
import re
from typing import Any

from classification_stats import ClassificationStats

from .config import RunSpec
from .paths import resolver_for
from .prompts import get_prompt


INVALID_LABEL = "(INVALID)"


def _parse_prediction_payload(response_text: str) -> dict[str, Any]:
    try:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except (AttributeError, json.JSONDecodeError):
        pass
    return {}


def _normalize_label(label: str | None, allowed: set[str]) -> str:
    if not label:
        return "(MISSING)"
    label = label.lower().strip()
    if label not in allowed:
        return INVALID_LABEL
    return label


def _active_roles(run_spec: RunSpec) -> tuple[str, ...]:
    role_mode = run_spec.dataset_spec.effective_role_mode(run_spec.options)
    if role_mode == "joint":
        return ("infant", "caregiver")
    return (role_mode,)


def score_fold(run_spec: RunSpec, fold_id: int) -> dict[str, Any]:
    resolver = resolver_for(run_spec)
    model_spec = run_spec.model_spec
    dataset_spec = run_spec.dataset_spec
    pred_path = resolver.fold_prediction_dir(model_spec, dataset_spec, fold_id, run_spec.options) / "generated_predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    prompt = get_prompt(
        role_mode=dataset_spec.effective_role_mode(run_spec.options),
        include_background=dataset_spec.include_background,
        excluded_labels=dataset_spec.effective_excluded_labels(run_spec.options),
    )
    allowed_infant = set(prompt["allowed_labels"]["infant"])
    allowed_caregiver = set(prompt["allowed_labels"]["caregiver"])

    stats = {"infant": ClassificationStats(), "caregiver": ClassificationStats()}
    predictions: list[dict[str, Any]] = []
    parse_errors = 0
    invalid_predictions = 0
    roles = _active_roles(run_spec)

    with pred_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            gt_payload = _parse_prediction_payload(entry.get("label", ""))
            pred_payload = _parse_prediction_payload(entry.get("predict", ""))

            if not pred_payload:
                parse_errors += 1

            prediction_record = {"raw_output": entry.get("predict", "")[:500]}
            for role in roles:
                code_key = f"{role}_code"
                allowed = allowed_infant if role == "infant" else allowed_caregiver
                gt = _normalize_label(gt_payload.get(code_key), allowed)
                pred = _normalize_label(pred_payload.get(code_key), allowed)
                if pred == INVALID_LABEL:
                    invalid_predictions += 1
                stats[role].update(gt, pred)
                prediction_record[f"gt_{role}"] = gt
                prediction_record[f"pred_{role}"] = pred
            predictions.append(prediction_record)

    session_stats = {role: stats[role].to_dict() for role in roles}
    report = {
        "fold": fold_id,
        "profile_id": run_spec.profile.id,
        "model_id": model_spec.id,
        "dataset_spec_id": resolver.variant_id(dataset_spec, run_spec.options),
        "role_mode": dataset_spec.effective_role_mode(run_spec.options),
        "val_samples": len(predictions),
        "parse_errors": parse_errors,
        "invalid_predictions": invalid_predictions,
        "session_stats": session_stats,
        "predictions": predictions,
    }
    report_path = resolver.fold_eval_report(model_spec, dataset_spec, fold_id, run_spec.options)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def score_test(run_spec: RunSpec) -> dict[str, Any]:
    resolver = resolver_for(run_spec)
    model_spec = run_spec.model_spec
    dataset_spec = run_spec.dataset_spec
    pred_path = resolver.final_prediction_dir(model_spec, dataset_spec, run_spec.options) / "generated_predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    prompt = get_prompt(
        role_mode=dataset_spec.effective_role_mode(run_spec.options),
        include_background=dataset_spec.include_background,
        excluded_labels=dataset_spec.effective_excluded_labels(run_spec.options),
    )
    allowed_infant = set(prompt["allowed_labels"]["infant"])
    allowed_caregiver = set(prompt["allowed_labels"]["caregiver"])

    stats = {"infant": ClassificationStats(), "caregiver": ClassificationStats()}
    predictions: list[dict[str, Any]] = []
    parse_errors = 0
    invalid_predictions = 0
    roles = _active_roles(run_spec)

    with pred_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            gt_payload = _parse_prediction_payload(entry.get("label", ""))
            pred_payload = _parse_prediction_payload(entry.get("predict", ""))

            if not pred_payload:
                parse_errors += 1

            prediction_record = {"raw_output": entry.get("predict", "")[:500]}
            for role in roles:
                code_key = f"{role}_code"
                allowed = allowed_infant if role == "infant" else allowed_caregiver
                gt = _normalize_label(gt_payload.get(code_key), allowed)
                pred = _normalize_label(pred_payload.get(code_key), allowed)
                if pred == INVALID_LABEL:
                    invalid_predictions += 1
                stats[role].update(gt, pred)
                prediction_record[f"gt_{role}"] = gt
                prediction_record[f"pred_{role}"] = pred
            predictions.append(prediction_record)

    report = {
        "profile_id": run_spec.profile.id,
        "model_id": model_spec.id,
        "dataset_spec_id": resolver.variant_id(dataset_spec, run_spec.options),
        "role_mode": dataset_spec.effective_role_mode(run_spec.options),
        "test_samples": len(predictions),
        "parse_errors": parse_errors,
        "invalid_predictions": invalid_predictions,
        "session_stats": {role: stats[role].to_dict() for role in roles},
        "predictions": predictions,
    }
    report_path = resolver.final_test_eval_report(model_spec, dataset_spec, run_spec.options)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def score_cv(run_spec: RunSpec) -> dict[str, Any]:
    resolver = resolver_for(run_spec)
    model_spec = run_spec.model_spec
    dataset_spec = run_spec.dataset_spec
    roles = _active_roles(run_spec)

    global_stats = {role: ClassificationStats() for role in roles}
    fold_reports: dict[str, Any] = {}

    for fold_id in run_spec.options.folds:
        try:
            fold_report = score_fold(run_spec, fold_id)
        except FileNotFoundError:
            continue
        fold_reports[str(fold_id)] = fold_report
        for role in roles:
            for prediction in fold_report["predictions"]:
                global_stats[role].update(prediction[f"gt_{role}"], prediction[f"pred_{role}"])

    aggregate = {
        "profile_id": run_spec.profile.id,
        "model": model_spec.id,
        "dataset_spec_id": resolver.variant_id(dataset_spec, run_spec.options),
        "role_mode": dataset_spec.effective_role_mode(run_spec.options),
        "folds_evaluated": [int(fold_id) for fold_id in fold_reports.keys()],
        "global_stats": {role: global_stats[role].to_dict() for role in roles},
        "fold_summaries": {
            fold_id: {
                "val_samples": report["val_samples"],
                "parse_errors": report["parse_errors"],
                "invalid_predictions": report["invalid_predictions"],
                **{
                    f"{role}_weighted_f1": report["session_stats"][role]["metrics"].get("_WEIGHTED_AVG", {}).get("f1_score")
                    for role in roles
                },
            }
            for fold_id, report in fold_reports.items()
        },
    }
    output_path = resolver.cv_summary_path(model_spec, dataset_spec, run_spec.options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate
