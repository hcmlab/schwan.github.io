from __future__ import annotations

import json
from typing import Any


INFANT_LABELS = {
    "ineg": "Negative (distress, crying)",
    "ipro": "Protest (fussing, negative face)",
    "iwit": "Withdrawn (averting gaze, pulling away)",
    "inon": "None (no interaction)",
    "ineu": "Neutral (neutral face, looking at partner)",
    "ipos": "Positive (smiling, laughing)",
    "islp": "Sleeping",
    "iusc": "Unscorable",
    "bg": "Background (no codable interaction)",
}

CAREGIVER_LABELS = {
    "cneg": "Negative (hostile, criticism)",
    "cwit": "Withdrawn (ignoring, flat affect)",
    "cint": "Intrusive (forcing attention)",
    "chos": "Hostile (angry)",
    "cnon": "None (passive)",
    "cneu": "Neutral (neutral affect)",
    "cpos": "Positive (warmth, praise)",
    "cpvc": "Physical Control",
    "ctch": "Touch",
    "bg": "Background (no codable interaction)",
}

VALID_ROLE_MODES = {"joint", "infant", "caregiver"}


def _filtered_labels(labels: dict[str, str], include_background: bool, excluded_labels: tuple[str, ...]) -> dict[str, str]:
    excluded = set(label.lower() for label in excluded_labels)
    result = {}
    for code, description in labels.items():
        if not include_background and code == "bg":
            continue
        if code in excluded:
            continue
        result[code] = description
    return result


def _label_block(title: str, labels: dict[str, str]) -> str:
    lines = [f"**{title}:**"]
    for code, description in labels.items():
        lines.append(f"- `{code}`: {description}")
    return "\n".join(lines)


def _schema_fields(role_mode: str) -> tuple[str, ...]:
    if role_mode == "infant":
        return ("infant_code", "infant_description")
    if role_mode == "caregiver":
        return ("caregiver_code", "caregiver_description")
    return ("infant_code", "caregiver_code", "infant_description", "caregiver_description")


def _build_prompt_text(role_mode: str, include_background: bool, excluded_labels: tuple[str, ...]) -> str:
    if role_mode not in VALID_ROLE_MODES:
        raise ValueError(f"Unsupported role mode: {role_mode}")

    infant_labels = _filtered_labels(INFANT_LABELS, include_background, excluded_labels)
    caregiver_labels = _filtered_labels(CAREGIVER_LABELS, include_background, excluded_labels)

    intro = "You are an expert behavior coder for Mother-Infant interactions using the ICEP-R (Infant and Caregiver Engagement Phases - Revised) scheme."
    context = "The clip may contain split-screen views of the same interaction. Use all available context from the provided media."
    if role_mode == "joint":
        task = "Analyze the clip and return the correct infant and caregiver codes plus short descriptions for both roles."
        blocks = [
            _label_block("Infant Codes", infant_labels),
            _label_block("Caregiver Codes", caregiver_labels),
        ]
    elif role_mode == "infant":
        task = "Analyze the clip and return only the Infant code plus a short infant behavior description. Ignore caregiver coding as a target."
        blocks = [_label_block("Infant Codes", infant_labels)]
    else:
        task = "Analyze the clip and return only the Caregiver code plus a short caregiver behavior description. Ignore infant coding as a target."
        blocks = [_label_block("Caregiver Codes", caregiver_labels)]

    fields = ", ".join(f"`{field}`" for field in _schema_fields(role_mode))
    outro = f"Return only a JSON object with keys {fields}."
    return "\n\n".join([intro, context, task, *blocks, outro])


def get_prompt(role_mode: str, include_background: bool, excluded_labels: tuple[str, ...] = ()) -> dict[str, Any]:
    infant_labels = _filtered_labels(INFANT_LABELS, include_background, excluded_labels)
    caregiver_labels = _filtered_labels(CAREGIVER_LABELS, include_background, excluded_labels)
    return {
        "role_mode": role_mode,
        "text": _build_prompt_text(role_mode, include_background, excluded_labels),
        "allowed_labels": {
            "infant": list(infant_labels.keys()),
            "caregiver": list(caregiver_labels.keys()),
        },
        "schema_fields": list(_schema_fields(role_mode)),
    }


def make_human_message(modalities: tuple[str, ...], prompt_text: str) -> str:
    prefix = []
    if "video" in modalities:
        prefix.append("<video>")
    if "audio" in modalities:
        prefix.append("<audio>")
    return "\n".join(prefix + [prompt_text]).strip()


def make_response(payload: dict[str, str]) -> str:
    return json.dumps(payload, ensure_ascii=False)
