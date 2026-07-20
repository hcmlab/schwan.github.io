from __future__ import annotations

import json
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path


def get_session_info(root: Path) -> list[dict[str, object]]:
    sessions: list[dict[str, object]] = []
    for name in sorted(os.listdir(root)):
        session_path = root / name
        if not session_path.is_dir():
            continue
        if "_MUC_" in name:
            location = "MUC"
        elif "_HD_" in name:
            location = "HD"
        else:
            continue

        duration = None
        meta_path = session_path / "metadata.json"
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                duration = float(meta.get("technical_metadata", {}).get("duration", 0))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                duration = None
        if not duration:
            duration = probe_video_duration(session_path)

        sessions.append(
            {
                "session_id": name,
                "path": str(session_path),
                "location": location,
                "duration": duration or 0.0,
            }
        )
    return sessions


def probe_video_duration(session_path: Path) -> float | None:
    videos = list(session_path.glob("*.mp4"))
    if not videos:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(videos[0]),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def assign_duration_quartile(sessions: list[dict[str, object]]) -> None:
    durations = sorted(float(session["duration"]) for session in sessions if float(session["duration"]) > 0)
    if len(durations) < 4:
        for session in sessions:
            session["quartile"] = 0
        return
    q1 = durations[len(durations) // 4]
    q2 = durations[len(durations) // 2]
    q3 = durations[3 * len(durations) // 4]
    for session in sessions:
        duration = float(session["duration"])
        if duration <= q1:
            session["quartile"] = 0
        elif duration <= q2:
            session["quartile"] = 1
        elif duration <= q3:
            session["quartile"] = 2
        else:
            session["quartile"] = 3


def _group_sessions(sessions: list[dict[str, object]]) -> dict[tuple[str, int], list[dict[str, object]]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for session in sessions:
        groups[(str(session["location"]), int(session.get("quartile", 0)))].append(session)
    return groups


def split_train_dev_test(
    sessions: list[dict[str, object]],
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if test_fraction == 0.0:
        return list(sessions), []

    rng = random.Random(seed)
    train_dev: list[dict[str, object]] = []
    test: list[dict[str, object]] = []
    for key in sorted(_group_sessions(sessions).keys()):
        group = list(_group_sessions(sessions)[key])
        rng.shuffle(group)
        n_group = len(group)
        n_test = int(round(n_group * test_fraction))
        if n_group >= 2 and n_test == 0:
            n_test = 1
        if n_group >= 2 and n_test >= n_group:
            n_test = n_group - 1
        test.extend(group[:n_test])
        train_dev.extend(group[n_test:])
    return sorted(train_dev, key=lambda item: str(item["session_id"])), sorted(test, key=lambda item: str(item["session_id"]))


def create_stratified_folds(sessions: list[dict[str, object]], n_folds: int = 5, seed: int = 42) -> list[list[str]]:
    rng = random.Random(seed)
    folds = [[] for _ in range(n_folds)]
    fold_idx = 0
    for key in sorted(_group_sessions(sessions).keys()):
        group = list(_group_sessions(sessions)[key])
        rng.shuffle(group)
        for session in group:
            folds[fold_idx % n_folds].append(str(session["session_id"]))
            fold_idx += 1
    return [sorted(fold) for fold in folds]


def write_folds(
    root: Path,
    output_path: Path,
    n_folds: int = 5,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    sessions = get_session_info(root)
    assign_duration_quartile(sessions)
    train_dev_sessions, test_sessions = split_train_dev_test(sessions, test_fraction=test_fraction, seed=seed)
    folds = create_stratified_folds(train_dev_sessions, n_folds=n_folds, seed=seed)
    session_map = {str(session["session_id"]): session for session in sessions}
    train_dev_ids = sorted(str(session["session_id"]) for session in train_dev_sessions)
    test_ids = sorted(str(session["session_id"]) for session in test_sessions)

    payload: dict[str, object] = {
        "n_folds": n_folds,
        "seed": seed,
        "test_fraction": test_fraction,
        "total_sessions": len(sessions),
        "train_dev_sessions": train_dev_ids,
        "test_sessions": test_ids,
        "train_dev_count": len(train_dev_ids),
        "test_count": len(test_ids),
        "folds": {},
    }
    for idx, fold_sessions in enumerate(folds):
        fold_muc = sum(1 for sid in fold_sessions if session_map[sid]["location"] == "MUC")
        fold_hd = sum(1 for sid in fold_sessions if session_map[sid]["location"] == "HD")
        payload["folds"][str(idx)] = {
            "val_sessions": sorted(fold_sessions),
            "train_sessions": sorted([sid for sid in train_dev_ids if sid not in fold_sessions]),
            "val_count": len(fold_sessions),
            "val_muc": fold_muc,
            "val_hd": fold_hd,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
