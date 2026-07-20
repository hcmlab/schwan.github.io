#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FT_DIR="${FT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=/dev/null
source "${FT_DIR}/slurm/common.sh"

REPO_ROOT="$(cd "${FT_DIR}/.." && pwd)"
swan_load_env "$REPO_ROOT"

PROFILE="${PROFILE:-slurm_h100}"
DATASET_SPEC="${DATASET_SPEC:-icep_no_bg_joint_omni}"
MODEL="${MODEL:-qwen25_omni_7b}"
APPLY="${APPLY:-1}"
CHECK_TEST="${CHECK_TEST:-1}"

CONDA_ACTIVATE="$(swan_conda_activate "$PROFILE")"
PYTHON_BIN="$(swan_python_bin "$PROFILE")"
if [ -f "$CONDA_ACTIVATE" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_ACTIVATE"
  conda activate "${CONDA_ENV_NAME:-swan2}"
fi

export PYTHONNOUSERSITE=1
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

swan_log_runtime "$PROFILE" "$FT_DIR"

echo "================================================================"
echo "Cleaning corrupted omni eval data"
echo "Profile: ${PROFILE}"
echo "Dataset spec: ${DATASET_SPEC}"
echo "Model: ${MODEL}"
echo "Apply changes: ${APPLY}"
echo "Check test split: ${CHECK_TEST}"
echo "Node: $(hostname)"
echo "================================================================"

"$PYTHON_BIN" - <<'PY'
import json
import subprocess
from pathlib import Path

from swan_ft.config import RunOptions, RunSpec, load_dataset_spec, load_model_spec, load_profile
from swan_ft.paths import resolver_for

profile_id = Path.cwd()  # placeholder to keep imports above grouped

import os
profile = load_profile(os.environ.get("PROFILE", "slurm_h100"))
dataset_spec = load_dataset_spec(os.environ.get("DATASET_SPEC", "icep_no_bg_joint_omni"))
model_spec = load_model_spec(os.environ.get("MODEL", "qwen25_omni_7b"))
options = RunOptions(folds=(0, 1, 2, 3, 4))
run_spec = RunSpec(profile=profile, dataset_spec=dataset_spec, model_spec=model_spec, options=options)
resolver = resolver_for(run_spec)
variant_root = resolver.variant_root(dataset_spec, options)
apply_changes = os.environ.get("APPLY", "1") == "1"
check_test = os.environ.get("CHECK_TEST", "1") == "1"

reports_root = variant_root / "media_audit_reports"
reports_root.mkdir(parents=True, exist_ok=True)


def ffprobe_streams(path: Path):
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    streams = payload.get("streams", [])
    has_video = any(stream.get("codec_type") == "video" for stream in streams)
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    return has_video, has_audio


def ffmpeg_decode_ok(path: Path):
    video_cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"]
    audio_cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-t", "1", "-f", "null", "-"]
    subprocess.run(video_cmd, check=True, capture_output=True, text=True)
    subprocess.run(audio_cmd, check=True, capture_output=True, text=True)


def sample_media_path(sample):
    videos = sample.get("videos") or []
    return Path(videos[0]) if videos else None


def sample_session(sample):
    for key in ("session_id", "session", "id"):
        value = sample.get(key)
        if value:
            return value
    conv = sample.get("conversations") or []
    if conv:
        return conv[0].get("value", "")[:120]
    return None


def audit_split(path: Path):
    if not path.exists():
        print(f"SKIP missing split: {path}")
        return

    original = json.loads(path.read_text(encoding="utf-8"))
    clean = []
    bad = []

    for index, sample in enumerate(original):
        media_path = sample_media_path(sample)
        record = {
            "index": index,
            "session": sample_session(sample),
            "path": str(media_path) if media_path is not None else None,
        }
        if media_path is None:
            record["reason"] = "missing_videos"
            bad.append(record)
            print(f"DROP {path.name} idx={index} reason=missing_videos")
            continue
        if not media_path.exists():
            record["reason"] = "video_missing"
            bad.append(record)
            print(f"DROP {path.name} idx={index} reason=video_missing path={media_path}")
            continue
        try:
            has_video, has_audio = ffprobe_streams(media_path)
        except Exception as exc:
            record["reason"] = "ffprobe_failed"
            record["error"] = str(exc)
            bad.append(record)
            print(f"DROP {path.name} idx={index} reason=ffprobe_failed path={media_path}")
            continue
        if not has_video:
            record["reason"] = "no_video_stream"
            bad.append(record)
            print(f"DROP {path.name} idx={index} reason=no_video_stream path={media_path}")
            continue
        if not has_audio:
            record["reason"] = "no_audio_stream"
            bad.append(record)
            print(f"DROP {path.name} idx={index} reason=no_audio_stream path={media_path}")
            continue
        try:
            ffmpeg_decode_ok(media_path)
        except Exception as exc:
            record["reason"] = "ffmpeg_decode_failed"
            record["error"] = str(exc)
            bad.append(record)
            print(f"DROP {path.name} idx={index} reason=ffmpeg_decode_failed path={media_path}")
            continue
        clean.append(sample)

    backup_path = path.with_suffix(path.suffix + ".bak")
    report_path = reports_root / f"{path.stem}_bad_samples.json"
    summary_path = reports_root / f"{path.stem}_summary.json"

    report_path.write_text(json.dumps(bad, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "split": str(path),
        "kept": len(clean),
        "removed": len(bad),
        "total": len(original),
        "backup": str(backup_path) if apply_changes else None,
        "report": str(report_path),
        "apply_changes": apply_changes,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if apply_changes:
        if not backup_path.exists():
            backup_path.write_text(json.dumps(original, indent=2, ensure_ascii=False), encoding="utf-8")
        path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"REWROTE {path} kept={len(clean)} removed={len(bad)} backup={backup_path}")
    else:
        print(f"DRY-RUN {path} would keep={len(clean)} remove={len(bad)}")


for fold in range(5):
    audit_split(variant_root / f"fold_{fold}_val.json")
if check_test:
    audit_split(variant_root / "test.json")

print(f"Reports written to: {reports_root}")
PY
