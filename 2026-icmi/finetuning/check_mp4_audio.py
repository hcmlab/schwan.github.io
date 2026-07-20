#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def has_audio_stream(path: Path) -> tuple[bool, str]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to inspect embedded audio streams in mp4 files.")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index,codec_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = stderr.splitlines()[-1] if stderr else f"ffprobe failed with exit code {exc.returncode}"
        return False, f"probe_error: {detail}"
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    if not streams:
        return False, "no audio stream"
    codecs = ",".join(str(stream.get("codec_name", "unknown")) for stream in streams)
    return True, codecs


def iter_mp4s(target: Path):
    if target.is_file():
        yield target
        return
    yield from sorted(target.rglob("*.mp4"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether mp4 clips contain embedded audio streams.")
    parser.add_argument("target", help="Path to an mp4 file or a directory containing mp4 clips.")
    parser.add_argument("--limit", type=int, default=None, help="Only inspect the first N mp4 files.")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only print mp4 paths that are missing audio or could not be probed.",
    )
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    checked = 0
    with_audio = 0
    without_audio = 0
    probe_errors = 0
    for path in iter_mp4s(target):
        ok, detail = has_audio_stream(path)
        checked += 1
        if ok:
            with_audio += 1
            if not args.only_missing:
                print(f"[AUDIO] {path} ({detail})")
        elif detail.startswith("probe_error:"):
            probe_errors += 1
            if args.only_missing:
                print(path)
            else:
                print(f"[PROBE_ERROR] {path} ({detail.removeprefix('probe_error: ').strip()})")
        else:
            without_audio += 1
            if args.only_missing:
                print(path)
            else:
                print(f"[NO_AUDIO] {path}")
        if args.limit is not None and checked >= args.limit:
            break

    print()
    print(f"checked={checked}")
    print(f"with_audio={with_audio}")
    print(f"without_audio={without_audio}")
    print(f"probe_errors={probe_errors}")
    return 0 if without_audio == 0 and probe_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
