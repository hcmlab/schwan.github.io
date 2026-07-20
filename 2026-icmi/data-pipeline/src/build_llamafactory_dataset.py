"""
build_llamafactory_dataset.py
=============================
Step E: Generate a LlamaFactory-ready multimodal ShareGPT JSON file
for Qwen2.5-Omni ICEP fine-tuning.

Each training example is one video chunk with:
  - ShareGPT conversations (system + human + assistant)
  - A `videos` list pointing to the MP4 path
  - No separate `audios` — Qwen2.5-Omni loads audio from MP4 via
    use_audio_in_video=True

Prompt design:
  Human:     <video> + instruction
  Assistant: JSON with track, short_code, label, rationale
             (rationale = full_description + video_description + audio_description
              from the enriched annotation)

Input env vars:
    DATA_ROOT, OUT_DIR

Output:
    {OUT_DIR}/llamafactory/schwan_icep_sft.json
    {OUT_DIR}/llamafactory/dataset_info.json  (copy from configs/)
"""

import json
import os
import shutil
from pathlib import Path


def get_env_paths():
    data_root = Path(os.environ.get("DATA_ROOT", "/mnt/dataset-swan/data/Schwan_T3_FineTune"))
    out_dir = Path(os.environ.get("OUT_DIR", "gpu_server/data"))
    return data_root, out_dir


# ── System prompt ───────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in the Infant Caregiver Engagement Phases (ICEP) coding system. "
    "Your task is to analyze a short video clip of a caregiver-infant interaction and "
    "predict the correct ICEP engagement code. Pay close attention to facial expressions, "
    "vocalizations, posture, gaze direction, and the overall affective tone. "
    "Respond only with a JSON object."
)


# ── Prompt / response builders ──────────────────────────────────

def build_human_message(track: str, camera_view: str | None = None) -> str:
    """Build the human prompt with a <video> placeholder and camera context."""
    if "Infant" in track:
        target = "infant's"
    elif "Caregiver" in track:
        target = "caregiver's"
    else:
        target = "participant's"

    # Camera view context sentence
    if camera_view == "splitscreen_hd":
        cam_context = (
            "This is a single split-screen recording showing both the infant "
            "and caregiver from a fixed HD camera position."
        )
    elif camera_view == "splitscreen":
        cam_context = (
            "This is a combined split-screen grid view showing all available "
            "camera angles (Kamera 1–4) simultaneously."
        )
    elif camera_view and camera_view.startswith("kamera"):
        cam_num = camera_view.replace("kamera", "")
        cam_context = (
            f"This is an individual camera angle (Kamera {cam_num}) from a "
            f"multi-camera recording setup."
        )
    else:
        cam_context = ""

    parts = [
        "<video>",
        cam_context,
        f"Predict the ICEP engagement label for the {target} behavior in this clip. "
        f"Respond with a JSON object containing: track, short_code, label, and rationale.",
    ]

    return "\n".join(p for p in parts if p)



def build_assistant_response(
    ann: dict,
    track: str,
    short_code: str,
) -> str:
    """
    Build the ground-truth assistant response as a JSON string.

    Supervision rule:
      - If track == Caregiver_Engagement → set caregiver_phase = short_code,
        infant_phase = null
      - If track == Infant_Engagement   → set infant_phase   = short_code,
        caregiver_phase = null
    """
    # Build rationale from annotation descriptions
    rationale_parts = []
    if ann.get("full_description"):
        rationale_parts.append(ann["full_description"])
    if ann.get("video_description"):
        rationale_parts.append(f"Visual cues: {ann['video_description']}")
    if ann.get("audio_description"):
        rationale_parts.append(f"Auditory cues: {ann['audio_description']}")

    rationale = " ".join(rationale_parts) if rationale_parts else ""

    response_obj = {
        "track": track,
        "short_code": short_code,
        "label": ann.get("label", short_code),
    }

    # Set the supervised phase, leave the other as null
    if "Caregiver" in track:
        response_obj["caregiver_phase"] = short_code
        response_obj["infant_phase"] = None
    elif "Infant" in track:
        response_obj["infant_phase"] = short_code
        response_obj["caregiver_phase"] = None

    response_obj["rationale"] = rationale

    return json.dumps(response_obj, ensure_ascii=False)


# ── Annotation loader ──────────────────────────────────────────

def load_session_annotations(session_dir: Path) -> dict | None:
    """
    Load the enriched annotation JSON for a session.
    Returns {idx -> annotation_dict} keyed by annotation index.
    """
    session_name = session_dir.name
    ann_file = session_dir / f"{session_name}_finetune_annotations.json"
    if not ann_file.exists():
        return None

    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build a lookup by (track, idx)
    annotations = data.get("annotations", [])
    lookup = {}
    for i, ann in enumerate(annotations):
        key = (ann.get("track", ""), i)
        lookup[key] = ann

    return lookup


# ── Main pipeline ───────────────────────────────────────────────

def main():
    data_root, out_dir = get_env_paths()

    # Load manifest
    manifest_file = out_dir / "chunk_manifest.jsonl"
    if not manifest_file.exists():
        print(f"Error: {manifest_file} not found — run build_manifest.py first.")
        return

    manifest = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest.append(json.loads(line))

    print(f"Loaded {len(manifest)} chunk manifest rows")

    # Load train split
    train_file = out_dir / "splits" / "train_sessions.json"
    if not train_file.exists():
        print(f"Error: {train_file} not found — run split_sessions.py first.")
        return

    with open(train_file, "r", encoding="utf-8") as f:
        train_sessions = set(json.load(f))

    print(f"Train split: {len(train_sessions)} sessions")

    # Load val split
    val_file = out_dir / "splits" / "val_sessions.json"
    if not val_file.exists():
        print(f"Error: {val_file} not found — run split_sessions.py first.")
        return

    with open(val_file, "r", encoding="utf-8") as f:
        val_sessions = set(json.load(f))

    print(f"Val split: {len(val_sessions)} sessions")

    # Load test split
    test_file = out_dir / "splits" / "test_sessions.json"
    if not test_file.exists():
        print(f"Error: {test_file} not found — run split_sessions.py first.")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        test_sessions = set(json.load(f))

    print(f"Test split: {len(test_sessions)} sessions")

    # Build annotation lookups per session (cached)
    ann_cache: dict[str, dict | None] = {}

    # Filter flags
    include_no_annotation = os.environ.get("INCLUDE_NO_ANNOTATION", "0") == "1"

    dataset_train = []
    dataset_val = []
    dataset_test = []
    corrupted_data = []
    
    skipped_no_video = 0
    skipped_no_annotation_filter = 0
    skipped_missing_split = 0
    skipped_no_ann_file = 0

    for row in manifest:
        session_id = row["session_id"]
        track = row["track"]
        idx = row["idx"]
        short_code = row["short_code"]
        video_path = row.get("video_path")
        camera_view = row.get("camera_view")

        # Determine split target
        if session_id in train_sessions:
            target_list = dataset_train
        elif session_id in val_sessions:
            target_list = dataset_val
        elif session_id in test_sessions:
            target_list = dataset_test
        else:
            skipped_missing_split += 1
            continue

        # Skip chunks without video
        if not video_path:
            skipped_no_video += 1
            continue

        # Skip corrupted/unreadable videos (ffprobe confirmed duration <= 0.1s OR PyAV failed)
        duration_sec = row.get("duration_sec")
        is_corrupted = row.get("corrupted", False)
        if is_corrupted or (duration_sec is not None and duration_sec <= 0.1):
            corrupted_data.append(row)
            continue

        # Skip no_annotation by default
        if short_code == "no_annotation" and not include_no_annotation:
            skipped_no_annotation_filter += 1
            continue

        # Load annotation data for rationale
        if session_id not in ann_cache:
            session_dir = data_root / session_id
            ann_cache[session_id] = load_session_annotations(session_dir)

        ann_lookup = ann_cache[session_id]
        if ann_lookup is None:
            skipped_no_ann_file += 1
            continue

        # Find matching annotation by (track, idx)
        ann = ann_lookup.get((track, idx))
        if ann is None:
            # Fallback: match by short_code if idx doesn't align
            ann = {
                "short_code": short_code,
                "label": short_code,
                "full_description": "",
                "video_description": "",
                "audio_description": "",
            }

        # Build ShareGPT entry
        entry = {
            "conversations": [
                {"from": "human", "value": build_human_message(track, camera_view)},
                {"from": "gpt", "value": build_assistant_response(ann, track, short_code)},
            ],
            "system": SYSTEM_PROMPT,
            "videos": [video_path],
        }

        target_list.append(entry)


    # Write output
    llama_dir = out_dir / "llamafactory"
    llama_dir.mkdir(parents=True, exist_ok=True)

    out_corrupt = llama_dir / "corrupt_data.json"
    with open(out_corrupt, "w", encoding="utf-8") as f:
        json.dump(corrupted_data, f, indent=2, ensure_ascii=False)

    out_train = llama_dir / "schwan_icep_sft_train.json"
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(dataset_train, f, indent=2, ensure_ascii=False)

    out_val = llama_dir / "schwan_icep_sft_val.json"
    with open(out_val, "w", encoding="utf-8") as f:
        json.dump(dataset_val, f, indent=2, ensure_ascii=False)

    out_test = llama_dir / "schwan_icep_sft_test.json"
    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(dataset_test, f, indent=2, ensure_ascii=False)

    print(f"Generated LlamaFactory datasets:")
    print(f"  Train chunks: {len(dataset_train)} (saved to {out_train})")
    print(f"  Val chunks:   {len(dataset_val)} (saved to {out_val})")
    print(f"  Test chunks:  {len(dataset_test)} (saved to {out_test})")
    print(f"\nSkipped (Missing Split mapping): {skipped_missing_split}")

    # Copy dataset_info.json from configs/
    script_dir = Path(__file__).resolve().parent.parent
    src_info = script_dir / "configs" / "dataset_info.json"
    dst_info = llama_dir / "dataset_info.json"
    if src_info.exists():
        shutil.copy2(src_info, dst_info)
        print(f"Copied dataset_info.json → {dst_info}")
    else:
        print(f"[WARN] configs/dataset_info.json not found at {src_info}")

    # Summary
    print(f"\n{'='*60}")
    print(f"LlamaFactory datasets created!")
    print(f"  Train:     {out_train}")
    print(f"  Val:       {out_val}")
    print(f"  Test:      {out_test}")
    print(f"  Corrupted: {out_corrupt}")
    print(f"  Total train examples            : {len(dataset_train)}")
    print(f"  Total val examples              : {len(dataset_val)}")
    print(f"  Total test examples             : {len(dataset_test)}")
    print(f"  Corrupted videos skipped        : {len(corrupted_data)}")
    print(f"  Skipped (missing split)         : {skipped_missing_split}")
    print(f"  Skipped (no video)              : {skipped_no_video}")
    print(f"  Skipped (no_annotation filtered): {skipped_no_annotation_filter}")
    print(f"  Skipped (no annotation file)    : {skipped_no_ann_file}")

    if dataset_train:
        print(f"\nSample Train Entry:")
        print(json.dumps(dataset_train[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
