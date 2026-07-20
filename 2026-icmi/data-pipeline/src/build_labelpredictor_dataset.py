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
    out_dir = Path(os.environ.get("OUT_DIR", "../data"))
    
    print(f"[Python] Parsing DATA_ROOT: {data_root}")
    print(f"[Python] Parsing OUT_DIR: {out_dir}")
    
    return data_root, out_dir


# ── System prompt ───────────────────────────────────────────────

def get_icep_codes():
    script_dir = Path(__file__).resolve().parent.parent
    with open(script_dir / "data_prep" / "icep_codes.json", "r", encoding="utf-8") as f:
        return json.load(f)

def build_system_prompt(track: str, icep_codes: dict) -> str:
    base = (
        "You are an expert in the Infant Caregiver Engagement Phases (ICEP) coding system. "
        "Your task is to analyze a video clip of a caregiver-infant interaction and "
        "predict the exact ICEP engagement short code.\n\n"
        "Here are the specific definitions for the possible codes:\n"
    )
    if "Infant" in track:
        codes = icep_codes.get("infant_codes", {})
    else:
        codes = icep_codes.get("caregiver_codes", {})
        
    for code, info in codes.items():
        base += f"- {code} ({info['label']}): {info['full_description']}\n"
    
    valid_codes = list(codes.keys())
    base += f"\nYou must choose EXACTLY ONE of the following valid codes: {valid_codes}. Respond ONLY with the exact short code. Do not output any other text or formatting."
    return base


# ── Prompt / response builders ──────────────────────────────────

def build_human_message(track: str) -> str:
    """Build the human prompt with a <video> placeholder."""
    if "Infant" in track:
        target = "infant's"
    elif "Caregiver" in track:
        target = "caregiver's"
    else:
        target = "participant's"

    return f"<video>\nPredict the ICEP engagement short code for the {target} behavior in this clip."

def build_assistant_response(short_code: str) -> str:
    """
    Build the ground-truth assistant response as just the short code string.
    """
    return short_code

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
    icep_codes = get_icep_codes()

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
                {"from": "human", "value": build_human_message(track)},
                {"from": "gpt", "value": build_assistant_response(short_code)},
            ],
            "system": build_system_prompt(track, icep_codes),
            "videos": [video_path],
        }

        target_list.append(entry)


    # Write output
    llama_dir = out_dir / "llamafactory"
    llama_dir.mkdir(parents=True, exist_ok=True)

    out_corrupt = llama_dir / "corrupt_data_labelpredictor.json"
    with open(out_corrupt, "w", encoding="utf-8") as f:
        json.dump(corrupted_data, f, indent=2, ensure_ascii=False)

    out_train = llama_dir / "schwan_icep_labelpredictor_train.json"
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(dataset_train, f, indent=2, ensure_ascii=False)

    out_val = llama_dir / "schwan_icep_labelpredictor_val.json"
    with open(out_val, "w", encoding="utf-8") as f:
        json.dump(dataset_val, f, indent=2, ensure_ascii=False)

    out_test = llama_dir / "schwan_icep_labelpredictor_test.json"
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
