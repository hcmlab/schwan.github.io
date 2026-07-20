import os
import json
from pathlib import Path
from load_config import FINETUNE_DIR

# Directories (derived from config)
CHUNKS_DIR = FINETUNE_DIR
DATASET_OUT = FINETUNE_DIR / "llamafactory_dataset_full.json"


def create_system_prompt():
    return (
        "You are an expert in the Infant Caregiver Engagement Phases (ICEP) coding system. "
        "Your task is to analyze the caregiver-infant interaction in the provided video and audio "
        "and determine the correct ICEP engagement code. Pay attention to facial expressions, "
        "vocalizations, posture, and gaze."
    )


def create_user_prompt(track_name, start_time, end_time):
    target = "infant's" if "Infant" in track_name else "caregiver's" if "Caregiver" in track_name else "interaction"
    return (
        f"<video><audio>Analyze the {target} engagement in this multimedia segment "
        f"(from {start_time:.1f}s to {end_time:.1f}s in the original session). "
        "Based on the visual and auditory cues, provide the ICEP engagement code and explain your reasoning."
    )


def create_assistant_response(code, label, video_desc, audio_desc):
    if code == "no_annotation":
        return (
            f"**Visual Analysis:** {video_desc}\n\n"
            f"**Audio Analysis:** {audio_desc}\n\n"
            f"Label: {code}"
        )
    else:
        return (
            f"**Code: {code} ({label})**\n\n"
            f"**Visual Analysis:** {video_desc}\n\n"
            f"**Audio Analysis:** {audio_desc}\n\n"
            f"Label: {code}"
        )


def main():
    dataset = []

    session_dirs = sorted([d for d in CHUNKS_DIR.iterdir() if d.is_dir() and (d / "chunks").exists()])
    print(f"Found {len(session_dirs)} session directories with chunks.")

    for session_dir in session_dirs:
        session_name = session_dir.name
        ann_file = session_dir / f"{session_name}_finetune_annotations.json"
        chunks_dir = session_dir / "chunks"

        if not ann_file.exists():
            continue

        with open(ann_file, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        annotations = ann_data.get("annotations", [])

        for idx, ann in enumerate(annotations):
            code = ann["short_code"]
            safe_code = code.replace(" ", "_").replace("/", "_")
            track = ann["track"]
            
            base_name = f"{session_name}_{track}_idx{idx:04d}_{safe_code}"
            
            # Find the shared audio chunk
            audio_file = chunks_dir / f"{base_name}.wav"
            if not audio_file.exists():
                # print(f"Missing audio: {audio_file.name}")
                continue
                
            audio_path_str = str(audio_file.absolute()).replace("\\", "/")

            # Find all video variants for this annotation (Kamera1, Kamera2, Splitscreen, etc)
            video_files = list(chunks_dir.glob(f"{base_name}_*.mp4"))
            
            # Create a separate multimodal entry for EACH camera angle
            for video_file in video_files:
                video_path_str = str(video_file.absolute()).replace("\\", "/")

                entry = {
                    "messages": [
                        {
                            "content": create_system_prompt(),
                            "role": "system"
                        },
                        {
                            "content": create_user_prompt(track, ann["original_start"], ann["original_end"]),
                            "role": "user"
                        },
                        {
                            "content": create_assistant_response(
                                ann["short_code"], 
                                ann["label"], 
                                ann["video_description"], 
                                ann["audio_description"]
                            ),
                            "role": "assistant"
                        }
                    ],
                    "videos": [video_path_str],
                    "audios": [audio_path_str]
                }

                dataset.append(entry)

    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Successfully created LlamaFactory dataset with {len(dataset)} multimodal entries")
    print(f"Target: {DATASET_OUT}")


if __name__ == "__main__":
    main()
