import os
import argparse
import time
import logging
import subprocess
import json
import re
from tqdm import tqdm
from dotenv import load_dotenv

# Import custom modules
import alm.model as alm
import data.processing as data_processing
import librosa

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Setup Logger
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'alm_processing.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Enable Transformers logging to show download progress
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()

def extract_audio_from_video(video_path, audio_path):
    if os.path.exists(audio_path):
        return True

    print(f"Extracting audio from {os.path.basename(video_path)}...", flush=True)
    command = [
        "ffmpeg", "-y", "-hwaccel", "auto", 
        "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        audio_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        return False

def parse_alm_response(response_text):
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return data
    except Exception:
        pass
    
    return {"raw": response_text}

def process_session(session_path, model, processor, limit=0, dry_run=False):
    # 1. Find Files
    video_path, anno_path = data_processing.find_session_files(session_path)
    if not video_path: return

    # 2. Extract Audio
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    if not dry_run:
        if not extract_audio_from_video(video_path, audio_path):
            logging.error(f"Failed to extract audio for {session_path}")
            return

    # 3. Load Events
    if not anno_path: return
    events = data_processing.load_events(anno_path)
    if not events: return
    
    # 4. Setup Output
    output_path = os.path.join(session_path, 'alm_annotations.json')
    
    events_to_process = events
    results = []
    
    if limit > 0:
        events_to_process = events_to_process[:limit]
        print(f"  Limiting to first {limit} events.", flush=True)

    # 5. Analyze Loop
    start_time = time.time()
    processed_count = 0
    
    # Get total duration of audio file to prevent out of bounds
    try:
        total_audio_duration = librosa.get_duration(filename=audio_path)
    except Exception as e:
        logging.error(f"Could not determine duration of {audio_path}: {e}")
        return

    MIN_CONTEXT_DURATION = 4.0

    for event in tqdm(events_to_process, desc="  Analyzing Audio Chunks", leave=False):
        processed_count += 1
        event_start = event['start']

        event_end = event['end']
        duration = event_end - event_start
        role = event.get('role', 'infant') # Default to infant if unknown
        
        if duration < 0.1:
            logging.warning(f"Skipping extremely short event at {event_start}: {duration}s")
            continue

        if dry_run:
            continue
            
        try:
            # Pass role to model
            metrics, response = alm.analyze_audio_chunk(model, processor, audio_path, event_start, event_end, role=role)
            
            prediction = parse_alm_response(response)

            result_entry = {
                "timestamp_start": event_start,
                "timestamp_end": event_end,
                "duration": duration,
                "human_code_ref": event['original_code'],
                "role": role,
                "alm_prediction_code": prediction.get(f"{role}_code", ""),
                "alm_reasoning": prediction.get("reasoning", ""),
                "metrics": metrics,
                "model_version": "Qwen2-Audio-7B-Instruct",
                "raw_response": response if "reasoning" not in prediction else None
            }
            
            results.append(result_entry)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error processing {os.path.basename(session_path)} at {event_start}-{event_end}: {e}")
            print(f"    Error at {event_start}-{event_end}: {e}", flush=True)

    end_time = time.time()
    total_duration = end_time - start_time
    avg_time = total_duration / processed_count if processed_count > 0 else 0
    
    log_msg = (f"Session: {os.path.basename(session_path)} | "
               f"Events: {len(events_to_process)} | "
               f"Processed: {processed_count} | "
               f"Time: {total_duration:.2f}s | "
               f"Avg: {avg_time:.2f}s/event | "
               f"DryRun: {dry_run}")
    
    logging.info(log_msg)
    print(f"\n[Performance] {log_msg}\n", flush=True)

def main(root_dir, target_session=None, limit=0, dry_run=False):
    print(f"Scanning directory: {root_dir}", flush=True)
    
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Filter out redundant_annotations as per user request
    subdirs = [d for d in subdirs if "redundant_annotations" not in os.path.basename(d)]
    
    if target_session:
        subdirs = [d for d in subdirs if target_session in os.path.basename(d)]
        if not subdirs:
            print(f"Error: Session '{target_session}' not found in {root_dir}", flush=True)
            return
    
    print(f"Found {len(subdirs)} sessions to process.", flush=True)
    
    if subdirs:
        if not dry_run:
            model, processor = alm.load_model()
        else:
            model, processor = None, None
            print("Dry run: Model loading skipped.", flush=True)

        for session_dir in tqdm(subdirs, desc="Overall Progress", unit="session"):
            process_session(session_dir, model, processor, limit, dry_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"X:\data\Schwan_T3_Clean", help="Root directory of the clean dataset")
    parser.add_argument("--session", help="Specific session name to process (optional)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of events to process per session (0 for all)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate run without model inference")
    args = parser.parse_args()

    if not os.path.exists(args.root):
        print(f"Error: Root directory {args.root} does not exist or is not accessible.", flush=True)
    else:
        main(args.root, args.session, args.limit, args.dry_run)
