import os
# Fix for OMP error #15 (conflicting OpenMP libraries between torch/decord/transformers)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time
import logging
from tqdm import tqdm
from dotenv import load_dotenv

# Import custom modules
import vlm.model as vlm
import data.processing as data_processing

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Setup Logger
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'vlm_processing.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# logging.getLogger('').addHandler(console) # Avoid duplicate if reusing existing handlers

def process_session(session_path, model, processor, limit=0, dry_run=False):
    # 1. Find Files
    video_path, anno_path = data_processing.find_session_files(session_path)
    if not video_path: return

    # 2. Extract Events (skip if no annotation found, though find_session_files checks this)
    if not anno_path: return

    events = data_processing.load_events(anno_path)
    if not events: return

    # 3. Setup Output
    output_path = data_processing.setup_output(session_path)
    
    # 4. Filter Already Processed
    events_to_process, results = data_processing.filter_processed_events(events, output_path)

    if limit > 0:
        events_to_process = events_to_process[:limit]
        print(f"  Limiting to first {limit} events.", flush=True)
    
    # 5. Analyze Loop
    start_time = time.time()
    processed_count = 0
    
    for event in tqdm(events_to_process, desc="  Analyzing Chunks"):
        processed_count += 1
        event_start = event['start']
        event_end = event['end']
        duration = event_end - event_start
        
        if duration < 0.5: continue 

        if dry_run:
            continue
            
        try:
            metrics, response = vlm.analyze_chunk(model, processor, video_path, event_start, event_end)
            
            clean_json, prediction = data_processing.parse_vlm_response(response)

            result_entry = {
                "timestamp_start": event_start,
                "timestamp_end": event_end,
                "duration": duration,
                "human_code_ref": event['original_code'],
                "vlm_infant_description": prediction.get("infant_description", ""),
                "vlm_caregiver_description": prediction.get("caregiver_description", ""),
                "vlm_prediction_infant": prediction.get("infant_code", ""),
                "vlm_prediction_caregiver": prediction.get("caregiver_code", ""),
                "metrics": metrics,
                "model_version": "Qwen2.5-VL-7B-Instruct",
                "raw_response": clean_json if "infant_description" not in prediction else None
            }
            
            results.append(result_entry)
            
            # Save incrementally
            data_processing.save_results(results, output_path)
                
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
    
    if target_session:
        subdirs = [d for d in subdirs if target_session in os.path.basename(d)]
        if not subdirs:
            print(f"Error: Session '{target_session}' not found in {root_dir}", flush=True)
            return
    
    print(f"Found {len(subdirs)} sessions to process.", flush=True)
    
    if subdirs:
        if not dry_run:
            model, processor = vlm.load_model()
        else:
            model, processor = None, None
            print("Dry run: Model loading skipped.", flush=True)

        for session_dir in subdirs:
            process_session(session_dir, model, processor, limit, dry_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"X:\data\Schwan_T3_Clean", help="Root directory of the clean dataset")
    parser.add_argument("--session", help="Specific session name to process (optional)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of events to process (0 for all)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate run without model inference")
    args = parser.parse_args()

    if not os.path.exists(args.root):
        print(f"Error: Root directory {args.root} does not exist or is not accessible.", flush=True)
    else:
        main(args.root, args.session, args.limit, args.dry_run)
