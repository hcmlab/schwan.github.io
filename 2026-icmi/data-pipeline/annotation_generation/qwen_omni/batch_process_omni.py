import os
import subprocess
import glob
import time
import json
import logging

# Configure path
DATA_ROOT = "/mnt/dataset-swan/data/Schwan_T3_Clean"
SCRIPT_PATH = "/home/hcai-admin/Documents/GitHub/schwan/run_omni_analysis.sh"
LOG_FILE = "batch_omni_log.txt"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def is_container_running(container_name="schwan_omni_run"):
    """Check if the docker container is currently running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={container_name}"],
            capture_output=True, text=True
        )
        return bool(result.stdout.strip())
    except Exception:
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Pass dry-run down to container")
    args = parser.parse_args()
    
    print("Starting batch Omni analysis monitor...")
    logging.info("Starting batch Omni analysis monitor...")
    
    # Wait for any existing analysis to finish
    if is_container_running():
        print("Detected running analysis (schwan_omni_run). Waiting for it to finish...")
        logging.info("Detected running analysis. Waiting...")
        while is_container_running():
            time.sleep(60) 
        print("Previous analysis finished. Starting batch processing.")
        logging.info("Previous analysis finished.")

    # Get all sessions
    sessions = sorted([d for d in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(d)])
    
    for i, session_path in enumerate(sessions):
        session_name = os.path.basename(session_path)
        output_file = os.path.join(session_path, "vlm_annotations", "omni_analysis.json")
        
        # Determine if we should process or skip
        if os.path.exists(output_file) and not args.dry_run:
            print(f"[{i+1}/{len(sessions)}] Skipping {session_name} (Output exists)")
            logging.info(f"Skipping {session_name} - Output exists")
            continue
            
        print(f"[{i+1}/{len(sessions)}] Processing {session_name}...")
        logging.info(f"Processing {session_name}...")
        
        start_time = time.time()
        try:
            # Force remove an old container state just in case
            subprocess.run(["docker", "rm", "-f", "schwan_omni_run"], 
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            
            # Using --limit 9999 to process all
            cmd = [SCRIPT_PATH, session_path, "--limit", "9999"]
            if args.dry_run:
                cmd.append("--dry-run")
                cmd[3] = "1" # Limit to 1 for dry runs to save time
                
            # Run and capture output to log
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            
            duration = time.time() - start_time
            print(f"Success: {session_name} ({duration:.1f}s)")
            logging.info(f"Success: {session_name} ({duration:.1f}s)")
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"Failed: {session_name} ({duration:.1f}s)")
            logging.error(f"Failed: {session_name} ({duration:.1f}s)")
            logging.error(f"Stderr: {e.stderr}")
        except Exception as e:
            print(f"Error: {session_name} - {str(e)}")
            logging.error(f"Error: {session_name} - {str(e)}")
            
        time.sleep(2)

if __name__ == "__main__":
    main()
