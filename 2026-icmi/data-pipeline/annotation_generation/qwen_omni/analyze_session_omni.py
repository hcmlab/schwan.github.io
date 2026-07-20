import os
import subprocess
import json
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
import logging
import traceback
import sys
import gc
import re
import time

# Append path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from icep_definitions import get_annotation_info

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vlm_omni_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- SYSTEM PROMPTS ---
SYSTEM_PROMPT_INFANT = """
You are an expert behavior coder for Mother-Infant interactions using the ICEP-R (Infant and Caregiver Engagement Phases - Revised) scheme.
You are watching a video chunk containing synchronized AUDIO and VIDEO.

**Goal:** Analyze the **INFANT's** facial expressions, bodily movements, and vocalizations. 

**INFANT Codes:**
- `ineg`: Negative (distress, crying, high-intensity fussing, screaming)
- `ipro`: Protest (fussing, whining, complaining, negative face)
- `iwit`: Withdrawn (averting gaze, pulling away, faint whimpering, silence while disengaged)
- `inon`: Object/Env Engagement (babbling directed at objects, looking at environment, ignoring caregiver)
- `ineu`: Neutral (neutral face, checking in with caregiver quietly)
- `ipos`: Positive (smiling, laughing, happy squealing)
- `islp`: Sleeping (rhythmic breathing, closed eyes)
- `iusc`: Unscorable

**Instructions:**
1. Focus ENTIRELY on the infant.
2. Integrate what you see (visual cues) with what you hear (audio cues).
3. Return ONLY a JSON object evaluating the infant.

**Output Format (JSON Only):**
{
  "infant_code": "code",
  "visual_cues": "What the infant is doing visually...",
  "audio_cues": "What the infant is doing acoustically..."
}
"""

SYSTEM_PROMPT_CAREGIVER = """
You are an expert behavior coder for Mother-Infant interactions using the ICEP-R (Infant and Caregiver Engagement Phases - Revised) scheme.
You are watching a video chunk containing synchronized AUDIO and VIDEO.

**Goal:** Analyze the **CAREGIVER's** tone of voice, verbal content (spoken in GERMAN), facial expressions, and physical gestures.

**CAREGIVER Codes:**
- `cneg`: Negative (hostile, critical, sharp tone, yelling, expressionless/cold voice)
- `cwit`: Withdrawn (ignoring infant, flat affect, silence, no Motherese)
- `cint`: Intrusive (forcing attention, overwhelming vocalizations, too loud/fast)
- `chos`: Hostile (aggressive, angry tone)
- `cnon`: Non-Infant Focused (talking to others, distracted)
- `cneu`: Neutral (normal adult speech, simple narration, neutral affect)
- `cpos`: Positive ("Motherese" / high pitch, warm smile, praise, singing, warmth)
- `cpvc`: Physical Control (manually moving the infant)
- `ctch`: Touch (gentle contact)

**Instructions:**
1. Focus ENTIRELY on the caregiver. The language spoken is German.
2. Integrate what you see (visual cues) with what you hear (prosody/tone audio cues).
3. Return ONLY a JSON object evaluating the caregiver.

**Output Format (JSON Only):**
{
  "caregiver_code": "code",
  "visual_cues": "What the caregiver is doing visually...",
  "audio_cues": "What the caregiver is doing acoustically/verbally..."
}
"""


class OmniAnalyzer:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        if self.dry_run:
            logger.info("Dry run enabled. Skipping model loading.")
            return
            
        logger.info("Loading Qwen2.5-Omni-7B model...")
        
        # Load Processor
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        
        # Optimize model loading for Blackwell based environments
        try:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-Omni-7B",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
            logger.info("Loaded model with Flash-Attention-2 and bfloat16")
        except Exception as e:
            logger.warning(f"Flash-Attention-2 failed: {e}. Falling back to float16/SDPA")
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-Omni-7B",
                device_map="auto",
                torch_dtype="auto"
            )
            
        # Optimization: disable audio generation layer
        try:
            self.model.disable_talker()
            logger.info("Disabled talker output to conserve GPU Memory (~2GB).")
        except Exception as e:
            logger.warning(f"Failed to turn off talker: {e}")
            
    def check_vram(self, threshold_gb=1.0):
        if not torch.cuda.is_available():
            return True
        free_mem, _ = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024**3)
        if free_gb < threshold_gb:
            logger.error(f"Low VRAM: {free_gb:.2f}GB / Threshold: {threshold_gb}GB")
            return False
        return True

    def extract_json(self, raw_text, role):
        # Fallback JSON parsing built directly in
        clean_json = raw_text.replace('\n', ' ').replace('\r', '')
        
        # Try native json load first
        try:
            idx1 = clean_json.find('{')
            idx2 = clean_json.rfind('}') + 1
            if idx1 != -1 and idx2 != 0:
                return json.loads(clean_json[idx1:idx2])
        except json.JSONDecodeError:
            pass

        # Regex fallback
        prediction = {}
        if role == "Infant":
            code_match = re.search(r'"(?:infant_code|code)":\s*"([^"]*)"', clean_json, re.IGNORECASE)
            if code_match: prediction["infant_code"] = code_match.group(1).strip()
        else:
            code_match = re.search(r'"(?:caregiver_code|code)":\s*"([^"]*)"', clean_json, re.IGNORECASE)
            if code_match: prediction["caregiver_code"] = code_match.group(1).strip()
            
        vis_match = re.search(r'"visual_cues":\s*"([^"]*)"', clean_json, re.IGNORECASE)
        if vis_match: prediction["visual_cues"] = vis_match.group(1).strip()
        
        aud_match = re.search(r'"audio_cues":\s*"([^"]*)"', clean_json, re.IGNORECASE)
        if aud_match: prediction["audio_cues"] = aud_match.group(1).strip()
        
        return prediction

    def analyze_chunk(self, video_path, duration, subject_type):
        if self.dry_run:
            # Fake payload
            return {"fake_video": f"chunk", "role": subject_type}, {"predicted_code": "unknown"}, "dry_run_text"
            
        min_frames = 4
        max_frames = 8
        fps = min(2.0, max_frames / duration)
        if fps * duration < min_frames: fps = min_frames / duration
        
        if not self.check_vram():
            raise MemoryError("VRAM below threshold for Omni Model")
            
        system_prompt = SYSTEM_PROMPT_INFANT if subject_type == "Infant" else SYSTEM_PROMPT_CAREGIVER
        
        # Format explicitly designed for Qwen-Omni with Audio In Video
        # Note: 'assistant' prompt needs 'You are Qwen' to enable audio perception properly per docs
        # But we customize it heavily for behavior coding.
        # Ensure we always add the Qwen preface
        system_prefix = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. "
        
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prefix + system_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 360,
                        "fps": fps,
                    },
                    {"type": "text", "text": "Analyze the provided video according to the provided instructions."}
                ]
            }
        ]
        
        USE_AUDIO_IN_VIDEO = True
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )

        inputs = inputs.to(self.model.device)
        if hasattr(self.model, 'dtype'):
             for key, val in inputs.items():
                if isinstance(val, torch.Tensor) and val.dtype == torch.float32:
                    inputs[key] = val.to(self.model.dtype)

        with torch.inference_mode():
            # Generate exclusively text, disabling audio generation natively
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                return_audio=False,
                max_new_tokens=1024,
                temperature=0.1
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Cleanup
        del inputs, text_ids, generated_ids_trimmed
        if 'audios' in locals(): del audios
        if 'images' in locals(): del images
        if 'videos' in locals(): del videos
        gc.collect()
        torch.cuda.empty_cache()
        
        parsed_json = self.extract_json(output_text, subject_type)
        
        metrics = {
            "processed_fps": fps,
            "input_duration": duration,
            "role": subject_type,
            "raw_response": output_text
        }
        
        return metrics, parsed_json, output_text

    def synthesize_events(self, narrative_texts, subject_type):
        if self.dry_run:
            return {"predicted_code": "unknown"}
            
        system_prompt = SYSTEM_PROMPT_INFANT if subject_type == "Infant" else SYSTEM_PROMPT_CAREGIVER
        system_prefix = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. "
        
        # Build synthesis prompt
        prompt = "Here is a sequence of observations from a continuous long video event, broken into consecutive 8-second chunks. Synthesize these observations to extract the final comprehensive JSON prediction for the entire duration, adhering to your coding manual instructions.\n\n"
        for i, text in enumerate(narrative_texts):
            prompt += f"--- Chunk {i+1} Observation ---\n{text}\n\n"
            
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prefix + system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        text_input = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
        # Use process_mm_info even for text-only to maintain internal consistency with processor/model expectations
        USE_AUDIO_IN_VIDEO = True
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        inputs = self.processor(
            text=text_input, 
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt", 
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = inputs.to(self.model.device)
        
        with torch.inference_mode():
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                return_audio=False,
                max_new_tokens=1024,
                temperature=0.1
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        parsed_json = self.extract_json(output_text, subject_type)
        return parsed_json, output_text
def load_session_data(session_path):
    meta_path = os.path.join(session_path, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}")
        
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    anno_filename = metadata.get("annotation_file_json", f"{metadata['session_id']}.json")
    anno_path = os.path.join(session_path, anno_filename)
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Annotation JSON not found at {anno_path}")
        
    with open(anno_path, 'r') as f:
        annotations = json.load(f)
        
    return metadata, annotations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_path")
    parser.add_argument("--output", default="omni_analysis.json")
    parser.add_argument("--log_file", default="vlm_omni_analysis.log")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    # Update logger to use session-specific path if preferred, or distinct global log
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    metadata, annotations = load_session_data(args.session_path)
    
    # Locate video
    video_filename = metadata['videos'][-1]
    for v in metadata['videos']:
        if "Splitscreen" in v:
            video_filename = v
            break
    video_path = os.path.join(args.session_path, video_filename)
    logger.info(f"Analyzing Video: {video_path}")
    
    # Setup Out Directory
    vlm_annotations_dir = os.path.join(args.session_path, "vlm_annotations")
    os.makedirs(vlm_annotations_dir, exist_ok=True)
    out_path = os.path.join(vlm_annotations_dir, args.output)
    
    analyzer = OmniAnalyzer(dry_run=args.dry_run)
    
    # Resume Logic
    results = []
    if os.path.exists(out_path):
        try:
            with open(out_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Resuming with {len(results)} processed events.")
        except:
             pass
    processed_keys = set([f"{r.get('start')}_{r.get('end')}_{r.get('track')}" for r in results])
    
    for track in annotations['tracks']:
        track_name = track['name']
        if track_name not in ["Infant_Engagement", "Caregiver_Engagement"]:
            continue
            
        subject_type = "Infant" if "Infant" in track_name else "Caregiver"
        logger.info(f"Processing Track: {track_name}")
        
        processed_for_track = 0
        
        for event in tqdm(track['events']):
            if args.limit > 0 and processed_for_track >= args.limit:
                break
                
            start = float(event['start'])
            end = float(event['end'])
            code = event['code']
            duration = end - start
            
            if duration < 0.5:
                continue
                
            if f"{start}_{end}_{track_name}" in processed_keys:
                continue
                
            logger.info(f"Analyzing {start:.1f}s to {end:.1f}s ({subject_type})")
            inference_start = time.time()
            
            try:
                aggregated_texts = []
                metrics = {"role": subject_type, "input_duration": duration}
                
                # Sliding Window Sub-chunking for events > 10s
                MAX_CHUNK_DUR = 8.5
                OVERLAP = 0.5
                
                if duration > 10.0:
                    chunks = []
                    curr_start = start
                    while curr_start < end:
                        curr_end = min(curr_start + MAX_CHUNK_DUR, end)
                        chunks.append((curr_start, curr_end))
                        if curr_end == end: break
                        curr_start = curr_end - OVERLAP
                        
                    logger.info(f"Event > 10s. Spliced into {len(chunks)} overlapping sub-chunks for memory safety.")
                    
                    for i, (c_start, c_end) in enumerate(chunks):
                        c_dur = c_end - c_start
                        chunk_path = f"/tmp/schwan_omni_chunk_{c_start}_{c_end}.mp4"
                        ffmpeg_cmd = [
                            "ffmpeg", "-y", "-i", video_path, 
                            "-ss", str(c_start), "-to", str(c_end), 
                            "-c:v", "libx264", "-preset", "ultrafast", 
                            "-c:a", "aac", "-b:a", "128k", 
                            chunk_path
                        ]
                        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        
                        logger.info(f"Inferring sub-chunk {i+1}/{len(chunks)}...")
                        _, _, raw_text = analyzer.analyze_chunk(chunk_path, c_dur, subject_type)
                        aggregated_texts.append(raw_text)
                        
                        if os.path.exists(chunk_path): os.remove(chunk_path)
                        
                    logger.info("Synthesizing sub-chunks (Chain of Thought)...")
                    parsed, synth_text = analyzer.synthesize_events(aggregated_texts, subject_type)
                    metrics["raw_response"] = synth_text
                    metrics["chunks_processed"] = len(chunks)
                    
                else:
                    # Normal Single-Chunk Execution
                    chunk_path = f"/tmp/schwan_omni_chunk_{start}_{end}.mp4"
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", video_path, 
                        "-ss", str(start), "-to", str(end), 
                        "-c:v", "libx264", "-preset", "ultrafast", 
                        "-c:a", "aac", "-b:a", "128k", 
                        chunk_path
                    ]
                    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    
                    metrics, parsed, _ = analyzer.analyze_chunk(chunk_path, duration, subject_type)
                    
                    if os.path.exists(chunk_path): os.remove(chunk_path)
                    
                inference_time = time.time() - inference_start
                
                # Fetch detailed code descriptions
                predicted_code = parsed.get("infant_code", "") if subject_type == "Infant" else parsed.get("caregiver_code", "")
                pred_code_info = get_annotation_info(predicted_code) if predicted_code else None
                human_code_info = get_annotation_info(code)
                
                results.append({
                    "track": track_name,
                    "start": start,
                    "end": end,
                    "human_code": code,
                    "human_code_description": human_code_info.get("full", human_code_info.get("brief", "Unknown")),
                    "omni_analysis": parsed,
                    "predicted_code": predicted_code,
                    "predicted_code_description": pred_code_info.get("full", pred_code_info.get("brief", "Unknown")) if pred_code_info else "Unknown",
                    "technical_params": metrics,
                    "inference_time_sec": round(inference_time, 2),
                    "model_id": "Qwen/Qwen2.5-Omni-7B"
                })
                
                # Incremental Save
                with open(out_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                processed_for_track += 1
                
            except Exception as e:
                logger.error(f"Error {start}-{end}: {e}")
                traceback.print_exc()
                continue
                
    logger.info(f"Analysis Routine Complete. Total outputs: {len(results)}")


if __name__ == "__main__":
    main()
