"""
evaluate_labelpredictor_metrics.py
==================================
Reads LlamaFactory evaluation output (generated_predictions.jsonl),
extracts exact classification from generated strings, and computes
Caregiver and Infant classification metrics via scikit-learn.
"""

import json
import argparse
from pathlib import Path
from sklearn.metrics import classification_report


def extract_label_from_llm(prediction_str: str) -> str:
    """Safely unwrap label from LLM which could be strict string or legacy JSON."""
    clean_str = prediction_str.strip('`"\' \n')
    if clean_str.startswith("json\n"):
        clean_str = clean_str[5:].strip('`"\' \n')
    elif clean_str.startswith("json"):
        clean_str = clean_str[4:].strip('`"\' \n')
    
    try:
        pred_obj = json.loads(clean_str)
        if isinstance(pred_obj, dict):
            return pred_obj.get("short_code", "UNKNOWN")
        else:
            return str(pred_obj)
    except Exception:
        return clean_str


def main():
    parser = argparse.ArgumentParser(description="Evaluate labelpredictor F1 scores.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to generated_predictions.jsonl")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"Error: Could not find predictions file at {pred_path}")
        return

    # Tracking exact matches
    caregiver_y_true = []
    caregiver_y_pred = []
    
    infant_y_true = []
    infant_y_pred = []

    malformed_count = 0
    total_count = 0

    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            total_count += 1
            row = json.loads(line)
            
            # Label represents ground truth
            label_obj = json.loads(row["label"])
            true_track = label_obj.get("track")
            true_short_code = label_obj.get("short_code")
            
            # Predict represents LLM response
            pred_short_code = extract_label_from_llm(row["predict"])
            
            if not pred_short_code:
                malformed_count += 1
                pred_short_code = "MALFORMED"

            if "Caregiver" in true_track:
                caregiver_y_true.append(true_short_code)
                caregiver_y_pred.append(pred_short_code)
            elif "Infant" in true_track:
                infant_y_true.append(true_short_code)
                infant_y_pred.append(pred_short_code)

    print(f"========================================")
    print(f" Evaluation Metrics Labelpredictor Study")
    print(f"========================================")
    print(f"Total entries parsed: {total_count}")
    print(f"Malformed LLM outputs: {malformed_count} ({(malformed_count/total_count)*100:.1f}%)")
    print("\n--- CAREGIVER TRACK (N={}) ---".format(len(caregiver_y_true)))
    if caregiver_y_true:
        print(classification_report(caregiver_y_true, caregiver_y_pred, zero_division=0))
    else:
        print("No caregiver samples.")

    print("\n--- INFANT TRACK (N={}) ---".format(len(infant_y_true)))
    if infant_y_true:
        print(classification_report(infant_y_true, infant_y_pred, zero_division=0))
    else:
        print("No infant samples.")
    print(f"========================================")

if __name__ == "__main__":
    main()
