"""
validate_xiact_vs_json.py
=========================
Parses the original .xiact annotation files and compares them against the
converted .json files to detect any conversion errors or timestamp discrepancies.

.xiact format:
  - Tab-separated, timestamps in 100ns ticks (divide by 10,000,000 → seconds)
  - Header line: Level  Onset  Offset  Memo  Caregiver_Engagement_Phases  Infant_Engagement_Phases  Add_Infant_Codes  Add_Caregiver_Codes
  - Level 2: Video reference rows
  - Level 3: Data rows (annotations)
  - Caregiver codes appear in column 4, Infant codes in column 5
  - Additional codes (Isc, Idis etc.) in columns 6-7

Outputs:
  - Per-session comparison table (terminal)
  - Discrepancy report JSON and CSV
  - Visualization of mismatches
"""

import json
import csv
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

RAW_DATA_DIR = Path(r"X:\data\Schwan_T3_Clean")
OUTPUT_DIR = Path(r"X:\data\Schwan_T3_FineTune\stats")

TICKS_PER_SECOND = 10_000_000.0  # 100ns ticks → seconds
TIME_TOLERANCE = 0.1  # 0.1s tolerance for timestamp comparison


def parse_xiact(filepath: Path) -> dict:
    """
    Parse a .xiact file and extract annotations for each track.
    Returns dict with:
      - infant_events: list of {code, start, end}
      - caregiver_events: list of {code, start, end}
      - additional_infant_events: list (Isc, Idis, etc.)
      - paradigm_events: list (Trans, SFP, RP, PP)
      - fps: int
    """
    result = {
        "infant_events": [],
        "caregiver_events": [],
        "additional_infant_events": [],
        "paradigm_events": [],
        "other_events": [],
        "fps": 25,
        "raw_lines": 0,
        "data_lines": 0,
    }

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    result["raw_lines"] = len(lines)

    # Parse FPS from the first line if present
    first_line = lines[0] if lines else ""
    fps_match = re.search(r"FPS,\s*(\d+)", first_line)
    if fps_match:
        result["fps"] = int(fps_match.group(1))

    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("[INDVAR]"):
            header_idx = i + 1  # next line is the actual header
            break

    if header_idx is None or header_idx >= len(lines):
        return result

    # Parse the header to understand column positions
    header = lines[header_idx].strip().split("\t")
    # Expected: Level  Onset  Offset  Memo  Caregiver_Engagement_Phases  Infant_Engagement_Phases  Add_Infant_Codes  Add_Caregiver_Codes
    # But column names may vary slightly

    # Find column indices
    col_map = {}
    for ci, col_name in enumerate(header):
        col_lower = col_name.strip().lower()
        if "caregiver_engagement" in col_lower or "caregiver_engagement_phases" in col_lower:
            col_map["caregiver"] = ci
        elif "infant_engagement" in col_lower or "infant_engagement_phases" in col_lower:
            col_map["infant"] = ci
        elif "add_infant" in col_lower:
            col_map["add_infant"] = ci
        elif "add_caregiver" in col_lower:
            col_map["add_caregiver"] = ci
        elif col_lower == "level":
            col_map["level"] = ci
        elif col_lower == "onset":
            col_map["onset"] = ci
        elif col_lower == "offset":
            col_map["offset"] = ci
        elif col_lower == "memo":
            col_map["memo"] = ci

    # Parse data rows (starting after header)
    for line in lines[header_idx + 1:]:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue

        try:
            level = int(parts[col_map.get("level", 0)].strip())
        except (ValueError, IndexError):
            continue

        if level != 3:
            continue

        result["data_lines"] += 1

        try:
            onset_ticks = int(parts[col_map.get("onset", 1)].strip())
            offset_ticks = int(parts[col_map.get("offset", 2)].strip())
        except (ValueError, IndexError):
            continue

        start_sec = round(onset_ticks / TICKS_PER_SECOND, 3)
        end_sec = round(offset_ticks / TICKS_PER_SECOND, 3)

        # Get codes from each column
        def get_col(key):
            idx = col_map.get(key)
            if idx is not None and idx < len(parts):
                return parts[idx].strip()
            return ""

        caregiver_code = get_col("caregiver")
        infant_code = get_col("infant")
        add_infant_code = get_col("add_infant")
        add_caregiver_code = get_col("add_caregiver")

        # Caregiver engagement
        if caregiver_code:
            # Paradigm markers are in the caregiver column in some files
            if caregiver_code in ("Trans", "SFP", "RP"):
                result["paradigm_events"].append({
                    "code": caregiver_code, "start": start_sec, "end": end_sec
                })
            elif caregiver_code == "PP":
                result["other_events"].append({
                    "code": "PP", "start": start_sec, "end": end_sec
                })
            else:
                result["caregiver_events"].append({
                    "code": caregiver_code, "start": start_sec, "end": end_sec
                })

        # Infant engagement
        if infant_code:
            result["infant_events"].append({
                "code": infant_code, "start": start_sec, "end": end_sec
            })

        # Additional infant codes (Isc, Idis)
        if add_infant_code:
            result["additional_infant_events"].append({
                "code": add_infant_code, "start": start_sec, "end": end_sec
            })

    return result


def parse_json_annotations(filepath: Path) -> dict:
    """Parse the .json annotation file and extract events per track."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {
        "infant_events": [],
        "caregiver_events": [],
        "paradigm_events": [],
        "other_events": [],
    }

    for track in data.get("tracks", []):
        name = track.get("name", "")
        events = track.get("events", [])
        for e in events:
            evt = {
                "code": e.get("code", ""),
                "start": round(float(e.get("start", 0)), 3),
                "end": round(float(e.get("end", 0)), 3),
            }
            if name == "Infant_Engagement":
                result["infant_events"].append(evt)
            elif name == "Caregiver_Engagement":
                result["caregiver_events"].append(evt)
            elif name == "Paradigm_Phases":
                result["paradigm_events"].append(evt)
            elif name == "Other":
                result["other_events"].append(evt)

    return result


def compare_events(xiact_events: list, json_events: list, track_name: str) -> list:
    """
    Compare two lists of events and find discrepancies.
    Returns list of discrepancy dicts.
    """
    discrepancies = []

    # Sort both by start time
    xi_sorted = sorted(xiact_events, key=lambda e: e["start"])
    js_sorted = sorted(json_events, key=lambda e: e["start"])

    xi_idx = 0
    js_idx = 0
    matched_xi = set()
    matched_js = set()

    # Match events by proximity
    for xi_i, xi_evt in enumerate(xi_sorted):
        best_match = None
        best_dist = float("inf")
        for js_i, js_evt in enumerate(js_sorted):
            if js_i in matched_js:
                continue
            dist = abs(xi_evt["start"] - js_evt["start"]) + abs(xi_evt["end"] - js_evt["end"])
            if dist < best_dist:
                best_dist = dist
                best_match = js_i

        if best_match is not None and best_dist < 2.0:  # within 2 seconds total
            matched_xi.add(xi_i)
            matched_js.add(best_match)

            js_evt = js_sorted[best_match]

            # Check for code mismatch
            if xi_evt["code"] != js_evt["code"]:
                discrepancies.append({
                    "type": "code_mismatch",
                    "track": track_name,
                    "xiact_code": xi_evt["code"],
                    "json_code": js_evt["code"],
                    "xiact_start": xi_evt["start"],
                    "json_start": js_evt["start"],
                    "xiact_end": xi_evt["end"],
                    "json_end": js_evt["end"],
                })

            # Check for timestamp discrepancy
            start_diff = abs(xi_evt["start"] - js_evt["start"])
            end_diff = abs(xi_evt["end"] - js_evt["end"])
            if start_diff > TIME_TOLERANCE or end_diff > TIME_TOLERANCE:
                discrepancies.append({
                    "type": "timestamp_mismatch",
                    "track": track_name,
                    "code": xi_evt["code"],
                    "xiact_start": xi_evt["start"],
                    "json_start": js_evt["start"],
                    "start_diff": round(start_diff, 3),
                    "xiact_end": xi_evt["end"],
                    "json_end": js_evt["end"],
                    "end_diff": round(end_diff, 3),
                })

    # Events in xiact but not in json
    for xi_i, xi_evt in enumerate(xi_sorted):
        if xi_i not in matched_xi:
            discrepancies.append({
                "type": "missing_in_json",
                "track": track_name,
                "code": xi_evt["code"],
                "start": xi_evt["start"],
                "end": xi_evt["end"],
            })

    # Events in json but not in xiact
    for js_i, js_evt in enumerate(js_sorted):
        if js_i not in matched_js:
            discrepancies.append({
                "type": "missing_in_xiact",
                "track": track_name,
                "code": js_evt["code"],
                "start": js_evt["start"],
                "end": js_evt["end"],
            })

    return discrepancies


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir()])

    all_results = []
    total_discrepancies = 0
    sessions_with_issues = []

    print(f"{'='*140}")
    print(f"XIACT vs JSON VALIDATION — Checking {len(session_dirs)} sessions")
    print(f"{'='*140}")
    header = (f"{'Session':<30} {'XI_Inf':>7} {'JS_Inf':>7} {'Inf_OK':>7} "
              f"{'XI_Cgv':>7} {'JS_Cgv':>7} {'Cgv_OK':>7} "
              f"{'XI_Para':>7} {'JS_Para':>8} "
              f"{'Discrepancies':>14}")
    print(header)
    print("-" * 140)

    for session_dir in session_dirs:
        session_name = session_dir.name
        xiact_file = session_dir / f"{session_name}.xiact"
        json_file = session_dir / f"{session_name}.json"

        if not xiact_file.exists() or not json_file.exists():
            continue

        # Parse both files
        xi_data = parse_xiact(xiact_file)
        js_data = parse_json_annotations(json_file)

        # Compare tracks
        infant_disc = compare_events(xi_data["infant_events"], js_data["infant_events"], "Infant")
        caregiver_disc = compare_events(xi_data["caregiver_events"], js_data["caregiver_events"], "Caregiver")
        paradigm_disc = compare_events(xi_data["paradigm_events"], js_data["paradigm_events"], "Paradigm")

        all_disc = infant_disc + caregiver_disc + paradigm_disc
        n_disc = len(all_disc)
        total_discrepancies += n_disc

        xi_inf = len(xi_data["infant_events"])
        js_inf = len(js_data["infant_events"])
        xi_cgv = len(xi_data["caregiver_events"])
        js_cgv = len(js_data["caregiver_events"])
        xi_para = len(xi_data["paradigm_events"])
        js_para = len(js_data["paradigm_events"])

        inf_match = "✓" if xi_inf == js_inf and not infant_disc else f"✗({len(infant_disc)})"
        cgv_match = "✓" if xi_cgv == js_cgv and not caregiver_disc else f"✗({len(caregiver_disc)})"

        result = {
            "session": session_name,
            "xiact_infant_events": xi_inf,
            "json_infant_events": js_inf,
            "xiact_caregiver_events": xi_cgv,
            "json_caregiver_events": js_cgv,
            "xiact_paradigm_events": xi_para,
            "json_paradigm_events": js_para,
            "xiact_additional_infant": len(xi_data["additional_infant_events"]),
            "xiact_other_events": len(xi_data["other_events"]),
            "xiact_data_lines": xi_data["data_lines"],
            "total_discrepancies": n_disc,
            "discrepancies": all_disc,
            "infant_count_match": xi_inf == js_inf,
            "caregiver_count_match": xi_cgv == js_cgv,
        }
        all_results.append(result)

        if n_disc > 0:
            sessions_with_issues.append(session_name)

        disc_str = f"{n_disc}" if n_disc == 0 else f"⚠ {n_disc}"
        print(f"{session_name:<30} {xi_inf:>7} {js_inf:>7} {inf_match:>7} "
              f"{xi_cgv:>7} {js_cgv:>7} {cgv_match:>7} "
              f"{xi_para:>7} {js_para:>8} "
              f"{disc_str:>14}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Sessions checked: {len(all_results)}")
    print(f"Sessions with discrepancies: {len(sessions_with_issues)}")
    print(f"Total discrepancies: {total_discrepancies}")

    if sessions_with_issues:
        print(f"\nSessions with issues:")
        for r in all_results:
            if r["total_discrepancies"] > 0:
                print(f"\n  ⚠ {r['session']} ({r['total_discrepancies']} discrepancies):")
                # Group discrepancies by type
                by_type = defaultdict(list)
                for d in r["discrepancies"]:
                    by_type[d["type"]].append(d)
                for dtype, items in by_type.items():
                    print(f"    {dtype}: {len(items)}")
                    for item in items[:5]:  # show first 5
                        if dtype == "code_mismatch":
                            print(f"      xiact={item['xiact_code']} vs json={item['json_code']} "
                                  f"at {item['xiact_start']:.1f}s")
                        elif dtype == "timestamp_mismatch":
                            print(f"      {item['code']} start_diff={item['start_diff']:.3f}s "
                                  f"end_diff={item['end_diff']:.3f}s")
                        elif dtype == "missing_in_json":
                            print(f"      {item['track']}: {item['code']} "
                                  f"[{item['start']:.1f}–{item['end']:.1f}s] in .xiact but NOT in .json")
                        elif dtype == "missing_in_xiact":
                            print(f"      {item['track']}: {item['code']} "
                                  f"[{item['start']:.1f}–{item['end']:.1f}s] in .json but NOT in .xiact")
                    if len(items) > 5:
                        print(f"      ... and {len(items)-5} more")
    else:
        print("\n✅ ALL SESSIONS MATCH PERFECTLY — no conversion errors detected!")

    # ── Visualization ──
    n = len(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(24, max(20, n * 0.3)))
    fig.suptitle(f"XIACT vs JSON Validation — {n} sessions", fontsize=16, fontweight="bold", y=1.01)

    names = [r["session"] for r in all_results]
    y_pos = np.arange(n)

    # Plot 1: Event count comparison (Infant)
    ax1 = axes[0]
    xi_inf_counts = [r["xiact_infant_events"] for r in all_results]
    js_inf_counts = [r["json_infant_events"] for r in all_results]
    bar_h = 0.35
    ax1.barh(y_pos - bar_h/2, xi_inf_counts, bar_h, color="#FF9800", label=".xiact", edgecolor="white")
    ax1.barh(y_pos + bar_h/2, js_inf_counts, bar_h, color="#4CAF50", label=".json", edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=6)
    ax1.set_xlabel("Event Count")
    ax1.set_title("Infant Events: .xiact vs .json", pad=12)
    ax1.legend(loc="lower right")
    ax1.invert_yaxis()

    # Highlight mismatches
    for i, r in enumerate(all_results):
        if r["xiact_infant_events"] != r["json_infant_events"]:
            ax1.add_patch(plt.Rectangle(
                (-1, i-0.45), max(xi_inf_counts[i], js_inf_counts[i]) + 5, 0.9,
                linewidth=2, edgecolor='red', facecolor='none'
            ))
            ax1.text(max(xi_inf_counts[i], js_inf_counts[i]) + 2, i,
                     f"Δ{abs(r['xiact_infant_events']-r['json_infant_events'])}",
                     va='center', fontsize=6, color='red', fontweight='bold')

    # Plot 2: Event count comparison (Caregiver)
    ax2 = axes[1]
    xi_cgv_counts = [r["xiact_caregiver_events"] for r in all_results]
    js_cgv_counts = [r["json_caregiver_events"] for r in all_results]
    ax2.barh(y_pos - bar_h/2, xi_cgv_counts, bar_h, color="#FF9800", label=".xiact", edgecolor="white")
    ax2.barh(y_pos + bar_h/2, js_cgv_counts, bar_h, color="#4CAF50", label=".json", edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=6)
    ax2.set_xlabel("Event Count")
    ax2.set_title("Caregiver Events: .xiact vs .json", pad=12)
    ax2.legend(loc="lower right")
    ax2.invert_yaxis()

    for i, r in enumerate(all_results):
        if r["xiact_caregiver_events"] != r["json_caregiver_events"]:
            ax2.add_patch(plt.Rectangle(
                (-1, i-0.45), max(xi_cgv_counts[i], js_cgv_counts[i]) + 5, 0.9,
                linewidth=2, edgecolor='red', facecolor='none'
            ))
            ax2.text(max(xi_cgv_counts[i], js_cgv_counts[i]) + 2, i,
                     f"Δ{abs(r['xiact_caregiver_events']-r['json_caregiver_events'])}",
                     va='center', fontsize=6, color='red', fontweight='bold')

    # Plot 3: Total discrepancies per session
    ax3 = axes[2]
    disc_counts = [r["total_discrepancies"] for r in all_results]
    colors = ["#EF5350" if d > 0 else "#4CAF50" for d in disc_counts]
    ax3.barh(y_pos, disc_counts, color=colors, edgecolor="white")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=6)
    ax3.set_xlabel("Number of Discrepancies")
    ax3.set_title("Total Discrepancies per Session", pad=12)
    ax3.invert_yaxis()

    for i, d in enumerate(disc_counts):
        if d > 0:
            ax3.text(d + 0.5, i, str(d), va='center', fontsize=6, color='red', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = OUTPUT_DIR / "xiact_vs_json_validation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization → {fig_path}")
    plt.close()

    # Save report JSON
    report = {
        "total_sessions": len(all_results),
        "sessions_with_discrepancies": len(sessions_with_issues),
        "total_discrepancies": total_discrepancies,
        "sessions": [{
            k: v for k, v in r.items() if k != "discrepancies"
        } for r in all_results],
        "discrepancy_details": {
            r["session"]: r["discrepancies"]
            for r in all_results if r["total_discrepancies"] > 0
        }
    }

    report_path = OUTPUT_DIR / "xiact_vs_json_validation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report JSON → {report_path}")

    # Save CSV
    csv_path = OUTPUT_DIR / "xiact_vs_json_validation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["session", "xiact_infant", "json_infant", "infant_match",
                          "xiact_caregiver", "json_caregiver", "caregiver_match",
                          "xiact_paradigm", "json_paradigm",
                          "xiact_add_infant", "xiact_other",
                          "total_discrepancies"])
        for r in all_results:
            writer.writerow([
                r["session"], r["xiact_infant_events"], r["json_infant_events"],
                r["infant_count_match"],
                r["xiact_caregiver_events"], r["json_caregiver_events"],
                r["caregiver_count_match"],
                r["xiact_paradigm_events"], r["json_paradigm_events"],
                r["xiact_additional_infant"], r["xiact_other_events"],
                r["total_discrepancies"]
            ])
    print(f"Saved CSV → {csv_path}")


if __name__ == "__main__":
    main()
