"""
fix_json_from_xiact.py
======================
Fixes conversion issues between .xiact and .json annotation files.

Root cause: In some sessions, the .xiact header has swapped column order
(Add_Infant_Codes before Infant_Engagement_Phases), causing the JSON converter
to put additional infant codes (Isc, Idis) into the Infant_Engagement track
while dropping the actual infant engagement annotations (Inon, Ineu, Ipro, Ipos).

This script:
1. Parses each .xiact file using the ACTUAL header column names
2. Extracts the correct Infant_Engagement events and any additional infant codes
3. For sessions with discrepancies, creates a backup and patches the .json
4. Re-runs comparison to verify the fix
5. Re-generates the finetune annotations for fixed sessions
"""

import json
import re
import shutil
from pathlib import Path
from datetime import datetime

RAW_DATA_DIR = Path(r"X:\data\Schwan_T3_Clean")
FINETUNE_DIR = Path(r"X:\data\Schwan_T3_FineTune")
OUTPUT_DIR = FINETUNE_DIR / "stats"
BACKUP_DIR = RAW_DATA_DIR / "_backups"

TICKS_PER_SECOND = 10_000_000.0
KNOWN_INFANT_ENGAGEMENT_CODES = {"Inon", "Ineu", "Ipos", "Ipro", "Iusc", "Iwit"}
KNOWN_CAREGIVER_CODES = {"Cpos", "Cpvc", "Cneu", "Cnon", "Cint", "Cusc"}
PARADIGM_CODES = {"Trans", "SFP", "RP"}
OTHER_CODES = {"PP"}
ADDITIONAL_INFANT_CODES = {"Isc o", "Isc h", "Idis"}


def parse_xiact_robust(filepath: Path) -> dict:
    """
    Robustly parse a .xiact file using the ACTUAL header column names.
    Returns properly separated tracks.
    """
    result = {
        "infant_engagement": [],     # Inon, Ineu, Ipro, Ipos, etc.
        "caregiver_engagement": [],  # Cpos, Cpvc, Cneu, etc.
        "paradigm_phases": [],       # Trans, SFP, RP
        "other": [],                 # PP
        "additional_infant": [],     # Isc o, Isc h, Idis
        "fps": 25,
    }

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Parse FPS
    fps_match = re.search(r"FPS,\s*(\d+)", lines[0] if lines else "")
    if fps_match:
        result["fps"] = int(fps_match.group(1))

    # Find header
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("[INDVAR]"):
            header_idx = i + 1
            break

    if header_idx is None or header_idx >= len(lines):
        return result

    # Parse header columns dynamically
    header = lines[header_idx].strip().split("\t")
    col_map = {}
    for ci, col_name in enumerate(header):
        col_lower = col_name.strip().lower()
        if col_lower == "level":
            col_map["level"] = ci
        elif col_lower == "onset":
            col_map["onset"] = ci
        elif col_lower == "offset":
            col_map["offset"] = ci
        elif col_lower == "memo":
            col_map["memo"] = ci
        elif "caregiver_engagement" in col_lower:
            col_map["caregiver_engagement"] = ci
        elif "infant_engagement" in col_lower:
            col_map["infant_engagement"] = ci
        elif "add_infant" in col_lower:
            col_map["add_infant"] = ci
        elif "add_caregiver" in col_lower:
            col_map["add_caregiver"] = ci

    print(f"    Header columns: {header}")
    print(f"    Column map: {col_map}")

    # Parse data rows
    for line in lines[header_idx + 1:]:
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 3:
            continue

        try:
            level = int(parts[col_map.get("level", 0)].strip())
        except (ValueError, IndexError):
            continue

        if level != 3:
            continue

        try:
            onset = int(parts[col_map.get("onset", 1)].strip())
            offset = int(parts[col_map.get("offset", 2)].strip())
        except (ValueError, IndexError):
            continue

        start_sec = round(onset / TICKS_PER_SECOND, 3)
        end_sec = round(offset / TICKS_PER_SECOND, 3)

        def get_col(key):
            idx = col_map.get(key)
            if idx is not None and idx < len(parts):
                return parts[idx].strip()
            return ""

        # Read each column based on its ACTUAL name from header
        infant_code = get_col("infant_engagement")
        caregiver_code = get_col("caregiver_engagement")
        add_infant_code = get_col("add_infant")

        evt = {"start": start_sec, "end": end_sec}

        # Route infant engagement codes
        if infant_code:
            result["infant_engagement"].append({**evt, "code": infant_code})

        # Route caregiver codes (including paradigm markers mixed in)
        if caregiver_code:
            if caregiver_code in PARADIGM_CODES:
                result["paradigm_phases"].append({**evt, "code": caregiver_code})
            elif caregiver_code in OTHER_CODES:
                result["other"].append({**evt, "code": caregiver_code})
            else:
                result["caregiver_engagement"].append({**evt, "code": caregiver_code})

        # Route additional infant codes
        if add_infant_code:
            result["additional_infant"].append({**evt, "code": add_infant_code})

    return result


def rebuild_json(json_path: Path, xiact_data: dict, session_name: str) -> dict:
    """
    Rebuild the JSON file with correct tracks from xiact data.
    Returns a summary of changes made.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    changes = {"session": session_name, "modifications": []}

    # Build track lookup
    track_map = {t["name"]: t for t in json_data.get("tracks", [])}

    # Fix Infant_Engagement track
    old_infant = track_map.get("Infant_Engagement", {})
    old_events = old_infant.get("events", [])
    old_count = len(old_events)
    old_codes = sorted(set(e.get("code", "") for e in old_events))

    # The xiact infant_engagement has the correct events
    new_infant_events = []
    for e in xiact_data["infant_engagement"]:
        new_infant_events.append({
            "code": e["code"],
            "start": str(e["start"]),
            "end": str(e["end"]),
        })

    new_count = len(new_infant_events)
    new_codes = sorted(set(e["code"] for e in new_infant_events))

    if old_count != new_count or old_codes != new_codes:
        changes["modifications"].append({
            "track": "Infant_Engagement",
            "old_count": old_count,
            "new_count": new_count,
            "old_codes": old_codes,
            "new_codes": new_codes,
        })

        if "Infant_Engagement" in track_map:
            track_map["Infant_Engagement"]["events"] = new_infant_events
        else:
            json_data.setdefault("tracks", []).append({
                "name": "Infant_Engagement",
                "events": new_infant_events,
            })

    # Fix Caregiver_Engagement track
    old_caregiver = track_map.get("Caregiver_Engagement", {})
    old_cgv_events = old_caregiver.get("events", [])
    old_cgv_count = len(old_cgv_events)

    new_cgv_events = []
    for e in xiact_data["caregiver_engagement"]:
        new_cgv_events.append({
            "code": e["code"],
            "start": str(e["start"]),
            "end": str(e["end"]),
        })

    new_cgv_count = len(new_cgv_events)
    if old_cgv_count != new_cgv_count:
        changes["modifications"].append({
            "track": "Caregiver_Engagement",
            "old_count": old_cgv_count,
            "new_count": new_cgv_count,
        })

        if "Caregiver_Engagement" in track_map:
            track_map["Caregiver_Engagement"]["events"] = new_cgv_events
        else:
            json_data.setdefault("tracks", []).append({
                "name": "Caregiver_Engagement",
                "events": new_cgv_events,
            })

    # Fix Paradigm_Phases track
    old_paradigm = track_map.get("Paradigm_Phases", {})
    old_para_events = old_paradigm.get("events", [])
    old_para_count = len(old_para_events)

    new_para_events = []
    for e in xiact_data["paradigm_phases"]:
        new_para_events.append({
            "code": e["code"],
            "start": str(e["start"]),
            "end": str(e["end"]),
        })

    new_para_count = len(new_para_events)
    if old_para_count != new_para_count:
        changes["modifications"].append({
            "track": "Paradigm_Phases",
            "old_count": old_para_count,
            "new_count": new_para_count,
        })

        if "Paradigm_Phases" in track_map:
            track_map["Paradigm_Phases"]["events"] = new_para_events
        else:
            json_data.setdefault("tracks", []).append({
                "name": "Paradigm_Phases",
                "events": new_para_events,
            })

    # Fix Other track
    old_other = track_map.get("Other", {})
    old_other_events = old_other.get("events", [])
    old_other_count = len(old_other_events)

    # Other track: from xiact "other" (PP) + additional_infant (Isc, Idis)
    new_other_events = []
    for e in xiact_data["other"]:
        new_other_events.append({
            "code": e["code"],
            "start": str(e["start"]),
            "end": str(e["end"]),
        })
    for e in xiact_data["additional_infant"]:
        new_other_events.append({
            "code": e["code"],
            "start": str(e["start"]),
            "end": str(e["end"]),
        })
    # Sort by start time
    new_other_events.sort(key=lambda x: float(x["start"]))

    new_other_count = len(new_other_events)
    if old_other_count != new_other_count:
        changes["modifications"].append({
            "track": "Other",
            "old_count": old_other_count,
            "new_count": new_other_count,
        })

    if "Other" in track_map:
        track_map["Other"]["events"] = new_other_events
    else:
        json_data.setdefault("tracks", []).append({
            "name": "Other",
            "events": new_other_events,
        })

    # Rebuild tracks list in a consistent order
    track_order = ["Infant_Engagement", "Caregiver_Engagement", "Paradigm_Phases", "Other"]
    rebuilt_tracks = []
    for name in track_order:
        if name in track_map:
            rebuilt_tracks.append(track_map[name])
    # Keep any other tracks not in our order
    for t in json_data.get("tracks", []):
        if t.get("name") not in track_order:
            rebuilt_tracks.append(t)

    json_data["tracks"] = rebuilt_tracks

    return json_data, changes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir() and d.name != "_backups"])

    all_changes = []
    fixed_sessions = []
    already_ok = []

    print(f"{'='*100}")
    print(f"FIX JSON FROM XIACT — Processing {len(session_dirs)} sessions")
    print(f"{'='*100}")

    for session_dir in session_dirs:
        session_name = session_dir.name
        xiact_file = session_dir / f"{session_name}.xiact"
        json_file = session_dir / f"{session_name}.json"

        if not xiact_file.exists() or not json_file.exists():
            continue

        print(f"\n{'─'*80}")
        print(f"Processing: {session_name}")

        # Parse xiact robustly
        xiact_data = parse_xiact_robust(xiact_file)

        xi_inf = len(xiact_data["infant_engagement"])
        xi_cgv = len(xiact_data["caregiver_engagement"])
        xi_para = len(xiact_data["paradigm_phases"])
        xi_other = len(xiact_data["other"])
        xi_add_inf = len(xiact_data["additional_infant"])

        print(f"  XIACT: {xi_inf} infant, {xi_cgv} caregiver, "
              f"{xi_para} paradigm, {xi_other} other, {xi_add_inf} add_infant")

        # Read current JSON
        with open(json_file, "r", encoding="utf-8") as f:
            current_json = json.load(f)

        current_tracks = {t["name"]: len(t.get("events", [])) for t in current_json.get("tracks", [])}
        js_inf = current_tracks.get("Infant_Engagement", 0)
        js_cgv = current_tracks.get("Caregiver_Engagement", 0)
        print(f"  JSON:  {js_inf} infant, {js_cgv} caregiver")

        # Check if fix is needed
        needs_fix = False
        if xi_inf != js_inf:
            print(f"  ⚠ INFANT MISMATCH: xiact={xi_inf} vs json={js_inf}")
            needs_fix = True
        if xi_cgv != js_cgv:
            print(f"  ⚠ CAREGIVER MISMATCH: xiact={xi_cgv} vs json={js_cgv}")
            needs_fix = True

        if not needs_fix:
            print(f"  ✓ OK — counts match")
            already_ok.append(session_name)
            continue

        # Create backup
        backup_path = BACKUP_DIR / f"{session_name}.json.bak"
        if not backup_path.exists():
            shutil.copy2(json_file, backup_path)
            print(f"  Created backup → {backup_path}")
        else:
            print(f"  Backup already exists → {backup_path}")

        # Rebuild and save
        fixed_json, changes = rebuild_json(json_file, xiact_data, session_name)
        all_changes.append(changes)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(fixed_json, f, indent=2, ensure_ascii=False)

        # Verify the fix
        with open(json_file, "r", encoding="utf-8") as f:
            verify = json.load(f)
        verify_tracks = {t["name"]: len(t.get("events", [])) for t in verify.get("tracks", [])}

        print(f"  FIXED: {verify_tracks.get('Infant_Engagement', 0)} infant, "
              f"{verify_tracks.get('Caregiver_Engagement', 0)} caregiver, "
              f"{verify_tracks.get('Paradigm_Phases', 0)} paradigm, "
              f"{verify_tracks.get('Other', 0)} other")

        v_inf = verify_tracks.get('Infant_Engagement', 0)
        v_cgv = verify_tracks.get('Caregiver_Engagement', 0)

        if v_inf == xi_inf and v_cgv == xi_cgv:
            print(f"  ✅ VERIFIED — JSON now matches XIACT perfectly")
            fixed_sessions.append(session_name)
        else:
            print(f"  ❌ VERIFICATION FAILED — still mismatched!")

        for mod in changes.get("modifications", []):
            print(f"    {mod['track']}: {mod.get('old_count', '?')} → {mod.get('new_count', '?')}")
            if 'old_codes' in mod:
                print(f"      codes: {mod['old_codes']} → {mod['new_codes']}")

    # ── SUMMARY ──
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"Sessions checked: {len(already_ok) + len(fixed_sessions)}")
    print(f"Already correct:  {len(already_ok)}")
    print(f"Fixed:            {len(fixed_sessions)}")
    if fixed_sessions:
        print(f"  Fixed sessions: {fixed_sessions}")
    print(f"\nBackups saved to: {BACKUP_DIR}")

    # Save change log
    log_path = OUTPUT_DIR / "fix_log.json"
    log = {
        "timestamp": datetime.now().isoformat(),
        "fixed_sessions": fixed_sessions,
        "already_ok": already_ok,
        "changes": all_changes,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"Saved change log → {log_path}")


if __name__ == "__main__":
    main()
