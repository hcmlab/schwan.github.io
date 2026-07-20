"""
session_stats_xiact.py
======================
Session-by-session analysis using XIACT annotation files directly.
Reads paradigm phase markers (Trans, SFP, RP) straight from .xiact files
rather than the converted .json, and generates per-session stats, CSV, JSON,
and a visualization saved as session_analysis_with_xiact.png.

IMPORTANT: All durations, coverage, and the SFP timeline are computed
against the ANNOTATION SPAN (first annotation start → last annotation end),
NOT the raw video file duration. This avoids untrimmed video recordings
(e.g. SAJARE01 at 38min) from skewing the reunion phase and coverage metrics.
"""

import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

RAW_DATA_DIR = Path(r"X:\data\Schwan_T3_Clean")
FINETUNE_DIR = Path(r"X:\data\Schwan_T3_FineTune")
OUTPUT_DIR = FINETUNE_DIR / "stats"

TICKS_PER_SECOND = 10_000_000.0

KNOWN_INFANT_ENGAGEMENT_CODES = {"Inon", "Ineu", "Ipos", "Ipro", "Iusc", "Iwit"}
KNOWN_CAREGIVER_CODES = {"Cpos", "Cpvc", "Cneu", "Cnon", "Cint", "Cusc"}
PARADIGM_CODES = {"Trans", "SFP", "RP"}
OTHER_CODES = {"PP"}


def parse_xiact(filepath: Path) -> dict:
    """
    Robustly parse a .xiact file using the ACTUAL header column names.
    Returns properly separated tracks with events as dicts {code, start, end}.
    """
    result = {
        "infant_engagement": [],
        "caregiver_engagement": [],
        "paradigm_phases": [],
        "other": [],
        "additional_infant": [],
        "fps": 25,
    }

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Parse FPS
    fps_match = re.search(r"FPS,\s*(\d+)", lines[0] if lines else "")
    if fps_match:
        result["fps"] = int(fps_match.group(1))

    # Find header line (follows [INDVAR])
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

        infant_code = get_col("infant_engagement")
        caregiver_code = get_col("caregiver_engagement")
        add_infant_code = get_col("add_infant")

        evt = {"start": start_sec, "end": end_sec}

        if infant_code:
            result["infant_engagement"].append({**evt, "code": infant_code})

        if caregiver_code:
            if caregiver_code in PARADIGM_CODES:
                result["paradigm_phases"].append({**evt, "code": caregiver_code})
            elif caregiver_code in OTHER_CODES:
                result["other"].append({**evt, "code": caregiver_code})
            else:
                result["caregiver_engagement"].append({**evt, "code": caregiver_code})

        if add_infant_code:
            result["additional_infant"].append({**evt, "code": add_infant_code})

    return result


def get_video_duration(session_dir: Path) -> float:
    meta_file = session_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        dur = meta.get("technical_metadata", {}).get("duration")
        if dur:
            return float(dur)
    return None


def compute_coverage(events: list, reference_duration: float) -> dict:
    """Compute how much of the reference span is covered by annotations (union of intervals).
    
    reference_duration can be the annotation span or video duration.
    """
    if not events or reference_duration is None:
        return {"covered_seconds": 0, "coverage_pct": 0, "num_events": 0}

    intervals = []
    for e in events:
        s, end = float(e.get("start", 0)), float(e.get("end", 0))
        if end > s:
            intervals.append((s, end))

    if not intervals:
        return {"covered_seconds": 0, "coverage_pct": 0, "num_events": len(events)}

    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    covered = sum(e - s for s, e in merged)
    return {
        "covered_seconds": round(covered, 2),
        "coverage_pct": round(100 * covered / reference_duration, 1) if reference_duration > 0 else 0,
        "num_events": len(events),
        "first_annotation_start": round(merged[0][0], 2),
        "last_annotation_end": round(merged[-1][1], 2),
    }


def parse_paradigm_phases_from_xiact(paradigm_events: list, last_annotation_end: float) -> dict:
    """Extract Still Face Paradigm phase info directly from xiact paradigm events.
    
    Reunion phase duration is computed as last_annotation_end - RP_start,
    NOT video_duration - RP_start, to avoid untrimmed video skew.
    """
    phases = {"has_paradigm": False}

    if not paradigm_events:
        return phases

    phases["has_paradigm"] = True
    sorted_events = sorted(paradigm_events, key=lambda e: float(e.get("start", 0)))

    trans_start = None
    sfp_start = None
    rp_start = None

    for e in sorted_events:
        code = e.get("code", "")
        start = float(e.get("start", 0))
        end = float(e.get("end", 0))

        if code == "Trans":
            trans_start = start
            phases["transition_start"] = round(start, 2)
            phases["transition_end"] = round(end, 2)
            phases["transition_duration"] = round(end - start, 2)
        elif code == "SFP":
            sfp_start = start
            phases["sfp_start"] = round(start, 2)
            phases["sfp_end"] = round(end, 2)
            phases["sfp_marker_duration"] = round(end - start, 2)
        elif code == "RP":
            rp_start = start
            phases["reunion_start"] = round(start, 2)
            phases["reunion_end"] = round(end, 2)
            phases["rp_marker_duration"] = round(end - start, 2)

    # Play phase: from 0 to transition start
    if trans_start is not None:
        phases["play_duration"] = round(trans_start, 2)

    # Still Face phase: from SFP marker to RP marker
    if sfp_start is not None and rp_start is not None:
        phases["sfp_duration"] = round(rp_start - sfp_start, 2)

    # Reunion phase: from RP marker to LAST ANNOTATION END (not video end)
    if rp_start is not None and last_annotation_end is not None and last_annotation_end > rp_start:
        phases["reunion_duration"] = round(last_annotation_end - rp_start, 2)

    return phases


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([d for d in RAW_DATA_DIR.iterdir()
                           if d.is_dir() and d.name != "_backups"])

    all_stats = []

    print(f"{'=' * 130}")
    print(f"SESSION-BY-SESSION ANALYSIS FROM XIACT FILES")
    print(f"{'=' * 130}")

    for session_dir in session_dirs:
        session_name = session_dir.name
        xiact_file = session_dir / f"{session_name}.xiact"

        if not xiact_file.exists():
            continue

        # Parse xiact directly
        xiact_data = parse_xiact(xiact_file)
        video_duration = get_video_duration(session_dir)

        # Compute coverage from xiact events
        infant_events = [{"start": e["start"], "end": e["end"]} for e in xiact_data["infant_engagement"]]
        caregiver_events = [{"start": e["start"], "end": e["end"]} for e in xiact_data["caregiver_engagement"]]
        all_engagement_events = infant_events + caregiver_events

        # Compute annotation span from infant/caregiver events
        # Start = first infant or caregiver annotation (whichever starts first)
        # End = last infant or caregiver annotation (whichever ends last)
        inf_starts = [float(e["start"]) for e in infant_events] if infant_events else []
        cgv_starts = [float(e["start"]) for e in caregiver_events] if caregiver_events else []
        inf_ends = [float(e["end"]) for e in infant_events] if infant_events else []
        cgv_ends = [float(e["end"]) for e in caregiver_events] if caregiver_events else []

        all_starts = inf_starts + cgv_starts
        all_ends = inf_ends + cgv_ends
        annotation_span_start = min(all_starts) if all_starts else 0
        annotation_span_end = max(all_ends) if all_ends else (video_duration or 0)

        # Compute coverage against VIDEO DURATION (for Plot 1)
        infant_coverage = compute_coverage(infant_events, video_duration)
        caregiver_coverage = compute_coverage(caregiver_events, video_duration)
        total_coverage = compute_coverage(all_engagement_events, video_duration)

        # Parse paradigm phases — reunion ends at last ANNOTATION, not video end (for Plot 3)
        paradigm = parse_paradigm_phases_from_xiact(
            xiact_data["paradigm_phases"], annotation_span_end
        )

        stat = {
            "session": session_name,
            "source": "xiact",
            "video_duration_sec": video_duration,
            "video_duration_min": round(video_duration / 60, 1) if video_duration else None,
            "annotation_span_start": round(annotation_span_start, 2),
            "annotation_span_end": round(annotation_span_end, 2),
            "infant": infant_coverage,
            "caregiver": caregiver_coverage,
            "total_engagement": total_coverage,
            "paradigm_phases": paradigm,
            "xiact_paradigm_events": xiact_data["paradigm_phases"],
        }
        all_stats.append(stat)

    # Save full stats JSON
    stats_json_path = OUTPUT_DIR / "session_stats_xiact.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    # ── Print summary table ──
    print(f"\n{'=' * 160}")
    print(f"SESSION-BY-SESSION ANALYSIS (from XIACT, annotation-based) — {len(all_stats)} sessions")
    print(f"{'=' * 160}")

    header = (f"{'Session':<30} {'VidDur':>8} {'Inf#':>5} {'InfCov':>7} {'Cgv#':>5} {'CgvCov':>7} "
              f"{'TotCov':>7} {'Play':>7} {'Trans':>7} {'SFP':>7} {'Reunion':>7}"
              f"  {'Paradigm Events'}")
    print(header)
    print("-" * 160)

    no_paradigm = []
    low_coverage = []

    for s in all_stats:
        vid_str = f"{s['video_duration_min']:.1f}m" if s['video_duration_min'] else "N/A"
        inf_n = s['infant']['num_events']
        inf_cov = f"{s['infant']['coverage_pct']:.0f}%"
        cgv_n = s['caregiver']['num_events']
        cgv_cov = f"{s['caregiver']['coverage_pct']:.0f}%"
        tot_cov = f"{s['total_engagement']['coverage_pct']:.0f}%"

        p = s['paradigm_phases']
        play = f"{p.get('play_duration', 0):.0f}s" if p.get('has_paradigm') else "—"
        trans = f"{p.get('transition_duration', 0):.0f}s" if p.get('has_paradigm') else "—"
        sfp = f"{p.get('sfp_duration', 0):.0f}s" if p.get('sfp_duration') else "—"
        reunion = f"{p.get('reunion_duration', 0):.0f}s" if p.get('reunion_duration') else "—"

        # Show raw paradigm events from xiact
        para_evts = s.get("xiact_paradigm_events", [])
        para_str = ", ".join(
            f"{e['code']}[{e['start']:.1f}-{e['end']:.1f}s]" for e in
            sorted(para_evts, key=lambda x: x['start'])
        ) if para_evts else "NONE"

        print(f"{s['session']:<30} {vid_str:>8} {inf_n:>5} {inf_cov:>7} {cgv_n:>5} {cgv_cov:>7} "
              f"{tot_cov:>7} {play:>7} {trans:>7} {sfp:>7} {reunion:>7}  {para_str}")

        if not p.get('has_paradigm'):
            no_paradigm.append(s['session'])
        if s['total_engagement']['coverage_pct'] < 50:
            low_coverage.append((s['session'], s['total_engagement']['coverage_pct']))

    # Summary
    vid_durations = [s['video_duration_sec'] for s in all_stats if s['video_duration_sec']]
    coverages = [s['total_engagement']['coverage_pct'] for s in all_stats]
    print(f"\n{'─' * 80}")
    print(f"Overall: {len(all_stats)} sessions (source: .xiact files)")
    print(f"  Video durations: min={min(vid_durations):.0f}s ({min(vid_durations)/60:.1f}m), "
          f"max={max(vid_durations):.0f}s ({max(vid_durations)/60:.1f}m), "
          f"mean={np.mean(vid_durations):.0f}s ({np.mean(vid_durations)/60:.1f}m), "
          f"total={sum(vid_durations)/3600:.1f}h")
    print(f"  Coverage (vs video duration): min={min(coverages):.0f}%, max={max(coverages):.0f}%, mean={np.mean(coverages):.0f}%")
    if no_paradigm:
        print(f"  Sessions WITHOUT paradigm phases: {len(no_paradigm)} → {no_paradigm}")
    if low_coverage:
        print(f"  Sessions with <50% coverage: {len(low_coverage)}")
        for name, pct in sorted(low_coverage, key=lambda x: x[1]):
            print(f"    {name}: {pct:.0f}%")

    # ──────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ──────────────────────────────────────────────────────────────
    sessions_sorted = sorted(all_stats, key=lambda s: s['video_duration_sec'] or 0, reverse=True)
    names = [s['session'] for s in sessions_sorted]
    n = len(names)
    y_pos = np.arange(n)

    # Expected paradigm durations (seconds)
    EXPECTED_PLAY = 120
    EXPECTED_SFP = 120
    EXPECTED_REUNION = 120
    TOLERANCE = 60

    def get_anomaly_notes(s):
        """Return list of short notes for anomalous phase durations."""
        p = s['paradigm_phases']
        if not p.get('has_paradigm'):
            return ["NO PARADIGM"]
        notes = []
        play_dur = p.get('play_duration', 0)
        sfp_dur = p.get('sfp_duration', 0)
        reunion_dur = p.get('reunion_duration', 0)
        cov = s['total_engagement']['coverage_pct']

        if play_dur > EXPECTED_PLAY + TOLERANCE:
            notes.append(f"Play={play_dur:.0f}s (>{EXPECTED_PLAY}s)")
        if sfp_dur > EXPECTED_SFP + TOLERANCE:
            notes.append(f"SFP={sfp_dur:.0f}s (>{EXPECTED_SFP}s)")
        elif sfp_dur < EXPECTED_SFP - TOLERANCE and sfp_dur > 0:
            notes.append(f"SFP={sfp_dur:.0f}s (<{EXPECTED_SFP}s)")
        if reunion_dur > EXPECTED_REUNION + TOLERANCE:
            notes.append(f"Reunion={reunion_dur:.0f}s (>{EXPECTED_REUNION}s)")
        if cov < 50:
            notes.append(f"Coverage={cov:.0f}%")
        return notes

    fig, axes = plt.subplots(1, 3, figsize=(28, max(22, n * 0.32)),
                              gridspec_kw={'width_ratios': [1, 0.8, 1.4]})
    fig.suptitle(
        f"Session-by-Session Analysis (from XIACT) — {n} sessions\n"
        f"Plot 1-2: Video-based coverage | Plot 3: SFP timeline (annotation-based, reunion ends at last annotation)",
        fontsize=15, fontweight="bold", y=1.01
    )

    # ── Plot 1: Video Duration + Coverage (video-based) ──
    ax1 = axes[0]
    vid_durs = [s['video_duration_sec'] or 0 for s in sessions_sorted]
    tot_covs = [s['total_engagement']['covered_seconds'] for s in sessions_sorted]

    ax1.barh(y_pos, vid_durs, color="#E0E0E0", edgecolor="white", label="Video Duration")
    ax1.barh(y_pos, tot_covs, color="#4CAF50", edgecolor="white", alpha=0.8, label="Annotated Coverage")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=6)
    ax1.set_xlabel("Seconds")
    ax1.set_title("Video Duration vs Annotation Coverage", pad=12)
    ax1.legend(loc="lower right")
    ax1.invert_yaxis()

    for i, s in enumerate(sessions_sorted):
        pct = s['total_engagement']['coverage_pct']
        is_low = pct < 50
        ax1.text(vid_durs[i] + 2, i, f"{pct:.0f}%", va='center', fontsize=5,
                 color='red' if is_low else 'black', fontweight='bold' if is_low else 'normal')
        if is_low:
            ax1.add_patch(plt.Rectangle(
                (-2, i - 0.45), vid_durs[i] + 4, 0.9,
                linewidth=1.5, edgecolor='red', facecolor='none', linestyle='-'
            ))

    # ── Plot 2: Infant vs Caregiver Coverage ──
    ax2 = axes[1]
    inf_covs = [s['infant']['coverage_pct'] for s in sessions_sorted]
    cgv_covs = [s['caregiver']['coverage_pct'] for s in sessions_sorted]

    bar_h = 0.35
    ax2.barh(y_pos - bar_h/2, inf_covs, bar_h, color="#4CAF50", label="Infant", edgecolor="white")
    ax2.barh(y_pos + bar_h/2, cgv_covs, bar_h, color="#2196F3", label="Caregiver", edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=6)
    ax2.set_xlabel("Coverage %")
    ax2.set_title("Infant vs Caregiver Coverage % (from XIACT)", pad=12)
    ax2.legend(loc="lower right")
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, linewidth=0.8, label="50% threshold")
    ax2.invert_yaxis()

    # ── Plot 3: Still Face Paradigm Phases (Timeline) — annotation-based ──
    ax3 = axes[2]
    phase_colors = {"play": "#66BB6A", "transition": "#FFA726", "sfp": "#EF5350", "reunion": "#42A5F5"}

    def add_segment_label(ax, left, width, y, label, min_width=15):
        """Add a small duration label centered on a segment if wide enough."""
        if width >= min_width:
            ax.text(left + width / 2, y, label, ha='center', va='center',
                    fontsize=4, color='white', fontweight='bold')

    for i, s in enumerate(sessions_sorted):
        ann_start = s.get('annotation_span_start', 0)
        ann_end = s.get('annotation_span_end', 0)
        p = s['paradigm_phases']

        if not p.get('has_paradigm'):
            bar_width = ann_end - ann_start
            ax3.barh(i, bar_width, left=ann_start, color="#E0E0E0", edgecolor="white", height=0.7)
            ax3.text(ann_start - 3, i, f"{ann_start:.0f}s", ha='right', va='center',
                     fontsize=4, color='#555555')
            ax3.text(ann_end + 5, i, "NO PARADIGM", va='center', fontsize=5,
                     color='gray', fontstyle='italic')
            continue

        # Show annotation start time near Y axis
        ax3.text(ann_start - 3, i, f"{ann_start:.0f}s", ha='right', va='center',
                 fontsize=4, color='#555555')

        # Play phase: annotation_start → transition_start
        play_end = p.get('transition_start', ann_start)
        play_dur = play_end - ann_start
        if play_dur > 0:
            ax3.barh(i, play_dur, left=ann_start, color=phase_colors["play"], edgecolor="white", height=0.7)
            add_segment_label(ax3, ann_start, play_dur, i, f"{play_dur:.0f}s")

        # Transition phase
        trans_start = p.get('transition_start', ann_start)
        trans_end = p.get('transition_end', trans_start)
        trans_dur = trans_end - trans_start
        ax3.barh(i, trans_dur, left=trans_start,
                 color=phase_colors["transition"], edgecolor="white", height=0.7)
        add_segment_label(ax3, trans_start, trans_dur, i, f"{trans_dur:.0f}s")

        # Still Face phase: SFP → RP
        sfp_start = p.get('sfp_start', trans_end)
        rp_start = p.get('reunion_start', sfp_start)
        sfp_dur = rp_start - sfp_start
        ax3.barh(i, sfp_dur, left=sfp_start,
                 color=phase_colors["sfp"], edgecolor="white", height=0.7)
        add_segment_label(ax3, sfp_start, sfp_dur, i, f"{sfp_dur:.0f}s")

        # Reunion phase: RP → last annotation end
        reunion_dur = ann_end - rp_start
        ax3.barh(i, reunion_dur, left=rp_start,
                 color=phase_colors["reunion"], edgecolor="white", height=0.7)
        add_segment_label(ax3, rp_start, reunion_dur, i, f"{reunion_dur:.0f}s")

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=6)
    ax3.set_xlabel("Seconds")
    ax3.set_title(
        "Still Face Paradigm Phases Timeline (annotation-based)\n"
        "Start = first annotation | End = last annotation — dotted lines at 2/4/6 min",
        pad=12, fontsize=11
    )
    ax3.invert_yaxis()

    # Reference lines at 2/4/6 min
    for mins, secs in [(2, 120), (4, 240), (6, 360)]:
        ax3.axvline(x=secs, color='black', linestyle=':', alpha=0.4, linewidth=1)
        ax3.text(secs, -1.5, f"{mins} min", ha='center', va='bottom', fontsize=7,
                 color='black', fontstyle='italic')

    # Legend
    legend_handles = [mpatches.Patch(color=c, label=l.capitalize())
                      for l, c in phase_colors.items()]
    legend_handles.append(mpatches.Patch(color="#E0E0E0", label="No paradigm info"))
    ax3.legend(handles=legend_handles, loc="lower right", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = OUTPUT_DIR / "session_analysis_with_xiact.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization → {fig_path}")

    # Print anomaly summary
    print(f"\n{'─' * 80}")
    print("ANOMALOUS SESSIONS (red-boxed in the plot):")
    for s in sessions_sorted:
        notes = get_anomaly_notes(s)
        if notes:
            print(f"  ⚠ {s['session']}: {' | '.join(notes)}")

    plt.close()

    # Save CSV
    csv_path = OUTPUT_DIR / "session_stats_xiact.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("session,video_duration_sec,video_duration_min,infant_events,infant_coverage_sec,"
                "infant_coverage_pct,caregiver_events,caregiver_coverage_sec,caregiver_coverage_pct,"
                "total_coverage_sec,total_coverage_pct,has_paradigm,play_duration,transition_start,"
                "transition_end,transition_duration,sfp_start,sfp_duration,reunion_start,"
                "reunion_duration,paradigm_events\n")
        for s in all_stats:
            p = s['paradigm_phases']
            para_evts = s.get("xiact_paradigm_events", [])
            para_str = "; ".join(
                f"{e['code']}[{e['start']:.1f}-{e['end']:.1f}]" for e in
                sorted(para_evts, key=lambda x: x['start'])
            )
            f.write(f"{s['session']},{s['video_duration_sec'] or ''},"
                    f"{s['video_duration_min'] or ''},"
                    f"{s['infant']['num_events']},{s['infant']['covered_seconds']},"
                    f"{s['infant']['coverage_pct']},"
                    f"{s['caregiver']['num_events']},{s['caregiver']['covered_seconds']},"
                    f"{s['caregiver']['coverage_pct']},"
                    f"{s['total_engagement']['covered_seconds']},{s['total_engagement']['coverage_pct']},"
                    f"{p.get('has_paradigm', False)},"
                    f"{p.get('play_duration', '')},"
                    f"{p.get('transition_start', '')},"
                    f"{p.get('transition_end', '')},"
                    f"{p.get('transition_duration', '')},"
                    f"{p.get('sfp_start', '')},"
                    f"{p.get('sfp_duration', '')},"
                    f"{p.get('reunion_start', '')},"
                    f"{p.get('reunion_duration', '')},"
                    f"\"{para_str}\"\n")
    print(f"Saved CSV → {csv_path}")
    print(f"Saved JSON → {stats_json_path}")


if __name__ == "__main__":
    main()
