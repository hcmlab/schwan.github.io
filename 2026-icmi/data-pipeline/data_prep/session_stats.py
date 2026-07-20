"""
session_stats.py
================
Session-by-session analysis of annotation coverage and Still Face Paradigm phases.
Produces per-session stats, a summary CSV, and visualizations.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

RAW_DATA_DIR = Path(r"X:\data\Schwan_T3_Clean")
FINETUNE_DIR = Path(r"X:\data\Schwan_T3_FineTune")
OUTPUT_DIR = FINETUNE_DIR / "stats"


def get_video_duration(session_dir: Path) -> float:
    meta_file = session_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        dur = meta.get("technical_metadata", {}).get("duration")
        if dur:
            return float(dur)
    return None


def compute_coverage(events: list, video_duration: float) -> dict:
    """Compute how much of the video is covered by annotations (union of intervals)."""
    if not events or video_duration is None:
        return {"covered_seconds": 0, "coverage_pct": 0, "num_events": 0}

    # Merge overlapping intervals
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
        "coverage_pct": round(100 * covered / video_duration, 1) if video_duration > 0 else 0,
        "num_events": len(events),
        "first_annotation_start": round(merged[0][0], 2),
        "last_annotation_end": round(merged[-1][1], 2),
    }


def parse_paradigm_phases(tracks: list, last_annotation_end: float) -> dict:
    """Extract Still Face Paradigm phase info from the Paradigm_Phases track.
    
    Reunion phase duration is computed as last_annotation_end - RP_start,
    NOT video_duration - RP_start, to avoid untrimmed video skew.
    """
    phases = {"has_paradigm": False}

    for t in tracks:
        if t.get("name") == "Paradigm_Phases":
            events = t.get("events", [])
            if not events:
                break

            phases["has_paradigm"] = True
            sorted_events = sorted(events, key=lambda e: float(e.get("start", 0)))

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
                elif code == "RP":
                    rp_start = start
                    phases["reunion_start"] = round(start, 2)

            # Compute phase durations from markers
            # Play phase: from start of video (or first annotation) to transition start
            if trans_start is not None:
                phases["play_duration"] = round(trans_start, 2)

            # Still Face phase: from SFP marker to RP marker
            if sfp_start is not None and rp_start is not None:
                phases["sfp_duration"] = round(rp_start - sfp_start, 2)

            # Reunion phase: from RP marker to LAST ANNOTATION END (not video end)
            if rp_start is not None and last_annotation_end is not None and last_annotation_end > rp_start:
                phases["reunion_duration"] = round(last_annotation_end - rp_start, 2)

            break

    return phases


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir()])

    all_stats = []

    for session_dir in session_dirs:
        session_name = session_dir.name
        ann_file = session_dir / f"{session_name}.json"
        if not ann_file.exists():
            continue

        with open(ann_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        video_duration = get_video_duration(session_dir)
        tracks = session_data.get("tracks", [])

        # Collect events per track
        infant_events = []
        caregiver_events = []
        all_engagement_events = []

        for t in tracks:
            name = t.get("name", "")
            events = t.get("events", [])
            if name == "Infant_Engagement":
                infant_events = events
                all_engagement_events.extend(events)
            elif name == "Caregiver_Engagement":
                caregiver_events = events
                all_engagement_events.extend(events)

        infant_coverage = compute_coverage(infant_events, video_duration)
        caregiver_coverage = compute_coverage(caregiver_events, video_duration)
        total_coverage = compute_coverage(all_engagement_events, video_duration)

        # Compute annotation span from infant/caregiver events
        # Start = first infant or caregiver annotation (whichever starts first)
        # End = last infant or caregiver annotation (whichever ends last)
        inf_starts = [float(e.get("start", 0)) for e in infant_events] if infant_events else []
        cgv_starts = [float(e.get("start", 0)) for e in caregiver_events] if caregiver_events else []
        inf_ends = [float(e.get("end", 0)) for e in infant_events] if infant_events else []
        cgv_ends = [float(e.get("end", 0)) for e in caregiver_events] if caregiver_events else []

        all_starts = inf_starts + cgv_starts
        all_ends = inf_ends + cgv_ends
        annotation_span_start = min(all_starts) if all_starts else 0
        annotation_span_end = max(all_ends) if all_ends else (video_duration or 0)

        paradigm = parse_paradigm_phases(tracks, annotation_span_end)

        stat = {
            "session": session_name,
            "video_duration_sec": video_duration,
            "video_duration_min": round(video_duration / 60, 1) if video_duration else None,
            "annotation_span_start": round(annotation_span_start, 2),
            "annotation_span_end": round(annotation_span_end, 2),
            "infant": infant_coverage,
            "caregiver": caregiver_coverage,
            "total_engagement": total_coverage,
            "paradigm_phases": paradigm,
        }
        all_stats.append(stat)

    # Save full stats JSON
    stats_json_path = OUTPUT_DIR / "session_stats.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    # ── Print summary table ──
    print(f"{'='*130}")
    print(f"SESSION-BY-SESSION ANALYSIS — {len(all_stats)} sessions")
    print(f"{'='*130}")
    header = (f"{'Session':<30} {'Duration':>8} {'Inf#':>5} {'InfCov':>7} {'Cgv#':>5} {'CgvCov':>7} "
              f"{'TotCov':>7} {'Play':>7} {'Trans':>7} {'SFP':>7} {'Reunion':>7}")
    print(header)
    print("-" * 130)

    no_paradigm = []
    low_coverage = []

    for s in all_stats:
        dur_str = f"{s['video_duration_min']:.1f}m" if s['video_duration_min'] else "N/A"
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

        print(f"{s['session']:<30} {dur_str:>8} {inf_n:>5} {inf_cov:>7} {cgv_n:>5} {cgv_cov:>7} "
              f"{tot_cov:>7} {play:>7} {trans:>7} {sfp:>7} {reunion:>7}")

        if not p.get('has_paradigm'):
            no_paradigm.append(s['session'])
        if s['total_engagement']['coverage_pct'] < 50:
            low_coverage.append((s['session'], s['total_engagement']['coverage_pct']))

    # Summary
    durations = [s['video_duration_sec'] for s in all_stats if s['video_duration_sec']]
    coverages = [s['total_engagement']['coverage_pct'] for s in all_stats]
    print(f"\n{'─'*80}")
    print(f"Overall: {len(all_stats)} sessions")
    print(f"  Video durations: min={min(durations):.0f}s ({min(durations)/60:.1f}m), "
          f"max={max(durations):.0f}s ({max(durations)/60:.1f}m), "
          f"mean={np.mean(durations):.0f}s ({np.mean(durations)/60:.1f}m), "
          f"total={sum(durations)/3600:.1f}h")
    print(f"  Coverage: min={min(coverages):.0f}%, max={max(coverages):.0f}%, mean={np.mean(coverages):.0f}%")
    if no_paradigm:
        print(f"  Sessions WITHOUT paradigm phases: {len(no_paradigm)} → {no_paradigm}")
    if low_coverage:
        print(f"  Sessions with <50% coverage: {len(low_coverage)}")
        for name, pct in sorted(low_coverage, key=lambda x: x[1]):
            print(f"    {name}: {pct:.0f}%")

    # ──────────────────────────────────────────────────────────────
    # VISUALIZATIONS
    # ──────────────────────────────────────────────────────────────
    sessions_sorted = sorted(all_stats, key=lambda s: s['video_duration_sec'] or 0, reverse=True)
    names = [s['session'] for s in sessions_sorted]
    n = len(names)
    y_pos = np.arange(n)

    # Expected paradigm durations (seconds)
    EXPECTED_PLAY = 120       # ~2 min
    EXPECTED_SFP = 120        # ~2 min
    EXPECTED_REUNION = 120    # ~2 min
    TOLERANCE = 60            # ±60s tolerance for flagging anomalies

    # Detect anomalous sessions
    def get_anomaly_notes(s):
        """Return a list of short notes for anomalous phase durations."""
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
        f"Session-by-Session Analysis — {n} sessions\n"
        f"Plot 1-2: Video-based coverage | Plot 3: SFP timeline (annotation-based, reunion ends at last annotation)",
        fontsize=15, fontweight="bold", y=1.01
    )

    # ── Plot 1: Video Duration + Coverage ──
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

    # Add coverage % labels + red box for low coverage
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
    ax2.set_title("Infant vs Caregiver Coverage %", pad=12)
    ax2.legend(loc="lower right")
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, linewidth=0.8, label="50% threshold")
    ax2.invert_yaxis()

    # ── Plot 3: Still Face Paradigm Phases (Timeline — annotation-based) ──
    ax3 = axes[2]
    phase_colors = {"play": "#66BB6A", "transition": "#FFA726", "sfp": "#EF5350", "reunion": "#42A5F5"}

    def add_segment_label(ax, left, width, y, label, min_width=15):
        """Add a small duration label centered on a segment if wide enough."""
        if width >= min_width:
            ax.text(left + width / 2, y, label, ha='center', va='center',
                    fontsize=4, color='white', fontweight='bold')

    for i, s in enumerate(sessions_sorted):
        ann_start = s.get('annotation_span_start', 0)
        ann_end = s.get('annotation_span_end', s['video_duration_sec'] or 0)
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

    # Add reference dotted lines at 2min (120s), 4min (240s), 6min (360s)
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
    fig_path = OUTPUT_DIR / "session_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization → {fig_path}")

    # Print anomaly summary
    print(f"\n{'─'*80}")
    print("ANOMALOUS SESSIONS (red-boxed in the plot):")
    for s in sessions_sorted:
        notes = get_anomaly_notes(s)
        if notes:
            print(f"  ⚠ {s['session']}: {' | '.join(notes)}")

    plt.close()

    # Save CSV for easy spreadsheet review
    csv_path = OUTPUT_DIR / "session_stats.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("session,video_duration_sec,video_duration_min,infant_events,infant_coverage_sec,"
                "infant_coverage_pct,caregiver_events,caregiver_coverage_sec,caregiver_coverage_pct,"
                "total_coverage_sec,total_coverage_pct,has_paradigm,play_duration,transition_duration,"
                "sfp_duration,reunion_duration\n")
        for s in all_stats:
            p = s['paradigm_phases']
            f.write(f"{s['session']},{s['video_duration_sec'] or ''},"
                    f"{s['video_duration_min'] or ''},"
                    f"{s['infant']['num_events']},{s['infant']['covered_seconds']},"
                    f"{s['infant']['coverage_pct']},"
                    f"{s['caregiver']['num_events']},{s['caregiver']['covered_seconds']},"
                    f"{s['caregiver']['coverage_pct']},"
                    f"{s['total_engagement']['covered_seconds']},{s['total_engagement']['coverage_pct']},"
                    f"{p.get('has_paradigm', False)},"
                    f"{p.get('play_duration', '')},"
                    f"{p.get('transition_duration', '')},"
                    f"{p.get('sfp_duration', '')},"
                    f"{p.get('reunion_duration', '')}\n")
    print(f"Saved CSV → {csv_path}")
    print(f"Saved JSON → {stats_json_path}")


if __name__ == "__main__":
    main()
