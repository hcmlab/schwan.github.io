"""
annotation_stats.py
===================
Quick analysis of the prepared annotation distribution across all sessions.
Produces per-label counts, durations, and a bar chart visualization.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

FINETUNE_DIR = Path(r"X:\data\Schwan_T3_FineTune")
OUTPUT_DIR = FINETUNE_DIR / "stats"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect stats
    code_counts = Counter()
    code_durations = defaultdict(float)  # total original duration per code
    track_counts = Counter()
    session_counts = {}  # session -> count
    code_labels = {}  # short_code -> label name

    session_dirs = sorted([
        d for d in FINETUNE_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}_finetune_annotations.json").exists()
    ])

    for session_dir in session_dirs:
        session_name = session_dir.name
        ann_file = session_dir / f"{session_name}_finetune_annotations.json"

        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        session_counts[session_name] = len(annotations)

        for ann in annotations:
            code = ann["short_code"]
            label = ann.get("label", code)
            track = ann["track"]
            orig_dur = ann["original_end"] - ann["original_start"]

            code_counts[code] += 1
            code_durations[code] += orig_dur
            track_counts[track] += 1
            code_labels[code] = label

    # ── Print summary ──
    total = sum(code_counts.values())
    print(f"{'='*70}")
    print(f"ANNOTATION STATISTICS — {len(session_dirs)} sessions, {total} total annotations")
    print(f"{'='*70}\n")

    # By code
    print(f"{'Code':<8} {'Label':<40} {'Count':>6} {'%':>7} {'Total Dur (s)':>14} {'Avg Dur (s)':>12}")
    print("-" * 90)
    for code, count in code_counts.most_common():
        label = code_labels.get(code, code)
        pct = 100 * count / total
        dur = code_durations[code]
        avg_dur = dur / count if count > 0 else 0
        print(f"{code:<8} {label:<40} {count:>6} {pct:>6.1f}% {dur:>13.1f} {avg_dur:>11.1f}")

    print(f"\n{'─'*70}")
    print("By Track:")
    for track, count in track_counts.most_common():
        print(f"  {track}: {count} ({100*count/total:.1f}%)")

    # Sessions with fewest/most annotations
    sorted_sessions = sorted(session_counts.items(), key=lambda x: x[1])
    print(f"\nSmallest sessions: {sorted_sessions[:3]}")
    print(f"Largest sessions:  {sorted_sessions[-3:]}")
    print(f"Average annotations/session: {total/len(session_dirs):.1f}")

    # ── Plot 1: Code distribution bar chart ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"ICEP Annotation Distribution — {len(session_dirs)} sessions, {total} annotations",
                 fontsize=16, fontweight="bold")

    # Sort codes by count
    sorted_codes = [c for c, _ in code_counts.most_common()]
    counts = [code_counts[c] for c in sorted_codes]
    labels = [f"{c}\n({code_labels.get(c, c)[:20]})" for c in sorted_codes]

    # Assign colors by track type
    colors = []
    for c in sorted_codes:
        if c.startswith("I"):
            colors.append("#4CAF50")  # green for infant
        elif c.startswith("C"):
            colors.append("#2196F3")  # blue for caregiver
        else:
            colors.append("#9E9E9E")  # gray for unknown

    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(sorted_codes)), counts, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(sorted_codes)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Count")
    ax1.set_title("Annotation Count per ICEP Code")
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha="center", va="bottom", fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4CAF50", label="Infant"),
                       Patch(facecolor="#2196F3", label="Caregiver"),
                       Patch(facecolor="#9E9E9E", label="Other/Unknown")]
    ax1.legend(handles=legend_elements, loc="upper right")

    # ── Plot 2: Percentage pie chart ──
    ax2 = axes[0, 1]
    # Group smaller codes into "Other" for readability
    threshold = 0.02 * total
    pie_labels = []
    pie_sizes = []
    other_size = 0
    for c in sorted_codes:
        if code_counts[c] >= threshold:
            pie_labels.append(f"{c} ({code_counts[c]})")
            pie_sizes.append(code_counts[c])
        else:
            other_size += code_counts[c]
    if other_size > 0:
        pie_labels.append(f"Other ({other_size})")
        pie_sizes.append(other_size)

    ax2.pie(pie_sizes, labels=pie_labels, autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 8})
    ax2.set_title("Annotation Distribution (%)")

    # ── Plot 3: Average duration per code ──
    ax3 = axes[1, 0]
    avg_durs = [code_durations[c] / code_counts[c] for c in sorted_codes]
    ax3.barh(range(len(sorted_codes)), avg_durs, color=colors, edgecolor="white")
    ax3.set_yticks(range(len(sorted_codes)))
    ax3.set_yticklabels([f"{c}" for c in sorted_codes], fontsize=9)
    ax3.set_xlabel("Average Duration (seconds)")
    ax3.set_title("Average Annotation Duration per Code")
    ax3.invert_yaxis()

    # ── Plot 4: Annotations per session ──
    ax4 = axes[1, 1]
    session_names_short = [s[:12] for s in sorted(session_counts.keys())]
    session_vals = [session_counts[s] for s in sorted(session_counts.keys())]
    ax4.barh(range(len(session_names_short)), session_vals, color="#FF9800", edgecolor="white", linewidth=0.3)
    ax4.set_yticks(range(len(session_names_short)))
    ax4.set_yticklabels(session_names_short, fontsize=5)
    ax4.set_xlabel("Number of Annotations")
    ax4.set_title("Annotations per Session")
    ax4.invert_yaxis()

    plt.tight_layout()
    out_fig = OUTPUT_DIR / "annotation_distribution.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot → {out_fig}")
    plt.close()

    # ── Save stats as JSON ──
    stats_json = {
        "total_sessions": len(session_dirs),
        "total_annotations": total,
        "code_counts": dict(code_counts.most_common()),
        "code_labels": code_labels,
        "track_counts": dict(track_counts),
        "avg_annotations_per_session": round(total / len(session_dirs), 1),
        "code_avg_duration_seconds": {c: round(code_durations[c]/code_counts[c], 2) for c in sorted_codes},
    }
    stats_out = OUTPUT_DIR / "annotation_stats.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)
    print(f"Saved stats JSON → {stats_out}")


if __name__ == "__main__":
    main()
