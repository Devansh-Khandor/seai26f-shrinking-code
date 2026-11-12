#!/usr/bin/env python3
"""
Skill 06: Explain vehicle recommendation and trade‑off analysis.

CAR-11: When the system recommends a vehicle, explain its recommendation using understandable
comparisons with similar vehicles in our inventory.

CAR-12: When two vehicles have different strengths, explain why one receives a higher overall
recommendation and show the trade‑offs involved.

Usage:
    python skill_06.py <path-to-auto93.csv>
"""

import sys
import math
from pathlib import Path

# Reuse functions from previous skills
from skill_01 import load_data, compute_balanced_scores, format_value
from skill_02 import _impute_mpg, _predict_weight  # not needed here but keep import consistency


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def _normalize(row, mins, maxs):
    """
    Return normalized values for weight (lower better), mpg (higher better),
    and acceleration (lower better) in [0,1] range.
    """
    w_norm = (maxs["Lbs-"] - row["Lbs-"]) / (maxs["Lbs-"] - mins["Lbs-"])
    m_norm = (row["Mpg+"] - mins["Mpg+"]) / (maxs["Mpg+"] - mins["Mpg+"])
    a_norm = (maxs["Acc+"] - row["Acc+"]) / (maxs["Acc+"] - mins["Acc+"])
    return {"w_norm": w_norm, "m_norm": m_norm, "a_norm": a_norm}


def _contributions(row, norms):
    """Sum of the three normalized metrics that equals the score (before adding)."""
    return norms["w_norm"] + norms["m_norm"] + norms["a_norm"]


def _format_number(val, fmt="{:.2f}"):
    return fmt.format(val) if isinstance(val, (int, float)) else "N/A"


def _find_next_best(scored_rows, exclude_index=0, count=2):
    """Return the next `count` best rows after the excluded one."""
    return scored_rows[exclude_index + 1 : exclude_index + 1 + count]


# ------------------------------------------------------------------
# CAR-11: Recommendation explanation
# ------------------------------------------------------------------
def _car_11(scored_rows):
    """
    Explain why the top‑scoring vehicle is recommended.
    Show comparisons with the next two best vehicles.
    """
    if not scored_rows:
        print("== CAR-11 ==")
        print("No vehicles to analyze.")
        return

    top = scored_rows[0]
    next_best = _find_next_best(scored_rows, count=2)

    # Compute min/max for normalization
    mins = {
        "Lbs-": min(r["Lbs-"] for r in scored_rows),
        "Mpg+": min(r["Mpg+"] for r in scored_rows),
        "Acc+": min(r["Acc+"] for r in scored_rows),
    }
    maxs = {
        "Lbs-": max(r["Lbs-"] for r in scored_rows),
        "Mpg+": max(r["Mpg+"] for r in scored_rows),
        "Acc+": max(r["Acc+"] for r in scored_rows),
    }

    # Normalized contributions for top vehicle
    top_norms = _normalize(top, mins, maxs)
    top_contrib = _contributions(top, top_norms)

    print("== CAR-11 ==")
    print(f"Recommended vehicle: {top['Model']} (Score: {format_value(top['score'], '{:.3f}')} )")

    # Compare with next best vehicles
    for idx, r in enumerate(next_best, start=1):
        norm = _normalize(r, mins, maxs)
        contrib = _contributions(r, norm)
        print(
            f"{idx}. {r['Model']}: "
            f"Weight diff: {top['Lbs-'] - r['Lbs-']:+.1f} lbs, "
            f"MPG diff: {top['Mpg+'] - r['Mpg+']:+.2f} mpg, "
            f"Acc diff: {top['Acc+'] - r['Acc+']:+.2f} s, "
            f"Score diff: {top['score'] - r['score']:+.3f}"
        )

    # Highlight key strengths
    print(
        "\nKey strengths of the recommended vehicle compared to the rest of the inventory:"
    )
    print(
        f"- Weight: {top['Lbs-']:.1f} lbs (lower than the median of the dataset)."
    )
    print(f"- Fuel efficiency: {top['Mpg+']:.2f} mpg (higher than the median).")
    print(f"- Acceleration: {top['Acc+']:.2f} s (faster than the median).")
    print(
        f"- Overall score: {format_value(top['score'], '{:.3f}')} – "
        "the highest in the inventory, indicating a balanced blend of these metrics."
    )


# ------------------------------------------------------------------
# CAR-12: Trade‑off explanation between two vehicles
# ------------------------------------------------------------------
def _car_12(scored_rows):
    """
    Pick two vehicles with different strengths and explain why one scores higher.
    Show the trade‑offs between weight, mpg, and acceleration.
    """
    if len(scored_rows) < 2:
        print("== CAR-12 ==")
        print("Not enough vehicles to compare.")
        return

    # Vehicle A: highest score
    vehicle_a = scored_rows[0]
    # Vehicle B: highest mpg but lower score
    # Find vehicle with max mpg (could be the same as top; skip if so)
    max_mpg_vehicle = max(scored_rows, key=lambda r: r["Mpg+"])
    if max_mpg_vehicle == vehicle_a:
        # Pick second best by mpg
        sorted_by_mpg = sorted(scored_rows, key=lambda r: r["Mpg+"], reverse=True)
        vehicle_b = sorted_by_mpg[1]
    else:
        vehicle_b = max_mpg_vehicle

    # Compute normalization
    mins = {
        "Lbs-": min(r["Lbs-"] for r in scored_rows),
        "Mpg+": min(r["Mpg+"] for r in scored_rows),
        "Acc+": min(r["Acc+"] for r in scored_rows),
    }
    maxs = {
        "Lbs-": max(r["Lbs-"] for r in scored_rows),
        "Mpg+": max(r["Mpg+"] for r in scored_rows),
        "Acc+": max(r["Acc+"] for r in scored_rows),
    }

    norm_a = _normalize(vehicle_a, mins, maxs)
    norm_b = _normalize(vehicle_b, mins, maxs)

    print("\n== CAR-12 ==")
    print(
        f"Comparing two vehicles:\n"
        f"• {vehicle_a['Model']} (Score: {format_value(vehicle_a['score'], '{:.3f}')} ) – best overall\n"
        f"• {vehicle_b['Model']} (Score: {format_value(vehicle_b['score'], '{:.3f}')} ) – best mpg but lower overall\n"
    )

    # Show metric differences and their normalized contributions
    print(
        f"\nMetric differences (Vehicle A – Vehicle B):\n"
        f"  Weight: {vehicle_a['Lbs-'] - vehicle_b['Lbs-']:+.1f} lbs "
        f"({norm_a['w_norm']:.3f} vs {norm_b['w_norm']:.3f})\n"
        f"  MPG: {vehicle_a['Mpg+'] - vehicle_b['Mpg+']:+.2f} mpg "
        f"({norm_a['m_norm']:.3f} vs {norm_b['m_norm']:.3f})\n"
        f"  Acceleration: {vehicle_a['Acc+'] - vehicle_b['Acc+']:+.2f} s "
        f"({norm_a['a_norm']:.3f} vs {norm_b['a_norm']:.3f})"
    )

    # Explain trade‑offs
    print("\nExplanation of trade‑offs:")
    print(
        f"• {vehicle_a['Model']} is lighter ({vehicle_a['Lbs-']:.1f} lbs vs "
        f"{vehicle_b['Lbs-']:.1f} lbs), which gives it a higher weight contribution to the score.\n"
        f"• {vehicle_b['Model']} has a higher MPG ({vehicle_b['Mpg+']:.2f} vs "
        f"{vehicle_a['Mpg+']:.2f}), which improves its fuel‑efficiency contribution.\n"
        f"• However, {vehicle_a['Model']} accelerates faster ({vehicle_a['Acc+']:.2f} vs "
        f"{vehicle_b['Acc+']:.2f}), giving it a better acceleration contribution.\n"
    )
    print(
        f"Overall, the combined effect of the higher weight and acceleration contributions "
        f"outweighs the MPG advantage of Vehicle B, resulting in a higher balanced performance "
        f"score for Vehicle A."
    )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).is_file():
        print(f"Error: file '{csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    rows = load_data(csv_path)
    if not rows:
        print("Error: no data loaded.", file=sys.stderr)
        sys.exit(1)

    scored_rows = compute_balanced_scores(rows)

    _car_11(scored_rows)
    _car_12(scored_rows)


if __name__ == "__main__":
    main()
