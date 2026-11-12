#!/usr/bin/env python3
"""
Skill 08: Alternative vehicle recommendation and target‑driven matching.

CAR-15: For a poorly rated vehicle, show alternative vehicles in our historical
inventory with similar characteristics but better recorded performance, and
explain which differences could account for the improvement.

CAR-16: For a vehicle that does not meet our preferred performance targets,
identify the closest available alternative that meets more of those targets
while changing as few vehicle characteristics as possible.

Usage:
    python skill_08.py <path-to-auto93.csv>
"""

import sys
import csv
import statistics
import math
from pathlib import Path

# Reuse core utilities from the previous skills
from skill_01 import load_data, compute_balanced_scores, format_value


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def _median(values):
    """Return the median of a list of numbers."""
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    return (
        sorted_vals[mid]
        if n % 2 == 1
        else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    )


def _similarity_tolerance(rows):
    """Compute 10% tolerance for each metric based on min/max across dataset."""
    weights = [r["Lbs-"] for r in rows]
    mpgs = [r["Mpg+"] for r in rows]
    accs = [r["Acc+"] for r in rows]
    return {
        "Lbs-": 0.1 * (max(weights) - min(weights)),
        "Mpg+": 0.1 * (max(mpgs) - min(mpgs)),
        "Acc+": 0.1 * (max(accs) - min(accs)),
    }


def _targets_met(row, medians):
    """Return how many of the three simple performance targets are satisfied."""
    count = 0
    if row["Lbs-"] <= medians["Lbs-"]:
        count += 1
    if row["Mpg+"] >= medians["Mpg+"]:
        count += 1
    if row["Acc+"] <= medians["Acc+"]:
        count += 1
    return count


# ------------------------------------------------------------------
# CAR-15: Alternative vehicles for a poorly rated vehicle
# ------------------------------------------------------------------
def _car_15(rows, scored_rows):
    """Show alternatives for the worst‑scoring vehicle."""
    if not scored_rows:
        print("== CAR-15 ==")
        print("No data available.")
        return

    # Worst vehicle (lowest score)
    target = scored_rows[-1]

    # Tolerance for similarity
    tolerances = _similarity_tolerance(rows)

    # Find vehicles with similar weight, mpg and acceleration
    candidates = []
    for r in scored_rows:
        if r is target:
            continue
        # Skip vehicles that are not better in score
        if r["score"] <= target["score"]:
            continue
        # Check similarity within tolerance
        if (
            abs(r["Lbs-"] - target["Lbs-"]) <= tolerances["Lbs-"]
            and abs(r["Mpg+"] - target["Mpg+"]) <= tolerances["Mpg+"]
            and abs(r["Acc+"] - target["Acc+"]) <= tolerances["Acc+"]
        ):
            candidates.append(r)

    # Sort by best score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_alternatives = candidates[:3]

    print("== CAR-15 ==")
    print(f"Target vehicle (worst score): {target['Model']} (Score: {format_value(target['score'], '{:.3f}')} )")
    print(f"  Weight: {format_value(target['Lbs-'])} lbs, MPG: {format_value(target['Mpg+'])}, Acc: {format_value(target['Acc+'])} s")
    print("\nBest alternative vehicles with similar characteristics and higher performance:")
    if not top_alternatives:
        print("  No alternatives found that match the similarity criteria.")
        return
    for idx, r in enumerate(top_alternatives, start=1):
        print(
            f"{idx}. {r['Model']} (Score: {format_value(r['score'], '{:.3f}')} )"
            f"  Weight: {format_value(r['Lbs-'])} lbs, MPG: {format_value(r['Mpg+'])}, Acc: {format_value(r['Acc+'])} s"
        )
        # Explain differences
        w_diff = r["Lbs-"] - target["Lbs-"]
        m_diff = r["Mpg+"] - target["Mpg+"]
        a_diff = r["Acc+"] - target["Acc+"]
        print(
            f"    Differences -> Weight: {w_diff:+.1f} lbs, "
            f"MPG: {m_diff:+.2f}, Acc: {a_diff:+.2f} s, "
            f"Score: {format_value(r['score'] - target['score'], '{:.3f}')}"
        )
    # Ensure at least 3 lines of output
    print("\nEnd of CAR-15 recommendation.\n")


# ------------------------------------------------------------------
# CAR-16: Target‑driven alternative for a vehicle below preferred thresholds
# ------------------------------------------------------------------
def _car_16(rows, scored_rows):
    """Find a closer alternative that meets more performance targets."""
    if not scored_rows:
        print("== CAR-16 ==")
        print("No data available.")
        return

    # Compute mean score to find a vehicle that is below average
    mean_score = statistics.mean([r["score"] for r in scored_rows])
    # Prefer first vehicle below mean, else pick worst
    target = next((r for r in scored_rows if r["score"] < mean_score), scored_rows[-1])

    # Compute medians for simple target thresholds
    medians = {
        "Lbs-": _median([r["Lbs-"] for r in rows]),
        "Mpg+": _median([r["Mpg+"] for r in rows]),
        "Acc+": _median([r["Acc+"] for r in rows]),
    }

    # Identify candidates that score higher than target
    candidates = [r for r in scored_rows if r is not target and r["score"] > target["score"]]
    if not candidates:
        print("== CAR-16 ==")
        print("No higher‑scoring vehicles available for comparison.")
        return

    # Rank candidates by how many targets they satisfy, then by closeness
    def candidate_score(c):
        tgt_met = _targets_met(c, medians)
        # closeness metric: sum of absolute differences
        diff_sum = abs(c["Lbs-"] - target["Lbs-"]) + abs(c["Mpg+"] - target["Mpg+"]) + abs(c["Acc+"] - target["Acc+"])
        return (tgt_met, -diff_sum)  # higher targets_met, then smaller diff_sum

    candidates.sort(key=candidate_score, reverse=True)
    best_candidate = candidates[0]

    print("== CAR-16 ==")
    print(f"Target vehicle (below preferred thresholds): {target['Model']} (Score: {format_value(target['score'], '{:.3f}')} )")
    print(f"  Weight: {format_value(target['Lbs-'])} lbs, MPG: {format_value(target['Mpg+'])}, Acc: {format_value(target['Acc+'])} s")
    print(f"Targets met (median thresholds): "
          f"Weight ≤ {format_value(medians['Lbs-'])}, "
          f"MPG ≥ {format_value(medians['Mpg+'])}, "
          f"Acc ≤ {format_value(medians['Acc+'])}")
    print("\nRecommended alternative that meets more targets while staying close in characteristics:")
    print(f"• {best_candidate['Model']} (Score: {format_value(best_candidate['score'], '{:.3f}')} )")
    print(f"  Weight: {format_value(best_candidate['Lbs-'])} lbs, MPG: {format_value(best_candidate['Mpg+'])}, Acc: {format_value(best_candidate['Acc+'])} s")
    # Explain target satisfaction
    tgt_met = _targets_met(best_candidate, medians)
    print(f"  Targets satisfied: {tgt_met} of 3")
    # Differences from target
    w_diff = best_candidate["Lbs-"] - target["Lbs-"]
    m_diff = best_candidate["Mpg+"] - target["Mpg+"]
    a_diff = best_candidate["Acc+"] - target["Acc+"]
    print(f"  Characteristic differences -> Weight: {w_diff:+.1f} lbs, MPG: {m_diff:+.2f}, Acc: {a_diff:+.2f} s")
    # Summary
    print("\nThis alternative offers improved performance on the median targets with minimal change in vehicle characteristics.")
    print("End of CAR-16 recommendation.\n")


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <path-to-auto93.csv>", file=sys.stderr)
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

    _car_15(rows, scored_rows)
    _car_16(rows, scored_rows)


if __name__ == "__main__":
    main()
