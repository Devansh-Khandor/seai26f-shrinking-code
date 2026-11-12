#!/usr/bin/env python3
"""
Extended data‑analysis script for the Auto93 dataset.

Adds two new client requests:
    CAR-15 – For a poorly rated vehicle, show alternative vehicles in our historical
             inventory with similar characteristics but better recorded performance,
             and explain which differences could account for the improvement.
    CAR-16 – For a vehicle that does not meet our preferred performance targets,
             identify the closest available alternative that meets more of those
             targets while changing as few vehicle characteristics as possible.

Usage:
    python skill_08.py <path_to_auto93.csv>
"""

import sys
from typing import List, Dict, Any, Tuple

# Import helpers from the earlier skill modules
from skill_01 import read_csv, compute_score, compute_stats, format_float


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _bounds(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Return min/max bounds for weight, mpg and acceleration, used for score calculation.
    """
    weight_bounds = {
        'min': min(r['Lbs-'] for r in rows if r['Lbs-'] is not None),
        'max': max(r['Lbs-'] for r in rows if r['Lbs-'] is not None),
    }
    mpg_bounds = {
        'min': min(r['Mpg+'] for r in rows if r['Mpg+'] is not None),
        'max': max(r['Mpg+'] for r in rows if r['Mpg+'] is not None),
    }
    accel_bounds = {
        'min': min(r['Acc+'] for r in rows if r['Acc+'] is not None),
        'max': max(r['Acc+'] for r in rows if r['Acc+'] is not None),
    }
    return weight_bounds, mpg_bounds, accel_bounds


def _compute_scores(rows: List[Dict[str, Any]]) -> None:
    """Compute composite score for every row that has all required metrics."""
    weight_bounds, mpg_bounds, accel_bounds = _bounds(rows)
    for r in rows:
        if None in (r['Lbs-'], r['Mpg+'], r['Acc+']):
            continue
        r['score'] = compute_score(r, weight_bounds, mpg_bounds, accel_bounds)


def _find_worst_vehicle(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the vehicle with the lowest composite score."""
    valid = [r for r in rows if 'score' in r]
    return min(valid, key=lambda r: r['score']) if valid else None


def _similarity_filter(v1: Dict[str, Any], v2: Dict[str, Any], tolerance: float = 0.10) -> bool:
    """
    Return True if v2 is within `tolerance` (relative) of v1 for
    weight, mpg and acceleration.
    """
    for key in ('Lbs-', 'Mpg+', 'Acc+'):
        if v1[key] is None or v2[key] is None:
            return False
        if abs(v2[key] - v1[key]) > tolerance * abs(v1[key]):
            return False
    return True


def _percentage_diff(a: float, b: float) -> float:
    """Return percent difference of a relative to b."""
    if b == 0:
        return 0.0
    return ((a - b) / b) * 100.0


def _explain_differences(v1: Dict[str, Any], v2: Dict[str, Any]) -> str:
    """Return a textual explanation of differences between two vehicles."""
    diff_parts = []
    for key in ('Lbs-', 'Mpg+', 'Acc+'):
        percent = _percentage_diff(v2[key], v1[key])
        diff_parts.append(f"{key} {percent:+.1f}%")
    return ", ".join(diff_parts)


def _target_meets(v: Dict[str, Any]) -> int:
    """Count how many of the preferred targets are met."""
    count = 0
    if v['Lbs-'] is not None and v['Lbs-'] <= 3500:
        count += 1
    if v['Mpg+'] is not None and v['Mpg+'] >= 25:
        count += 1
    if v['Acc+'] is not None and v['Acc+'] <= 15:
        count += 1
    return count


def _target_difference_sum(v: Dict[str, Any]) -> float:
    """
    Sum of absolute differences from the preferred thresholds
    (positive means the vehicle is further from the target).
    """
    diff = 0.0
    if v['Lbs-'] is not None:
        diff += max(0, v['Lbs-'] - 3500)   # heavier
    if v['Mpg+'] is not None:
        diff += max(0, 25 - v['Mpg+'])     # lower mpg
    if v['Acc+'] is not None:
        diff += max(0, v['Acc+'] - 15)     # slower
    return diff


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Compute scores for all vehicles
    _compute_scores(rows)

    # Identify the poorly rated vehicle (lowest score)
    worst = _find_worst_vehicle(rows)
    if worst is None:
        print("No vehicles with complete data to compute scores.")
        return

    # ----------------------------------------------------------------------- #
    # CAR-15 – Similar vehicles with better performance
    # ----------------------------------------------------------------------- #
    print("== CAR-15 ==")
    alternatives = []
    for r in rows:
        if r is worst:
            continue
        if not _similarity_filter(worst, r):
            continue
        if 'score' not in r or r['score'] <= worst['score']:
            continue
        alternatives.append(r)

    # Sort by score descending
    alternatives.sort(key=lambda r: r['score'], reverse=True)
    top3 = alternatives[:3] if alternatives else []

    if not top3:
        print("No suitable alternatives found for the poorly rated vehicle.")
    else:
        for idx, v in enumerate(top3, start=1):
            diff_desc = _explain_differences(worst, v)
            print(f"{idx}. Alternative Vehicle: Clndrs={int(v['Clndrs'])}, "
                  f"Lbs-={int(v['Lbs-'])}, Mpg+={int(v['Mpg+'])}, Acc+={format_float(v['Acc+'])}, "
                  f"score={format_float(v['score'])} | Differences: {diff_desc}")

    # ----------------------------------------------------------------------- #
    # CAR-16 – Closest alternative meeting more performance targets
    # ----------------------------------------------------------------------- #
    print("\n== CAR-16 ==")
    # Find all vehicles that meet at least one target
    candidate_rows = [r for r in rows if r is not worst and _target_meets(r) > 0]
    if not candidate_rows:
        print("No alternatives meet any of the preferred performance targets.")
        return

    # Sort primarily by number of targets met (descending),
    # then by total difference from thresholds (ascending).
    candidate_rows.sort(key=lambda r: (-_target_meets(r), _target_difference_sum(r)))

    top3_alt = candidate_rows[:3]
    for idx, v in enumerate(top3_alt, start=1):
        # Explain which targets are improved relative to the worst vehicle
        improvements = []
        if v['Lbs-'] is not None and worst['Lbs-'] is not None:
            if v['Lbs-'] < worst['Lbs-']:
                improvements.append(f"lighter by {int(worst['Lbs-'] - v['Lbs-'])} lbs")
        if v['Mpg+'] is not None and worst['Mpg+'] is not None:
            if v['Mpg+'] > worst['Mpg+']:
                improvements.append(f"higher mpg by {int(v['Mpg+'] - worst['Mpg+'])}")
        if v['Acc+'] is not None and worst['Acc+'] is not None:
            if v['Acc+'] < worst['Acc+']:
                improvements.append(f"faster acceleration by {format_float(worst['Acc+'] - v['Acc+'])} s")
        improvement_str = "; ".join(improvements) if improvements else "no improvement in key metrics"
        print(f"{idx}. Alternative Vehicle: Clndrs={int(v['Clndrs'])}, "
              f"Lbs-={int(v['Lbs-'])}, Mpg+={int(v['Mpg+'])}, Acc+={format_float(v['Acc+'])}, "
              f"score={format_float(v['score'])} | {improvement_str}")


if __name__ == "__main__":
    main()
