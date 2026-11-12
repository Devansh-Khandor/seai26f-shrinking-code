#!/usr/bin/env python3
"""
Skill 08 – Comparative Recommendations and Alternatives

This script implements two new diagnostic components for the
used‑car dealership data:

* CAR-15 – For a poorly rated vehicle, show alternative vehicles in our historical
  inventory with similar characteristics but better recorded performance, and
  explain which differences could account for the improvement.

* CAR-16 – For a vehicle that does not meet our preferred performance targets,
  identify the closest available alternative that meets more of those targets
  while changing as few vehicle characteristics as possible.

The script is self‑contained and uses only the Python standard library.
"""

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read the CSV file into a list of dictionaries."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def safe_float(val: str) -> Optional[float]:
    """Return a float or None if the conversion fails."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def median(values: List[float]) -> float:
    """Return the median of a list of numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    return (
        (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        if n % 2 == 0
        else sorted_vals[mid]
    )


# ----------------------------------------------------------------------
# Composite score calculation (identical to skill_01)
# ----------------------------------------------------------------------
def compute_scores(rows: List[Dict[str, str]]) -> List[Tuple[float, int, Dict[str, str]]]:
    """Return a list of tuples: (score, 1‑based index, row)."""
    # Extract numeric columns, ignoring rows with missing data
    weights = [safe_float(r["Lbs-"]) for r in rows]
    mpgs = [safe_float(r["Mpg+"]) for r in rows]
    accs = [safe_float(r["Acc+"]) for r in rows]

    weights = [v for v in weights if v is not None]
    mpgs = [v for v in mpgs if v is not None]
    accs = [v for v in accs if v is not None]

    if not weights or not mpgs or not accs:
        return []

    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    def norm(val: float, low: float, high: float, higher_is_better: bool = True) -> float:
        if high == low:
            return 0.5
        return (
            (val - low) / (high - low)
            if higher_is_better
            else (high - val) / (high - low)
        )

    scores: List[Tuple[float, int, Dict[str, str]]] = []
    for idx, r in enumerate(rows, start=1):
        w = safe_float(r["Lbs-"])
        m = safe_float(r["Mpg+"])
        a = safe_float(r["Acc+"])
        if None in (w, m, a):
            continue
        w_score = norm(w, w_min, w_max, higher_is_better=False)
        m_score = norm(m, mpg_min, mpg_max)
        a_score = norm(a, acc_min, acc_max)
        composite = (w_score + m_score + a_score) / 3
        scores.append((composite, idx, r))
    return scores


# ----------------------------------------------------------------------
# CAR-15 – Find alternatives for the worst vehicle
# ----------------------------------------------------------------------
def find_worst_vehicle(scores: List[Tuple[float, int, Dict[str, str]]]) -> Tuple[Optional[Tuple[float, int, Dict[str, str]]], List[Tuple[float, int, Dict[str, str]]]]:
    """Return the worst vehicle and all vehicles sorted by similarity."""
    if not scores:
        return None, []

    # Find the vehicle with the lowest composite score (excluding NaN)
    worst = min(scores, key=lambda x: x[0])
    # Compute bounds for normalization
    weights = [safe_float(r["Lbs-"]) for _, _, r in scores]
    mpgs = [safe_float(r["Mpg+"]) for _, _, r in scores]
    accs = [safe_float(r["Acc+"]) for _, _, r in scores]
    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    def norm(val: float, low: float, high: float) -> float:
        if high == low:
            return 0.5
        return (val - low) / (high - low)

    # Compute Euclidean distance in normalized space
    def distance(idx: int, w: float, m: float, a: float) -> float:
        return math.sqrt(
            (norm(w, w_min, w_max) - norm(worst[2]["Lbs-"], w_min, w_max)) ** 2
            + (norm(m, mpg_min, mpg_max) - norm(worst[2]["Mpg+"], mpg_min, mpg_max)) ** 2
            + (norm(a, acc_min, acc_max) - norm(worst[2]["Acc+"], acc_min, acc_max)) ** 2
        )

    # Filter vehicles that have a better score than the worst
    candidates = [
        (score, idx, r)
        for score, idx, r in scores
        if score > worst[0]
    ]

    # Sort by distance (ascending)
    candidates.sort(key=lambda x: distance(x[1], safe_float(x[2]["Lbs-"]), safe_float(x[2]["Mpg+"]), safe_float(x[2]["Acc+"])))
    return worst, candidates[:3]  # top 3 alternatives


def explain_car_15(worst: Tuple[float, int, Dict[str, str]], alternatives: List[Tuple[float, int, Dict[str, str]]]) -> None:
    """Print CAR‑15 output."""
    print("== CAR-15 ==")
    if worst is None or not alternatives:
        print("No alternatives could be found.")
        return

    w_worst = safe_float(worst[2]["Lbs-"])
    m_worst = safe_float(worst[2]["Mpg+"])
    a_worst = safe_float(worst[2]["Acc+"])
    print(f"Poorly rated vehicle: Row {worst[1]} – Weight {w_worst:.1f} lbs, MPG {m_worst:.1f}, Acc {a_worst:.2f}s")

    for alt_idx, (score, idx, r) in enumerate(alternatives, start=1):
        w_alt = safe_float(r["Lbs-"])
        m_alt = safe_float(r["Mpg+"])
        a_alt = safe_float(r["Acc+"])
        print(f"Alternative {alt_idx}: Row {idx} – Weight {w_alt:.1f} lbs, MPG {m_alt:.1f}, Acc {a_alt:.2f}s, Score {score:.3f}")
        # Explain differences
        diff_w = w_worst - w_alt
        diff_m = m_alt - m_worst
        diff_a = a_worst - a_alt
        msgs = []
        if diff_w > 0:
            msgs.append(f"{abs(diff_w):.1f} lbs lighter")
        if diff_m > 0:
            msgs.append(f"{abs(diff_m):.1f} mpg better")
        if diff_a > 0:
            msgs.append(f"{abs(diff_a):.2f}s faster")
        if msgs:
            print(f"  Differences: {', '.join(msgs)}")
        else:
            print("  No clear improvements in the three metrics.")


# ----------------------------------------------------------------------
# CAR-16 – Find alternative meeting more performance targets
# ----------------------------------------------------------------------
def find_best_alternative_for_worst(worst: Tuple[float, int, Dict[str, str]], scores: List[Tuple[float, int, Dict[str, str]]]) -> Tuple[Optional[Tuple[float, int, Dict[str, str]]], int]:
    """Return the closest alternative meeting more performance targets and count of targets met."""
    if worst is None:
        return None, 0

    # Compute median thresholds as preferred targets
    weights = [safe_float(r["Lbs-"]) for _, _, r in scores]
    mpgs = [safe_float(r["Mpg+"]) for _, _, r in scores]
    accs = [safe_float(r["Acc+"]) for _, _, r in scores]
    w_thresh = median([v for v in weights if v is not None])
    mpg_thresh = median([v for v in mpgs if v is not None])
    acc_thresh = median([v for v in accs if v is not None])

    # Helper to count targets met
    def targets_met(r: Dict[str, str]) -> int:
        w = safe_float(r["Lbs-"])
        m = safe_float(r["Mpg+"])
        a = safe_float(r["Acc+"])
        if w is None or m is None or a is None:
            return -1  # invalid
        count = 0
        if w <= w_thresh:  # lower weight better
            count += 1
        if m >= mpg_thresh:  # higher mpg better
            count += 1
        if a <= acc_thresh:  # lower acceleration better
            count += 1
        return count

    worst_targets = targets_met(worst[2])
    if worst_targets == -1:
        return None, 0

    # Find alternatives that meet strictly more targets
    best = None
    best_dist = None
    best_targets = 0
    w_worst = safe_float(worst[2]["Lbs-"])
    m_worst = safe_float(worst[2]["Mpg+"])
    a_worst = safe_float(worst[2]["Acc+"])
    for score, idx, r in scores:
        if score == worst[0]:
            continue  # skip itself
        t = targets_met(r)
        if t <= worst_targets:
            continue
        w = safe_float(r["Lbs-"])
        m = safe_float(r["Mpg+"])
        a = safe_float(r["Acc+"])
        # Euclidean distance in raw space
        dist = math.sqrt((w - w_worst) ** 2 + (m - m_worst) ** 2 + (a - a_worst) ** 2)
        if best is None or dist < best_dist:
            best = (score, idx, r)
            best_dist = dist
            best_targets = t

    return best, best_targets


def explain_car_16(best: Tuple[float, int, Dict[str, str]], best_targets: int, worst: Tuple[float, int, Dict[str, str]]) -> None:
    """Print CAR‑16 output."""
    print("== CAR-16 ==")
    if best is None:
        print("No alternative vehicle meets more performance targets.")
        return

    w_best = safe_float(best[2]["Lbs-"])
    m_best = safe_float(best[2]["Mpg+"])
    a_best = safe_float(best[2]["Acc+"])
    print(f"Alternative vehicle: Row {best[1]} – Weight {w_best:.1f} lbs, MPG {m_best:.1f}, Acc {a_best:.2f}s")
    print(f"Targets met: {best_targets} / 3")

    # Compare differences to the original worst vehicle
    w_worst = safe_float(worst[2]["Lbs-"])
    m_worst = safe_float(worst[2]["Mpg+"])
    a_worst = safe_float(worst[2]["Acc+"])
    diff_w = w_best - w_worst
    diff_m = m_best - m_worst
    diff_a = a_best - a_worst
    msgs = []
    if diff_w != 0:
        msgs.append(f"{'increase' if diff_w > 0 else 'decrease'} by {abs(diff_w):.1f} lbs")
    if diff_m != 0:
        msgs.append(f"{'increase' if diff_m > 0 else 'decrease'} by {abs(diff_m):.1f} mpg")
    if diff_a != 0:
        msgs.append(f"{'increase' if diff_a > 0 else 'decrease'} by {abs(diff_a):.2f}s")
    if msgs:
        print(f"Changes compared to the poor vehicle: {', '.join(msgs)}")
    else:
        print("No noticeable changes in the three metrics.")


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("Error: CSV file is empty or has no data rows.")
        sys.exit(1)

    # Compute composite scores
    scores = compute_scores(rows)

    # CAR-15
    worst_vehicle, alternatives = find_worst_vehicle(scores)
    explain_car_15(worst_vehicle, alternatives)

    # CAR-16
    best_alternative, best_targets = find_best_alternative_for_worst(worst_vehicle, scores)
    explain_car_16(best_alternative, best_targets, worst_vehicle)


if __name__ == "__main__":
    main()
