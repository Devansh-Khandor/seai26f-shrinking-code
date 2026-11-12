#!/usr/bin/env python3
"""
Skill 07 – Targeted Vehicle Identification & Inspection Prioritisation

This script implements two new diagnostic components for the
used‑car dealership data:

* CAR-13 – Identify a vehicle with a strong balance of fuel efficiency,
  weight, and acceleration while evaluating as few candidate vehicles as
  reasonably possible.

* CAR-14 – When we have a limited inspection budget, recommend which
  vehicles should be evaluated first so that we can discover promising
  inventory options early.

Usage:
    python skill_07.py <csv_path>
"""

import csv
import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV file into a list of row dictionaries."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def safe_float(val: str) -> Optional[float]:
    """Return float or None if conversion fails."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def median(values: List[float]) -> float:
    """Return the median of a list of numbers."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


def percentile(values: List[float], p: float) -> float:
    """Return the p-th percentile (0<=p<=1)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(math.ceil(p * len(sorted_vals))) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


# ----------------------------------------------------------------------
# Composite score calculation
# ----------------------------------------------------------------------
def composite_score(row: Dict[str, str],
                    w_min: float, w_max: float,
                    mpg_min: float, mpg_max: float,
                    acc_min: float, acc_max: float) -> Optional[float]:
    """Return composite score for a single row."""
    w = safe_float(row["Lbs-"])
    mpg = safe_float(row["Mpg+"])
    acc = safe_float(row["Acc+"])
    if None in (w, mpg, acc):
        return None

    def norm(val: float, low: float, high: float,
             higher_is_better: bool = True) -> float:
        if high == low:
            return 0.5
        return (val - low) / (high - low) if higher_is_better else (high - val) / (high - low)

    w_score = norm(w, w_min, w_max, higher_is_better=False)
    mpg_score = norm(mpg, mpg_min, mpg_max)
    acc_score = norm(acc, acc_min, acc_max)
    return (w_score + mpg_score + acc_score) / 3


# ----------------------------------------------------------------------
# CAR-13 – Balanced vehicle identification
# ----------------------------------------------------------------------
def find_balanced_vehicle(rows: List[Dict[str, str]]) -> Tuple[int, float, float, float, float]:
    """
    Return (row_index, composite_score, weight, mpg, acc) for the vehicle
    that best balances weight, mpg, and acceleration while limiting
    evaluation to a small subset.
    """
    # Collect numeric lists
    weights = [safe_float(r["Lbs-"]) for r in rows if safe_float(r["Lbs-"]) is not None]
    mpgs = [safe_float(r["Mpg+"]) for r in rows if safe_float(r["Mpg+"]) is not None]
    accs = [safe_float(r["Acc+"]) for r in rows if safe_float(r["Acc+"]) is not None]

    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    # Define thresholds to keep the candidate set small
    w_thresh = percentile(weights, 0.25)      # lightest 25%
    mpg_thresh = percentile(mpgs, 0.75)       # best 25% mpg
    acc_thresh = percentile(accs, 0.25)       # fastest 25% (lower acceleration)

    # Filter candidates
    candidate_indices = []
    for idx, r in enumerate(rows):
        w = safe_float(r["Lbs-"])
        mpg = safe_float(r["Mpg+"])
        acc = safe_float(r["Acc+"])
        if None in (w, mpg, acc):
            continue
        if w <= w_thresh and mpg >= mpg_thresh and acc <= acc_thresh:
            candidate_indices.append(idx)

    # If no candidates, fall back to all rows
    indices_to_evaluate = candidate_indices or list(range(len(rows)))

    # Compute composite scores for selected rows
    best_idx = None
    best_score = -1.0
    best_metrics = (0.0, 0.0, 0.0)

    for idx in indices_to_evaluate:
        r = rows[idx]
        score = composite_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max)
        if score is None or math.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_metrics = (safe_float(r["Lbs-"]), safe_float(r["Mpg+"]), safe_float(r["Acc+"]))
            best_idx = idx

    return best_idx, best_score, *best_metrics


# ----------------------------------------------------------------------
# CAR-14 – Inspection prioritisation
# ----------------------------------------------------------------------
def top_inspection_priority(rows: List[Dict[str, str]], top_n: int = 5) -> List[Tuple[int, float, float, float, float]]:
    """
    Return a list of the top `top_n` rows sorted by composite score
    (best first). Each entry is (row_index, score, weight, mpg, acc).
    """
    # Collect bounds
    weights = [safe_float(r["Lbs-"]) for r in rows if safe_float(r["Lbs-"]) is not None]
    mpgs = [safe_float(r["Mpg+"]) for r in rows if safe_float(r["Mpg+"]) is not None]
    accs = [safe_float(r["Acc+"]) for r in rows if safe_float(r["Acc+"]) is not None]

    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    # Score all rows
    scored_rows = []
    for idx, r in enumerate(rows):
        score = composite_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max)
        if score is None or math.isnan(score):
            continue
        scored_rows.append((idx, score, safe_float(r["Lbs-"]), safe_float(r["Mpg+"]), safe_float(r["Acc+"])))

    scored_rows.sort(key=lambda x: x[1], reverse=True)
    return scored_rows[:top_n]


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("Error: CSV file is empty or has no data rows.")
        sys.exit(1)

    # CAR-13
    idx, score, weight, mpg, acc = find_balanced_vehicle(rows)
    print("== CAR-13 ==")
    if idx is None or math.isnan(score):
        print("No suitable vehicle found.")
    else:
        print(f"Vehicle with best balance: Row {idx + 1}")
        print(f"  Weight: {weight:7.1f} lbs")
        print(f"  MPG   : {mpg:7.1f}")
        print(f"  Acc   : {acc:7.2f} s")
        print(f"Composite score: {score:7.3f}")

    print("\n== CAR-14 ==")
    top_rows = top_inspection_priority(rows, top_n=5)
    if not top_rows:
        print("No vehicles with complete data to evaluate.")
    else:
        print("Prioritised vehicles for inspection (best first):")
        for rank, (idx, score, weight, mpg, acc) in enumerate(top_rows, 1):
            print(f"{rank:2d}. Row {idx + 1:<3d} | Score: {score:6.3f} | "
                  f"Weight: {weight:>7.1f} lbs | MPG: {mpg:>5.1f} | Acc: {acc:>5.2f}s")

if __name__ == "__main__":
    main()
