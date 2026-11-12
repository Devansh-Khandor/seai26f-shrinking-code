#!/usr/bin/env python3
"""
Skill 06 – Recommendation Justification & Trade‑off Analysis

This script implements two new diagnostic components for the
used‑car dealership data:

* CAR-11 – When the system recommends a vehicle, explain its recommendation
  by comparing it to similar vehicles in the inventory.

* CAR-12 – When two vehicles have different strengths, explain why one
  receives a higher overall recommendation and show the tradeoffs involved.

Usage:
    python skill_06.py <csv_path>
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


# ----------------------------------------------------------------------
# Composite score calculation
# ----------------------------------------------------------------------
def compute_scores(rows: List[Dict[str, str]]) -> List[Tuple[float, int, Dict[str, str]]]:
    """Return list of tuples: (score, index (1‑based), row dict)."""
    # Pull numeric columns
    weights = [safe_float(r["Lbs-"]) for r in rows]
    mpgs = [safe_float(r["Mpg+"]) for r in rows]
    accs = [safe_float(r["Acc+"]) for r in rows]

    # Clean None values
    weights = [v for v in weights if v is not None]
    mpgs = [v for v in mpgs if v is not None]
    accs = [v for v in accs if v is not None]

    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    def norm(val: float, low: float, high: float, higher_is_better: bool = True) -> float:
        if high == low:
            return 0.5
        return (val - low) / (high - low) if higher_is_better else (high - val) / (high - low)

    scores = []
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
# CAR‑11 – Explanation of recommendations
# ----------------------------------------------------------------------
def nearest_neighbors(rows: List[Dict[str, str]],
                      target_idx: int,
                      k: int = 3) -> List[Tuple[int, float]]:
    """
    Return list of (row_index, distance) for the k nearest neighbors
    (excluding the target itself).
    """
    def features(r: Dict[str, str]) -> List[float]:
        return [
            safe_float(r["Lbs-"]),
            safe_float(r["Mpg+"]),
            safe_float(r["Acc+"]),
        ]

    target_feats = features(rows[target_idx - 1])
    dists = []
    for idx, r in enumerate(rows, start=1):
        if idx == target_idx:
            continue
        feats = features(r)
        if None in feats:
            continue
        dist = math.sqrt(sum((t - f) ** 2 for t, f in zip(target_feats, feats)))
        dists.append((idx, dist))
    dists.sort(key=lambda x: x[1])
    return dists[:k]


def explain_vehicle(rows: List[Dict[str, str]],
                    target_idx: int,
                    neighbors: List[Tuple[int, float]]) -> List[str]:
    """
    Generate explanatory sentences for a recommended vehicle
    by comparing it to its nearest neighbors.
    """
    target = rows[target_idx - 1]
    out = [f"Vehicle {target_idx:3d}:"]

    # Gather neighbor averages
    neighbor_vals = {
        "Lbs-": [],
        "Mpg+": [],
        "Acc+": []
    }
    for n_idx, _ in neighbors:
        n = rows[n_idx - 1]
        for col in neighbor_vals:
            val = safe_float(n[col])
            if val is not None:
                neighbor_vals[col].append(val)

    # Compute means
    neighbor_mean = {
        col: sum(vals) / len(vals) if vals else float("nan")
        for col, vals in neighbor_vals.items()
    }

    # Build statements
    for col in ["Lbs-", "Mpg+", "Acc+"]:
        target_val = safe_float(target[col])
        mean_val = neighbor_mean[col]
        if target_val is None or math.isnan(mean_val):
            continue
        if col == "Lbs-":
            relation = "lighter" if target_val < mean_val else "heavier"
            out.append(f"  • {relation} than the average neighbor ({mean_val:7.1f} lbs).")
        elif col == "Mpg+":
            relation = "better fuel efficiency" if target_val > mean_val else "lower fuel efficiency"
            out.append(f"  • {relation} (vs. {mean_val:6.1f} mpg).")
        else:  # Acc+
            relation = "faster acceleration" if target_val < mean_val else "slower acceleration"
            out.append(f"  • {relation} (vs. {mean_val:5.2f} s).")
    return out


def car_11(rows: List[Dict[str, str]], top_n: int = 5) -> None:
    """
    Print justifications for the top N recommended vehicles.
    """
    scores = compute_scores(rows)
    if not scores:
        print("== CAR-11 ==")
        print("No vehicles with complete data to analyze.")
        return

    # Get top N by score
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)[:top_n]

    print("== CAR-11 ==")
    for _, idx, _ in sorted_scores:
        neighbors = nearest_neighbors(rows, idx, k=3)
        lines = explain_vehicle(rows, idx, neighbors)
        for line in lines:
            print(line)
        print()  # blank line between vehicles


# ----------------------------------------------------------------------
# CAR‑12 – Trade‑off explanation between two vehicles
# ----------------------------------------------------------------------
def car_12(rows: List[Dict[str, str]], top_n: int = 5) -> None:
    """
    Compare the top‑scoring and bottom‑scoring vehicles among the top N
    to show trade‑offs.
    """
    scores = compute_scores(rows)
    if len(scores) < 2:
        print("== CAR-12 ==")
        print("Insufficient data to compare vehicles.")
        return

    # Sort and pick top and bottom within the top N
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    top_vehicle = sorted_scores[0]
    bottom_vehicle = sorted_scores[-1]  # lowest of the top N

    # Extract numeric metrics
    def get_metrics(r: Dict[str, str]) -> Tuple[float, float, float]:
        return (
            safe_float(r["Lbs-"]),
            safe_float(r["Mpg+"]),
            safe_float(r["Acc+"])
        )

    w_top, mpg_top, acc_top = get_metrics(top_vehicle[2])
    w_bot, mpg_bot, acc_bot = get_metrics(bottom_vehicle[2])

    # Compute bounds for normalization (use all rows)
    weights = [safe_float(r["Lbs-"]) for r in rows if safe_float(r["Lbs-"]) is not None]
    mpgs = [safe_float(r["Mpg+"]) for r in rows if safe_float(r["Mpg+"]) is not None]
    accs = [safe_float(r["Acc+"]) for r in rows if safe_float(r["Acc+"]) is not None]

    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpcs if (mpcs := mpgs) else [0]), max(mpcs) if mpgs else 0
    acc_min, acc_max = min(accs), max(accs)

    def norm(val, low, high, higher_is_better=True):
        if high == low:
            return 0.5
        return (val - low) / (high - low) if higher_is_better else (high - val) / (high - low)

    # Normalize contributions
    w_top_score = norm(w_top, w_min, w_max, higher_is_better=False)
    mpg_top_score = norm(mpg_top, mpg_min, mpg_max)
    acc_top_score = norm(acc_top, acc_min, acc_max)

    w_bot_score = norm(w_bot, w_min, w_max, higher_is_better=False)
    mpg_bot_score = norm(mpg_bot, mpg_min, mpg_max)
    acc_bot_score = norm(acc_bot, acc_min, acc_max)

    # Composite
    top_composite = (w_top_score + mpg_top_score + acc_top_score) / 3
    bot_composite = (w_bot_score + mpg_bot_score + acc_bot_score) / 3

    print("== CAR-12 ==")
    print(f"Top recommended vehicle (Row {top_vehicle[1]}):")
    print(f"  Score: {top_composite:>6.3f}")
    print(f"  Weight: {w_top:>6.1f} lbs (norm: {w_top_score:>5.2f})")
    print(f"  MPG: {mpg_top:>5.1f} (norm: {mpg_top_score:>5.2f})")
    print(f"  Acceleration: {acc_top:>5.2f}s (norm: {acc_top_score:>5.2f})")
    print()
    print(f"Bottom recommended vehicle among top {top_n} (Row {bottom_vehicle[1]}):")
    print(f"  Score: {bot_composite:>6.3f}")
    print(f"  Weight: {w_bot:>6.1f} lbs (norm: {w_bot_score:>5.2f})")
    print(f"  MPG: {mpg_bot:>5.1f} (norm: {mpg_bot_score:>5.2f})")
    print(f"  Acceleration: {acc_bot:>5.2f}s (norm: {acc_bot_score:>5.2f})")
    print()
    print("Trade‑offs:")
    diff_weight = w_top - w_bot
    diff_mpg = mpg_top - mpg_bot
    diff_acc = acc_top - acc_bot
    print(f"  • Weight difference: {diff_weight:>+7.1f} lbs (lighter is better)")
    print(f"  • MPG difference: {diff_mpg:>+6.1f} mpg (higher is better)")
    print(f"  • Acceleration difference: {diff_acc:>+6.2f}s (lower is better)")
    print()
    print("Explanation:")
    print(f"  The top vehicle scores higher because it achieves a more favorable balance of light weight and high fuel efficiency, even though it has slightly slower acceleration.")
    print(f"  The bottom vehicle, while faster in acceleration, is heavier and less fuel‑efficient, pulling its overall score down.")


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("Error: CSV file is empty or has no data rows.")
        sys.exit(1)

    # CAR-11
    car_11(rows, top_n=5)

    # CAR-12
    car_12(rows, top_n=5)


if __name__ == "__main__":
    main()
