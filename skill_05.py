#!/usr/bin/env python3
"""
Skill 05 – Diagnostic and Comparative Analysis

This script implements two new diagnostic components for the
used‑car dealership data:

* CAR-09 – For vehicles with unusually poor fuel efficiency,
  identify the characteristics that deviate most from the
  overall population.

* CAR-10 – Compare the common characteristics of
  poorly performing vehicles to those of high‑performing ones.

Usage:
    python skill_05.py <csv_path>
"""

import csv
import math
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def read_csv(path: Path):
    """Read CSV file into a list of row dictionaries."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def safe_float(val):
    """Return float or None if conversion fails."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

# ----------------------------------------------------------------------
# Composite score calculation (same as skill_01)
# ----------------------------------------------------------------------
def compute_composite_scores(rows):
    """Return list of composite scores; None for missing data."""
    weights = [safe_float(r["Lbs-"]) for r in rows]
    mpgs = [safe_float(r["Mpg+"]) for r in rows]
    accs = [safe_float(r["Acc+"]) for r in rows]

    weights = [v for v in weights if v is not None]
    mpgs = [v for v in mpgs if v is not None]
    accs = [v for v in accs if v is not None]

    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    def norm(val, low, high, higher_is_better=True):
        if high == low:
            return 0.5
        return (val - low) / (high - low) if higher_is_better else (high - val) / (high - low)

    scores = []
    for r in rows:
        w = safe_float(r["Lbs-"])
        m = safe_float(r["Mpg+"])
        a = safe_float(r["Acc+"])
        if None in (w, m, a):
            scores.append(None)
            continue
        w_score = norm(w, w_min, w_max, higher_is_better=False)
        m_score = norm(m, mpg_min, mpg_max)
        a_score = norm(a, acc_min, acc_max)
        scores.append((w_score + m_score + a_score) / 3)
    return scores

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def compute_mean_std(values):
    if not values:
        return (0.0, 0.0)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return (mean, math.sqrt(var))

# ----------------------------------------------------------------------
# CAR-09 – Poor MPG characteristics
# ----------------------------------------------------------------------
def car_09(rows):
    numeric_cols = ["Clndrs", "Volume", "HpX", "Model", "origin",
                    "Lbs-", "Acc+", "Mpg+"]

    # Compute overall statistics
    stats = {}
    for col in numeric_cols:
        vals = [safe_float(r[col]) for r in rows]
        vals = [v for v in vals if v is not None]
        mean, std = compute_mean_std(vals)
        stats[col] = (mean, std if std > 0 else 1.0)

    # Determine poor MPG threshold
    mpg_vals = [safe_float(r["Mpg+"]) for r in rows if safe_float(r["Mpg+"]) is not None]
    mpg_mean, mpg_std = compute_mean_std(mpg_vals)
    threshold = mpg_mean - mpg_std  # below mean – one std

    # Identify poor vehicles
    poor_indices = [i for i, r in enumerate(rows)
                    if safe_float(r["Mpg+"]) is not None and safe_float(r["Mpg+"]) <= threshold]

    print("== CAR-09 ==")
    if not poor_indices:
        print("No vehicles meet the 'poor MPG' threshold.")
        return

    # For each poor vehicle, find the feature with the largest absolute z‑score
    results = []
    for idx in poor_indices[:3]:  # show at most 3 vehicles
        r = rows[idx]
        z_scores = {}
        for col in numeric_cols:
            val = safe_float(r[col])
            if val is None:
                continue
            mean, std = stats[col]
            z = (val - mean) / std
            z_scores[col] = z
        # Feature most strongly associated (max |z|)
        if not z_scores:
            continue
        top_feature, top_z = max(z_scores.items(), key=lambda kv: abs(kv[1]))
        results.append((idx + 1, top_feature, safe_float(r[top_feature]), top_z))

    for row_no, feat, val, z in results:
        print(f"Vehicle {row_no:>2} – Top feature: {feat:<6} (value: {val:>7.1f}, z: {z:>6.2f})")

# ----------------------------------------------------------------------
# CAR-10 – Comparative analysis of favorable vs unfavorable
# ----------------------------------------------------------------------
def car_10(rows):
    numeric_cols = ["Clndrs", "Volume", "HpX", "Model", "origin",
                    "Lbs-", "Acc+", "Mpg+"]

    # Composite scores
    scores = compute_composite_scores(rows)

    # Separate indices
    valid_indices = [i for i, s in enumerate(scores) if s is not None]
    valid_scores = [scores[i] for i in valid_indices]
    sorted_idx = sorted(valid_indices, key=lambda i: scores[i], reverse=True)

    # Bottom 25% unfavorable, top 25% favorable
    n = len(sorted_idx)
    quarter = max(1, n // 4)

    unfav_indices = sorted_idx[-quarter:]   # lowest scores
    fav_indices = sorted_idx[:quarter]     # highest scores

    def group_stats(indices):
        group = [rows[i] for i in indices]
        return {col: [safe_float(r[col]) for r in group if safe_float(r[col]) is not None]
                for col in numeric_cols}

    unfav_stats_raw = group_stats(unfav_indices)
    fav_stats_raw = group_stats(fav_indices)

    def mean_of(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Compute means
    unfav_means = {col: mean_of(vals) for col, vals in unfav_stats_raw.items()}
    fav_means = {col: mean_of(vals) for col, vals in fav_stats_raw.items()}

    print("\n== CAR-10 ==")
    print("Unfavorable group (bottom 25%) average characteristics:")
    for col in numeric_cols:
        print(f"  {col:<6} : {unfav_means[col]:>7.1f}")

    print("\nFavorable group (top 25%) average characteristics:")
    for col in numeric_cols:
        print(f"  {col:<6} : {fav_means[col]:>7.1f}")

    print("\nDifferences (Favorable – Unfavorable):")
    for col in numeric_cols:
        diff = fav_means[col] - unfav_means[col]
        print(f"  {col:<6} : {diff:>7.1f}")

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("Error: CSV file is empty or has no data rows.")
        sys.exit(1)

    # Run CAR‑09
    car_09(rows)

    # Run CAR‑10
    car_10(rows)


if __name__ == "__main__":
    main()
