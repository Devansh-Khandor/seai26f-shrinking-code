#!/usr/bin/env python3
"""
Skill 02 – Fuel‑Efficiency & Weight Estimation

Usage:
    python skill_02.py <csv_path>
"""

import csv
import math
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def read_csv(path: Path):
    """Read the CSV and return a list of rows as dictionaries."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def safe_float(val: str):
    """Return float or None if the string is empty or cannot be converted."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------
# CAR‑03 – Estimating missing MPG
# ----------------------------------------------------------------------
def estimate_mpg(rows):
    """Return a dict mapping row indices to estimated MPG values."""
    # Build a lookup: model → list of known MPG values
    model_to_mpg = {}
    for i, r in enumerate(rows):
        mpg = safe_float(r["Mpg+"])
        if mpg is not None:
            model = r["Model"]
            model_to_mpg.setdefault(model, []).append(mpg)

    # Compute overall mean MPG for fallback
    all_mpgs = [mpg for mpgs in model_to_mpg.values() for mpg in mpgs]
    overall_mean = sum(all_mpgs) / len(all_mpgs) if all_mpgs else 0.0

    estimates = {}
    for i, r in enumerate(rows):
        if safe_float(r["Mpg+"]) is None:
            model = r["Model"]
            if model in model_to_mpg:
                est = sum(model_to_mpg[model]) / len(model_to_mpg[model])
            else:
                est = overall_mean
            estimates[i] = est
    return estimates


def print_car_03(rows, mpg_estimates):
    print("== CAR-03 ==")
    # Print the first three estimations (or fewer if not enough)
    for idx, (row_idx, est) in enumerate(sorted(mpg_estimates.items())[:3], 1):
        print(
            f"{idx:2d}. Row {row_idx + 1:<3d} | "
            f"Estimated MPG: {est:>5.2f}"
        )
    if len(mpg_estimates) > 3:
        print(f"... and {len(mpg_estimates) - 3} more estimates.")


# ----------------------------------------------------------------------
# CAR‑04 – Estimating weight and uncertainty
# ----------------------------------------------------------------------
def compute_feature_stats(rows, feature_names):
    """Compute mean and variance for each numerical feature."""
    stats = {}
    n = len(rows)
    for name in feature_names:
        vals = [safe_float(r[name]) for r in rows if safe_float(r[name]) is not None]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            stats[name] = (mean, var if var > 0 else 1.0)
        else:
            stats[name] = (0.0, 1.0)
    return stats


def estimate_weight_and_uncertainty(rows):
    """
    For each row, estimate weight using weighted nearest‑neighbour
    based on the five numeric features.
    Returns a dict mapping row index to (est_weight, uncertainty).
    """
    feature_names = ["HpX", "Volume", "Clndrs", "origin", "Acc+"]
    stats = compute_feature_stats(rows, feature_names)

    estimates = {}
    for i, r in enumerate(rows):
        # Extract features for this row
        try:
            feats_i = [safe_float(r[name]) for name in feature_names]
            if None in feats_i:
                raise ValueError
        except ValueError:
            # If any feature missing, skip estimation
            continue

        # Build weighted sum over all other rows
        w_total = 0.0
        w_weighted = 0.0
        w_weighted_sq = 0.0
        for j, r2 in enumerate(rows):
            if i == j:
                continue
            try:
                feats_j = [safe_float(r2[name]) for name in feature_names]
                if None in feats_j:
                    continue
            except ValueError:
                continue
            # Compute squared, variance‑normalised distance
            dist_sq = 0.0
            for k, name in enumerate(feature_names):
                mean, var = stats[name]
                diff = feats_i[k] - feats_j[k]
                dist_sq += (diff * diff) / var
            dist = math.sqrt(dist_sq)
            # Similarity weight (avoid division by zero)
            w = 1.0 / (dist + 1e-6)

            # Target weight
            w_target = safe_float(r2["Lbs-"])
            if w_target is None:
                continue

            w_total += w
            w_weighted += w * w_target
            w_weighted_sq += w * (w_target * w_target)

        if w_total == 0.0:
            continue

        est = w_weighted / w_total
        # Weighted variance: E[w * (x - E[x])^2] = (E[w*x^2] / E[w]) - (E[x])^2
        mean_sq = (w_weighted_sq / w_total)
        variance = mean_sq - est * est
        uncertainty = math.sqrt(variance) if variance > 0 else 0.0

        estimates[i] = (est, uncertainty)
    return estimates


def print_car_04(rows, weight_estimates):
    print("== CAR-04 ==")
    for idx, (row_idx, (est, unc)) in enumerate(sorted(weight_estimates.items())[:3], 1):
        print(
            f"{idx:2d}. Row {row_idx + 1:<3d} | "
            f"Estimated Weight: {est:>7.1f} lbs (±{unc:>6.1f} lbs)"
        )
    if len(weight_estimates) > 3:
        print(f"... and {len(weight_estimates) - 3} more estimates.")


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("Error: CSV file is empty or has no data rows.")
        sys.exit(1)

    # CAR‑03
    mpg_estimates = estimate_mpg(rows)
    print_car_03(rows, mpg_estimates)

    # CAR‑04
    weight_estimates = estimate_weight_and_uncertainty(rows)
    print_car_04(rows, weight_estimates)


if __name__ == "__main__":
    main()
