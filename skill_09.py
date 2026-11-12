#!/usr/bin/env python3
"""
skill_09.py

A self‑contained script that adds two additional NASA analyses:

- NASA-17: Flag historical projects whose combinations of project characteristics
           are unusually different from the rest of our records so that we can review them.
- NASA-18: Identify project records whose reported effort or defect counts differ
           substantially from what comparable projects would lead us to expect,
           so we can investigate possible measurement or recording issues.

Usage:
    python skill_09.py path/to/nasa93dem.csv
"""

import csv
import sys
import statistics
import math
from pathlib import Path

# --------------------------------------------------------------------------- #
# CSV reading
# --------------------------------------------------------------------------- #

def _read_csv(path: Path):
    """Read a CSV file and convert numeric values to float where possible."""
    data = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if v is None:
                    cleaned[k] = None
                    continue
                val = v.strip()
                try:
                    cleaned[k] = float(val)
                except ValueError:
                    cleaned[k] = val
            data.append(cleaned)
    return data

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _numeric_columns(data):
    """Return list of columns that contain at least two numeric values."""
    counts = {}
    for row in data:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                counts[k] = counts.get(k, 0) + 1
    return [k for k, c in counts.items() if c >= 2]

def _column_values(data, column):
    """Return a list of numeric values for the given column."""
    return [row[column] for row in data if isinstance(row.get(column), (int, float))]

def _mean_std(values):
    """Return (mean, stddev).  If <2 samples, stddev is None."""
    n = len(values)
    if n == 0:
        return (None, None)
    mean_val = statistics.mean(values)
    if n < 2:
        return (mean_val, None)
    std_val = statistics.stdev(values)
    return (mean_val, std_val)

def _linear_regression(x_vals, y_vals):
    """Return (intercept, slope, se, mean_x, ssx)."""
    n = len(x_vals)
    if n < 2:
        raise ValueError("Need at least two data points for regression.")
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    ssx = sum((x - mean_x) ** 2 for x in x_vals)
    if ssx == 0:
        slope = 0.0
    else:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / ssx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(x_vals, y_vals)]
    sse = sum(r ** 2 for r in residuals)
    se = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
    return intercept, slope, se, mean_x, ssx

def _predict(x, intercept, slope):
    return intercept + slope * x

# --------------------------------------------------------------------------- #
# NASA-17: Unusual characteristic combinations
# --------------------------------------------------------------------------- #

def _nasa_17(data):
    print("== NASA-17 ==")

    # Identify numeric columns with enough data
    num_cols = _numeric_columns(data)
    if not num_cols:
        print("No numeric columns available for analysis.")
        return

    # Compute mean and std for each numeric column
    means = {}
    stds  = {}
    for col in num_cols:
        vals = _column_values(data, col)
        mean, std = _mean_std(vals)
        if mean is None:
            continue
        means[col] = mean
        stds[col]  = std

    # Compute distance (sum of squared z‑scores) for each record
    distances = []
    for row in data:
        dist_sq = 0.0
        count = 0
        for col in means:
            val = row.get(col)
            if isinstance(val, (int, float)) and stds[col]:
                z = (val - means[col]) / stds[col]
                dist_sq += z * z
                count += 1
        if count > 0:
            distances.append((row.get("idX", "Unknown"), math.sqrt(dist_sq)))
    if not distances:
        print("Insufficient data to compute distances.")
        return

    # Sort by distance descending
    distances.sort(key=lambda kv: kv[1], reverse=True)

    # Show top 5 (or all if fewer than 5)
    top_n = min(5, len(distances))
    print(f"Top {top_n} projects with the most unusual characteristic combinations:")
    for idx in range(top_n):
        pid, dist = distances[idx]
        print(f"  {idx+1}. id {pid} → distance = {dist:.3f}")
    print()

# --------------------------------------------------------------------------- #
# NASA-18: Outliers in Effort or Defect counts
# --------------------------------------------------------------------------- #

def _nasa_18(data):
    print("== NASA-18 ==")

    # Separate completed projects (with numeric Effort- and Defects-)
    completed = [
        r for r in data
        if isinstance(r.get("Effort-"), (int, float))
        and isinstance(r.get("Defects-"), (int, float))
        and isinstance(r.get("Kloc+"), (int, float))
    ]

    if not completed:
        print("No completed projects with complete numeric data.")
        return

    # Regression models
    try:
        e_int, e_slope, e_se, e_mean_x, e_ssx = _linear_regression(
            _column_values(completed, "Kloc+"),
            _column_values(completed, "Effort-")
        )
    except Exception as exc:
        e_int = e_slope = None
        print(f"Could not compute Effort regression: {exc}")

    try:
        d_int, d_slope, d_se, d_mean_x, d_ssx = _linear_regression(
            _column_values(completed, "Effort-"),
            _column_values(completed, "Defects-")
        )
    except Exception as exc:
        d_int = d_slope = None
        print(f"Could not compute Defects regression: {exc}")

    # Prepare lists of residuals
    effort_residuals = []
    defect_residuals = []

    # Compute predictions and residuals for each project
    for r in completed:
        pid = r.get("idX", "Unknown")
        k = r["Kloc+"]
        effort_actual = r["Effort-"]
        defect_actual = r["Defects-"]

        if e_int is not None:
            effort_pred = _predict(k, e_int, e_slope)
            effort_residuals.append((pid, effort_actual, effort_pred, effort_actual - effort_pred))
        if d_int is not None:
            # Predicted defect uses predicted effort
            effort_pred_for_defect = _predict(k, e_int, e_slope) if e_int is not None else None
            if effort_pred_for_defect is not None:
                defect_pred = _predict(effort_pred_for_defect, d_int, d_slope)
                defect_residuals.append((pid, defect_actual, defect_pred, defect_actual - defect_pred))

    # Compute thresholds (2*std of residuals)
    def compute_threshold(resid_list):
        if not resid_list:
            return None
        residual_vals = [abs(r[3]) for r in resid_list]
        mean = statistics.mean(residual_vals)
        std  = statistics.stdev(residual_vals) if len(residual_vals) > 1 else 0.0
        return mean + 2 * std

    effort_threshold = compute_threshold(effort_residuals)
    defect_threshold = compute_threshold(defect_residuals)

    # Flag projects exceeding threshold
    flagged_effort = [
        (pid, act, pred, res) for (pid, act, pred, res) in effort_residuals
        if effort_threshold is None or abs(res) > effort_threshold
    ]

    flagged_defect = [
        (pid, act, pred, res) for (pid, act, pred, res) in defect_residuals
        if defect_threshold is None or abs(res) > defect_threshold
    ]

    # Output results
    if flagged_effort:
        print(f"Projects with Effort outliers (|residual| > 2σ):")
        for idx, (pid, act, pred, res) in enumerate(flagged_effort[:5], start=1):
            print(f"  {idx}. id {pid} → actual={act:.1f}, pred={pred:.1f}, res={res:+.1f}")
        if len(flagged_effort) > 5:
            print(f"  ... and {len(flagged_effort)-5} more.")
    else:
        print("No projects flagged as Effort outliers.")

    print()

    if flagged_defect:
        print(f"Projects with Defect outliers (|residual| > 2σ):")
        for idx, (pid, act, pred, res) in enumerate(flagged_defect[:5], start=1):
            print(f"  {idx}. id {pid} → actual={act:.1f}, pred={pred:.1f}, res={res:+.1f}")
        if len(flagged_defect) > 5:
            print(f"  ... and {len(flagged_defect)-5} more.")
    else:
        print("No projects flagged as Defect outliers.")
    print()

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    data = _read_csv(csv_path)

    _nasa_17(data)
    _nasa_18(data)

    sys.exit(0)

if __name__ == "__main__":
    main()
