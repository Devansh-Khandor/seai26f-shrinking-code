#!/usr/bin/env python3
"""
skill_02.py

A self‑contained script that extends the NASA data analysis to provide
predictive estimates for a proposed software project.

- NASA-03: Estimate development effort based on the project's size
           (Kloc+) using historical data.
- NASA-04: Estimate the likely number of software defects for a
           proposed project and provide an uncertainty estimate.

Usage:
    python skill_02.py path/to/nasa93dem.csv
"""

import csv
import sys
from pathlib import Path
import math
import statistics

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _read_csv(path: Path):
    """
    Read the CSV file and return a list of dicts with numeric fields
    converted to float. Non‑numeric values remain strings.
    """
    data = []
    with path.open(newline='') as f:
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

def _column_values(data, column):
    """Return a list of numeric values for the given column."""
    return [row[column] for row in data
            if isinstance(row.get(column), (int, float))]

def _linear_regression(x_vals, y_vals):
    """
    Perform ordinary least squares linear regression.

    Returns:
        intercept, slope, SE (standard error of estimate),
        mean_x, SSx (sum of squares of x about its mean)
    """
    n = len(x_vals)
    if n < 2:
        raise ValueError("Need at least two data points for regression.")
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    ssx = sum((x - mean_x) ** 2 for x in x_vals)
    ssy = sum((y - mean_y) ** 2 for y in y_vals)
    if ssx == 0:
        slope = 0.0
    else:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / ssx
    intercept = mean_y - slope * mean_x
    # Standard error of estimate
    residuals = [y - (intercept + slope * x) for x, y in zip(x_vals, y_vals)]
    sse = sum(r ** 2 for r in residuals)
    se = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
    return intercept, slope, se, mean_x, ssx

def _prediction_interval(x, intercept, slope, se, mean_x, ssx, n, conf_factor=2.0):
    """
    Compute a prediction interval for a new observation at x.

    conf_factor is the approximate z value for the desired confidence
    (2.0 ~ 95%).
    """
    if ssx == 0 or n <= 2:
        return None, None
    # Standard error of prediction
    se_pred = se * math.sqrt(1 + 1/n + ((x - mean_x) ** 2) / ssx)
    margin = conf_factor * se_pred
    y_hat = intercept + slope * x
    return y_hat, margin

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    data = _read_csv(csv_path)

    # Extract numeric columns
    kloc_values = _column_values(data, "Kloc+")
    effort_values = _column_values(data, "Effort-")
    defect_values = _column_values(data, "Defects-")

    # 1️⃣ NASA-03: Effort prediction based on size
    try:
        eff_intercept, eff_slope, eff_se, eff_mean_x, eff_ssx = _linear_regression(
            kloc_values, effort_values)
    except ValueError as e:
        print(f"Error calculating NASA-03 regression: {e}")
        eff_intercept = eff_slope = eff_se = eff_mean_x = eff_ssx = None

    # 2️⃣ NASA-04: Defect prediction based on effort
    try:
        def_intercept, def_slope, def_se, def_mean_x, def_ssx = _linear_regression(
            effort_values, defect_values)
    except ValueError as e:
        print(f"Error calculating NASA-04 regression: {e}")
        def_intercept = def_slope = def_se = def_mean_x = def_ssx = None

    # Median values for “typical” project
    median_kloc = statistics.median(kloc_values)
    median_effort = statistics.median(effort_values)

    # ----------------------------------------------------------------------- #
    # NASA-03 output
    # ----------------------------------------------------------------------- #
    print("== NASA-03 ==")
    if eff_intercept is not None:
        y_hat, margin = _prediction_interval(
            median_kloc, eff_intercept, eff_slope, eff_se,
            eff_mean_x, eff_ssx, len(kloc_values))
        if margin is not None:
            lower, upper = y_hat - margin, y_hat + margin
            print(f"Median Kloc+ = {median_kloc:.1f}")
            print(f"Predicted Effort = {y_hat:.1f} (+/- {margin:.1f})")
            print(f"95% prediction interval: {lower:.1f} – {upper:.1f}")
        else:
            print(f"Median Kloc+ = {median_kloc:.1f}")
            print("Insufficient data to compute prediction interval.")
            print(f"Predicted Effort = {y_hat:.1f}")
    else:
        print("Could not compute NASA-03 regression due to insufficient data.")
    print()

    # ----------------------------------------------------------------------- #
    # NASA-04 output
    # ----------------------------------------------------------------------- #
    print("== NASA-04 ==")
    if def_intercept is not None:
        y_hat, margin = _prediction_interval(
            median_effort, def_intercept, def_slope, def_se,
            def_mean_x, def_ssx, len(effort_values))
        if margin is not None:
            lower, upper = y_hat - margin, y_hat + margin
            print(f"Median Effort = {median_effort:.1f}")
            print(f"Predicted Defects = {y_hat:.1f} (+/- {margin:.1f})")
            print(f"95% prediction interval: {lower:.1f} – {upper:.1f}")
        else:
            print(f"Median Effort = {median_effort:.1f}")
            print("Insufficient data to compute prediction interval.")
            print(f"Predicted Defects = {y_hat:.1f}")
    else:
        print("Could not compute NASA-04 regression due to insufficient data.")
    print()

if __name__ == "__main__":
    main()
