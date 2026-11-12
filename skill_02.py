#!/usr/bin/env python3
"""
skill_02.py - NASA-03 and NASA-04 estimates for a proposed project.

The program reads the historical project data from a CSV file, builds simple
linear regression models to predict effort and defects based on project size
(KLOC), and outputs estimates for a hypothetical project with median
characteristics.

Usage:
    python skill_02.py <path-to-nasa93dem.csv>
"""

import csv
import sys
import statistics
from pathlib import Path
from typing import List, Dict, Tuple

# Import the helpers from skill_01.py
# Assuming skill_01.py is in the same directory
try:
    from skill_01 import (
        load_data,
        compute_statistics,
        find_best_worst,
        compute_medians,
        find_favorable,
    )
except Exception as exc:
    raise RuntimeError("Could not import helpers from skill_01.py") from exc

# --------------------------------------------------------------------
# Helper functions for linear regression
# --------------------------------------------------------------------

def linear_regression(projects: List[Dict], x_field: str, y_field: str) -> Tuple[float, float, float]:
    """
    Compute slope and intercept of a simple linear regression y ~ x
    and return (slope, intercept, std_err of residuals).

    Args:
        projects: List of project dicts with numeric fields.
        x_field: Name of the independent variable field.
        y_field: Name of the dependent variable field.

    Returns:
        slope, intercept, residual standard error
    """
    x_vals = [float(p[x_field]) for p in projects]
    y_vals = [float(p[y_field]) for p in projects]
    n = len(x_vals)
    if n < 2:
        raise ValueError("Not enough data points for regression.")
    x_mean = statistics.mean(x_vals)
    y_mean = statistics.mean(y_vals)
    # Compute covariance and variance
    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    var_x = sum((x - x_mean) ** 2 for x in x_vals)
    if var_x == 0:
        raise ValueError("Zero variance in X; cannot perform regression.")
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    # Residuals and standard error
    residuals = [y - (slope * x + intercept) for x, y in zip(x_vals, y_vals)]
    sse = sum(r ** 2 for r in residuals)
    std_err = (sse / (n - 2)) ** 0.5
    return slope, intercept, std_err


def estimate_for_median(projects: List[Dict], slope: float, intercept: float) -> Tuple[float, float]:
    """
    Estimate the dependent variable using the model and the median value
    of the independent variable. Return (estimate, +/- std_err).
    """
    kloc_vals = [float(p["Kloc+"]) for p in projects]
    kloc_median = statistics.median(kloc_vals)
    estimate = slope * kloc_median + intercept
    return kloc_median, estimate


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <path-to-nasa93dem.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    projects = load_data(csv_path)

    if not projects:
        print("Error: No valid project data found.")
        sys.exit(1)

    # ----------------- NASA-03: Effort estimate -----------------
    try:
        slope_eff, intercept_eff, std_err_eff = linear_regression(
            projects, "Kloc+", "Effort-"
        )
    except Exception as exc:
        print(f"Error computing effort regression: {exc}")
        sys.exit(1)

    kloc_median, effort_est = estimate_for_median(
        projects, slope_eff, intercept_eff
    )

    print("== NASA-03 ==")
    print(f"Median project size: {kloc_median:.1f} KLOC")
    print(f"Estimated effort: {effort_est:.1f} person‑months")
    print(f"Uncertainty (±1σ): {std_err_eff:.1f} person‑months")

    # ----------------- NASA-04: Defect estimate -----------------
    try:
        slope_def, intercept_def, std_err_def = linear_regression(
            projects, "Kloc+", "Defects-"
        )
    except Exception as exc:
        print(f"Error computing defect regression: {exc}")
        sys.exit(1)

    _, defect_est = estimate_for_median(
        projects, slope_def, intercept_def
    )

    print("\n== NASA-04 ==")
    print(f"Median project size: {kloc_median:.1f} KLOC")
    print(f"Estimated defects: {defect_est:.1f}")
    print(f"Uncertainty (±1σ): {std_err_def:.1f}")

if __name__ == "__main__":
    main()
