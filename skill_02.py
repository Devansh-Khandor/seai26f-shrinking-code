#!/usr/bin/env python3
"""
Skill 02: Estimate effort and defects for a proposed project based on historical records.
"""

import csv
import math
import sys

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def parse_float(val: str) -> float | None:
    """Safely parse a string to float, return None on failure."""
    try:
        return float(val.strip())
    except (AttributeError, ValueError):
        return None


def read_historical_data(csv_path: str):
    """
    Read the CSV and return three lists:
    - kloc : list of Kloc+ values
    - effort : list of Effort- values
    - defects : list of Defects- values
    Only rows where all three fields are present are kept.
    """
    kloc = []
    effort = []
    defects = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = parse_float(row.get("Kloc+"))
            e = parse_float(row.get("Effort-"))
            d = parse_float(row.get("Defects-"))
            if k is not None and e is not None and d is not None:
                kloc.append(k)
                effort.append(e)
                defects.append(d)

    return kloc, effort, defects


def linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Return slope and intercept for simple linear regression y = a + b * x.
    """
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(v * v for v in x)
    sum_xy = sum(v * w for v, w in zip(x, y))

    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0, 0.0  # fallback

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def predict_with_confidence(
    slope: float,
    intercept: float,
    x0: float,
    x: list[float],
    y: list[float],
) -> tuple[float, float]:
    """
    Return predicted y at x0 and its standard error of prediction.
    """
    y0 = intercept + slope * x0

    n = len(x)
    if n < 2:
        return y0, 0.0

    mean_x = sum(x) / n
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)

    # Residual sum of squares
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    sigma_sq = sum(r * r for r in residuals) / (n - 2) if n > 2 else 0.0

    if sum_sq_x == 0 or sigma_sq == 0:
        return y0, 0.0

    se = math.sqrt(
        sigma_sq * (1 + 1 / n + ((x0 - mean_x) ** 2) / sum_sq_x)
    )
    return y0, se


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def main(csv_path: str):
    kloc, effort, defects = read_historical_data(csv_path)

    if not kloc:
        print("No valid historical data found.")
        sys.exit(1)

    # --------------------------------------------------------------------- #
    # Prompt user for proposed Kloc+ value
    # --------------------------------------------------------------------- #
    try:
        proposed = input("Enter proposed KLOC+ (numeric): ")
        proposed_kloc = float(proposed.strip())
    except Exception:
        print("Invalid input; using default KLOC+ = 100.")
        proposed_kloc = 100.0

    # --------------------------------------------------------------------- #
    # NASA-03: Effort estimation
    # --------------------------------------------------------------------- #
    slope_eff, intercept_eff = linear_regression(kloc, effort)
    est_effort, se_effort = predict_with_confidence(
        slope_eff, intercept_eff, proposed_kloc, kloc, effort
    )
    ci_lower_eff = est_effort - 1.96 * se_effort
    ci_upper_eff = est_effort + 1.96 * se_effort

    print("== NASA-03 ==")
    print(f"Proposed KLOC+: {proposed_kloc:.2f}")
    print(
        f"Estimated Effort-: {est_effort:.2f} (95% CI: {ci_lower_eff:.2f} - {ci_upper_eff:.2f})"
    )
    print()

    # --------------------------------------------------------------------- #
    # NASA-04: Defect estimation
    # --------------------------------------------------------------------- #
    slope_def, intercept_def = linear_regression(kloc, defects)
    est_defects, se_defects = predict_with_confidence(
        slope_def, intercept_def, proposed_kloc, kloc, defects
    )
    ci_lower_def = est_defects - 1.96 * se_defects
    ci_upper_def = est_defects + 1.96 * se_defects

    print("== NASA-04 ==")
    print(
        f"Estimated Defects-: {est_defects:.2f} (95% CI: {ci_lower_def:.2f} - {ci_upper_def:.2f})"
    )
    print()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
