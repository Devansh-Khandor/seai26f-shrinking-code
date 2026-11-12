#!/usr/bin/env python3
"""
Skill 09: Detect unusual projects and potential measurement errors.

NASA-17: Flag historical projects whose combinations of project characteristics
are unusually different from the rest of our records so that we can review them.

NASA-18: Identify project records whose reported effort or defect counts differ
substantially from what comparable projects would lead us to expect, so we can
investigate possible measurement or recording issues.

Usage:
    python skill_09.py <csv_path>
"""

import csv
import math
import sys

# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_float(val: str | None) -> float | None:
    """Safely parse a string to float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val.strip())
    except (AttributeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Basic statistics
# --------------------------------------------------------------------------- #
def mean(values):
    return sum(values) / len(values) if values else 0.0


def stddev(values, mean_val):
    n = len(values)
    if n < 2:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (n - 1))


# --------------------------------------------------------------------------- #
# Regression utilities (same as previous skills)
# --------------------------------------------------------------------------- #
def linear_regression(x, y):
    """Return slope and intercept for y = intercept + slope * x."""
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(v * v for v in x)
    sum_xy = sum(v * w for v, w in zip(x, y))
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def predict_with_confidence(slope, intercept, x0, x, y):
    """Return predicted y at x0 and its standard error of prediction."""
    y0 = intercept + slope * x0
    n = len(x)
    if n < 2:
        return y0, 0.0
    mean_x = sum(x) / n
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    sigma_sq = sum(r * r for r in residuals) / (n - 2) if n > 2 else 0.0
    if sum_sq_x == 0 or sigma_sq == 0:
        return y0, 0.0
    se = math.sqrt(sigma_sq * (1 + 1 / n + ((x0 - mean_x) ** 2) / sum_sq_x))
    return y0, se


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(csv_path):
    """Read CSV, return list of dicts with parsed numeric fields."""
    numeric_cols = ["Kloc+", "Effort-", "Defects-", "MonthsX", "YearX", "centerX"]
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            r = raw.copy()
            for col in numeric_cols:
                r[col] = parse_float(r.get(col))
            rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# NASA-17: Multivariate outlier detection using z‑scores
# --------------------------------------------------------------------------- #
def nasa_17(rows):
    print("== NASA-17 ==")

    # Identify projects with any missing numeric fields – skip them
    valid_rows = [r for r in rows if all(r.get(col) is not None for col in
                                          ["Kloc+", "Effort-", "Defects-", "MonthsX", "YearX", "centerX"])]
    if not valid_rows:
        print("No valid records to analyze.")
        print()
        return

    # Compute means and std devs for each numeric variable
    numeric_cols = ["Kloc+", "Effort-", "Defects-", "MonthsX", "YearX", "centerX"]
    stats = {}
    for col in numeric_cols:
        vals = [r[col] for r in valid_rows]
        stats[col] = {"mean": mean(vals), "std": stddev(vals, mean(vals))}

    # Compute squared z‑score sum for each project
    outliers = []
    for r in valid_rows:
        sum_sq_z = 0.0
        for col in numeric_cols:
            std = stats[col]["std"]
            if std == 0:
                # skip if no variation
                continue
            z = (r[col] - stats[col]["mean"]) / std
            sum_sq_z += z * z
        r["_sqz"] = sum_sq_z
        if sum_sq_z > 16.81:  # chi‑square 0.99 threshold for df=6
            outliers.append(r)

    if not outliers:
        print("No projects flagged as unusually different.")
    else:
        print(f"Found {len(outliers)} projects with unusually different characteristics:")
        for r in outliers[:10]:
            print(f"  ID={r.get('id', 'N/A')}, Squared Z‑Sum={r['_sqz']:.2f}")
    print()


# --------------------------------------------------------------------------- #
# NASA-18: Residual outliers for effort and defects
# --------------------------------------------------------------------------- #
def nasa_18(rows):
    print("== NASA-18 ==")

    # Use only projects with complete numeric data
    valid_rows = [r for r in rows if all(r.get(col) is not None for col in
                                          ["Kloc+", "Effort-", "Defects-"])]
    if not valid_rows:
        print("No valid records to analyze.")
        print()
        return

    # Separate data for regression
    kloc_vals = [r["Kloc+"] for r in valid_rows]
    effort_vals = [r["Effort-"] for r in valid_rows]
    defect_vals = [r["Defects-"] for r in valid_rows]

    # Fit regressions
    slope_eff, intercept_eff = linear_regression(kloc_vals, effort_vals)
    slope_def, intercept_def = linear_regression(kloc_vals, defect_vals)

    # Compute residual thresholds (2×SE)
    eff_outliers = []
    def_outliers = []

    for r in valid_rows:
        k = r["Kloc+"]
        # Effort residual
        pred_eff, se_eff = predict_with_confidence(slope_eff, intercept_eff, k,
                                                  kloc_vals, effort_vals)
        res_eff = r["Effort-"] - pred_eff
        if abs(res_eff) > 2 * se_eff:
            eff_outliers.append((r, pred_eff, res_eff, se_eff))

        # Defect residual
        pred_def, se_def = predict_with_confidence(slope_def, intercept_def, k,
                                                   kloc_vals, defect_vals)
        res_def = r["Defects-"] - pred_def
        if abs(res_def) > 2 * se_def:
            def_outliers.append((r, pred_def, res_def, se_def))

    # Output results
    if not eff_outliers and not def_outliers:
        print("No projects with effort or defect counts deviating beyond 2σ.")
    else:
        if eff_outliers:
            print(f"Projects with unusually high/low Effort- (|residual| > 2σ):")
            for r, pred, res, se in eff_outliers[:10]:
                print(f"  ID={r.get('id', 'N/A')}: Observed={r['Effort-']:.2f}, "
                      f"Predicted={pred:.2f}, Residual={res:.2f}, SE={se:.2f}")
        else:
            print("No effort outliers detected.")

        print()

        if def_outliers:
            print(f"Projects with unusually high/low Defects- (|residual| > 2σ):")
            for r, pred, res, se in def_outliers[:10]:
                print(f"  ID={r.get('id', 'N/A')}: Observed={r['Defects-']:.2f}, "
                      f"Predicted={pred:.2f}, Residual={res:.2f}, SE={se:.2f}")
        else:
            print("No defect outliers detected.")
    print()


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_data(csv_path)
    if not rows:
        print("No data loaded from CSV.")
        sys.exit(1)

    nasa_17(rows)
    nasa_18(rows)


if __name__ == "__main__":
    main()
