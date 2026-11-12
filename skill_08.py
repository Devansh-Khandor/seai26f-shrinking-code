#!/usr/bin/env python3
"""
Skill 08: Identify changes that can reduce effort or defects for a proposed project.

NASA-15: For a project proposal predicted to require excessive effort, identify plausible
adjustable planning or process characteristics that might reduce its predicted effort.

NASA-16: For a project proposal with a high predicted defect count, identify the
smallest plausible combination of changes to adjustable planning or process
characteristics that could improve its predicted quality, while explaining potential tradeoffs.
"""

import csv
import math
import sys
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def parse_float(val: str | None) -> float | None:
    """Parse a string into a float, return None if it fails."""
    if val is None:
        return None
    try:
        return float(val.strip())
    except (ValueError, AttributeError):
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float], mean_val: float) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (n - 1))


# --------------------------------------------------------------------------- #
# Regression helpers
# --------------------------------------------------------------------------- #
def linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
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


def predict_with_confidence(slope: float, intercept: float, x0: float, x: list[float], y: list[float]) -> tuple[float, float]:
    """Return predicted y at x0 and its standard error."""
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
def load_data(csv_path: str):
    """Return rows as list of dicts with numeric fields parsed."""
    numeric_cols = ["Kloc+", "Effort-", "Defects-"]
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
# Identify categorical columns
# --------------------------------------------------------------------------- #
def get_categorical_columns(rows: list[dict]) -> list[str]:
    numeric = {"Kloc+", "Effort-", "Defects-"}
    keys = rows[0].keys()
    return [k for k in keys if k not in numeric and k != "id"]


# --------------------------------------------------------------------------- #
# Compute group statistics
# --------------------------------------------------------------------------- #
def compute_group_stats(rows, col, numeric_col):
    """Return dict mapping category value to (mean, std, n)."""
    groups = defaultdict(list)
    for r in rows:
        val = r.get(col)
        num = r.get(numeric_col)
        if val is not None and num is not None:
            groups[val].append(num)

    stats = {}
    for key, values in groups.items():
        if values:
            stats[key] = {
                "n": len(values),
                "mean": mean(values),
                "std": stddev(values, mean(values)),
            }
    return stats


# --------------------------------------------------------------------------- #
# NASA‑15 implementation
# --------------------------------------------------------------------------- #
def nasa_15(rows):
    print("== NASA-15 ==")

    # 1. Prompt for proposal KLOC+
    try:
        inp = input("Enter proposed KLOC+ (numeric): ").strip()
        prop_kloc = float(inp)
    except Exception:
        print("Invalid input; default KLOC+ = 100.")
        prop_kloc = 100.0

    # 2. Predict effort using simple regression on KLOC+
    kloc_vals = [r["Kloc+"] for r in rows if r["Kloc+"] is not None]
    effort_vals = [r["Effort-"] for r in rows if r["Effort-"] is not None]
    slope, intercept = linear_regression(kloc_vals, effort_vals)
    pred_effort, se_effort = predict_with_confidence(slope, intercept, prop_kloc, kloc_vals, effort_vals)
    ci_low, ci_up = pred_effort - 1.96 * se_effort, pred_effort + 1.96 * se_effort

    print(f"Proposed KLOC+ : {prop_kloc:.2f}")
    print(f"Predicted Effort-: {pred_effort:.2f} (95% CI: {ci_low:.2f} - {ci_up:.2f})")
    print()

    # 3. Compute overall mean Effort- to measure potential reduction
    overall_effort_mean = mean(effort_vals)

    # 4. Identify categorical changes that lower the mean Effort- the most
    cat_cols = get_categorical_columns(rows)
    if not cat_cols:
        print("No categorical variables available to propose changes.")
        return

    suggestions = []
    for col in cat_cols:
        stats = compute_group_stats(rows, col, "Effort-")
        # Find category with lowest mean Effort-
        best_cat, best_stats = min(stats.items(), key=lambda kv: kv[1]["mean"])
        # Reduction relative to overall mean
        reduction = overall_effort_mean - best_stats["mean"]
        if reduction > 0:
            suggestions.append((col, best_cat, best_stats["mean"], reduction))

    # Sort by potential reduction
    suggestions.sort(key=lambda x: x[3], reverse=True)

    if not suggestions:
        print("No actionable categorical changes found that reduce Effort-.")
        return

    print("Recommended changes to reduce effort:")
    for i, (col, cat, new_mean, red) in enumerate(suggestions[:3], 1):
        print(f"  {i}. Change '{col}' to '{cat}' → expected mean Effort- = {new_mean:.2f} hrs (reduction {red:.2f} hrs)")
    print("\nTrade‑offs: changing a process or planning attribute may affect other quality or schedule dimensions.")
    print()


# --------------------------------------------------------------------------- #
# NASA‑16 implementation
# --------------------------------------------------------------------------- #
def nasa_16(rows):
    print("== NASA-16 ==")

    # 1. Prompt for proposal KLOC+
    try:
        inp = input("Enter proposed KLOC+ (numeric): ").strip()
        prop_kloc = float(inp)
    except Exception:
        print("Invalid input; default KLOC+ = 100.")
        prop_kloc = 100.0

    # 2. Predict defects using simple regression on KLOC+
    kloc_vals = [r["Kloc+"] for r in rows if r["Kloc+"] is not None]
    defect_vals = [r["Defects-"] for r in rows if r["Defects-"] is not None]
    slope, intercept = linear_regression(kloc_vals, defect_vals)
    pred_defects, se_defects = predict_with_confidence(slope, intercept, prop_kloc, kloc_vals, defect_vals)
    ci_low, ci_up = pred_defects - 1.96 * se_defects, pred_defects + 1.96 * se_defects

    print(f"Proposed KLOC+ : {prop_kloc:.2f}")
    print(f"Predicted Defects-: {pred_defects:.2f} (95% CI: {ci_low:.2f} - {ci_up:.2f})")
    print()

    # 3. Compute overall mean Defects- to measure potential improvement
    overall_defect_mean = mean(defect_vals)

    # 4. Identify categorical changes that lower the mean Defects- the most
    cat_cols = get_categorical_columns(rows)
    if not cat_cols:
        print("No categorical variables available to propose changes.")
        return

    suggestions = []
    for col in cat_cols:
        stats = compute_group_stats(rows, col, "Defects-")
        best_cat, best_stats = min(stats.items(), key=lambda kv: kv[1]["mean"])
        improvement = overall_defect_mean - best_stats["mean"]
        if improvement > 0:
            suggestions.append((col, best_cat, best_stats["mean"], improvement))

    suggestions.sort(key=lambda x: x[3], reverse=True)

    if not suggestions:
        print("No actionable categorical changes found that improve defect quality.")
        return

    print("Recommended changes to improve quality (reduce defects):")
    for i, (col, cat, new_mean, imp) in enumerate(suggestions[:3], 1):
        print(f"  {i}. Change '{col}' to '{cat}' → expected mean Defects- = {new_mean:.2f} (improvement {imp:.2f})")
    print("\nTrade‑offs: tightening a process attribute may increase effort or delay delivery.")
    print()


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_data(csv_path)
    if not rows:
        print("No data loaded from the CSV file.")
        sys.exit(1)

    nasa_15(rows)
    nasa_16(rows)


if __name__ == "__main__":
    main()
