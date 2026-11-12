#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <csv_path>

This script extends the previous analysis with two new NASA queries:

  NASA-03 – Predict the development effort for a proposed project
            based on its known size (KLOC) and historical records.
  NASA-04 – Predict the expected defect count for a proposed project
            and provide an uncertainty estimate.

The script re‑uses helper functions from skill_01.py whenever possible.
"""

import csv
import sys
import statistics
import math
from collections import namedtuple

# --------------------------------------------------------------------
# Import helpers from skill_01
# --------------------------------------------------------------------

# We copy the minimal set of helpers that are needed from the previous
# module.  This keeps the implementation self‑contained while still
# re‑using the parsing logic.

NUMERIC_FIELDS = ["idX", "centerX", "YearX", "Kloc+", "Effort-", "Defects-", "MonthsX"]

Row = namedtuple(
    "Row",
    [
        "id",
        "center",
        "year",
        "kloc",
        "effort",
        "defects",
        "months",
    ],
)

def parse_row(raw: dict) -> Row | None:
    try:
        id_ = int(raw["idX"])
        center = int(raw["centerX"])
        year = int(raw["YearX"])
        kloc = float(raw["Kloc+"])
        effort = float(raw["Effort-"])
        defects = float(raw["Defects-"])
        months = float(raw["MonthsX"])
    except (ValueError, KeyError):
        return None
    return Row(id_, center, year, kloc, effort, defects, months)

def read_csv(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = parse_row(raw)
            if row:
                rows.append(row)
    return rows

# --------------------------------------------------------------------
# New statistical helpers
# --------------------------------------------------------------------

def linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """
    Compute a simple linear regression (y = a + b*x).
    Returns (a, b) where a is the intercept and b is the slope.
    """
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least two data points for regression")
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        raise ValueError("Variance of x is zero; cannot compute slope")
    b = cov_xy / var_x
    a = mean_y - b * mean_x
    return a, b

def predict(x: float, a: float, b: float) -> float:
    return a + b * x

def residuals(xs, ys, a, b):
    return [y - (a + b * x) for x, y in zip(xs, ys)]

def standard_error(residuals_list: list[float]) -> float:
    n = len(residuals_list)
    if n <= 1:
        return 0.0
    s2 = sum(r ** 2 for r in residuals_list) / (n - 1)
    return math.sqrt(s2)

# --------------------------------------------------------------------
# NASA-03 – Effort prediction
# --------------------------------------------------------------------

def nasa_03(rows: list[Row]):
    """
    Predict effort for a proposed project based on its KLOC.
    Uses a simple linear regression of Effort vs KLOC.
    """
    klocs = [r.kloc for r in rows]
    efforts = [r.effort for r in rows]
    a, b = linear_regression(klocs, efforts)

    print("== NASA-03 ==")
    print(f"Regression model: Effort = {a:.2f} + {b:.4f} * KLOC")
    print("Enter the estimated project size (KLOC) for the proposed project:")

    try:
        size_str = input().strip()
        proposed_size = float(size_str)
    except Exception:
        print("  (Invalid input – using median KLOC from historical data)")
        proposed_size = statistics.median(klocs)

    predicted_effort = predict(proposed_size, a, b)
    print(f"  Project size: {proposed_size:.1f} KLOC")
    print(f"  Estimated effort: {predicted_effort:.1f} person‑months")

# --------------------------------------------------------------------
# NASA-04 – Defect prediction with uncertainty
# --------------------------------------------------------------------

def nasa_04(rows: list[Row]):
    """
    Predict defect count for a proposed project and estimate
    uncertainty using the standard deviation of the regression residuals.
    """
    klocs = [r.kloc for r in rows]
    defects = [r.defects for r in rows]
    a, b = linear_regression(klocs, defects)
    res = residuals(klocs, defects, a, b)
    se = standard_error(res)

    print("\n== NASA-04 ==")
    print(f"Regression model: Defects = {a:.2f} + {b:.4f} * KLOC")
    print(f"  Standard error of the estimate: {se:.2f}")

    print("Enter the estimated project size (KLOC) for the proposed project:")
    try:
        size_str = input().strip()
        proposed_size = float(size_str)
    except Exception:
        print("  (Invalid input – using median KLOC from historical data)")
        proposed_size = statistics.median(klocs)

    predicted_defects = predict(proposed_size, a, b)
    lower = predicted_defects - se
    upper = predicted_defects + se

    print(f"  Project size: {proposed_size:.1f} KLOC")
    print(f"  Estimated defect count: {predicted_defects:.1f}")
    print(f"  1‑sigma uncertainty: ±{se:.1f} (range {lower:.1f} – {upper:.1f})")

# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main(csv_path: str):
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.")
        return

    nasa_03(rows)
    nasa_04(rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
