#!/usr/bin/env python3
"""
skill_05.py

Usage:
    python skill_05.py <csv_path>

This script answers two client requests:
    NASA-09 – Identify characteristics most strongly associated with unusually
              high development effort projects.
    NASA-10 – Identify characteristics most strongly associated with unusually
              high defect counts.

The script uses only the Python standard library.  For each request it prints
the request header, at least three lines of results, and the numeric values
used in the analysis.
"""

import csv
import math
import sys
from statistics import mean

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_float(value: str | None) -> float | None:
    """Parse a string into a float, or return None if parsing fails."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean_std(values):
    """Return (mean, stddev) ignoring None."""
    clean = [v for v in values if v is not None]
    if not clean:
        return (None, None)
    m = sum(clean) / len(clean)
    var = sum((x - m) ** 2 for x in clean) / len(clean)
    return (m, math.sqrt(var))


def _pearson_corr(x_vals, y_vals):
    """Return Pearson correlation coefficient for two numeric lists."""
    if len(x_vals) < 2:
        return None
    mean_x = mean(x_vals)
    mean_y = mean(y_vals)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
    var_x = sum((x - mean_x) ** 2 for x in x_vals)
    var_y = sum((y - mean_y) ** 2 for y in y_vals)
    denom = math.sqrt(var_x * var_y)
    return cov / denom if denom != 0 else None


def _load_csv(csv_path):
    """Load CSV, parse numeric fields, and return list of dicts."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for r in reader:
            parsed = {
                "idX": r.get("idX"),
                "Kloc+": _parse_float(r.get("Kloc+")),
                "Effort-": _parse_float(r.get("Effort-")),
                "Defects-": _parse_float(r.get("Defects-")),
                "MonthsX": _parse_float(r.get("MonthsX")),
                "acap": r.get("acap"),
                "cplx": r.get("cplx"),
                "time": r.get("time"),
            }
            rows.append(parsed)
    return rows


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def main(csv_path):
    rows = _load_csv(csv_path)

    # Gather numeric lists
    effort_vals = [r["Effort-"] for r in rows]
    defect_vals = [r["Defects-"] for r in rows]
    kloc_vals = [r["Kloc+"] for r in rows]
    months_vals = [r["MonthsX"] for r in rows]

    # Compute overall statistics
    effort_mean, effort_std = _mean_std(effort_vals)
    defect_mean, defect_std = _mean_std(defect_vals)

    # ------------------------------------------------------------------ NASA-09
    print("== NASA-09 ==")
    if effort_mean is None or effort_std is None:
        print("Insufficient data to compute effort statistics.")
    else:
        high_effort_thresh = effort_mean + 2 * effort_std
        # Find projects above threshold
        high_effort_projects = [
            r for r in rows
            if r["Effort-"] is not None and r["Effort-"] > high_effort_thresh
        ]
        print(f"Unusually high effort threshold: {high_effort_thresh:.2f}")
        print(f"Number of projects above threshold: {len(high_effort_projects)}")

        # Correlations with Effort-
        # Build lists of values where both predictors and Effort- are present
        def valid_pairs(predictor, target):
            return [
                (p, t) for p, t in zip(predictor, target)
                if p is not None and t is not None
            ]

        kloc_pairs = valid_pairs(kloc_vals, effort_vals)
        months_pairs = valid_pairs(months_vals, effort_vals)

        kloc_corr = (
            _pearson_corr(
                [p for p, _ in kloc_pairs],
                [t for _, t in kloc_pairs],
            )
            if kloc_pairs else None
        )
        months_corr = (
            _pearson_corr(
                [p for p, _ in months_pairs],
                [t for _, t in months_pairs],
            )
            if months_pairs else None
        )

        print("Correlations with Effort-:")
        print(f"  Kloc+      : {kloc_corr:.3f}" if kloc_corr is not None else "  Kloc+      : N/A")
        print(f"  MonthsX    : {months_corr:.3f}" if months_corr is not None else "  MonthsX    : N/A")

    # ------------------------------------------------------------------ NASA-10
    print("\n== NASA-10 ==")
    if defect_mean is None or defect_std is None:
        print("Insufficient data to compute defect statistics.")
    else:
        high_defect_thresh = defect_mean + 2 * defect_std
        high_defect_projects = [
            r for r in rows
            if r["Defects-"] is not None and r["Defects-"] > high_defect_thresh
        ]
        print(f"Unusually high defect threshold: {high_defect_thresh:.2f}")
        print(f"Number of projects above threshold: {len(high_defect_projects)}")

        # Correlations with Defects-
        kloc_pairs_def = valid_pairs(kloc_vals, defect_vals)
        months_pairs_def = valid_pairs(months_vals, defect_vals)

        kloc_corr_def = (
            _pearson_corr(
                [p for p, _ in kloc_pairs_def],
                [t for _, t in kloc_pairs_def],
            )
            if kloc_pairs_def else None
        )
        months_corr_def = (
            _pearson_corr(
                [p for p, _ in months_pairs_def],
                [t for _, t in months_pairs_def],
            )
            if months_pairs_def else None
        )

        print("Correlations with Defects-:")
        print(f"  Kloc+      : {kloc_corr_def:.3f}" if kloc_corr_def is not None else "  Kloc+      : N/A")
        print(f"  MonthsX    : {months_corr_def:.3f}" if months_corr_def is not None else "  MonthsX    : N/A")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
