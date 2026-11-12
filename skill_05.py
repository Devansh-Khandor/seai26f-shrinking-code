#!/usr/bin/env python3
"""
Skill 05: Analyze poor fuel efficiency and unfavorable performance.

CAR-09: Identify characteristics most strongly associated with unusually poor fuel efficiency.
CAR-10: Compare common characteristics between vehicles with unfavorable and favorable overall performance.

Usage:
    python skill_05.py <path-to-auto93.csv>
"""

import sys
import math
from pathlib import Path

# Reuse functionality from earlier skills
from skill_01 import load_data, compute_stats, compute_balanced_scores, format_value


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def _numeric_fields():
    """Return a list of numeric field names that are present in the dataset."""
    return ["Clndrs", "Volume", "HpX", "Lbs-", "Acc+", "Mpg+"]


def _mean_std(values):
    """Return mean and standard deviation of a list of numbers."""
    if not values:
        return (None, None)
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return (mean, math.sqrt(var))


def _pearson_correlation(xs, ys):
    """Compute Pearson correlation coefficient between two lists of numbers."""
    n = len(xs)
    if n == 0:
        return 0.0
    mean_x, std_x = _mean_std(xs)
    mean_y, std_y = _mean_std(ys)
    if std_x == 0 or std_y == 0:
        return 0.0
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
    return cov / (std_x * std_y)


# ------------------------------------------------------------------
# CAR-09: Poor fuel efficiency analysis
# ------------------------------------------------------------------
def _car_09(rows):
    """Identify characteristics most associated with unusually poor mpg."""
    # Compute mpg stats to define 'unusually poor'
    mpgs = [r["Mpg+"] for r in rows if isinstance(r["Mpg+"], (int, float))]
    mpg_mean, mpg_std = _mean_std(mpgs)
    if mpg_std is None:
        mpg_std = 1.0
    poor_threshold = mpg_mean - 2 * mpg_std

    # Gather data for correlation
    correlations = []
    for field in _numeric_fields():
        if field == "Mpg+":
            continue  # skip target
        xs = []
        ys = []
        for r in rows:
            val = r[field]
            mpg = r["Mpg+"]
            if isinstance(val, (int, float)) and isinstance(mpg, (int, float)):
                xs.append(val)
                ys.append(mpg)
        if xs:
            corr = _pearson_correlation(xs, ys)
            correlations.append((field, corr))

    # Sort by absolute correlation descending, then by correlation ascending (most negative)
    correlations.sort(key=lambda t: (abs(t[1]), t[1]), reverse=True)

    print("== CAR-09 ==")
    print("Characteristics most strongly associated with unusually poor fuel efficiency:")
    for idx, (field, corr) in enumerate(correlations[:3], start=1):
        print(
            f"{idx}. {field}: correlation {format_value(corr, '{:.3f}')} "
            f"(lower values linked to worse mpg)"
        )


# ------------------------------------------------------------------
# CAR-10: Unfavorable vs. favorable performance comparison
# ------------------------------------------------------------------
def _car_10(rows):
    """Compare common characteristics of unfavorable vs favorable vehicles."""
    scored_rows = compute_balanced_scores(rows)

    # Determine threshold for bottom third
    scores = sorted([r["score"] for r in scored_rows])
    threshold_idx = int(len(scores) * 2 / 3)
    bad_threshold = scores[threshold_idx]

    bad_rows = [r for r in scored_rows if r["score"] <= bad_threshold]
    good_rows = [r for r in scored_rows if r["score"] > bad_threshold]

    print("\n== CAR-10 ==")
    print("Common characteristics of vehicles with unfavorable overall performance:")

    # For each numeric field compute mean in both groups
    for field in _numeric_fields():
        bad_vals = [r[field] for r in bad_rows if isinstance(r[field], (int, float))]
        good_vals = [r[field] for r in good_rows if isinstance(r[field], (int, float))]

        bad_mean, _ = _mean_std(bad_vals)
        good_mean, _ = _mean_std(good_vals)

        diff = (bad_mean - good_mean) if bad_mean is not None and good_mean is not None else None

        # Only print if we have data
        if bad_mean is not None and good_mean is not None:
            print(
                f"- {field}: avg bad = {format_value(bad_mean)}, "
                f"avg good = {format_value(good_mean)}, "
                f"diff = {format_value(diff)}"
            )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).is_file():
        print(f"Error: file '{csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    rows = load_data(csv_path)
    if not rows:
        print("Error: no data loaded.", file=sys.stderr)
        sys.exit(1)

    _car_09(rows)
    _car_10(rows)


if __name__ == "__main__":
    main()
