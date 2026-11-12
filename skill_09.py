#!/usr/bin/env python3
"""
Skill 09: Outlier and discrepancy detection in the inventory.

CAR-17: Flag vehicle records whose combinations of characteristics are unusually
different from the rest of our inventory so that staff can review them.

CAR-18: Identify vehicles whose reported fuel efficiency or weight differs
substantially from what similar vehicles would lead us to expect, so we can
check for possible recording mistakes.

Usage:
    python skill_09.py <path-to-auto93.csv>
"""

import csv
import sys
import math
import statistics
from pathlib import Path

# Reuse core utilities from the earlier skills
from skill_01 import load_data, format_value
from skill_02 import _prepare_regression, _fit_regression, _predict


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def _numeric_fields(rows):
    """Return list of numeric field names that are present in the dataset."""
    # The dataset always contains these fields
    return ["Clndrs", "Volume", "HpX", "Lbs-", "Acc+", "Mpg+"]


def _mean_stdev(values):
    """Return mean and sample standard deviation of a list of numbers."""
    n = len(values)
    if n == 0:
        return (None, None)
    mean = sum(values) / n
    if n < 2:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return (mean, math.sqrt(var))


def _compute_stats(rows, fields):
    """Return a dict mapping field -> (mean, stdev) for numeric fields."""
    stats = {}
    for f in fields:
        vals = [r[f] for r in rows if isinstance(r[f], (int, float))]
        mean, stdev = _mean_stdev(vals)
        stats[f] = (mean, stdev if stdev > 0 else 1e-6)  # avoid division by zero
    return stats


def _standardized_z(value, mean, stdev):
    return (value - mean) / stdev if stdev else 0.0


def _distance_from_mean(row, stats):
    """Euclidean distance in standardized z‑space from the mean of all fields."""
    s = 0.0
    for field, (mean, stdev) in stats.items():
        z = _standardized_z(row[field], mean, stdev)
        s += z ** 2
    return math.sqrt(s)


# ------------------------------------------------------------------
# CAR-17: outlier detection
# ------------------------------------------------------------------
def _car_17(rows):
    """Flag vehicles whose multivariate characteristics are far from the norm."""
    print("== CAR-17 ==")

    fields = _numeric_fields(rows)
    stats = _compute_stats(rows, fields)

    # Compute distances for each row
    distances = []
    for r in rows:
        d = _distance_from_mean(r, stats)
        distances.append((r, d))

    # Determine threshold: mean + 2 * std of distances
    d_vals = [d for _, d in distances]
    d_mean, d_std = _mean_stdev(d_vals)
    threshold = d_mean + 2 * d_std if d_std else d_mean

    # Select outliers
    outliers = [(r, d) for r, d in distances if d > threshold]
    if not outliers:
        # Fallback: pick top 5 by distance
        outliers = sorted(distances, key=lambda x: x[1], reverse=True)[:5]

    # Print at least three results
    top = outliers[:5]
    print(f"Identified {len(top)} anomalous vehicles (distance > {format_value(threshold)}):")
    for idx, (r, d) in enumerate(top, start=1):
        print(
            f"{idx}. Model: {r['Model']} (Score: {format_value(r.get('score', 'N/A'), '{:.3f}')})\n"
            f"   Distance from mean: {format_value(d)}\n"
            f"   Characteristics: Clndrs={r['Clndrs']}, Volume={r['Volume']}, "
            f"HpX={r['HpX']}, Lbs-={r['Lbs-']}, Acc+={r['Acc+']}, Mpg+={r['Mpg+']}"
        )
    print()


# ------------------------------------------------------------------
# CAR-18: discrepancy detection for MPG and weight
# ------------------------------------------------------------------
def _car_18(rows):
    """Identify vehicles whose MPG or weight differs substantially from predictions."""
    print("== CAR-18 ==")

    # --- MPG regression ---
    mpg_predictors = ["Lbs-", "Acc+", "Volume", "HpX", "Clndrs"]
    X_mpg, y_mpg = _prepare_regression(rows, "Mpg+", mpg_predictors)
    if X_mpg:
        beta_mpg, std_mpg = _fit_regression(X_mpg, y_mpg)
    else:
        beta_mpg, std_mpg = None, None

    mpg_discrepancies = []
    for r in rows:
        actual = r.get("Mpg+")
        if isinstance(actual, (int, float)):
            pred = _predict(r, beta_mpg, mpg_predictors)
            resid = actual - pred
            mpg_discrepancies.append((r, resid))

    # --- Weight regression ---
    weight_predictors = ["Mpg+", "Acc+", "Volume", "HpX", "Clndrs"]
    X_wt, y_wt = _prepare_regression(rows, "Lbs-", weight_predictors)
    if X_wt:
        beta_wt, std_wt = _fit_regression(X_wt, y_wt)
    else:
        beta_wt, std_wt = None, None

    wt_discrepancies = []
    for r in rows:
        actual = r.get("Lbs-")
        if isinstance(actual, (int, float)):
            pred = _predict(r, beta_wt, weight_predictors)
            resid = actual - pred
            wt_discrepancies.append((r, resid))

    # Determine thresholds: mean + 2*std of absolute residuals
    def threshold_and_sort(discrepancies, std):
        abs_resids = [abs(res) for _, res in discrepancies]
        if not abs_resids:
            return 0.0, []
        mean, stdev = _mean_stdev(abs_resids)
        thr = mean + 2 * stdev if stdev else mean
        sorted_list = sorted(discrepancies, key=lambda x: abs(x[1]), reverse=True)
        return thr, sorted_list

    mpg_thr, mpg_sorted = threshold_and_sort(mpg_discrepancies, std_mpg)
    wt_thr, wt_sorted = threshold_and_sort(wt_discrepancies, std_wt)

    # Print MPG discrepancies
    print(f"\nMPG discrepancies (threshold > {format_value(mpg_thr)}):")
    for idx, (r, resid) in enumerate(mpg_sorted[:5], start=1):
        print(
            f"{idx}. Model: {r['Model']}\n"
            f"   Actual MPG: {format_value(r['Mpg+'])}, Predicted MPG: {format_value(_predict(r, beta_mpg, mpg_predictors))}\n"
            f"   Residual: {format_value(resid)}\n"
            f"   (Positive residual indicates reported MPG is higher than expected)"
        )

    # Print Weight discrepancies
    print(f"\nWeight discrepancies (threshold > {format_value(wt_thr)}):")
    for idx, (r, resid) in enumerate(wt_sorted[:5], start=1):
        print(
            f"{idx}. Model: {r['Model']}\n"
            f"   Actual Weight: {format_value(r['Lbs-'])} lbs, Predicted Weight: {format_value(_predict(r, beta_wt, weight_predictors))}\n"
            f"   Residual: {format_value(resid)} lbs\n"
            f"   (Positive residual indicates reported weight is heavier than expected)"
        )

    print()


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).is_file():
        print(f"Error: file '{csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    rows = load_data(csv_path)
    if not rows:
        print("Error: no data loaded.", file=sys.stderr)
        sys.exit(1)

    _car_17(rows)
    _car_18(rows)


if __name__ == "__main__":
    main()
