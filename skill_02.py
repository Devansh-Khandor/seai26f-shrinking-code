#!/usr/bin/env python3
"""
Small data‑analysis skill for NASA project data – extended for NASA-03 and NASA-04.

Usage:
    python skill_02.py <csv_path>
"""

import csv
import sys
import statistics
import math
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Re‑use helpers from skill_01.py
# --------------------------------------------------------------------------- #

def load_data(csv_path: str) -> List[Dict[str, str]]:
    """Read CSV and return list of records (dicts)."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_float(value: str) -> float:
    """Convert a string to float, stripping spaces. Return NaN for empty strings."""
    try:
        return float(value.strip())
    except ValueError:
        return float('nan')


def extract_numeric_fields(records: List[Dict[str, str]]) -> Dict[str, List[float]]:
    """Return dictionary mapping numeric column name to list of values."""
    numeric = {
        'Kloc+': [],
        'Effort-': [],
        'Defects-': [],
        'MonthsX': []
    }
    for r in records:
        numeric['Kloc+'].append(parse_float(r['Kloc+']))
        numeric['Effort-'].append(parse_float(r['Effort-']))
        numeric['Defects-'].append(parse_float(r['Defects-']))
        numeric['MonthsX'].append(parse_float(r['MonthsX']))
    return numeric


def compute_stats(values: List[float]) -> Tuple[float, float, float]:
    """Return mean, median, standard deviation for a list of numbers."""
    vals = [v for v in values if not v != v]  # filter NaNs
    if not vals:
        return (float('nan'), float('nan'), float('nan'))
    mean = statistics.mean(vals)
    median = statistics.median(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (mean, median, stdev)


def find_best_worst(records: List[Dict[str, str]]) -> Tuple[Tuple[int, float], Tuple[int, float]]:
    """Return best and worst project id and defect count."""
    best = None  # (id, defects)
    worst = None
    for r in records:
        pid = int(r['idX'])
        defects = parse_float(r['Defects-'])
        if defects != defects:
            continue
        if best is None or defects < best[1]:
            best = (pid, defects)
        if worst is None or defects > worst[1]:
            worst = (pid, defects)
    return best, worst


def percentile(values: List[float], p: float) -> float:
    """Return the p-th percentile (0‑100) of a list of numbers."""
    vals = sorted(v for v in values if v == v)  # drop NaN
    if not vals:
        return float('nan')
    k = (len(vals) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(vals):
        return vals[f] + c * (vals[f + 1] - vals[f])
    else:
        return vals[f]


# --------------------------------------------------------------------------- #
# New helpers for regression (used by NASA‑03 and NASA‑04)
# --------------------------------------------------------------------------- #

def _filter_nan(xs: List[float]) -> List[float]:
    """Return list without NaNs."""
    return [x for x in xs if not x != x]


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """
    Perform ordinary least squares regression y = a * x + b.
    Returns (slope, intercept, residual_std).
    """
    x = _filter_nan(x)
    y = _filter_nan(y)
    if len(x) != len(y) or not x:
        return (float('nan'), float('nan'), float('nan'))

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    if sxx == 0:
        a = 0.0
    else:
        a = sxy / sxx

    b = mean_y - a * mean_x

    # Compute residual standard deviation
    residuals = [yi - (a * xi + b) for xi, yi in zip(x, y)]
    if len(residuals) > 1:
        res_std = statistics.stdev(residuals)
    else:
        res_std = 0.0

    return (a, b, res_std)


def predict(slope: float, intercept: float, x_val: float) -> float:
    """Compute y = slope * x + intercept."""
    return slope * x_val + intercept


# --------------------------------------------------------------------------- #
# Main routine: NASA‑03 & NASA‑04
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_data(csv_path)
    numeric = extract_numeric_fields(records)

    # Compute basic stats for reference
    k_mean, k_med, k_sd = compute_stats(numeric['Kloc+'])
    e_mean, e_med, e_sd = compute_stats(numeric['Effort-'])
    d_mean, d_med, d_sd = compute_stats(numeric['Defects-'])

    # NASA‑03: Estimate effort from size (Kloc+)
    slope_ek, intercept_ek, resid_std_ek = linear_regression(numeric['Kloc+'], numeric['Effort-'])
    pred_effort_med_size = predict(slope_ek, intercept_ek, k_med)

    print("== NASA-03 ==")
    print(f"Regression of Effort on Size: slope={slope_ek:.3f}, intercept={intercept_ek:.3f}")
    print(f"Median project size (Kloc+={k_med:.1f}) predicts effort ≈ {pred_effort_med_size:.1f}")
    print(f"Residual standard deviation (effort): {resid_std_ek:.1f}")

    # NASA‑04: Estimate defects from effort
    slope_de, intercept_de, resid_std_de = linear_regression(numeric['Effort-'], numeric['Defects-'])
    pred_defects_med_effort = predict(slope_de, intercept_de, e_med)

    print("\n== NASA-04 ==")
    print(f"Regression of Defects on Effort: slope={slope_de:.3f}, intercept={intercept_de:.3f}")
    print(f"Median effort (Effort-={e_med:.1f}) predicts defects ≈ {pred_defects_med_effort:.1f}")
    print(f"Residual standard deviation (defects): {resid_std_de:.1f}")


if __name__ == "__main__":
    main()
