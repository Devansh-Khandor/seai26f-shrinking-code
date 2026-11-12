#!/usr/bin/env python3
"""
skill_08.py

NASA-15 & NASA-16 analyses.

- NASA-15: For proposals predicted to require excessive effort,
  identify plausible changes to adjustable characteristics that could
  reduce the predicted effort.
- NASA-16: For proposals with a high predicted defect count,
  suggest the smallest plausible set of changes that could improve
  predicted quality and explain trade‑offs.

Usage:
    python skill_08.py <path/to/nasa93dem.csv>
"""

import csv
import sys
import math
import statistics
from pathlib import Path

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _read_csv(path: Path):
    """Read CSV and return list of dicts with numeric conversion."""
    data = []
    with path.open(newline="") as f:
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

def _numeric_values(data, column):
    """Return list of numeric values for a column, excluding None."""
    return [row[column] for row in data
            if isinstance(row.get(column), (int, float))]

def _linear_regression(x_vals, y_vals):
    """Return (intercept, slope, SE, mean_x, ssx)."""
    n = len(x_vals)
    if n < 2:
        raise ValueError("Need at least two data points for regression.")
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    ssx = sum((x - mean_x) ** 2 for x in x_vals)
    if ssx == 0:
        slope = 0.0
    else:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / ssx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(x_vals, y_vals)]
    sse = sum(r ** 2 for r in residuals)
    se = (sse / (n - 2)) ** 0.5 if n > 2 else 0.0
    return intercept, slope, se, mean_x, ssx

def _prediction_interval(x, intercept, slope, se, mean_x, ssx, n, conf_factor=2.0):
    """Compute prediction interval for a new observation at x."""
    if ssx == 0 or n <= 2:
        return None, None
    se_pred = se * math.sqrt(1 + 1/n + ((x - mean_x) ** 2) / ssx)
    margin = conf_factor * se_pred
    y_hat = intercept + slope * x
    return y_hat, margin

def _categorical_columns(data):
    """Return list of columns that are string in at least one row."""
    cat_cols = set()
    for row in data:
        for k, v in row.items():
            if isinstance(v, str):
                cat_cols.add(k)
    return list(cat_cols)

def _group_avg(data, group_col, value_col):
    """Return mapping category -> average of value_col among completed projects."""
    totals = {}
    counts = {}
    for row in data:
        grp = row.get(group_col)
        val = row.get(value_col)
        if grp is None or not isinstance(val, (int, float)):
            continue
        totals[grp] = totals.get(grp, 0.0) + val
        counts[grp] = counts.get(grp, 0) + 1
    return {grp: totals[grp] / counts[grp] for grp in totals}

def _best_category_for_metric(completed, proposal, metric_col):
    """
    For each categorical column, find the category with the lowest
    average metric value. Return a list of suggestion strings.
    """
    suggestions = []
    cat_cols = _categorical_columns(completed)
    # Exclude numeric columns like idX, Kloc+, Effort- etc.
    # Also exclude columns that are the same as metric_col
    cat_cols = [c for c in cat_cols if c not in ("idX", "Kloc+", "Effort-", "Defects-")]

    # Overall average to decide if current is higher
    overall = statistics.mean([row[metric_col] for row in completed
                               if isinstance(row.get(metric_col), (int, float))])

    for col in cat_cols:
        # Current value
        current = proposal.get(col)
        if current is None:
            continue
        # Compute group averages
        grp_avgs = _group_avg(completed, col, metric_col)
        # Skip if we have no data for this column
        if not grp_avgs:
            continue
        # Current avg
        curr_avg = grp_avgs.get(current)
        if curr_avg is None:
            continue
        # If current avg already below overall, no need to change
        if curr_avg <= overall:
            continue
        # Find the best (lowest avg) category
        best_cat, best_avg = min(grp_avgs.items(), key=lambda kv: kv[1])
        if best_cat == current:
            continue
        suggestions.append(
            f"Change '{col}' from '{current}' to '{best_cat}' "
            f"(avg {best_avg:.1f} vs current {curr_avg:.1f})"
        )
    return suggestions or ["No actionable suggestions found."]

def _analyze_proposals(data):
    """Return list of proposals with predictions and high flags."""
    # Split completed vs proposals
    completed = [
        r for r in data
        if isinstance(r.get("Effort-"), (int, float))
        and isinstance(r.get("Defects-"), (int, float))
    ]
    proposals = [
        r for r in data
        if not isinstance(r.get("Effort-"), (int, float))
        or not isinstance(r.get("Defects-"), (int, float))
    ]

    if not completed or not proposals:
        return [], [], []

    # Statistics for thresholds
    eff_vals = _numeric_values(completed, "Effort-")
    defect_vals = _numeric_values(completed, "Defects-")
    eff_mean, eff_std = statistics.mean(eff_vals), statistics.stdev(eff_vals) if len(eff_vals) > 1 else 0.0
    def_mean, def_std = statistics.mean(defect_vals), statistics.stdev(defect_vals) if len(defect_vals) > 1 else 0.0

    # Regression models
    try:
        e_int, e_slope, e_se, e_mean_x, e_ssx = _linear_regression(
            _numeric_values(completed, "Kloc+"),
            _numeric_values(completed, "Effort-")
        )
    except Exception:
        e_int = e_slope = e_se = e_mean_x = e_ssx = None

    try:
        d_int, d_slope, d_se, d_mean_x, d_ssx = _linear_regression(
            _numeric_values(completed, "Effort-"),
            _numeric_values(completed, "Defects-")
        )
    except Exception:
        d_int = d_slope = d_se = d_mean_x = d_ssx = None

    high_effort = []
    high_defect = []

    for prop in proposals:
        pid = prop.get("idX", "Unknown")
        k = prop.get("Kloc+")
        if not isinstance(k, (int, float)):
            continue
        # Predict effort
        if e_int is not None:
            eff_pred, eff_margin = _prediction_interval(
                k, e_int, e_slope, e_se, e_mean_x, e_ssx, len(eff_vals)
            )
            eff_pred = eff_pred if eff_pred is not None else None
        else:
            eff_pred = eff_margin = None
        # Predict defect
        if eff_pred is not None and d_int is not None:
            defect_pred, defect_margin = _prediction_interval(
                eff_pred, d_int, d_slope, d_se, d_mean_x, d_ssx, len(defect_vals)
            )
            defect_pred = defect_pred if defect_pred is not None else None
        else:
            defect_pred = defect_margin = None

        # Flags
        if eff_pred is not None and eff_pred > eff_mean + 2 * eff_std:
            high_effort.append((prop, eff_pred, eff_margin))
        if defect_pred is not None and defect_pred > def_mean + 2 * def_std:
            high_defect.append((prop, defect_pred, defect_margin))

    return high_effort, high_defect, completed

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    data = _read_csv(csv_path)

    high_effort, high_defect, completed = _analyze_proposals(data)

    # NASA-15: Excessive effort
    print("== NASA-15 ==")
    if not high_effort:
        print("No proposals predicted to require excessive effort.")
    else:
        for prop, eff_pred, eff_margin in high_effort:
            pid = prop.get("idX", "Unknown")
            k = prop.get("Kloc+")
            print(f"Proposal ID {pid} (Kloc+={k})")
            if eff_margin is not None:
                lower, upper = eff_pred - eff_margin, eff_pred + eff_margin
                print(f"  Predicted Effort = {eff_pred:.1f} (+/- {eff_margin:.1f}) "
                      f"≈ [{lower:.1f}, {upper:.1f}] person‑months")
            else:
                print(f"  Predicted Effort = {eff_pred:.1f} person‑months (no interval)")
            # Suggestions
            suggestions = _best_category_for_metric(completed, prop, "Effort-")
            print("  Suggested changes to reduce effort:")
            for s in suggestions:
                print(f"    - {s}")
            print()

    # NASA-16: High defect count
    print("== NASA-16 ==")
    if not high_defect:
        print("No proposals predicted to have a high defect count.")
    else:
        for prop, defect_pred, defect_margin in high_defect:
            pid = prop.get("idX", "Unknown")
            k = prop.get("Kloc+")
            print(f"Proposal ID {pid} (Kloc+={k})")
            if defect_margin is not None:
                lower, upper = defect_pred - defect_margin, defect_pred + defect_margin
                print(f"  Predicted Defects = {defect_pred:.1f} (+/- {defect_margin:.1f}) "
                      f"≈ [{lower:.1f}, {upper:.1f}] defects")
            else:
                print(f"  Predicted Defects = {defect_pred:.1f} defects (no interval)")
            # Suggestions
            suggestions = _best_category_for_metric(completed, prop, "Defects-")
            print("  Suggested changes to improve quality:")
            for s in suggestions:
                print(f"    - {s}")
            print()

    sys.exit(0)

if __name__ == "__main__":
    main()
