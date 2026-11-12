#!/usr/bin/env python3
"""
skill_04.py

A self‑contained script that extends the NASA data analysis to
provide real‑time project assessment and historical updates.

- NASA-07: Assess new project proposals using available characteristics
           and alert when predicted effort or defects are unusually high.
- NASA-08: Update historical summaries and predictive models with all
           completed projects in the dataset, so future assessments reflect
           the latest information.

Usage:
    python skill_04.py path/to/nasa93dem.csv
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
    with path.open(newline='') as f:
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
    """Return a list of numeric values for a column, excluding None."""
    return [row[column] for row in data if isinstance(row.get(column), (int, float))]

def _mean_std(values):
    """Return (mean, stddev).  If <2 samples, stddev is None."""
    n = len(values)
    if n == 0:
        return None, None
    mean_val = statistics.mean(values)
    if n < 2:
        return mean_val, None
    std_val = statistics.stdev(values)
    return mean_val, std_val

def _linear_regression(x_vals, y_vals):
    """Return (intercept, slope, SE of estimate, mean_x, ssx)."""
    n = len(x_vals)
    if n < 2:
        raise ValueError("Need at least two data points for regression.")
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    ssx = sum((x - mean_x) ** 2 for x in x_vals)
    ssy = sum((y - mean_y) ** 2 for y in y_vals)
    if ssx == 0:
        slope = 0.0
    else:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / ssx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(x_vals, y_vals)]
    sse = sum(r ** 2 for r in residuals)
    se = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
    return intercept, slope, se, mean_x, ssx

def _predict(x, intercept, slope):
    """Linear prediction without confidence interval."""
    return intercept + slope * x

def _is_unusually_high(pred, mean, std, factor=2.0):
    """Return True if pred > mean + factor*std (std may be None)."""
    if mean is None or std is None:
        return False
    return pred > mean + factor * std

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    data = _read_csv(csv_path)

    # Separate completed projects (both Effort- and Defects- present)
    completed = [
        row for row in data
        if isinstance(row.get("Effort-"), (int, float))
        and isinstance(row.get("Defects-"), (int, float))
    ]
    # Proposals: any missing Effort- or Defects-
    proposals = [
        row for row in data
        if not isinstance(row.get("Effort-"), (int, float))
        or not isinstance(row.get("Defects-"), (int, float))
    ]

    # ------------------------------------------------------------------- #
    # NASA-08: Update historical summaries and regression models
    # ------------------------------------------------------------------- #
    print("== NASA-08 ==")
    if not completed:
        print("No completed projects available to update summaries.")
    else:
        # Statistics
        kloc_vals = _numeric_values(completed, "Kloc+")
        effort_vals = _numeric_values(completed, "Effort-")
        defect_vals = _numeric_values(completed, "Defects-")

        kloc_mean, kloc_std = _mean_std(kloc_vals)
        effort_mean, effort_std = _mean_std(effort_vals)
        defect_mean, defect_std = _mean_std(defect_vals)

        # Regression Effort ~ Kloc
        try:
            e_intercept, e_slope, e_se, e_mean_x, e_ssx = _linear_regression(
                _numeric_values(completed, "Kloc+"),
                _numeric_values(completed, "Effort-")
            )
        except ValueError:
            e_intercept = e_slope = e_se = None

        # Regression Defects ~ Effort
        try:
            d_intercept, d_slope, d_se, d_mean_x, d_ssx = _linear_regression(
                _numeric_values(completed, "Effort-"),
                _numeric_values(completed, "Defects-")
            )
        except ValueError:
            d_intercept = d_slope = d_se = None

        # Median values
        median_kloc = statistics.median(kloc_vals)
        median_effort = statistics.median(effort_vals)

        # Predictions for median values
        pred_effort_median = _predict(median_kloc, e_intercept, e_slope) if e_intercept is not None else None
        pred_defect_median = _predict(pred_effort_median, d_intercept, d_slope) if d_intercept is not None else None

        # Output
        print(f"Completed projects: {len(completed)}")
        print(f"Kloc+ mean = {kloc_mean:.2f} (std = {kloc_std:.2f if kloc_std is not None else 'NA'})")
        print(f"Effort- mean = {effort_mean:.2f} (std = {effort_std:.2f if effort_std is not None else 'NA'})")
        print(f"Defects- mean = {defect_mean:.2f} (std = {defect_std:.2f if defect_std is not None else 'NA'})")
        if e_intercept is not None:
            print(f"Effort regression: Effort = {e_intercept:.2f} + {e_slope:.2f}*Kloc")
            print(f"  Median Kloc ({median_kloc:.1f}) → Pred Effort = {pred_effort_median:.1f}")
        else:
            print("Not enough data to compute Effort regression.")
        if d_intercept is not None:
            print(f"Defects regression: Defects = {d_intercept:.2f} + {d_slope:.2f}*Effort")
            print(f"  Pred Effort ({pred_effort_median:.1f}) → Pred Defects = {pred_defect_median:.1f}")
        else:
            print("Not enough data to compute Defects regression.")
    print()

    # ------------------------------------------------------------------- #
    # NASA-07: Evaluate proposals
    # ------------------------------------------------------------------- #
    print("== NASA-07 ==")
    if not proposals:
        print("No new project proposals detected.")
    else:
        # Ensure we have regression models
        if e_intercept is None or d_intercept is None:
            print("Insufficient data to evaluate proposals (missing regressions).")
        else:
            alerts = []
            for row in proposals:
                pid = row.get("idX", f"Row{data.index(row)+1}")
                kloc = row.get("Kloc+")
                if kloc is None or not isinstance(kloc, (int, float)):
                    # Cannot predict without size
                    print(f"Proposal {pid}: Missing Kloc+; cannot predict.")
                    continue
                # Predict effort
                eff_pred = _predict(kloc, e_intercept, e_slope)
                # Predict defects from predicted effort
                defect_pred = _predict(eff_pred, d_intercept, d_slope)

                # Check for unusually high values
                high_eff = _is_unusually_high(eff_pred, effort_mean, effort_std)
                high_def = _is_unusually_high(defect_pred, defect_mean, defect_std)

                alert_msg = ""
                if high_eff and high_def:
                    alert_msg = "ALERT: Effort & Defects unusually high!"
                elif high_eff:
                    alert_msg = "ALERT: Effort unusually high!"
                elif high_def:
                    alert_msg = "ALERT: Defects unusually high!"
                else:
                    alert_msg = "Within normal range."

                alerts.append((pid, eff_pred, defect_pred, alert_msg))
                print(f"Proposal {pid}: Pred Effort={eff_pred:.1f}, Pred Defects={defect_pred:.1f} → {alert_msg}")

            # Summary of alerts
            high_count = sum(1 for _, _, _, msg in alerts if "ALERT" in msg)
            if high_count:
                print(f"\nSummary: {high_count} proposal(s) triggered alerts.")
            else:
                print("\nSummary: No proposals exceeded the high‑value thresholds.")

    # End of script
    sys.exit(0)

if __name__ == "__main__":
    main()
