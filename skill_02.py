#!/usr/bin/env python3
"""
NASA-03 and NASA-04 prediction skill
"""

import csv
import json
import math
import sys

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV into list of dicts."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def to_float(val):
    """Convert to float, handle '?' and missing."""
    if val is None:
        return None
    val = val.strip()
    if val == '' or val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def linear_regression(x_vals, y_vals):
    """Return slope and intercept of least squares fit."""
    n = len(x_vals)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
    var_x = sum((x - mean_x) ** 2 for x in x_vals)
    if var_x == 0:
        return 0.0, mean_y  # all x same
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept

def prediction_std(x_vals, y_vals, slope, intercept):
    """Standard deviation of residuals."""
    n = len(x_vals)
    if n == 0:
        return 0.0
    resid_sq = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_vals, y_vals))
    return math.sqrt(resid_sq / n)

# ----------------------------------------------------------------------
# Main prediction logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    total_rows = len(rows)

    # Identify held‑out rows (index 1‑based)
    held_out_indices = [i + 1 for i in range(total_rows) if (i + 1) % 5 == 0]
    training_rows = [r for i, r in enumerate(rows) if (i + 1) not in held_out_indices]

    # --------------------------------------------------------------
    # Pre‑compute statistics for training data
    # --------------------------------------------------------------
    # Kloc+ mean for imputing missing values
    kloc_vals = [to_float(r["Kloc+"]) for r in training_rows]
    kloc_vals = [v for v in kloc_vals if v is not None]
    mean_kloc = sum(kloc_vals) / len(kloc_vals) if kloc_vals else 0.0

    # Effort‑ regression
    effort_x, effort_y = [], []
    for r in training_rows:
        x = to_float(r["Kloc+"])
        y = to_float(r["Effort-"])
        if x is not None and y is not None:
            effort_x.append(x)
            effort_y.append(y)
    eff_slope, eff_intercept = linear_regression(effort_x, effort_y)
    eff_sigma = prediction_std(effort_x, effort_y, eff_slope, eff_intercept)

    # Defects‑ regression
    defect_x, defect_y = [], []
    for r in training_rows:
        x = to_float(r["Kloc+"])
        y = to_float(r["Defects-"])
        if x is not None and y is not None:
            defect_x.append(x)
            defect_y.append(y)
    def_slope, def_intercept = linear_regression(defect_x, defect_y)
    def_sigma = prediction_std(defect_x, defect_y, def_slope, def_intercept)

    # --------------------------------------------------------------
    # Predictions for held‑out rows
    # --------------------------------------------------------------
    effort_preds = {}
    def_preds = {}
    mae_sum = 0.0
    mae_count = 0

    for idx in held_out_indices:
        r = rows[idx - 1]  # 0‑based
        kloc = to_float(r["Kloc+"])
        if kloc is None:
            kloc = mean_kloc

        # Effort prediction
        eff_est = eff_slope * kloc + eff_intercept
        effort_preds[str(idx)] = eff_est

        # Defects prediction
        def_est = def_slope * kloc + def_intercept
        def_low = def_est - def_sigma
        def_high = def_est + def_sigma
        def_preds[str(idx)] = {
            "estimate": def_est,
            "low": def_low,
            "high": def_high
        }

        # MAE calculation if actual known
        actual_eff = to_float(r["Effort-"])
        if actual_eff is not None:
            mae_sum += abs(eff_est - actual_eff)
            mae_count += 1

    mae_value = (mae_sum / mae_count) if mae_count else None

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------
    # NASA-03
    print("== NASA-03 ==")
    print("Effort predictions for held‑out rows:")
    # Print first 3 predictions (or all if fewer)
    for i, (row_no, pred) in enumerate(effort_preds.items()):
        if i >= 3:
            break
        print(f"Row {row_no}: {pred:.2f}")
    if mae_value is not None:
        print(f"MAE: {mae_value:.2f}")
    else:
        print("MAE: None")
    answer_03 = {
        "predictions": effort_preds,
        "mae": mae_value
    }
    print("ANSWER NASA-03", json.dumps(answer_03, separators=(',', ':')))

    # NASA-04
    print("\n== NASA-04 ==")
    print("Defects predictions for held‑out rows:")
    for i, (row_no, rec) in enumerate(def_preds.items()):
        if i >= 3:
            break
        print(f"Row {row_no}: estimate={rec['estimate']:.2f}, low={rec['low']:.2f}, high={rec['high']:.2f}")
    answer_04 = {
        "predictions": def_preds
    }
    print("ANSWER NASA-04", json.dumps(answer_04, separators=(',', ':')))

if __name__ == "__main__":
    main()
