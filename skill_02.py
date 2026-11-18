#!/usr/bin/env python3
"""
skill_02.py

Usage: python skill_02.py <csv path>
"""

import csv
import json
import math
import sys
from collections import defaultdict

# ----------------------- helpers --------------------------------

def parse_float(v: str):
    """Parse a CSV cell into a float or None."""
    if v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def stddev(xs):
    """Population standard deviation."""
    if not xs:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

# ----------------------- regression --------------------------------

def simple_linear_regression(xys):
    """
    Perform a simple linear regression y = a * x + b
    xys: list of (x, y) tuples where both are not None
    Returns a, b
    """
    xs = [x for x, _ in xys]
    ys = [y for _, y in xys]
    if not xs:
        return 0.0, 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in xys)
    den = sum((x - mx) ** 2 for x in xs)
    a = num / den if den != 0 else 0.0
    b = my - a * mx
    return a, b

def predict(a, b, x):
    return a * x + b

def residuals(a, b, xys):
    return [y - (a * x + b) for x, y in xys]

# ----------------------- main --------------------------------

def main(csv_path):
    # Read the CSV
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = []
        for r in rdr:
            rows.append(r)

    # Identify column indices
    col_index = {name: idx for idx, name in enumerate(header)}

    # Determine held‑out rows (1‑based index divisible by 5)
    held_out_indices = {i for i in range(len(rows)) if (i + 1) % 5 == 0}
    training_rows = [rows[i] for i in range(len(rows)) if i not in held_out_indices]

    # Parse numeric columns we will use
    numeric_cols = ["Kloc+", "Effort-", "Defects-", "YearX", "MonthsX"]
    numeric_idx = {c: col_index[c] for c in numeric_cols}

    # Gather data for regression
    eff_xys = []
    defect_xys = []
    kloc_list = []

    for r in training_rows:
        kloc = parse_float(r[numeric_idx["Kloc+"]])
        if kloc is None:
            continue
        kloc_list.append(kloc)
        effort = parse_float(r[numeric_idx["Effort-"]])
        if effort is not None:
            eff_xys.append((kloc, effort))
        defect = parse_float(r[numeric_idx["Defects-"]])
        if defect is not None:
            defect_xys.append((kloc, defect))

    # Compute regression models
    eff_a, eff_b = simple_linear_regression(eff_xys)
    def_a, def_b = simple_linear_regression(defect_xys)

    # Residuals for uncertainty
    def_resid = residuals(def_a, def_b, defect_xys)
    def_sd = stddev(def_resid)

    # Mean Kloc for missing cases
    mean_kloc = mean(kloc_list) if kloc_list else 0.0

    # ---------------- NASA-03 --------------------------------

    nasa03_preds = {}
    mae_sum = 0.0
    mae_count = 0

    print("== NASA-03 ==")
    for idx in sorted(held_out_indices):
        r = rows[idx]
        row_num = str(idx + 1)
        kloc = parse_float(r[numeric_idx["Kloc+"]])
        eff_actual = parse_float(r[numeric_idx["Effort-"]])
        if kloc is None:
            kloc = mean_kloc
        eff_pred = predict(eff_a, eff_b, kloc)
        nasa03_preds[row_num] = round(eff_pred, 3)
        print(f"row {row_num}: Effort- predicted={eff_pred:.3f}")
        if eff_actual is not None:
            mae_sum += abs(eff_actual - eff_pred)
            mae_count += 1
    mae = mae_sum / mae_count if mae_count else 0.0
    print(f"MAE={mae:.3f}\n")

    print(f"ANSWER NASA-03 {json.dumps({'predictions': nasa03_preds, 'mae': round(mae, 3)})}\n")

    # ---------------- NASA-04 --------------------------------

    nasa04_preds = {}
    print("== NASA-04 ==")
    for idx in sorted(held_out_indices):
        r = rows[idx]
        row_num = str(idx + 1)
        kloc = parse_float(r[numeric_idx["Kloc+"]])
        if kloc is None:
            kloc = mean_kloc
        estimate = predict(def_a, def_b, kloc)
        low = estimate - 1.96 * def_sd
        high = estimate + 1.96 * def_sd
        low = max(0.0, low)
        high = max(0.0, high)
        nasa04_preds[row_num] = {
            "estimate": round(estimate, 3),
            "low": round(low, 3),
            "high": round(high, 3)
        }
        print(f"row {row_num}: estimate={estimate:.3f} low={low:.3f} high={high:.3f}")
    print()

    print(f"ANSWER NASA-04 {json.dumps({'predictions': nasa04_preds})}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])
