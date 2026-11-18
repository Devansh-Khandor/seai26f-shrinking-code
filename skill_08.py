#!/usr/bin/env python3
"""
skill_08.py

Usage: python skill_08.py <csv path>
"""

import csv
import json
import math
import sys
from collections import defaultdict

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def parse_value(v: str):
    """Parse a CSV cell: float if possible, else keep as string."""
    if v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return v

def round3(x):
    if isinstance(x, (int, float)):
        return round(x, 3)
    return x

# -------------------------------------------------------------
# Data loading
# -------------------------------------------------------------
def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = []
        for r in rdr:
            rows.append({h: parse_value(v) for h, v in zip(header, r)})
    return header, rows

# -------------------------------------------------------------
# Regression helpers (multiple linear regression using correlations)
# -------------------------------------------------------------
def compute_regression(rows, target_col, predictor_cols):
    """
    Simple linear regression using correlation coefficients.
    Returns intercept, betas dict, means dict.
    """
    # Filter rows with target and all predictors not None
    data = []
    for r in rows:
        t = r[target_col]
        if t is None:
            continue
        good = True
        for pc in predictor_cols:
            if r[pc] is None:
                good = False
                break
        if good:
            data.append(r)

    if not data:
        return 0.0, {}, {}

    # compute mean and std of target
    y_vals = [r[target_col] for r in data]
    n = len(y_vals)
    y_mean = sum(y_vals) / n
    y_var = sum((y - y_mean) ** 2 for y in y_vals) / (n - 1 if n > 1 else 1)
    y_std = math.sqrt(y_var) if y_var > 0 else 0.0

    betas = {}
    means = {}
    # compute for each predictor
    for pc in predictor_cols:
        x_vals = [r[pc] for r in data]
        x_mean = sum(x_vals) / n
        x_var = sum((x - x_mean) ** 2 for x in x_vals) / (n - 1 if n > 1 else 1)
        x_std = math.sqrt(x_var) if x_var > 0 else 0.0
        if x_std == 0 or y_std == 0:
            continue
        # covariance
        cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)) / (n - 1)
        corr = cov / (x_std * y_std)
        beta = corr * (y_std / x_std)
        betas[pc] = beta
        means[pc] = x_mean

    # intercept
    intercept = y_mean
    for pc, beta in betas.items():
        intercept -= beta * means[pc]

    return intercept, betas, means

def predict(row, intercept, betas, means):
    y = intercept
    for pc, beta in betas.items():
        x = row.get(pc)
        if x is None:
            continue
        y += beta * (x - means.get(pc, 0))
    return y

# -------------------------------------------------------------
# Change finder
# -------------------------------------------------------------
def best_change(row, intercept, betas, means, adjust_cols, unique_vals, target_pred_fn):
    """Return best single change to reduce prediction."""
    cur_pred = target_pred_fn(row)
    best = None
    best_reduce = 0.0
    best_col = None
    best_val = None
    for col in adjust_cols:
        cur = row.get(col)
        for v in unique_vals[col]:
            if v == cur:
                continue
            new_row = row.copy()
            new_row[col] = v
            new_pred = target_pred_fn(new_row)
            reduce = cur_pred - new_pred
            if reduce > best_reduce:
                best_reduce = reduce
                best_col = col
                best_val = v
    if best_col is None:
        return None
    return best_col, best_val, best_reduce

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main(csv_path):
    header, rows = load_csv(csv_path)

    # Determine column types
    goal_cols = [c for c in header if c.endswith(('+', '-'))]
    x_cols = [c for c in header if c.endswith('X')]
    adjust_cols = [c for c in header if c not in goal_cols + x_cols and c != 'Kloc+']

    # Unique values for each column
    unique_vals = {c: set() for c in adjust_cols}
    for r in rows:
        for c in adjust_cols:
            v = r.get(c)
            if v is not None:
                unique_vals[c].add(v)
    # Convert to list
    unique_vals = {c: list(vals) for c, vals in unique_vals.items()}

    # Predictor columns for regression (numeric columns except goals and X)
    predictor_cols = [c for c in header
                      if c not in goal_cols + x_cols
                      and c != 'Kloc+'
                      and isinstance(next((r[c] for r in rows if r[c] is not None), None), (int, float))]

    # Build models
    eff_intercept, eff_betas, eff_means = compute_regression(rows, 'Effort-', predictor_cols)
    defect_intercept, defect_betas, defect_means = compute_regression(rows, 'Defects-', predictor_cols)

    # Predict all rows
    for r in rows:
        r['pred_effort'] = predict(r, eff_intercept, eff_betas, eff_means)
        r['pred_defect'] = predict(r, defect_intercept, defect_betas, defect_means)

    # ---------- NASA-15 ----------
    # Project with maximum predicted effort
    max_eff_row = max(rows, key=lambda r: r['pred_effort'])
    max_eff_index = rows.index(max_eff_row) + 1
    cur_eff = max_eff_row['pred_effort']

    # Find up to 3 changes
    changes = {}
    temp_row = max_eff_row.copy()
    for _ in range(3):
        result = best_change(temp_row, eff_intercept, eff_betas, eff_means,
                             adjust_cols, unique_vals, lambda r: predict(r, eff_intercept, eff_betas, eff_means))
        if result is None:
            break
        col, new_val, _ = result
        changes[col] = [temp_row[col], new_val]
        temp_row[col] = new_val

    # After changes
    after_eff = predict(temp_row, eff_intercept, eff_betas, eff_means)

    # ---------- NASA-16 ----------
    # Project with maximum predicted defect
    max_def_row = max(rows, key=lambda r: r['pred_defect'])
    max_def_index = rows.index(max_def_row) + 1
    cur_def = max_def_row['pred_defect']
    cur_eff_def = max_def_row['pred_effort']

    # Find up to 3 changes to reduce defect
    changes_def = {}
    temp_def_row = max_def_row.copy()
    for _ in range(3):
        result = best_change(temp_def_row, defect_intercept, defect_betas, defect_means,
                             adjust_cols, unique_vals, lambda r: predict(r, defect_intercept, defect_betas, defect_means))
        if result is None:
            break
        col, new_val, _ = result
        changes_def[col] = [temp_def_row[col], new_val]
        temp_def_row[col] = new_val

    after_def = predict(temp_def_row, defect_intercept, defect_betas, defect_means)
    after_eff_def = predict(temp_def_row, eff_intercept, eff_betas, eff_means)

    # ---------- Output ----------
    # NASA-15
    print("== NASA-15 ==")
    print(f"Row {max_eff_index}: Predicted Effort before={round3(cur_eff)} after={round3(after_eff)}")
    for c, (old, new) in changes.items():
        print(f"{c}: {old} -> {new}")
    print(f"ANSWER NASA-15 {json.dumps({'project': max_eff_index, 'changes': changes, 'predicted_effort': {'before': round3(cur_eff), 'after': round3(after_eff)}})}")

    # NASA-16
    print("\n== NASA-16 ==")
    print(f"Row {max_def_index}: Predicted Defects before={round3(cur_def)} after={round3(after_def)}")
    print(f"Predicted Effort before={round3(cur_eff_def)} after={round3(after_eff_def)}")
    for c, (old, new) in changes_def.items():
        print(f"{c}: {old} -> {new}")
    print(f"ANSWER NASA-16 {json.dumps({'project': max_def_index, 'changes': changes_def, 'predicted_defects': {'before': round3(cur_def), 'after': round3(after_def)}, 'predicted_effort': {'before': round3(cur_eff_def), 'after': round3(after_eff_def)}})}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])
