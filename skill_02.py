#!/usr/bin/env python3
"""
Apache performance prediction and pairwise comparison.

Usage:  python skill_02.py <csv_path>
"""

import csv
import json
import sys
import math
from collections import defaultdict

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """Read CSV and return list of row dicts and fieldnames."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ---------------------------------------------------------------------------

def to_float(val):
    """Convert string to float or np.nan."""
    if val == '?' or val == '':
        return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan

# ---------------------------------------------------------------------------

def find_goal_column(fieldnames):
    """Return the column that ends with '+' or '-'."""
    for c in fieldnames:
        if c.endswith('+') or c.endswith('-'):
            return c
    return None

# ---------------------------------------------------------------------------

def build_feature_matrix(rows, feature_cols):
    """Return numpy array of shape (n_rows, n_features) with imputed NaNs."""
    data = []
    for row in rows:
        row_vals = [to_float(row[col]) for col in feature_cols]
        data.append(row_vals)
    X = np.array(data, dtype=float)
    # Impute missing values with mean
    imp = SimpleImputer(strategy='mean')
    X_imputed = imp.fit_transform(X)
    return X_imputed

# ---------------------------------------------------------------------------

def compute_mae(y_true, y_pred):
    """Compute mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))

# ---------------------------------------------------------------------------

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    goal_col = find_goal_column(fieldnames)
    if goal_col is None:
        print("No goal column found", file=sys.stderr)
        sys.exit(1)

    # Determine setting columns: all except goal and any ending with X
    setting_cols = [c for c in fieldnames if c != goal_col and not c.endswith('X')]
    # Feature columns for model: all setting columns
    feature_cols = setting_cols

    # Assign row numbers (1-based)
    row_numbers = list(range(1, len(rows)+1))

    # Identify held-out rows: divisible by 5
    held_out_indices = [i for i, rn in enumerate(row_numbers) if rn % 5 == 0]

    # Build training data (exclude held-out and missing goal)
    train_rows = []
    train_indices = []
    for idx, row in enumerate(rows):
        if idx in held_out_indices:
            continue
        goal_val = to_float(row[goal_col])
        if math.isnan(goal_val):
            continue
        train_rows.append(row)
        train_indices.append(idx)

    # Build X and y for training
    X_train = build_feature_matrix(train_rows, feature_cols)
    y_train = np.array([to_float(r[goal_col]) for r in train_rows], dtype=float)

    # Fit simple linear regression
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Predict on held-out rows
    X_hold = build_feature_matrix([rows[i] for i in held_out_indices], feature_cols)
    y_pred_hold = reg.predict(X_hold)

    # Actual goal values for held-out
    y_true_hold = np.array([to_float(rows[i][goal_col]) for i in held_out_indices], dtype=float)
    # Filter out any missing actual values (shouldn't happen but just in case)
    mask = ~np.isnan(y_true_hold)
    y_pred_hold = y_pred_hold[mask]
    y_true_hold = y_true_hold[mask]
    held_out_nums = [row_numbers[i] for i, m in zip(held_out_indices, mask) if m]

    mae = compute_mae(y_true_hold, y_pred_hold)

    # Print APACHE-03 section
    print("== APACHE-03 ==")
    for num, pred in zip(held_out_nums, y_pred_hold):
        print(f"Row {num}: predicted {pred:.4f}")
    print(f"MAE: {mae:.4f}")

    # Prepare JSON answer
    predictions_json = {str(num): float(pred) for num, pred in zip(held_out_nums, y_pred_hold)}
    answer_03 = {"predictions": predictions_json, "mae": mae}
    print(f"ANSWER APACHE-03 {json.dumps(answer_03, separators=(',', ':'))}")

    # -----------------------------------------------------------------------
    # APACHE-04: find pairs that differ by one setting

    # Precompute features for all rows (including those with missing goal)
    X_all = build_feature_matrix(rows, feature_cols)
    # Train on all rows with non-missing goal
    all_train_rows = [r for r in rows if not math.isnan(to_float(r[goal_col]))]
    X_all_train = build_feature_matrix(all_train_rows, feature_cols)
    y_all_train = np.array([to_float(r[goal_col]) for r in all_train_rows], dtype=float)
    reg_all = LinearRegression()
    reg_all.fit(X_all_train, y_all_train)
    # Predict on all rows (will be close to actual for training rows)
    y_all_pred = reg_all.predict(X_all)

    # Residual std dev for uncertainty
    residuals = y_all_train - reg_all.predict(X_all_train)
    sigma = float(np.std(residuals, ddof=1))

    pairs = []
    n = len(rows)
    # Build list of setting values for quick access
    setting_values = {col: [r[col] for r in rows] for col in setting_cols}

    for i in range(n):
        for j in range(i+1, n):
            # both rows must have non-missing goal
            gi = to_float(rows[i][goal_col])
            gj = to_float(rows[j][goal_col])
            if math.isnan(gi) or math.isnan(gj):
                continue
            diff_cols = []
            for col in setting_cols:
                vi = rows[i][col]
                vj = rows[j][col]
                if vi == '?' or vj == '?':
                    continue  # skip if missing in any
                if vi != vj:
                    diff_cols.append(col)
            if len(diff_cols) == 1:
                setting = diff_cols[0]
                diff_est = y_all_pred[j] - y_all_pred[i]
                low = diff_est - sigma
                high = diff_est + sigma
                pairs.append({
                    "a": row_numbers[i],
                    "b": row_numbers[j],
                    "setting": setting,
                    "difference": diff_est,
                    "low": low,
                    "high": high
                })
                if len(pairs) >= 50:  # gather some pairs
                    break
        if len(pairs) >= 50:
            break

    # Take first 5 pairs
    selected_pairs = pairs[:5]

    # Print APACHE-04 section
    print("== APACHE-04 ==")
    for p in selected_pairs:
        print(f"Pair {p['a']} vs {p['b']}: setting {p['setting']} diff {p['difference']:.4f} (±{sigma:.4f})")
    answer_04 = {"pairs": selected_pairs}
    print(f"ANSWER APACHE-04 {json.dumps(answer_04, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
