#!/usr/bin/env python3
"""
skill_02.py

Adds two new requests to the Apache analysis toolkit:
  - APACHE-03: Predict Performance- for held‑out rows and compute MAE.
  - APACHE-04: Estimate the performance difference between pairs of
               historically observed configurations that differ in exactly
               one setting, with an uncertainty interval.

The code builds on the helper functions defined in skill_01.py.
"""

import sys
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# Import helper functions from the previous skill module
from skill_01 import read_csv, parse_column_values

def feature_matrix(rows, setting_cols):
    """
    Convert setting columns to a binary feature matrix.
    Missing values ('?') are treated as 0.
    """
    X = np.zeros((len(rows), len(setting_cols)), dtype=float)
    for i, r in enumerate(rows):
        for j, col in enumerate(setting_cols):
            val = r[col]
            X[i, j] = 1.0 if val == '1' else 0.0
    return X

def predict_and_mae(rows, setting_cols, perf_col, heldout_indices, X, y):
    """
    Fit a linear regression on training data (not held‑out),
    predict for all held‑out rows, and compute MAE on
    held‑out rows that have known target values.
    Returns a dict mapping 1‑based row number to predicted value,
    and the MAE.
    """
    # Training indices: not held‑out and target known
    train_mask = np.ones(len(rows), dtype=bool)
    train_mask[heldout_indices] = False
    train_mask &= ~np.isnan(y)

    if np.sum(train_mask) == 0:
        # No training data: fallback predictions as 0
        predictions = {i+1: 0.0 for i in heldout_indices}
        mae = 0.0
        return predictions, mae

    X_train = X[train_mask]
    y_train = y[train_mask]

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predict all held‑out rows
    predictions = {}
    for idx in heldout_indices:
        row_num = idx + 1
        pred = float(lr.predict(X[idx:idx+1])[0])
        predictions[row_num] = pred

    # Compute MAE over held‑out rows with known target
    mae_vals = []
    for idx in heldout_indices:
        if not np.isnan(y[idx]):
            mae_vals.append(abs(y[idx] - predictions[idx+1]))
    mae = float(np.mean(mae_vals)) if mae_vals else 0.0

    return predictions, mae

def estimate_pair_differences(rows, setting_cols, perf_col, X, y):
    """
    For every pair of rows that differ in exactly one setting,
    estimate the Performance- difference using a model trained
    on all rows with known target values. Returns a list of
    dictionaries with the required information.
    """
    # Train on all rows with known target
    mask = ~np.isnan(y)
    if np.sum(mask) == 0:
        return []

    lr = LinearRegression()
    lr.fit(X[mask], y[mask])
    preds = lr.predict(X)

    # Residual standard deviation for uncertainty
    res = y[mask] - preds[mask]
    res_std = float(np.std(res))

    # Build pairs
    pairs = []
    n = len(rows)
    # Precompute setting values for each row for quick diff
    setting_vals = [[rows[i][col] for col in setting_cols] for i in range(n)]

    for i in range(n):
        if np.isnan(y[i]):
            continue
        for j in range(i+1, n):
            if np.isnan(y[j]):
                continue
            # Count differences in settings
            diffs = [k for k, (a,b) in enumerate(zip(setting_vals[i], setting_vals[j]))
                     if a != b]
            if len(diffs) == 1:
                diff_col = setting_cols[diffs[0]]
                diff_est = float(preds[j] - preds[i])
                low = diff_est - 2 * res_std
                high = diff_est + 2 * res_std
                pairs.append({
                    "a": i+1,
                    "b": j+1,
                    "setting": diff_col,
                    "difference": diff_est,
                    "low": low,
                    "high": high
                })
                if len(pairs) >= 5:
                    return pairs
    return pairs

def main(csv_path):
    rows, headers = read_csv(csv_path)

    # Identify goal columns
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    perf_col = goal_cols[0]  # Assume the first goal column is Performance-

    # Identify setting columns (exclude goals and X columns)
    setting_cols = [h for h in headers if not h.endswith('+') and not h.endswith('-') and not h.endswith('X')]

    # Build feature matrix and target array
    X = feature_matrix(rows, setting_cols)
    y = parse_column_values(rows, perf_col)

    # Determine held‑out indices (1‑based divisible by 5)
    heldout_indices = [i for i in range(len(rows)) if (i+1) % 5 == 0]

    # ---------- APACHE-03 ----------
    print("== APACHE-03 ==")
    preds, mae = predict_and_mae(rows, setting_cols, perf_col, heldout_indices, X, y)
    # Print a few example predictions
    for row_num in sorted(preds)[:3]:
        print(f"Row {row_num} predicted Performance- = {preds[row_num]:.4f}")
    print(f"MAE on held‑out rows with known values: {mae:.4f}")
    print("ANSWER APACHE-03", json.dumps({"predictions": preds, "mae": mae}, separators=(',', ':')))

    # ---------- APACHE-04 ----------
    print("\n== APACHE-04 ==")
    pairs = estimate_pair_differences(rows, setting_cols, perf_col, X, y)
    if not pairs:
        print("No suitable pairs found.")
    else:
        for p in pairs[:3]:
            print(f"Rows {p['a']} & {p['b']} differ in {p['setting']}: "
                  f"Δ≈{p['difference']:.2f} (±{2*(p['high']-p['low'])/2:.2f})")
    print("ANSWER APACHE-04", json.dumps({"pairs": pairs}, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
