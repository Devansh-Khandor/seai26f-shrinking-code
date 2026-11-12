#!/usr/bin/env python3
"""
skill_09.py

Usage:
    python skill_09.py <path-to-auto93.csv>

This script implements two new features for the dealership:

  CAR-17 : Flag vehicle records whose combinations of characteristics are unusually
           different from the rest of our inventory so that staff can review them.
  CAR-18 : Identify vehicles whose reported fuel efficiency or weight differs
           substantially from what similar vehicles would lead us to expect,
           so we can check for possible recording mistakes.

The output follows the requested format exactly: for each request, a line that is
exactly `== <ID> ==` followed by at least three lines of results, including
the actual numbers.
"""

import csv
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------

def read_data(csv_path):
    """
    Reads the CSV file and returns a list of records.
    Each record is a dict with numeric fields and a 'row' key indicating the original line number.
    Rows that cannot be parsed are skipped.
    """
    records = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader, start=1):
            if not row or all(cell.strip() == '' for cell in row):
                continue
            try:
                rec = {
                    'row': idx,
                    'Clndrs': float(row[0]),
                    'Volume': float(row[1]),
                    'HpX': float(row[2]),
                    'Model': float(row[3]),
                    'origin': float(row[4]),
                    'Lbs-': float(row[5]),
                    'Acc+': float(row[6]),
                    'Mpg+': float(row[7]),
                }
            except (IndexError, ValueError):
                continue
            records.append(rec)
    return records

# ---------------------------------------------------------------------------

def compute_combined_score(records, features):
    """
    For each record, compute a combined anomaly score based on the sum of the
    absolute z‑scores of the specified features.
    """
    data = np.array([[rec[f] for f in features] for rec in records], dtype=float)
    mean = data.mean(axis=0)
    std  = data.std(axis=0, ddof=1)
    # Avoid division by zero for constant columns
    std_adj = np.where(std == 0, 1, std)
    z = (data - mean) / std_adj
    scores = np.sum(np.abs(z), axis=1)
    return scores

# ---------------------------------------------------------------------------

def car_17(records):
    """
    Flag records that are far from the bulk of the distribution
    based on all numeric characteristics.
    """
    feature_names = ['Clndrs', 'Volume', 'HpX', 'Model',
                     'origin', 'Lbs-', 'Acc+', 'Mpg+']
    scores = compute_combined_score(records, feature_names)
    mean_score = np.mean(scores)
    std_score  = np.std(scores, ddof=1)
    threshold  = mean_score + 2 * std_score

    flagged = [(rec, score) for rec, score in zip(records, scores) if score > threshold]
    return flagged, threshold

# ---------------------------------------------------------------------------

def build_regressor(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    return reg

# ---------------------------------------------------------------------------

def car_18(records):
    """
    Identify records whose weight or mpg differs substantially from what a
    simple linear model would predict given the other features.
    Returns a list of tuples:
        (record, weight_pred, weight_residual, mpg_pred, mpg_residual)
    """
    # Prepare data
    X_w = np.column_stack([rec['Clndrs'], rec['Volume'], rec['HpX'],
                           rec['Model'], rec['origin'], rec['Mpg+']
                           for rec in records])
    y_w = np.array([rec['Lbs-'] for rec in records], dtype=float)

    X_m = np.column_stack([rec['Clndrs'], rec['Volume'], rec['HpX'],
                           rec['Model'], rec['origin'], rec['Lbs-']
                           for rec in records])
    y_m = np.array([rec['Mpg+'] for rec in records], dtype=float)

    # Fit regressors
    reg_w = build_regressor(X_w, y_w)
    reg_m = build_regressor(X_m, y_m)

    # Predictions
    w_pred = reg_w.predict(X_w)
    m_pred = reg_m.predict(X_m)

    # Residuals
    w_resid = y_w - w_pred
    m_resid = y_m - m_pred

    # Standard deviations of residuals
    w_std = np.sqrt(mean_squared_error(y_w, w_pred))
    m_std = np.sqrt(mean_squared_error(y_m, m_pred))

    threshold_w = 2 * w_std
    threshold_m = 2 * m_std

    anomalies = []
    for rec, w_p, w_r, m_p, m_r in zip(records, w_pred, w_resid, m_pred, m_resid):
        flag = False
        if abs(w_r) > threshold_w:
            flag = True
        if abs(m_r) > threshold_m:
            flag = True
        if flag:
            anomalies.append((rec, w_p, w_r, m_p, m_r))

    return anomalies, threshold_w, threshold_m

# ---------------------------------------------------------------------------

def print_car_17(flagged, threshold):
    print("== CAR-17 ==")
    if not flagged:
        print("  No anomalous vehicles detected.")
        return
    # Show top 5 most anomalous
    flagged_sorted = sorted(flagged, key=lambda x: x[1], reverse=True)
    for i, (rec, score) in enumerate(flagged_sorted[:5], start=1):
        print(f"  {i}) Row {rec['row']}: score={score:.3f}")
    print(f"  Total flagged: {len(flagged)}")

def print_car_18(anomalies, thr_w, thr_m):
    print("\n== CAR-18 ==")
    if not anomalies:
        print("  No significant discrepancies detected.")
        return
    for i, (rec, w_p, w_r, m_p, m_r) in enumerate(anomalies[:5], start=1):
        parts = []
        if abs(w_r) > thr_w:
            parts.append(f"Weight anomaly (pred={w_p:.1f}, actual={rec['Lbs-']:.1f}, resid={w_r:.1f})")
        if abs(m_r) > thr_m:
            parts.append(f"MPG anomaly (pred={m_p:.1f}, actual={rec['Mpg+']:.1f}, resid={m_r:.1f})")
        print(f"  {i}) Row {rec['row']}: " + "; ".join(parts))
    print(f"  Total anomalies: {len(anomalies)}")

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = read_data(csv_path)

    if not records:
        print("No valid data found in the CSV.", file=sys.stderr)
        sys.exit(1)

    # CAR-17
    flagged, thr = car_17(records)
    print_car_17(flagged, thr)

    # CAR-18
    anomalies, thr_w, thr_m = car_18(records)
    print_car_18(anomalies, thr_w, thr_m)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
