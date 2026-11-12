#!/usr/bin/env python3
"""
Skill 06: NASA-11 and NASA-12 – Explain effort estimates and risk differences.
Author: Your Name
"""

import sys
import csv
import math
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Read CSV and convert values to numeric where possible
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Reads the CSV at *path* and returns:
        header: list of column names
        rows: list of dict mapping column -> value (numeric where possible)
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                v = v.strip() if v is not None else ''
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return header, rows

# ----------------------------------------------------------------------
# Identify numeric columns (excluding target columns and ids)
# ----------------------------------------------------------------------
def numeric_columns(rows, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    numeric = []
    for key in rows[0].keys():
        if key in exclude_cols:
            continue
        # A column is numeric if all non-empty values are numbers
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            numeric.append(key)
    return numeric

# ----------------------------------------------------------------------
# Build feature matrix X for a given target column
# ----------------------------------------------------------------------
def build_matrix(rows, target_col, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    numeric = numeric_columns(rows, exclude_cols + [target_col])
    X_rows = []
    y_rows = []
    for r in rows:
        tgt = r.get(target_col)
        if not isinstance(tgt, (int, float)):
            continue
        features = []
        missing = False
        for col in numeric:
            val = r.get(col)
            if isinstance(val, (int, float)):
                features.append(val)
            else:
                missing = True
                break
        if missing:
            continue
        X_rows.append(features)
        y_rows.append(tgt)
    if not X_rows:
        raise ValueError(f"No valid rows for target {target_col}")
    X = np.array(X_rows)
    y = np.array(y_rows)
    return X, y, numeric

# ----------------------------------------------------------------------
# Find nearest neighbors (Euclidean distance) for a given row
# ----------------------------------------------------------------------
def nearest_neighbors(target_row, rows, numeric, k=3):
    """
    Returns a list of (distance, row) tuples sorted ascending.
    """
    def dist(row):
        return math.sqrt(sum((target_row.get(col, 0) - row.get(col, 0)) ** 2 for col in numeric))
    dists = [(dist(row), row) for row in rows]
    dists.sort(key=lambda x: x[0])
    return dists[:k]

# ----------------------------------------------------------------------
# NASA-11: Explain effort estimate with similar historical projects
# ----------------------------------------------------------------------
def nasa_11(header, rows):
    """
    For each project, predict effort, then find three nearest historical
    projects (excluding itself).  Print an explanation for the first
    project in the dataset as an example.
    """
    target = 'Effort-'
    exclude = ['idX', 'YearX', 'MonthsX']
    X, y, feat_names = build_matrix(rows, target, exclude)
    model = LinearRegression()
    model.fit(X, y)

    # Pick the first row as the "proposed" project
    proj = rows[0]
    proj_features = [proj.get(c, 0) for c in feat_names]
    proj_X = np.array([proj_features])
    pred = model.predict(proj_X)[0]
    actual = proj.get(target, 0.0)

    # Compute nearest neighbors among all historical rows (including itself, but we'll skip it later)
    all_neighbors = nearest_neighbors(proj, rows, feat_names, k=4)  # get 4 to skip self
    neighbors = [(d, r) for d, r in all_neighbors if r.get('idX') != proj.get('idX')][:3]

    print("== NASA-11 ==")
    print(f"Project ID={int(proj.get('idX',0))} | Predicted Effort={pred:.2f} | Actual Effort={actual:.2f} | Residual={pred - actual:.2f}")
    print("Nearest historical projects (most similar features):")
    for i, (d, n) in enumerate(neighbors, start=1):
        feat_str = ", ".join(f"{c}={n.get(c,0):.1f}" for c in feat_names)
        print(f"  {i}. ID={int(n.get('idX',0))} | Effort={n.get(target,0):.2f} | {feat_str}")
    print()

# ----------------------------------------------------------------------
# NASA-12: Explain differences between two risk assessments
# ----------------------------------------------------------------------
def nasa_12(header, rows):
    """
    Identify one high‑risk and one low‑risk project (based on predicted effort)
    and explain the main differences in their attributes.
    """
    target = 'Effort-'
    exclude = ['idX', 'YearX', 'MonthsX']
    X, y, feat_names = build_matrix(rows, target, exclude)
    model = LinearRegression()
    model.fit(X, y)

    # Predict effort for all rows
    preds = model.predict(X)
    mean_pred = np.mean(preds)
    std_pred = np.std(preds, ddof=1) if len(preds) > 1 else 0.0

    # Risk flag: high if predicted > mean + 2*std
    high_idx = None
    low_idx = None
    for idx, (row, p) in enumerate(zip(rows, preds)):
        if high_idx is None and p > mean_pred + 2 * std_pred:
            high_idx = idx
        if low_idx is None and p <= mean_pred:
            low_idx = idx
        if high_idx is not None and low_idx is not None:
            break

    if high_idx is None or low_idx is None:
        print("== NASA-12 ==")
        print("Insufficient data to identify contrasting risk levels.\n")
        return

    high_row = rows[high_idx]
    low_row = rows[low_idx]
    high_pred = preds[high_idx]
    low_pred = preds[low_idx]
    high_actual = high_row.get(target, 0.0)
    low_actual = low_row.get(target, 0.0)

    # Compute absolute differences for numeric features
    diffs = []
    for col in feat_names:
        h = high_row.get(col, 0)
        l = low_row.get(col, 0)
        if isinstance(h, (int, float)) and isinstance(l, (int, float)):
            diffs.append((abs(h - l), col, h, l))
    diffs.sort(key=lambda x: x[0], reverse=True)

    print("== NASA-12 ==")
    print(f"High‑risk Project ID={int(high_row.get('idX',0))} | Predicted Effort={high_pred:.2f} | Actual Effort={high_actual:.2f}")
    print(f"Low‑risk Project ID={int(low_row.get('idX',0))} | Predicted Effort={low_pred:.2f} | Actual Effort={low_actual:.2f}")
    print("\nTop differing characteristics:")
    for i, (diff, col, hval, lval) in enumerate(diffs[:3], start=1):
        print(f"  {i}. {col}: high={hval:.1f}, low={lval:.1f}, diff={diff:.1f}")

    # Simple trade‑off statement
    print("\nTrade‑off: The high‑risk project has higher values in key attributes (e.g., size, complexity, etc.) leading to greater predicted effort and associated defect risk, while the low‑risk project is leaner and thus projected to be easier and less costly.")
    print()

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        header, rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    # Execute NASA‑11 and NASA‑12 analyses
    try:
        nasa_11(header, rows)
    except Exception as e:
        print(f"NASA-11 error: {e}", file=sys.stderr)

    try:
        nasa_12(header, rows)
    except Exception as e:
        print(f"NASA-12 error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
