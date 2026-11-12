#!/usr/bin/env python3
"""
Skill 09: NASA-17 and NASA-18 – Outlier detection and residual analysis
Author: Your Name
"""

import sys
import csv
import math
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Read CSV and convert numeric values where possible
# ----------------------------------------------------------------------
def read_csv(path):
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
# Identify numeric columns (excluding IDs, year, month)
# ----------------------------------------------------------------------
def numeric_columns(rows):
    exclude = {'idX', 'YearX', 'MonthsX'}
    numeric = []
    for key in rows[0].keys():
        if key in exclude:
            continue
        # Column is numeric if all non-None values are numeric
        if all(isinstance(r.get(key, None), (int, float)) or r.get(key, None) is None
               for r in rows):
            numeric.append(key)
    return numeric

# ----------------------------------------------------------------------
# Build feature matrix and target vector
# ----------------------------------------------------------------------
def build_matrix(rows, target_col, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    numeric = [c for c in rows[0].keys()
               if c not in exclude_cols and c != target_col
               and all(isinstance(r.get(c, None), (int, float)) or r.get(c, None) is None
                       for r in rows)]

    X_rows = []
    y_rows = []
    for r in rows:
        tgt = r.get(target_col)
        if not isinstance(tgt, (int, float)):
            continue
        feat_vals = []
        missing = False
        for c in numeric:
            val = r.get(c)
            if isinstance(val, (int, float)):
                feat_vals.append(val)
            else:
                missing = True
                break
        if missing:
            continue
        X_rows.append(feat_vals)
        y_rows.append(tgt)

    if not X_rows:
        raise ValueError(f"No valid rows for target '{target_col}'.")
    return np.array(X_rows), np.array(y_rows), numeric

# ----------------------------------------------------------------------
# NASA-17: Flag projects with unusual characteristic combinations
# ----------------------------------------------------------------------
def nasa_17(rows):
    numeric = numeric_columns(rows)
    if not numeric:
        print("== NASA-17 ==")
        print("No numeric columns to analyze for outliers.")
        print()
        return

    X, _, feat_names = build_matrix(rows, target_col='Effort-', exclude_cols=['idX', 'YearX', 'MonthsX'])
    # Standardize each column
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=1)
    stds[stds == 0] = 1.0  # avoid division by zero
    z_matrix = (X - means) / stds
    # Euclidean norm of z-scores as anomaly score
    norms = np.linalg.norm(z_matrix, axis=1)

    mean_norm = np.mean(norms)
    std_norm = np.std(norms, ddof=1) if len(norms) > 1 else 0.0
    threshold = mean_norm + 2 * std_norm

    # Collect projects exceeding threshold
    flagged = [rows[i] for i, n in enumerate(norms) if n > threshold]

    print("== NASA-17 ==")
    if not flagged:
        print("No projects with unusually different characteristic combinations were found.")
        print()
        return

    print(f"Threshold (mean + 2*std) for anomaly score: {threshold:.2f}")
    print("Flagged projects (ID, anomaly score, key features):")
    for r in flagged[:10]:  # limit output to 10 for readability
        fid = int(r.get('idX', 0))
        # Compute score again
        idx = rows.index(r)
        score = norms[idx]
        feature_vals = {name: r.get(name, None) for name in feat_names[:3]}
        feat_str = ", ".join(f"{k}={v:.2f}" for k, v in feature_vals.items())
        print(f"  ID={fid} | Score={score:.2f} | {feat_str}")
    print()

# ----------------------------------------------------------------------
# NASA-18: Flag projects with residuals in effort or defects
# ----------------------------------------------------------------------
def nasa_18(rows):
    numeric = numeric_columns(rows)
    if not numeric:
        print("== NASA-18 ==")
        print("No numeric columns to train models for residual analysis.")
        print()
        return

    # Train effort model
    X_eff, y_eff, eff_feats = build_matrix(rows, 'Effort-', exclude_cols=['idX', 'YearX', 'MonthsX'])
    model_eff = LinearRegression().fit(X_eff, y_eff)
    preds_eff = model_eff.predict(X_eff)
    resid_eff = y_eff - preds_eff
    std_res_eff = np.std(resid_eff, ddof=1) if len(resid_eff) > 1 else 0.0
    thresh_eff = 2 * std_res_eff

    # Train defects model
    X_def, y_def, def_feats = build_matrix(rows, 'Defects-', exclude_cols=['idX', 'YearX', 'MonthsX'])
    model_def = LinearRegression().fit(X_def, y_def)
    preds_def = model_def.predict(X_def)
    resid_def = y_def - preds_def
    std_res_def = np.std(resid_def, ddof=1) if len(resid_def) > 1 else 0.0
    thresh_def = 2 * std_res_def

    flagged = []
    for i, r in enumerate(rows):
        eff_res = resid_eff[i]
        def_res = resid_def[i]
        if abs(eff_res) > thresh_eff or abs(def_res) > thresh_def:
            flagged.append((r, eff_res, def_res))

    print("== NASA-18 ==")
    if not flagged:
        print("No projects with substantial residuals in effort or defect counts were found.")
        print()
        return

    print(f"Residual thresholds: Effort={thresh_eff:.2f}, Defects={thresh_def:.2f}")
    print("Flagged projects (ID, effort residual, defect residual):")
    for r, eff_res, def_res in flagged[:10]:
        fid = int(r.get('idX', 0))
        print(f"  ID={fid} | EffRes={eff_res:.2f} | DefRes={def_res:.2f}")
    print()

# ----------------------------------------------------------------------
# Main entry
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

    try:
        nasa_17(rows)
    except Exception as e:
        print(f"NASA-17 error: {e}", file=sys.stderr)

    try:
        nasa_18(rows)
    except Exception as e:
        print(f"NASA-18 error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
