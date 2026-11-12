#!/usr/bin/env python3
"""
Skill 05: Blame analysis for NASA data.

NASA-09: Identify which project characteristics are most strongly associated with unusually high development effort.
NASA-10: Identify which project characteristics are most strongly associated with unusually high defect counts.

Usage:
    python skill_05.py <path_to_nasa93dem.csv>
"""

import csv
import sys
import math
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """Read the NASA CSV and return a list of rows (dicts) with stripped values."""
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                cleaned[k] = v.strip()
            rows.append(cleaned)
    return rows

# ---------------------------------------------------------------------------

def detect_column_types(rows):
    """
    Detect numeric vs categorical columns.
    Returns:
        numeric_cols: list of column names detected as numeric
        categorical_cols: list of column names detected as categorical
    """
    if not rows:
        return [], []

    all_cols = set(rows[0].keys())
    # remove id if present
    all_cols.discard('idX')
    numeric_cols = []
    categorical_cols = []

    for col in all_cols:
        numeric = 0
        total = 0
        for r in rows:
            val = r.get(col, "")
            if val == "":
                continue
            total += 1
            try:
                float(val)
                numeric += 1
            except ValueError:
                pass
        if total == 0:
            # no data – treat as categorical
            categorical_cols.append(col)
            continue
        if numeric / total >= 0.8:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols

# ---------------------------------------------------------------------------

def encode_categorical(rows, categorical_cols):
    """
    Map each categorical column to integer codes.
    Returns a dict: col -> list of encoded ints
    """
    encodings = {}
    for col in categorical_cols:
        mapping = {}
        encoded = []
        code = 0
        for r in rows:
            val = r.get(col, "")
            if val == "":
                # unknown category – assign a special code
                val_key = "__missing__"
            else:
                val_key = val
            if val_key not in mapping:
                mapping[val_key] = code
                code += 1
            encoded.append(mapping[val_key])
        encodings[col] = encoded
    return encodings

# ---------------------------------------------------------------------------

def build_feature_matrix(rows, numeric_cols, categorical_encodings):
    """
    Build X matrix (rows x features) and feature names list.
    Numeric values are cast to float; categorical values are integers.
    """
    n = len(rows)
    # Build columns lists
    cols = numeric_cols + list(categorical_encodings.keys())
    X = np.zeros((n, len(cols)), dtype=float)
    for i, col in enumerate(cols):
        if col in numeric_cols:
            # numeric
            for j, r in enumerate(rows):
                val = r.get(col, "")
                X[j, i] = float(val) if val != "" else np.nan
        else:
            # categorical encoded
            X[:, i] = categorical_encodings[col]
    # Remove rows with NaN in any column
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    return X, [c for c in cols]

# ---------------------------------------------------------------------------

def extract_target(rows, target_col):
    """Return list of floats for target column, excluding rows where value missing."""
    values = []
    for r in rows:
        val = r.get(target_col, "")
        if val == "":
            continue
        try:
            values.append(float(val))
        except ValueError:
            continue
    return np.array(values, dtype=float)

# ---------------------------------------------------------------------------

def top_correlations(X, feature_names, y, top_n=3):
    """
    Compute Pearson correlation between each feature column and target y.
    Return list of tuples (feature_name, corr_coeff) sorted by absolute correlation desc.
    """
    if X.shape[0] < 2:
        return []

    # Compute correlation matrix
    corr_mat = np.corrcoef(X.T, y)
    # corr_mat last row/col are correlations with y
    corr_with_y = corr_mat[-1, :-1]  # exclude self correlation
    abs_corr = np.abs(corr_with_y)
    idx = np.argsort(-abs_corr)  # descending
    results = []
    for i in idx[:top_n]:
        results.append((feature_names[i], corr_with_y[i]))
    return results

# ---------------------------------------------------------------------------

def format_corr(name, coeff):
    """Return nicely formatted string for a feature correlation."""
    return f"  {name:<15}  Corr: {coeff:+.4f}"

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)

    # Detect columns
    numeric_cols, categorical_cols = detect_column_types(rows)

    # Encode categorical columns
    categorical_encodings = encode_categorical(rows, categorical_cols)

    # Build feature matrix
    X, feature_names = build_feature_matrix(rows, numeric_cols, categorical_encodings)

    # Extract target vectors
    effort = extract_target(rows, "Effort-")
    defects = extract_target(rows, "Defects-")

    # Align X with target lengths
    # Some rows may have been dropped due to NaNs in features
    # We need to keep only rows where target also exists
    # Build a mask for effort
    effort_mask = np.isfinite(effort)
    if effort_mask.sum() > 0:
        X_effort = X[effort_mask]
        y_effort = effort[effort_mask]
    else:
        X_effort = np.array([])
        y_effort = np.array([])

    # Align for defects
    defects_mask = np.isfinite(defects)
    if defects_mask.sum() > 0:
        X_defects = X[defects_mask]
        y_defects = defects[defects_mask]
    else:
        X_defects = np.array([])
        y_defects = np.array([])

    # NASA-09
    print("== NASA-09 ==")
    if X_effort.size == 0:
        print("  No valid data for effort analysis.")
    else:
        top_effort = top_correlations(X_effort, feature_names, y_effort, top_n=3)
        for name, coeff in top_effort:
            print(format_corr(name, coeff))

    # NASA-10
    print("\n== NASA-10 ==")
    if X_defects.size == 0:
        print("  No valid data for defect analysis.")
    else:
        top_defects = top_correlations(X_defects, feature_names, y_defects, top_n=3)
        for name, coeff in top_defects:
            print(format_corr(name, coeff))

if __name__ == "__main__":
    main()
