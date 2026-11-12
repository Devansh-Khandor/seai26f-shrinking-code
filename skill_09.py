#!/usr/bin/env python3
"""
skill_09.py

Implements the NASA-17 and NASA-18 requests:

NASA-17: Identify historical projects whose combinations of characteristics
         are unusually different from the rest of the records.

NASA-18: Identify project records whose reported effort or defect counts
         differ substantially from what comparable projects would predict.

The script re‑uses helpers from earlier skills and the
training routine from skill_04.  It prints a header line
for each request followed by at least three lines of results.
"""

import sys
import numpy as np
from sklearn.ensemble import IsolationForest

# Helpers from earlier skills
from skill_01 import read_csv, extract_numeric_columns, compute_stats
from skill_04 import train_and_predict


def encode_features(rows):
    """
    Convert the list of dict rows into a numeric feature matrix suitable
    for outlier detection.

    Numeric columns are kept as floats (NaNs replaced by column means).
    Categorical columns are encoded with integer codes (consistent across all rows).
    """
    # Identify numeric vs categorical columns
    all_keys = rows[0].keys()
    numeric_cols = []
    categorical_cols = []

    for key in all_keys:
        # Try to parse the first non-empty value
        for val in (row[key] for row in rows):
            if val is None or val == '':
                continue
            try:
                float(val)
                numeric_cols.append(key)
                break
            except ValueError:
                categorical_cols.append(key)
                break

    n_rows = len(rows)
    n_numeric = len(numeric_cols)
    n_cat = len(categorical_cols)
    X = np.empty((n_rows, n_numeric + n_cat), dtype=float)

    # Numeric values
    for j, key in enumerate(numeric_cols):
        col_vals = np.array([row[key] for row in rows], dtype=object)
        # Convert to float, set errors to NaN
        col_num = np.array([float(v) if v not in (None, '') else np.nan
                            for v in col_vals], dtype=float)
        # Impute missing with column mean
        mean_val = np.nanmean(col_num)
        col_num[np.isnan(col_num)] = mean_val
        X[:, j] = col_num

    # Categorical values
    cat_mappings = {}
    for j, key in enumerate(categorical_cols, start=n_numeric):
        col_vals = np.array([row[key] for row in rows], dtype=object)
        unique_vals = sorted(set(v for v in col_vals if v not in (None, '')))
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        cat_mappings[key] = mapping
        # Encode
        encoded = np.array([mapping.get(v, 0) for v in col_vals], dtype=float)
        X[:, j] = encoded

    return X, [row['idX'] for row in rows], numeric_cols, categorical_cols, cat_mappings


def nasa_17(rows):
    """
    Flag projects whose overall characteristic combination is an outlier
    according to an IsolationForest model.
    """
    X, ids, _, _, _ = encode_features(rows)

    # Train Isolation Forest; contamination can be set to 5%
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    scores = iso.decision_function(X)  # higher scores = less abnormal
    predictions = iso.predict(X)      # 1 = normal, -1 = outlier

    # Get indices of outliers sorted by lowest score (most abnormal)
    outlier_idx = np.where(predictions == -1)[0]
    outlier_scores = scores[outlier_idx]
    sorted_indices = outlier_idx[np.argsort(outlier_scores)]  # ascending

    # Prepare output lines
    lines = []
    for idx in sorted_indices[:5]:   # report up to 5 outliers
        lines.append(
            f"  ID={ids[idx]}: Anomaly score={scores[idx]:.3f}"
        )
    if not lines:
        lines.append("  No outlier projects detected.")
    return lines


def nasa_18(rows, reg_effort, reg_defect):
    """
    Flag projects whose reported effort or defects deviate significantly
    from predictions based on their size.
    """
    # Extract numeric columns
    size = extract_numeric_columns(rows, ["Kloc+"])
    effort = extract_numeric_columns(rows, ["Effort-"])
    defects = extract_numeric_columns(rows, ["Defects-"])
    ids = np.array([int(r["idX"]) for r in rows], dtype=int)

    # Mask for records with complete effort and defect data
    mask = ~np.isnan(size) & ~np.isnan(effort) & ~np.isnan(defects)
    if not np.any(mask):
        return ["  No records with complete effort and defect data."]

    size_h = size[mask].reshape(-1, 1)
    effort_h = effort[mask]
    defect_h = defects[mask]

    # Predict using the models
    effort_pred = reg_effort.predict(size_h).flatten()
    defect_pred = reg_defect.predict(size_h).flatten()

    # Residuals
    effort_resid = effort_h - effort_pred
    defect_resid = defect_h - defect_pred

    # Standard deviations of residuals
    eff_std = np.sqrt(np.mean(effort_resid ** 2))
    def_std = np.sqrt(np.mean(defect_resid ** 2))

    # Thresholds (2*std)
    eff_thresh = 2 * eff_std
    def_thresh = 2 * def_std

    # Identify significant deviations
    eff_bad = np.abs(effort_resid) > eff_thresh
    def_bad = np.abs(defect_resid) > def_thresh

    flagged = np.where(eff_bad | def_bad)[0]
    lines = []
    for i, idx in enumerate(flagged[:5]):  # report up to 5 records
        rid = ids[mask][idx]
        er = effort_resid[idx]
        dr = defect_resid[idx]
        parts = [f"ID={rid}"]
        if er > eff_thresh:
            parts.append(f"Effort resid={er:.2f} mo")
        if er < -eff_thresh:
            parts.append(f"Effort resid={er:.2f} mo")
        if dr > def_thresh:
            parts.append(f"Defects resid={dr:.2f}")
        if dr < -def_thresh:
            parts.append(f"Defects resid={dr:.2f}")
        lines.append("  " + ", ".join(parts))

    if not lines:
        lines.append("  No records show significant residuals.")
    return lines


def main(csv_path):
    rows = read_csv(csv_path)

    # Train models for NASA-18
    reg_effort, reg_defect, _, _ = train_and_predict(rows)

    # ---- NASA-17 ----
    print("== NASA-17 ==")
    for line in nasa_17(rows):
        print(line)

    # ---- NASA-18 ----
    print("\n== NASA-18 ==")
    for line in nasa_18(rows, reg_effort, reg_defect):
        print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
