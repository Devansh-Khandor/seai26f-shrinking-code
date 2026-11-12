#!/usr/bin/env python3
"""
Skill 09: Detect unusual project records.

NASA-17: Flag historical projects whose combinations of project characteristics are unusually different
        from the rest of our records so that we can review them.

NASA-18: Identify project records whose reported effort or defect counts differ substantially
        from what comparable projects would lead us to expect, so we can investigate possible
        measurement or recording issues.

Usage:
    python skill_09.py <path_to_nasa93dem.csv>
"""

import csv
import sys
import math
import numpy as np
from sklearn.linear_model import LinearRegression


def read_csv(csv_path):
    """
    Read CSV and return a list of dictionaries (rows) with all fields stripped.
    """
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k: v.strip() for k, v in row.items()}
            rows.append(cleaned)
    return rows


def numeric_columns(rows):
    """
    Return list of column names that can be parsed as float in >80% of rows.
    """
    if not rows:
        return []

    cols = list(rows[0].keys())
    numeric = []
    for col in cols:
        if col in ("idX", "id"):  # skip id columns
            continue
        n = 0
        m = 0
        for r in rows:
            val = r.get(col, "")
            if val == "":
                continue
            try:
                float(val)
                n += 1
            except ValueError:
                pass
            m += 1
        if m == 0:
            continue
        if n / m >= 0.8:
            numeric.append(col)
    return numeric


def build_feature_matrix(rows, cols):
    """
    Build a 2‑D numpy array for numeric columns. Rows with any missing value
    for those columns are skipped.
    """
    data = []
    ids = []
    for r in rows:
        vals = []
        skip = False
        for c in cols:
            val = r.get(c, "")
            if val == "":
                skip = True
                break
            try:
                vals.append(float(val))
            except ValueError:
                skip = True
                break
        if not skip:
            data.append(vals)
            ids.append(r.get("idX", ""))
    if not data:
        return np.array([]), []
    return np.array(data, dtype=float), ids


def compute_mahalanobis_distances(X):
    """
    Compute Mahalanobis distance from the mean for each sample in X.
    """
    if X.size == 0:
        return np.array([])
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    # Add a small regularization to covariance if singular
    if np.linalg.det(cov) == 0:
        cov += np.eye(cov.shape[0]) * 1e-8
    inv_cov = np.linalg.inv(cov)
    diff = X - mean
    left = np.dot(diff, inv_cov)
    distances = np.sqrt(np.sum(left * diff, axis=1))
    return distances


def fit_regression(X, y):
    """Fit a linear regression model y = a*X + b."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def residuals(model, X, y):
    preds = model.predict(X)
    return y - preds


def format_number(val, decimals=3):
    if math.isnan(val) or math.isinf(val):
        return "N/A"
    return f"{val:.{decimals}f}".rstrip('0').rstrip('.')


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No data found.")
        sys.exit(1)

    # --------------------------------------------------------------------
    # NASA-17: Unusual characteristic combinations
    print("== NASA-17 ==")
    num_cols = numeric_columns(rows)
    # Build matrix for all numeric columns
    X_all, ids_all = build_feature_matrix(rows, num_cols)
    if X_all.size == 0:
        print("  No numeric data available for distance calculation.")
    else:
        distances = compute_mahalanobis_distances(X_all)
        # Flag top 5% projects with highest distances
        threshold = np.percentile(distances, 95)
        flagged = [(id_, d) for id_, d in zip(ids_all, distances) if d > threshold]
        if not flagged:
            print("  No projects exceed the 95th percentile distance threshold.")
        else:
            print(f"  Flagged {len(flagged)} projects (distance > {format_number(threshold)})")
            # Show at least 3 examples
            for pid, dist in flagged[:3]:
                print(f"  {pid} : distance = {format_number(dist)}")
            if len(flagged) > 3:
                print("  ...")

    # --------------------------------------------------------------------
    # NASA-18: Unexpected effort or defect counts
    print("\n== NASA-18 ==")

    # Build feature matrix for regression models (use all numeric columns)
    # We need Kloc+ to predict Effort and Defects
    # Build X and y for effort
    effort_rows = [r for r in rows if r.get("Effort-") and r.get("Kloc+")]
    X_eff, ids_eff = build_feature_matrix(effort_rows, ["Kloc+"])
    y_eff = np.array([float(r["Effort-"]) for r in effort_rows], dtype=float)
    if X_eff.size == 0:
        print("  No completed projects with effort data available.")
    else:
        eff_model = fit_regression(X_eff, y_eff)
        eff_res = residuals(eff_model, X_eff, y_eff)
        eff_std = np.std(eff_res, ddof=0)
        eff_threshold = 2 * eff_std
        flagged_eff = [(pid, obs, pred, res) for pid, obs, pred, res in
                       zip(ids_eff, y_eff,
                           eff_model.predict(X_eff),
                           eff_res) if abs(res) > eff_threshold]
        if not flagged_eff:
            print("  No projects exceed the effort residual threshold.")
        else:
            print(f"  Flagged {len(flagged_eff)} projects (|residual| > {format_number(eff_threshold)})")
            for pid, obs, pred, res in flagged_eff[:3]:
                print(f"  {pid} : observed {format_number(obs)} → predicted {format_number(pred)} (Δ{format_number(res)})")
            if len(flagged_eff) > 3:
                print("  ...")

    # Build feature matrix for defects
    defect_rows = [r for r in rows if r.get("Defects-") and r.get("Kloc+")]
    X_def, ids_def = build_feature_matrix(defect_rows, ["Kloc+"])
    y_def = np.array([float(r["Defects-"]) for r in defect_rows], dtype=float)
    if X_def.size == 0:
        print("\n  No completed projects with defect data available.")
    else:
        def_model = fit_regression(X_def, y_def)
        def_res = residuals(def_model, X_def, y_def)
        def_std = np.std(def_res, ddof=0)
        def_threshold = 2 * def_std
        flagged_def = [(pid, obs, pred, res) for pid, obs, pred, res in
                       zip(ids_def, y_def,
                           def_model.predict(X_def),
                           def_res) if abs(res) > def_threshold]
        if not flagged_def:
            print("\n  No projects exceed the defect residual threshold.")
        else:
            print(f"\n  Flagged {len(flagged_def)} projects (|residual| > {format_number(def_threshold)})")
            for pid, obs, pred, res in flagged_def[:3]:
                print(f"  {pid} : observed {format_number(obs)} → predicted {format_number(pred)} (Δ{format_number(res)})")
            if len(flagged_def) > 3:
                print("  ...")

    # End
    print()


if __name__ == "__main__":
    main()
