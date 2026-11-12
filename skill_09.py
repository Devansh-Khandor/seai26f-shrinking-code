#!/usr/bin/env python3
"""
skill_09.py - NASA‑17 and NASA‑18: Anomaly and outlier detection.

Usage:
    python skill_09.py <csv_path>
"""

import csv
import sys
import numpy as np

# ----------------------------------------------------------------------
# Basic CSV loading
# ----------------------------------------------------------------------
try:
    from skill_01 import load_csv, get_numeric_column
except Exception:  # Fallback
    def load_csv(csv_path: str):
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = [row for row in reader]
        return rows, header

    def get_numeric_column(rows, col):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (ValueError, TypeError):
                pass
        return vals

# ----------------------------------------------------------------------
# Feature helpers from skill_02
# ----------------------------------------------------------------------
try:
    from skill_02 import (
        identify_numeric_columns,
        prepare_data,
        train_regression,
        build_mappings,
        encode_row,
    )
except Exception:  # Minimal local re‑implementation
    def identify_numeric_columns(rows, header):
        numeric = set()
        categorical = set()
        for col in header:
            if col == 'idX':
                continue
            sample = None
            for r in rows:
                val = r.get(col, '')
                if val != '':
                    sample = val
                    break
            if sample is None:
                categorical.add(col)
            else:
                try:
                    float(sample)
                    numeric.add(col)
                except ValueError:
                    categorical.add(col)
        return numeric, categorical

    def build_mappings(rows, categorical):
        mappings = {}
        for col in categorical:
            mapping = {}
            idx = 0
            for r in rows:
                val = r.get(col, '')
                if val == '':
                    continue
                if val not in mapping:
                    mapping[val] = idx
                    idx += 1
            mappings[col] = mapping
        return mappings

    def encode_row(r, numeric, categorical, mappings):
        vec = []
        for col in numeric:
            try:
                vec.append(float(r.get(col, '0')))
            except ValueError:
                vec.append(0.0)
        for col in categorical:
            val = r.get(col, '')
            mapping = mappings[col]
            vec.append(float(mapping.get(val, -1)))  # unseen → -1
        return vec

    def prepare_data(rows, header, target_col):
        numeric, categorical = identify_numeric_columns(rows, header)
        numeric.discard(target_col)
        categorical.discard(target_col)
        numeric.discard('idX')
        categorical.discard('idX')
        mappings = build_mappings(rows, categorical)
        X, y = [], []
        for r in rows:
            try:
                y_val = float(r.get(target_col, '0'))
            except ValueError:
                continue
            X.append(encode_row(r, numeric, categorical, mappings))
            y.append(y_val)
        return np.array(X, dtype=float), np.array(y, dtype=float), numeric, categorical, mappings

    def train_regression(X, y):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model

# ----------------------------------------------------------------------
# NASA‑17: anomaly detection in feature space
# ----------------------------------------------------------------------
def report_nasa_17(rows, header):
    print("\n== NASA-17 ==")

    # Identify numeric features (excluding id and target columns)
    numeric, categorical = identify_numeric_columns(rows, header)
    if not numeric:
        print("No numeric attributes available for anomaly analysis.")
        return

    # Build feature matrix (only numeric columns)
    X = []
    ids = []
    for r in rows:
        vec = []
        skip = False
        for col in numeric:
            try:
                vec.append(float(r.get(col, '0')))
            except ValueError:
                skip = True
                break
        if skip:
            continue
        X.append(vec)
        ids.append(r.get('idX', 'N/A'))

    X = np.array(X, dtype=float)
    if X.shape[0] < 2:
        print("Insufficient data for statistical anomaly detection.")
        return

    # Compute z‑scores for each feature
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=1)
    # Avoid division by zero
    stds[stds == 0] = 1.0
    z = (X - means) / stds

    # Mahalanobis‑like anomaly score: sum of squared z‑scores
    scores = np.sum(z**2, axis=1)

    # Threshold: e.g., 3 standard deviations (score > 9)
    threshold = 9.0
    anomalies = [(ids[i], scores[i]) for i in range(len(scores)) if scores[i] > threshold]

    if not anomalies:
        print("No projects flagged as anomalous.")
        return

    anomalies.sort(key=lambda x: x[1], reverse=True)
    print(f"Number of anomalous projects: {len(anomalies)}")
    print("Top anomalous projects (by score):")
    for i, (pid, score) in enumerate(anomalies[:5], 1):
        print(f"  {i}. id={pid} -> anomaly score={score:.2f}")

# ----------------------------------------------------------------------
# NASA‑18: projects with unexpected effort/defect counts
# ----------------------------------------------------------------------
def report_nasa_18(rows, header):
    print("\n== NASA-18 ==")

    # Prepare data for effort and defect prediction
    X, y_effort, _, _, _ = prepare_data(rows, header, 'Effort-')
    model_effort = train_regression(X, y_effort)
    preds_effort = model_effort.predict(X)
    resid_eff = y_effort - preds_effort
    std_eff = np.std(resid_eff, ddof=1)

    X, y_defects, _, _, _ = prepare_data(rows, header, 'Defects-')
    model_defect = train_regression(X, y_defects)
    preds_defect = model_defect.predict(X)
    resid_def = y_defects - preds_defect
    std_def = np.std(resid_def, ddof=1)

    # Identify outliers: |actual - predicted| > 2*std
    outliers = []
    for idx, (actual, pred, sd) in enumerate(
        zip(zip(y_effort, preds_effort), zip(y_defects, preds_defect), [std_eff]*len(preds_effort) + [std_def]*len(preds_defect))
    ):
        pass  # placeholder

    # We'll collect separately for effort and defects
    effort_out = [(idx, rows[idx].get('idX', 'N/A'), y_effort[idx], preds_effort[idx], abs(y_effort[idx] - preds_effort[idx]))
                  for idx in range(len(preds_effort))
                  if abs(y_effort[idx] - preds_effort[idx]) > 2 * std_eff]

    defect_out = [(idx, rows[idx].get('idX', 'N/A'), y_defects[idx], preds_defect[idx], abs(y_defects[idx] - preds_defect[idx]))
                  for idx in range(len(preds_defect))
                  if abs(y_defects[idx] - preds_defect[idx]) > 2 * std_def]

    if not effort_out and not defect_out:
        print("No projects have unexpectedly high effort or defect counts.")
        return

    if effort_out:
        effort_out.sort(key=lambda x: x[4], reverse=True)
        print(f"Number of projects with unexpectedly high effort: {len(effort_out)}")
        print("Top effort anomalies:")
        for i, (idx, pid, act, pred, diff) in enumerate(effort_out[:5], 1):
            print(f"  {i}. id={pid} -> actual={act:.2f}, predicted={pred:.2f}, diff={diff:.2f}")

    if defect_out:
        defect_out.sort(key=lambda x: x[4], reverse=True)
        print(f"Number of projects with unexpectedly high defects: {len(defect_out)}")
        print("Top defect anomalies:")
        for i, (idx, pid, act, pred, diff) in enumerate(defect_out[:5], 1):
            print(f"  {i}. id={pid} -> actual={act:.2f}, predicted={pred:.2f}, diff={diff:.2f}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, header = load_csv(csv_path)

    report_nasa_17(rows, header)
    report_nasa_18(rows, header)

if __name__ == "__main__":
    main()
