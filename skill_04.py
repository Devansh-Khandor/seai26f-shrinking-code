#!/usr/bin/env python3
"""
Skill 04: NASA-07 and NASA-08 – Real‑time assessment of proposals and
incremental updating of historical summaries.
Author: Your Name
"""

import sys
import csv
import math
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Utility: read CSV and convert numeric values where possible
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Reads a CSV file and returns:
        header: list of column names
        rows: list of dicts mapping column -> value (numeric where possible)
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                if v is None:
                    row[k] = v
                    continue
                v = v.strip()
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return header, rows

# ----------------------------------------------------------------------
# Build feature matrix X and target vector y for a given target column
# ----------------------------------------------------------------------
def build_matrix(rows, target_col, exclude_cols=None):
    """
    Given rows, produce:
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,)
        feature_names: list of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []

    # Determine numeric columns that can serve as predictors
    numeric_cols = []
    for key in rows[0].keys():
        if key in exclude_cols or key == target_col:
            continue
        # A column is numeric if all its values are numeric (or missing)
        if all(
            isinstance(r.get(key, None), (int, float)) or r.get(key, None) is None
            for r in rows
        ):
            numeric_cols.append(key)

    # Build data arrays, skipping rows with missing target or any missing feature
    X_rows = []
    y_rows = []
    for r in rows:
        target_val = r.get(target_col, None)
        if not isinstance(target_val, (int, float)):
            continue  # skip rows without valid target
        feature_vals = []
        missing = False
        for col in numeric_cols:
            val = r.get(col, None)
            if isinstance(val, (int, float)):
                feature_vals.append(val)
            else:
                missing = True
                break
        if missing:
            continue
        X_rows.append(feature_vals)
        y_rows.append(target_val)

    if not X_rows:
        raise ValueError(f"No valid rows found for target '{target_col}'.")

    X = np.array(X_rows)
    y = np.array(y_rows)
    return X, y, numeric_cols

# ----------------------------------------------------------------------
# Helper: compute mean and standard deviation of predictions
# ----------------------------------------------------------------------
def prediction_stats(preds):
    """Return mean and std of a 1‑D array."""
    n = len(preds)
    if n == 0:
        return 0.0, 0.0
    m = np.mean(preds)
    s = np.std(preds, ddof=1) if n > 1 else 0.0
    return m, s

# ----------------------------------------------------------------------
# NASA-07: Real‑time assessment of proposals
# ----------------------------------------------------------------------
def nasa_07(header, rows, effort_model, effort_stats,
            defect_model, defect_stats):
    """
    For each row in *rows*, predict effort and defects using the provided
    models.  Alert if the predictions are unusually high (mean + 2*std).
    """
    eff_mean, eff_std = effort_stats
    def_mean, def_std = defect_stats

    # Thresholds for "unusually high"
    eff_thresh = eff_mean + 2 * eff_std
    def_thresh = def_mean + 2 * def_std

    print("== NASA-07 ==")
    # Header line
    print(f"Thresholds – Effort: {eff_thresh:.2f}, Defects: {def_thresh:.2f}")
    print()
    # Process each row
    for r in rows:
        # Build feature vector for this row
        try:
            X_row, _, feat_names = build_matrix([r], 'Effort-', exclude_cols=['idX', 'YearX', 'MonthsX'])
            eff_pred = effort_model.predict(X_row)[0]
        except Exception:
            eff_pred = None

        try:
            X_row_def, _, _ = build_matrix([r], 'Defects-', exclude_cols=['idX', 'YearX', 'MonthsX'])
            def_pred = defect_model.predict(X_row_def)[0]
        except Exception:
            def_pred = None

        alerts = []
        if eff_pred is not None and eff_pred > eff_thresh:
            alerts.append("Effort high")
        if def_pred is not None and def_pred > def_thresh:
            alerts.append("Defects high")

        alert_text = " | ".join(alerts) if alerts else "Normal"

        print(f"Project ID={int(r.get('idX',0))} | Effort={eff_pred: .2f} | Defects={def_pred: .2f} | {alert_text}")

    print()

# ----------------------------------------------------------------------
# NASA-08: Update historical summaries and retrain models
# ----------------------------------------------------------------------
def nasa_08(header, rows):
    """
    Update the historical summary statistics and retrain the linear
    regression models for effort and defects.  Then print the updated
    statistics and model coefficients.
    """
    # Summary for Effort-
    effort_vals = [r['Effort-'] for r in rows if isinstance(r.get('Effort-', None), (int, float))]
    effort_mean = sum(effort_vals) / len(effort_vals) if effort_vals else 0.0
    effort_std = math.sqrt(sum((x - effort_mean) ** 2 for x in effort_vals) / (len(effort_vals) - 1)) if len(effort_vals) > 1 else 0.0

    # Summary for Defects-
    defect_vals = [r['Defects-'] for r in rows if isinstance(r.get('Defects-', None), (int, float))]
    defect_mean = sum(defect_vals) / len(defect_vals) if defect_vals else 0.0
    defect_std = math.sqrt(sum((x - defect_mean) ** 2 for x in defect_vals) / (len(defect_vals) - 1)) if len(defect_vals) > 1 else 0.0

    # Build and train models
    effort_model = LinearRegression()
    X_eff, y_eff, eff_feat_names = build_matrix(rows, target_col='Effort-', exclude_cols=['idX', 'YearX', 'MonthsX'])
    effort_model.fit(X_eff, y_eff)

    defect_model = LinearRegression()
    X_def, y_def, def_feat_names = build_matrix(rows, target_col='Defects-', exclude_cols=['idX', 'YearX', 'MonthsX'])
    defect_model.fit(X_def, y_def)

    # Print updated summary
    print("== NASA-08 ==")
    print(f"Updated Effort-: mean={effort_mean:.2f}, std={effort_std:.2f}")
    print(f"Updated Defects-: mean={defect_mean:.2f}, std={defect_std:.2f}")
    print()

    # Print model coefficients
    print("Effort Prediction Model Coefficients:")
    for name, coef in zip(eff_feat_names, effort_model.coef_):
        print(f"  {name} -> {coef:.4f}")
    print(f"  Intercept: {effort_model.intercept_:.4f}")
    print()

    print("Defect Prediction Model Coefficients:")
    for name, coef in zip(def_feat_names, defect_model.coef_):
        print(f"  {name} -> {coef:.4f}")
    print(f"  Intercept: {defect_model.intercept_:.4f}")
    print()

    # Return models and stats for NASA-07
    return effort_model, (effort_mean, effort_std), defect_model, (defect_mean, defect_std)

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

    # Update historical summaries and train models (NASA-08)
    effort_model, effort_stats, defect_model, defect_stats = nasa_08(header, rows)

    # Assess proposals (NASA-07)
    nasa_07(header, rows, effort_model, effort_stats, defect_model, defect_stats)

if __name__ == "__main__":
    main()
