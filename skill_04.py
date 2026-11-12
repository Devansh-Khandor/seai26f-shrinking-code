#!/usr/bin/env python3
"""
skill_04.py

Implements NASA-07 (proposal assessment) and NASA-08 (historical update)
on the NASA software project dataset.

Usage:
    python skill_04.py <csv_path>
"""

import sys
import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def parse_csv(csv_path: Path):
    """Read CSV and return rows and numeric column names."""
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        numeric_cols = []
        for col in reader.fieldnames:
            try:
                for r in rows:
                    val = r[col]
                    if val != '':
                        float(val.replace(',', ''))
                        numeric_cols.append(col)
                        break
            except (ValueError, TypeError):
                continue
    return rows, numeric_cols

def to_numpy(rows, cols):
    """Return (n_samples, n_features) array of floats, NaN for missing."""
    arr = []
    for r in rows:
        row = []
        for c in cols:
            val = r[c]
            if val == '':
                row.append(np.nan)
            else:
                try:
                    row.append(float(val.replace(',', '')))
                except ValueError:
                    row.append(np.nan)
        arr.append(row)
    return np.array(arr, dtype=float)

def model_stats(X, y, model):
    """Return R², RMSE, residuals."""
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    residuals = y - y_pred
    return r2, rmse, residuals, y_pred

def predict_with_interval(model, X, residuals, x0):
    """
    Compute prediction for x0 and an approximate 95% prediction interval.
    """
    n, p = X.shape
    df = n - p - 1
    s = np.sqrt(np.sum(residuals ** 2) / df)

    Xfull = np.hstack([np.ones((n, 1)), X])
    inv_xtx = np.linalg.pinv(Xfull.T @ Xfull)

    x0_full = np.hstack([1.0, x0])
    se_pred = s * np.sqrt(1 + x0_full @ inv_xtx @ x0_full.T)

    # 95% t critical approx 2 for large df
    t_crit = 2.0
    pred = model.predict(x0_full.reshape(1, -1))[0]
    return pred, t_crit * se_pred

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: {csv_path} does not exist or is not a file.")
        sys.exit(1)

    rows, numeric_cols = parse_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    # Identify required columns
    eff_col = 'Effort-'
    defect_col = 'Defects-'
    if eff_col not in numeric_cols or defect_col not in numeric_cols:
        print("Required numeric columns 'Effort-' and 'Defects-' not found.")
        sys.exit(1)

    # Feature columns: all numeric except the two targets
    feature_cols = [c for c in numeric_cols if c not in (eff_col, defect_col)]

    if not feature_cols:
        print("No feature columns available for modeling.")
        sys.exit(1)

    # Build data matrices
    X_all = to_numpy(rows, feature_cols)
    y_eff_all = to_numpy(rows, [eff_col]).flatten()
    y_def_all = to_numpy(rows, [defect_col]).flatten()

    # Masks for completed projects (targets present and no missing features)
    mask_eff = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_eff_all)
    mask_def = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_def_all)

    X_eff = X_all[mask_eff]
    y_eff = y_eff_all[mask_eff]
    X_def = X_all[mask_def]
    y_def = y_def_all[mask_def]

    # Train models
    model_eff = LinearRegression(fit_intercept=True)
    model_eff.fit(X_eff, y_eff)
    r2_eff, rmse_eff, resid_eff, y_eff_pred = model_stats(X_eff, y_eff, model_eff)

    model_def = LinearRegression(fit_intercept=True)
    model_def.fit(X_def, y_def)
    r2_def, rmse_def, resid_def, y_def_pred = model_stats(X_def, y_def, model_def)

    # Statistics of predictions for completed projects
    eff_pred_mean = np.nanmean(y_eff_pred)
    eff_pred_std = np.nanstd(y_eff_pred, ddof=1)
    def_pred_mean = np.nanmean(y_def_pred)
    def_pred_std = np.nanstd(y_def_pred, ddof=1)

    # ---------------- NASA-07 ----------------
    print("== NASA-07 ==")

    # Identify new proposals: rows missing either Effort- or Defects-
    proposal_rows = [r for r in rows if r[eff_col] == '' or r[defect_col] == '']

    if not proposal_rows:
        print("No new proposals to evaluate.")
    else:
        for r in proposal_rows:
            # Build feature vector
            x_vals = []
            missing_feat = False
            for c in feature_cols:
                val = r[c]
                if val == '':
                    missing_feat = True
                    break
                try:
                    x_vals.append(float(val.replace(',', '')))
                except ValueError:
                    missing_feat = True
                    break
            if missing_feat:
                continue  # cannot predict this proposal

            x0 = np.array(x_vals, dtype=float)

            # Predict effort
            pred_eff, unc_eff = predict_with_interval(model_eff, X_eff, resid_eff, x0)
            # Predict defects
            pred_def, unc_def = predict_with_interval(model_def, X_def, resid_def, x0)

            # Determine alerts
            alert_eff = pred_eff > eff_pred_mean + 2 * eff_pred_std
            alert_def = pred_def > def_pred_mean + 2 * def_pred_std

            print(f"Proposal ID {r['idX']}:")
            print(f"  Predicted effort: {pred_eff:.2f} ± {unc_eff:.2f} person‑months")
            print(f"  Predicted defects: {pred_def:.2f} ± {unc_def:.2f}")
            if alert_eff or alert_def:
                alerts = []
                if alert_eff:
                    alerts.append("effort")
                if alert_def:
                    alerts.append("defects")
                print(f"  ALERT: unusually high {', '.join(alerts)}")
            print()

    # ---------------- NASA-08 ----------------
    print("== NASA-08 ==")

    total_completed = mask_eff.sum()  # same as mask_def
    if total_completed == 0:
        print("No completed projects available to update summaries.")
    else:
        # Summary statistics
        mean_eff = np.nanmean(y_eff)
        med_eff = np.nanmedian(y_eff)
        mean_def = np.nanmean(y_def)
        med_def = np.nanmedian(y_def)

        print(f"Total completed projects: {total_completed}")
        print(f"Effort: mean {mean_eff:.2f} person‑months, median {med_eff:.2f}")
        print(f"Defects: mean {mean_def:.2f}, median {med_def:.2f}")
        print()
        print(f"Effort model: R² = {r2_eff:.3f}, RMSE = {rmse_eff:.3f}")
        print(f"Defect model: R² = {r2_def:.3f}, RMSE = {rmse_def:.3f}")
        print()

if __name__ == "__main__":
    main()
