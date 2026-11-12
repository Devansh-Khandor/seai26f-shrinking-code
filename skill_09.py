#!/usr/bin/env python3
"""
skill_09.py

Implements NASA-17 and NASA-18 analyses on the NASA software project dataset.

NASA-17: Flag historical projects whose combinations of project characteristics are unusually different from the rest of our records
         so that we can review them.

NASA-18: Identify project records whose reported effort or defect counts differ substantially from what comparable projects would lead us to expect,
         so we can investigate possible measurement or recording issues.

Usage:
    python skill_09.py <csv_path>
"""

import sys
import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def parse_csv(csv_path: Path):
    """
    Read CSV and return rows (list of dict) and a list of numeric column names.
    """
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
    """
    Return (n_samples, n_features) array of floats, NaN for missing values.
    """
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

# ---------------------------------------------------------------------------

def nasa_17(rows, numeric_cols):
    """
    Flag projects whose feature vectors are outliers in Mahalanobis distance
    from the multivariate normal of the dataset.
    """
    # Exclude target columns from feature set
    target_cols = {'Effort-', 'Defects-'}
    feature_cols = [c for c in numeric_cols if c not in target_cols]
    if len(feature_cols) < 2:
        print("== NASA-17 ==")
        print("Insufficient numeric features to perform anomaly detection.")
        print()
        return

    X = to_numpy(rows, feature_cols)

    # Remove rows with any missing feature values for this calculation
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]
    ids_clean = [rows[i]['idX'] for i, m in enumerate(mask) if m]

    # Compute mean and covariance
    mu = np.nanmean(X_clean, axis=0)
    cov = np.cov(X_clean, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Covariance matrix singular; use pseudo-inverse
        inv_cov = np.linalg.pinv(cov)

    # Mahalanobis distances
    diff = X_clean - mu
    dists = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

    # Statistics
    mean_dist = np.mean(dists)
    std_dist = np.std(dists, ddof=1)

    # Identify top 5 outliers
    top_n = 5
    idx_sorted = np.argsort(-dists)
    top_idx = idx_sorted[:top_n]

    print("== NASA-17 ==")
    print(f"Total projects with complete features: {len(ids_clean)}")
    print(f"Mean Mahalanobis distance: {mean_dist:.3f}, std: {std_dist:.3f}")
    print("\nTop 5 anomalous projects (high distance):")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank}. ID {ids_clean[i]} – distance = {dists[i]:.3f}")
    print()

def nasa_18(rows, numeric_cols):
    """
    For each completed project (both Effort- and Defects- present), compare actual values to model predictions
    and flag those with large residuals.
    """
    target_eff = 'Effort-'
    target_def = 'Defects-'

    # Feature columns exclude targets
    feature_cols = [c for c in numeric_cols if c not in (target_eff, target_def)]
    if not feature_cols:
        print("== NASA-18 ==")
        print("Insufficient numeric features to build predictive models.")
        print()
        return

    # Build full feature matrix for all rows (may contain NaNs)
    X_all = to_numpy(rows, feature_cols)
    # Build target vectors
    y_eff_all = to_numpy(rows, [target_eff]).flatten()
    y_def_all = to_numpy(rows, [target_def]).flatten()

    # Mask for completed projects (no NaNs in features or targets)
    mask_eff = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_eff_all)
    mask_def = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_def_all)
    mask_complete = mask_eff & mask_def

    if mask_complete.sum() == 0:
        print("== NASA-18 ==")
        print("No completed projects with complete data available.")
        print()
        return

    X_comp = X_all[mask_complete]
    y_eff = y_eff_all[mask_complete]
    y_def = y_def_all[mask_complete]
    ids_comp = [rows[i]['idX'] for i, m in enumerate(mask_complete) if m]

    # Train linear regression models
    model_eff = LinearRegression(fit_intercept=True).fit(X_comp, y_eff)
    model_def = LinearRegression(fit_intercept=True).fit(X_comp, y_def)

    # Predict for all completed projects
    pred_eff = model_eff.predict(X_comp)
    pred_def = model_def.predict(X_comp)

    # Residuals
    resid_eff = y_eff - pred_eff
    resid_def = y_def - pred_def

    # Compute residual statistics
    mean_res_eff = np.mean(resid_eff)
    std_res_eff = np.std(resid_eff, ddof=1)
    mean_res_def = np.mean(resid_def)
    std_res_def = np.std(resid_def, ddof=1)

    # Thresholds: 2*std beyond mean (absolute)
    thresh_eff = 2 * std_res_eff
    thresh_def = 2 * std_res_def

    # Identify projects with large residuals (either effort or defect)
    large_res_idx = np.where((np.abs(resid_eff) > thresh_eff) | (np.abs(resid_def) > thresh_def))[0]

    print("== NASA-18 ==")
    print(f"Completed projects considered: {len(ids_comp)}")
    print(f"Effort residuals: mean {mean_res_eff:.3f}, std {std_res_eff:.3f}")
    print(f"Defect residuals: mean {mean_res_def:.3f}, std {std_res_def:.3f}\n")

    if large_res_idx.size == 0:
        print("No projects exhibit unusually large residuals.")
        print()
        return

    # Show up to top 5 projects with biggest absolute residuals
    # Compute a combined score for ranking
    combined_score = np.sqrt(resid_eff[large_res_idx]**2 + resid_def[large_res_idx]**2)
    top_order = np.argsort(-combined_score)[:5]
    print("Projects with large prediction residuals:")
    for rank, idx in enumerate(top_order, 1):
        i = large_res_idx[idx]
        pid = ids_comp[i]
        eff_act = y_eff[i]
        eff_pred = pred_eff[i]
        eff_res = resid_eff[i]
        def_act = y_def[i]
        def_pred = pred_def[i]
        def_res = resid_def[i]
        print(f"  {rank}. ID {pid}")
        print(f"       Effort: actual={eff_act:.2f}, predicted={eff_pred:.2f}, residual={eff_res:+.2f}")
        print(f"       Defects: actual={def_act:.2f}, predicted={def_pred:.2f}, residual={def_res:+.2f}")
    print()

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: {csv_path} does not exist or is not a file.")
        sys.exit(1)

    rows, numeric_cols = parse_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    nasa_17(rows, numeric_cols)
    nasa_18(rows, numeric_cols)

if __name__ == "__main__":
    main()
