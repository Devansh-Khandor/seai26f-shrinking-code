#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <csv_path>

Prints:
    CAR-03: Estimates missing MPG values with prediction intervals.
    CAR-04: Estimates vehicle weight and its uncertainty for the inventory.
"""

import csv
import sys
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return list of dicts with numeric conversion."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Convert numeric fields; missing values become None
            try:
                row['Clndrs'] = int(row['Clndrs'])
            except (ValueError, KeyError):
                row['Clndrs'] = None
            for col in ['Volume', 'HpX', 'Model', 'origin', 'Lbs-', 'Acc+', 'Mpg+']:
                try:
                    val = float(row[col])
                except (ValueError, KeyError):
                    val = None
                row[col] = val
            # keep row index for display
            row['_idx'] = i
            rows.append(row)
    return rows

def build_design_matrix(rows, feature_names, target_name):
    """
    Build X and y matrices for regression.
    X: n_samples x (n_features + 1) with bias term
    y: n_samples vector
    Returns X, y, mask of valid rows
    """
    X_list = []
    y_list = []
    mask = []
    for row in rows:
        # Skip rows with missing target
        if row[target_name] is None:
            continue
        # Build feature vector
        try:
            features = [row[fn] for fn in feature_names]
        except KeyError:
            continue
        if None in features:
            continue  # skip if any feature missing
        X_list.append([1.0] + features)  # bias term
        y_list.append(row[target_name])
        mask.append(True)
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y, mask

def fit_model(X, y):
    """
    Fit linear regression, compute sigma^2 and inverse(X^T X).
    Returns model, sigma2, inv_XTX
    """
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    y_pred = reg.predict(X)
    n, p = X.shape
    residuals = y - y_pred
    rss = residuals.T @ residuals
    sigma2 = rss / (n - p) if n > p else float('nan')
    inv_XTX = np.linalg.inv(X.T @ X) if n > p else None
    return reg, sigma2, inv_XTX

def predict_with_ci(reg, sigma2, inv_XTX, x_new, n, p):
    """
    Predict y for x_new (including bias), return y_hat and 95% confidence interval.
    """
    y_hat = reg.predict(x_new.reshape(1, -1))[0]
    # Standard error of prediction
    se = np.sqrt(sigma2 * (1.0 + x_new @ inv_XTX @ x_new.T)) if inv_XTX is not None else float('nan')
    ci = 1.96 * se  # 95% interval
    return y_hat, ci

# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    # Define feature sets
    all_features = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin', 'Lbs-', 'Acc+']
    mpg_features = [f for f in all_features if f != 'Mpg+']
    weight_features = [f for f in all_features if f != 'Lbs-']

    # ------------------------------
    # CAR-03: Missing MPG estimation
    # ------------------------------
    X_mpg, y_mpg, mask_mpg = build_design_matrix(rows, mpg_features, 'Mpg+')
    if X_mpg.shape[0] == 0:
        print("== CAR-03 ==")
        print("No data available to train MPG model.")
    else:
        reg_mpg, sigma2_mpg, inv_XTX_mpg = fit_model(X_mpg, y_mpg)
        print("== CAR-03 ==")
        # Find rows with missing MPG
        missing_mpg = [r for r in rows if r['Mpg+'] is None]
        if not missing_mpg:
            print("No missing MPG values found.")
        else:
            # Predict for first up to 3 missing entries
            printed = 0
            for r in missing_mpg:
                if printed >= 3:
                    break
                # Build feature vector
                try:
                    feats = [r[f] for f in mpg_features]
                except KeyError:
                    continue
                if None in feats:
                    continue
                x_new = np.array([1.0] + feats, dtype=float)
                y_hat, ci = predict_with_ci(reg_mpg, sigma2_mpg, inv_XTX_mpg,
                                            x_new, X_mpg.shape[0], X_mpg.shape[1])
                print(f"Vehicle #{r['_idx']} (ID={r['Clndrs']}) -> Estimated MPG: "
                      f"{y_hat:.2f} ± {ci:.2f}")
                printed += 1
            # If fewer than 3, add summary lines
            if printed == 0:
                print("No estimable missing MPG entries due to insufficient features.")
            elif printed < 3:
                for _ in range(3 - printed):
                    print("(No additional missing MPG entries.)")

    # ------------------------------
    # CAR-04: Weight estimation
    # ------------------------------
    X_wt, y_wt, mask_wt = build_design_matrix(rows, weight_features, 'Lbs-')
    if X_wt.shape[0] == 0:
        print("\n== CAR-04 ==")
        print("No data available to train weight model.")
    else:
        reg_wt, sigma2_wt, inv_XTX_wt = fit_model(X_wt, y_wt)
        print("\n== CAR-04 ==")
        # For demonstration, show predictions for first 3 vehicles
        printed = 0
        for r in rows:
            if printed >= 3:
                break
            # Build feature vector
            try:
                feats = [r[f] for f in weight_features]
            except KeyError:
                continue
            if None in feats:
                continue
            x_new = np.array([1.0] + feats, dtype=float)
            y_hat, ci = predict_with_ci(reg_wt, sigma2_wt, inv_XTX_wt,
                                        x_new, X_wt.shape[0], X_wt.shape[1])
            print(f"Vehicle #{r['_idx']} (ID={r['Clndrs']}) -> Estimated Weight: "
                  f"{y_hat:.2f} lbs ± {ci:.2f} lbs")
            printed += 1
        if printed == 0:
            print("No estimable weight entries due to insufficient features.")
        elif printed < 3:
            for _ in range(3 - printed):
                print("(No additional entries to predict.)")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle inventory predictive analysis")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()
    main(args.csv_path)
