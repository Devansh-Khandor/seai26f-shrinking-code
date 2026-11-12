#!/usr/bin/env python3
"""
skill_10.py

Implements NASA-19 and NASA-20 analyses on the NASA software project dataset.

NASA-19: Compare our project‑effort estimation system against a simple historical‑average estimate
         and show whether it makes meaningfully better predictions on previously unseen projects.

NASA-20: Compare our project‑selection approach with randomly selecting proposals under the same evaluation budget,
         and show whether our system identifies more desirable project outcomes.

Usage:
    python skill_10.py <csv_path>
"""

import sys
import csv
import random
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def parse_csv(csv_path: Path):
    """Read CSV and return list of rows (dict) and list of numeric column names."""
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


# ---------------------------------------------------------------------------

def nasa_19(rows, numeric_cols):
    """NASA-19: Compare model vs historical average for Effort-."""
    target_col = 'Effort-'
    if target_col not in numeric_cols:
        print("== NASA-19 ==")
        print("Target column 'Effort-' not found.")
        print()
        return

    # Build feature matrix excluding target
    feature_cols = [c for c in numeric_cols if c != target_col]
    X = to_numpy(rows, feature_cols)
    y = to_numpy(rows, [target_col]).flatten()

    # Drop rows with missing feature or target
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    if mask.sum() < 2:
        print("== NASA-19 ==")
        print("Insufficient data after removing missing values.")
        print()
        return

    X_clean = X[mask]
    y_clean = y[mask]

    # Train/test split (70/30)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(y_clean))
    split = int(0.7 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X_clean[train_idx], X_clean[test_idx]
    y_train, y_test = y_clean[train_idx], y_clean[test_idx]

    # Train linear regression
    model = LinearRegression(fit_intercept=True).fit(X_train, y_train)

    # Predictions on test set
    pred_test = model.predict(X_test)

    # Baseline: historical average from training set
    avg_train = np.mean(y_train)
    baseline_pred = np.full_like(y_test, avg_train)

    # Errors
    mae_model = np.mean(np.abs(pred_test - y_test))
    mae_baseline = np.mean(np.abs(baseline_pred - y_test))
    rmse_model = np.sqrt(np.mean((pred_test - y_test) ** 2))
    rmse_baseline = np.sqrt(np.mean((baseline_pred - y_test) ** 2))
    improvement = (mae_baseline - mae_model) / mae_baseline * 100 if mae_baseline != 0 else 0

    # Output
    print("== NASA-19 ==")
    print(f"Model MAE: {mae_model:.2f}, RMSE: {rmse_model:.2f}")
    print(f"Baseline MAE: {mae_baseline:.2f}, RMSE: {rmse_baseline:.2f}")
    print(f"MAE improvement over baseline: {improvement:.1f}%")
    print()


def nasa_20(rows, numeric_cols):
    """NASA-20: Compare scoring‑based selection vs random selection."""
    target_eff_col = 'Effort-'
    target_def_col = 'Defects-'
    if target_eff_col not in numeric_cols or target_def_col not in numeric_cols:
        print("== NASA-20 ==")
        print("Required target columns 'Effort-' and 'Defects-' not found.")
        print()
        return

    # Build feature matrix excluding targets
    feature_cols = [c for c in numeric_cols if c not in (target_eff_col, target_def_col)]
    X = to_numpy(rows, feature_cols)
    y_eff = to_numpy(rows, [target_eff_col]).flatten()
    y_def = to_numpy(rows, [target_def_col]).flatten()

    # Drop rows with missing features or targets
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_eff) & ~np.isnan(y_def)
    if mask.sum() < 2:
        print("== NASA-20 ==")
        print("Insufficient complete data for selection evaluation.")
        print()
        return

    X_clean = X[mask]
    y_eff_clean = y_eff[mask]
    y_def_clean = y_def[mask]
    ids_clean = [r['idX'] for i, r in enumerate(rows) if mask[i]]

    # Train linear regression models on all complete data (used for scoring)
    model_eff = LinearRegression(fit_intercept=True).fit(X_clean, y_eff_clean)
    model_def = LinearRegression(fit_intercept=True).fit(X_clean, y_def_clean)

    # Predict on all complete projects (simulate unseen proposals)
    pred_eff = model_eff.predict(X_clean)
    pred_def = model_def.predict(X_clean)

    # Score: lower predicted effort + defects is better
    scores = pred_eff + pred_def

    # Evaluation budget: select 10% of projects or at least 1
    budget = max(1, int(0.1 * len(ids_clean)))
    if budget > len(ids_clean):
        budget = len(ids_clean)

    # Scoring‑based selection (choose projects with lowest predicted sum)
    idx_sorted = np.argsort(scores)[:budget]
    selected_ids_scoring = [ids_clean[i] for i in idx_sorted]
    eff_scoring = y_eff_clean[idx_sorted]
    def_scoring = y_def_clean[idx_sorted]
    avg_eff_scoring = np.mean(eff_scoring)
    avg_def_scoring = np.mean(def_scoring)

    # Random selection
    random.seed(42)
    rand_indices = random.sample(range(len(ids_clean)), budget)
    selected_ids_rand = [ids_clean[i] for i in rand_indices]
    eff_rand = y_eff_clean[rand_indices]
    def_rand = y_def_clean[rand_indices]
    avg_eff_rand = np.mean(eff_rand)
    avg_def_rand = np.mean(def_rand)

    # Output
    print("== NASA-20 ==")
    print(f"Budget: {budget} proposals")
    print(f"Scoring‑based selection – avg Effort: {avg_eff_scoring:.2f}, avg Defects: {avg_def_scoring:.2f}")
    print(f"Random selection          – avg Effort: {avg_eff_rand:.2f}, avg Defects: {avg_def_rand:.2f}")
    print()
    # Show relative improvement
    eff_improvement = ((avg_eff_rand - avg_eff_scoring) / avg_eff_rand * 100
                       if avg_eff_rand != 0 else 0)
    def_improvement = ((avg_def_rand - avg_def_scoring) / avg_def_rand * 100
                       if avg_def_rand != 0 else 0)
    print(f"Effort reduction over random: {eff_improvement:.1f}%")
    print(f"Defect reduction over random: {def_improvement:.1f}%")
    print()


# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: {csv_path} does not exist or is not a file.")
        sys.exit(1)

    rows, numeric_cols = parse_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    nasa_19(rows, numeric_cols)
    nasa_20(rows, numeric_cols)


if __name__ == "__main__":
    main()
