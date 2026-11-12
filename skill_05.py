#!/usr/bin/env python3
"""
skill_05.py

Implements NASA-09 and NASA-10 analyses on the NASA software project dataset.
NASA-09: Identify project characteristics most strongly associated with unusually high effort.
NASA-10: Identify characteristics distinguishing projects with unusually high defect counts.

Usage:
    python skill_05.py <csv_path>
"""

import sys
import csv
from pathlib import Path
import numpy as np

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

def compute_correlations(X, y):
    """Return dict: feature name -> Pearson r with y (ignoring NaNs)."""
    correlations = {}
    # For each feature column
    for i, col in enumerate(feature_cols):
        x = X[:, i]
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        correlations[col] = r
    return correlations

def top_features_by_diff(high_mask, X, overall_mean, top_n=5):
    """Return list of tuples: (feature, mean_high, mean_all, diff)."""
    means_high = np.nanmean(X[high_mask], axis=0)
    diff = means_high - overall_mean
    stats = []
    for i, col in enumerate(feature_cols):
        if np.isnan(means_high[i]) or np.isnan(overall_mean[i]):
            continue
        stats.append((col, means_high[i], overall_mean[i], diff[i]))
    stats.sort(key=lambda t: abs(t[3]), reverse=True)
    return stats[:top_n]

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: {csv_path} does not exist or is not a file.")
        sys.exit(1)

    rows, numeric_cols = parse_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    # Target columns
    eff_col = 'Effort-'
    defect_col = 'Defects-'
    if eff_col not in numeric_cols or defect_col not in numeric_cols:
        print("Required numeric columns 'Effort-' and 'Defects-' not found.")
        sys.exit(1)

    # Feature columns: all numeric except the two targets
    global feature_cols
    feature_cols = [c for c in numeric_cols if c not in (eff_col, defect_col)]

    if not feature_cols:
        print("No feature columns available for analysis.")
        sys.exit(1)

    # Build data matrix
    X_all = to_numpy(rows, feature_cols)
    y_eff = to_numpy(rows, [eff_col]).flatten()
    y_def = to_numpy(rows, [defect_col]).flatten()

    # NASA-09: high effort analysis
    eff_percentile = 90
    eff_threshold = np.nanpercentile(y_eff, eff_percentile)
    high_eff_mask = y_eff > eff_threshold
    n_high_eff = np.sum(high_eff_mask)
    eff_min = np.nanmin(y_eff[high_eff_mask]) if n_high_eff else np.nan
    eff_max = np.nanmax(y_eff[high_eff_mask]) if n_high_eff else np.nan

    print("== NASA-09 ==")
    if n_high_eff == 0:
        print(f"No projects exceed the {eff_percentile}th percentile of effort.")
    else:
        print(f"High‑effort projects: {n_high_eff} (top {eff_percentile}%)")
        print(f"Effort range: {eff_min:.2f} – {eff_max:.2f} person‑months")

        overall_mean_eff = np.nanmean(X_all, axis=0)
        top_diff = top_features_by_diff(high_eff_mask, X_all, overall_mean_eff, top_n=5)

        if top_diff:
            print("\nTop characteristics associated with high effort:")
            for i, (col, mean_h, mean_a, diff) in enumerate(top_diff, 1):
                print(f"{i}. {col}: mean in high effort = {mean_h:.2f}, overall mean = {mean_a:.2f}, diff = {diff:.2f}")
        else:
            print("No distinguishable characteristics found.")

        # Correlations
        corr_eff = compute_correlations(X_all, y_eff)
        sorted_corr = sorted(corr_eff.items(), key=lambda kv: abs(kv[1]), reverse=True)
        print("\nTop 5 features by Pearson correlation with effort:")
        for i, (col, r) in enumerate(sorted_corr[:5], 1):
            print(f"{i}. {col}: r = {r:.3f}")

    print()

    # NASA-10: high defect analysis
    defect_percentile = 90
    defect_threshold = np.nanpercentile(y_def, defect_percentile)
    high_def_mask = y_def > defect_threshold
    n_high_def = np.sum(high_def_mask)
    def_min = np.nanmin(y_def[high_def_mask]) if n_high_def else np.nan
    def_max = np.nanmax(y_def[high_def_mask]) if n_high_def else np.nan

    print("== NASA-10 ==")
    if n_high_def == 0:
        print(f"No projects exceed the {defect_percentile}th percentile of defects.")
    else:
        print(f"High‑defect projects: {n_high_def} (top {defect_percentile}%)")
        print(f"Defect range: {def_min:.2f} – {def_max:.2f}")

        overall_mean_def = np.nanmean(X_all, axis=0)
        top_diff_def = top_features_by_diff(high_def_mask, X_all, overall_mean_def, top_n=5)

        if top_diff_def:
            print("\nTop characteristics associated with high defect counts:")
            for i, (col, mean_h, mean_a, diff) in enumerate(top_diff_def, 1):
                print(f"{i}. {col}: mean in high defects = {mean_h:.2f}, overall mean = {mean_a:.2f}, diff = {diff:.2f}")
        else:
            print("No distinguishable characteristics found.")

        # Correlations
        corr_def = compute_correlations(X_all, y_def)
        sorted_corr_def = sorted(corr_def.items(), key=lambda kv: abs(kv[1]), reverse=True)
        print("\nTop 5 features by Pearson correlation with defects:")
        for i, (col, r) in enumerate(sorted_corr_def[:5], 1):
            print(f"{i}. {col}: r = {r:.3f}")

    print()

if __name__ == "__main__":
    main()
