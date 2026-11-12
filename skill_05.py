#!/usr/bin/env python3
"""
skill_05.py - Advanced outlier analysis (NASA‑09 and NASA‑10).

Usage:
    python skill_05.py <csv_path>
"""

import csv
import sys
import numpy as np
from statistics import mean, stdev, StatisticsError
from typing import List, Dict, Tuple, Set

# ----------------------------------------------------------------------
# Basic helpers (imported from skill_01 if available)
# ----------------------------------------------------------------------
try:
    from skill_01 import load_csv, get_numeric_column
except Exception:
    def load_csv(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = [row for row in reader]
        return rows, header

    def get_numeric_column(rows: List[Dict[str, str]], col: str) -> List[float]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (ValueError, TypeError):
                pass
        return vals

# ----------------------------------------------------------------------
# Feature preparation utilities
# ----------------------------------------------------------------------
def identify_columns(rows: List[Dict[str, str]], header: List[str]
                    ) -> Tuple[List[str], List[str]]:
    """Return ordered lists of numeric and categorical column names."""
    numeric: List[str] = []
    categorical: List[str] = []
    for col in header:
        if col == 'idX':
            continue
        sample: str | None = None
        for r in rows:
            val = r.get(col, '')
            if val != '':
                sample = val
                break
        if sample is None:
            categorical.append(col)
            continue
        try:
            float(sample)
            numeric.append(col)
        except ValueError:
            categorical.append(col)
    return numeric, categorical

def build_mappings(rows: List[Dict[str, str]], categorical: List[str]
                  ) -> Dict[str, Dict[str, int]]:
    """Create integer mapping dictionaries for each categorical column."""
    mappings: Dict[str, Dict[str, int]] = {}
    for col in categorical:
        mapping: Dict[str, int] = {}
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

def encode_row(r: Dict[str, str],
               numeric: List[str],
               categorical: List[str],
               mappings: Dict[str, Dict[str, int]]) -> List[float]:
    """Encode a single row into a numeric feature vector."""
    vec: List[float] = []
    for col in numeric:
        try:
            val = float(r.get(col, '0'))
        except ValueError:
            val = 0.0
        vec.append(val)
    for col in categorical:
        val = r.get(col, '')
        mapping = mappings[col]
        vec.append(float(mapping.get(val, -1)))  # unseen categories map to -1
    return vec

def prepare_data_with_names(rows: List[Dict[str, str]],
                            header: List[str],
                            target_col: str
                           ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return X, y and the ordered list of feature names."""
    numeric, categorical = identify_columns(rows, header)

    # Remove target and id from features
    if target_col in numeric:
        numeric.remove(target_col)
    if target_col in categorical:
        categorical.remove(target_col)
    if 'idX' in numeric:
        numeric.remove('idX')
    if 'idX' in categorical:
        categorical.remove('idX')

    mappings = build_mappings(rows, categorical)

    X_list: List[List[float]] = []
    y_list: List[float] = []
    for r in rows:
        try:
            y_val = float(r.get(target_col, '0'))
        except ValueError:
            continue
        X_list.append(encode_row(r, numeric, categorical, mappings))
        y_list.append(y_val)

    # Build feature name list: numeric first, then categorical
    feature_names = numeric + categorical
    return np.array(X_list, dtype=float), np.array(y_list, dtype=float), feature_names

# ----------------------------------------------------------------------
# Statistical helpers
# ----------------------------------------------------------------------
def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Return mean and standard deviation (0 if insufficient data)."""
    if not values:
        return (float('nan'), float('nan'))
    try:
        m = mean(values)
        s = stdev(values) if len(values) > 1 else 0.0
    except StatisticsError:
        m, s = float('nan'), float('nan')
    return m, s

# ----------------------------------------------------------------------
# NASA‑09: High effort analysis
# ----------------------------------------------------------------------
def nasa_09(rows: List[Dict[str, str]], header: List[str]) -> None:
    """Identify characteristics most associated with unusually high effort."""
    print("\n== NASA-09 ==")

    target_col = 'Effort-'
    values = get_numeric_column(rows, target_col)
    if not values:
        print("No effort data available.")
        return

    # Define 'unusually high' as > mean + 2*std
    m, s = compute_mean_std(values)
    threshold = m + 2 * s
    high_projects = [r for r in rows if float(r.get(target_col, '0')) >= threshold]
    print(f"Threshold for high effort: {threshold:.2f}")
    print(f"Number of projects exceeding threshold: {len(high_projects)}")

    # Build regression to discover strong associations
    X, y, feat_names = prepare_data_with_names(rows, header, target_col)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    coeffs = model.coef_

    # Pair features with coefficients, sort by absolute magnitude
    feat_coeff = list(zip(feat_names, coeffs))
    feat_coeff.sort(key=lambda x: abs(x[1]), reverse=True)

    top_n = 5
    print(f"Top {top_n} features most strongly associated with effort:")
    for name, coeff in feat_coeff[:top_n]:
        print(f"  {name:>12} : coefficient = {coeff:+.4f}")

# ----------------------------------------------------------------------
# NASA‑10: High defect analysis
# ----------------------------------------------------------------------
def nasa_10(rows: List[Dict[str, str]], header: List[str]) -> None:
    """Identify characteristics most associated with unusually high defect counts."""
    print("\n== NASA-10 ==")

    target_col = 'Defects-'
    values = get_numeric_column(rows, target_col)
    if not values:
        print("No defect data available.")
        return

    # Define 'unusually high' as > mean + 2*std
    m, s = compute_mean_std(values)
    threshold = m + 2 * s
    high_projects = [r for r in rows if float(r.get(target_col, '0')) >= threshold]
    print(f"Threshold for high defects: {threshold:.2f}")
    print(f"Number of projects exceeding threshold: {len(high_projects)}")

    # Regression analysis
    X, y, feat_names = prepare_data_with_names(rows, header, target_col)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    coeffs = model.coef_

    feat_coeff = list(zip(feat_names, coeffs))
    feat_coeff.sort(key=lambda x: abs(x[1]), reverse=True)

    top_n = 5
    print(f"Top {top_n} features most strongly associated with defects:")
    for name, coeff in feat_coeff[:top_n]:
        print(f"  {name:>12} : coefficient = {coeff:+.4f}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, header = load_csv(csv_path)

    nasa_09(rows, header)
    nasa_10(rows, header)

if __name__ == "__main__":
    main()
