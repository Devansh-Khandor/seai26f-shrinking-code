#!/usr/bin/env python3
"""
Skill 07: NASA-13 and NASA-14 – Identify promising project characteristics and
prioritize proposals under limited evaluation budget.

Author: Your Name
"""

import csv
import math
import sys
import itertools
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Basic CSV reading – numeric conversion where possible
# ----------------------------------------------------------------------
def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                v = v.strip() if v is not None else ''
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return header, rows

# ----------------------------------------------------------------------
# Column classification
# ----------------------------------------------------------------------
def classify_columns(rows):
    numeric_cols = []
    categorical_cols = []
    for key in rows[0].keys():
        # skip identifiers
        if key in ('idX', 'YearX', 'MonthsX'):
            continue
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        if all(isinstance(v, (int, float)) for v in vals):
            numeric_cols.append(key)
        else:
            categorical_cols.append(key)
    return numeric_cols, categorical_cols

# ----------------------------------------------------------------------
# Group statistics for categorical combinations
# ----------------------------------------------------------------------
def group_stats(rows, group_cols, target_cols):
    """
    Compute mean and std for each target column within groups defined
    by values of group_cols (a tuple of column names).
    Returns a dict mapping group tuple -> {target_col: (mean, std, n)}.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = tuple(r[col] for col in group_cols)
        for t in target_cols:
            val = r.get(t)
            if isinstance(val, (int, float)):
                groups[key][t].append(val)

    stats = {}
    for key, targets in groups.items():
        stats[key] = {}
        for t, vals in targets.items():
            n = len(vals)
            if n == 0:
                continue
            mean = sum(vals) / n
            if n > 1:
                std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1))
            else:
                std = 0.0
            stats[key][t] = (mean, std, n)
    return stats

# ----------------------------------------------------------------------
# Compute z-scores for a list of numeric values
# ----------------------------------------------------------------------
def z_scores(values):
    if not values:
        return []
    m = sum(values) / len(values)
    if len(values) > 1:
        s = math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))
    else:
        s = 0.0
    return [0.0 if s == 0 else (x - m) / s for x in values]

# ----------------------------------------------------------------------
# NASA-13 – Identify promising characteristic combinations
# ----------------------------------------------------------------------
def nasa_13(rows, numeric_cols, categorical_cols):
    # Focus on categorical columns
    if len(categorical_cols) < 2:
        print("== NASA-13 ==")
        print("Not enough categorical columns to form combinations.")
        print()
        return

    target_cols = ('Kloc+', 'Effort-', 'Defects-')
    # Consider all pairs of categorical columns
    comb_stats = {}
    for pair in itertools.combinations(categorical_cols, 2):
        stats = group_stats(rows, pair, target_cols)
        if not stats:
            continue

        # Compute scores for each group: high size, low effort/size, low defects/size
        group_scores = {}
        size_values = [v[0] for v in stats.values()]
        eff_values = [v['Effort-'][0] for v in stats.values()]
        def_values = [v['Defects-'][0] for v in stats.values()]
        size_z = z_scores(size_values)
        eff_z = z_scores(eff_values)
        def_z = z_scores(def_values)

        for idx, key in enumerate(stats.keys()):
            size, _, _ = stats[key]['Kloc+']
            effort, _, _ = stats[key]['Effort-']
            defects, _, _ = stats[key]['Defects-']
            eff_per_k = effort / size if size else math.inf
            def_per_k = defects / size if size else math.inf
            # We also compute z for effort and defects per size
            # Use raw values for simplicity
            score = size_z[idx] - (eff_per_k * 0.5) - (def_per_k * 0.5)
            group_scores[key] = score

        if group_scores:
            best_group = max(group_scores, key=group_scores.get)
            comb_stats[pair] = (best_group, group_scores[best_group])

    if not comb_stats:
        print("== NASA-13 ==")
        print("No significant characteristic combinations found.")
        print()
        return

    # Sort by score
    sorted_combs = sorted(comb_stats.items(), key=lambda x: x[1][1], reverse=True)

    print("== NASA-13 ==")
    print("Top 3 promising combinations of characteristics (based on size, effort, defects):")
    for i, ((col1, col2), (group, score)) in enumerate(sorted_combs[:3], start=1):
        val1, val2 = group
        print(f"{i}. {col1}={val1}, {col2}={val2} | Score={score:.3f}")
    print()

# ----------------------------------------------------------------------
# Build feature matrix for regression
# ----------------------------------------------------------------------
def build_matrix(rows, target_col, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    numeric = [c for c in rows[0].keys()
               if c not in exclude_cols and c != target_col
               and all(isinstance(r.get(c, None), (int, float)) or r.get(c, None) is None
                       for r in rows)]
    X_rows = []
    y_rows = []
    for r in rows:
        tgt = r.get(target_col)
        if not isinstance(tgt, (int, float)):
            continue
        features = []
        missing = False
        for c in numeric:
            val = r.get(c)
            if isinstance(val, (int, float)):
                features.append(val)
            else:
                missing = True
                break
        if missing:
            continue
        X_rows.append(features)
        y_rows.append(tgt)
    if not X_rows:
        raise ValueError(f"No valid rows for target {target_col}")
    return np.array(X_rows), np.array(y_rows), numeric

# ----------------------------------------------------------------------
# NASA-14 – Prioritization of proposals under limited evaluation
# ----------------------------------------------------------------------
def nasa_14(rows, numeric_cols):
    # Train simple linear regression models for effort and defects
    try:
        X_eff, y_eff, feats_eff = build_matrix(rows, 'Effort-', exclude_cols=['idX', 'YearX', 'MonthsX'])
        model_eff = LinearRegression().fit(X_eff, y_eff)
        X_def, y_def, feats_def = build_matrix(rows, 'Defects-', exclude_cols=['idX', 'YearX', 'MonthsX'])
        model_def = LinearRegression().fit(X_def, y_def)
    except Exception as e:
        print("== NASA-14 ==")
        print(f"Unable to train predictive models: {e}")
        print()
        return

    # Compute predicted effort and defects for each project
    scores = []
    for r in rows:
        # Prepare feature vectors
        X_eff_row, _, _ = build_matrix([r], 'Effort-', exclude_cols=['idX', 'YearX', 'MonthsX'])
        X_def_row, _, _ = build_matrix([r], 'Defects-', exclude_cols=['idX', 'YearX', 'MonthsX'])
        pred_eff = model_eff.predict(X_eff_row)[0]
        pred_def = model_def.predict(X_def_row)[0]
        size = r.get('Kloc+', 0)
        if size <= 0:
            continue
        # Benefit metric: larger size, smaller effort & defects
        benefit = size / (pred_eff * (pred_def + 1e-6))  # add tiny to avoid div by zero
        scores.append((benefit, r))

    if not scores:
        print("== NASA-14 ==")
        print("No valid proposals for prioritization.")
        print()
        return

    scores.sort(reverse=True, key=lambda x: x[0])

    print("== NASA-14 ==")
    print("Top 5 proposals to evaluate first (based on predicted benefit):")
    for i, (ben, r) in enumerate(scores[:5], start=1):
        print(f"{i}. ID={int(r.get('idX',0))} | Size={r.get('Kloc+',0):.1f} | "
              f"Effort={r.get('Effort-',0):.1f} | Defects={r.get('Defects-',0):.1f} | "
              f"Benefit={ben:.3f}")
    print()

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

    numeric_cols, categorical_cols = classify_columns(rows)

    try:
        nasa_13(rows, numeric_cols, categorical_cols)
    except Exception as e:
        print(f"NASA-13 error: {e}", file=sys.stderr)

    try:
        nasa_14(rows, numeric_cols)
    except Exception as e:
        print(f"NASA-14 error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
