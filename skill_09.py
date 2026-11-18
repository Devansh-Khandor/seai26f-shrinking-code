#!/usr/bin/env python3
"""
skill_09.py <csv_path>

Performs two NASA analysis requests on the provided CSV file:
NASA-17 – Flag historical projects whose combinations of project characteristics
          are unusually different from the rest of our records so that we can review them.
NASA-18 – Identify project records whose reported effort or defect counts differ
          substantially from what comparable projects would lead us to expect,
          so we can investigate possible measurement or recording issues.
"""

import sys
import csv
import json
import numpy as np
from sklearn.impute import SimpleImputer

# Reuse helpers from skill_06
from skill_06 import load_csv, get_feature_columns, parse_value

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def build_feature_matrix(rows, feature_cols):
    """Return imputed matrix and z‑score matrix for the given rows."""
    X = []
    for r in rows:
        X.append([r[c] if r[c] is not None else np.nan for c in feature_cols])
    X = np.array(X, dtype=float)
    imp = SimpleImputer(strategy='mean')
    X_imp = imp.fit_transform(X)
    means = np.mean(X_imp, axis=0)
    stds = np.std(X_imp, axis=0, ddof=0)
    # avoid division by zero
    stds[stds == 0] = 1.0
    Z = (X_imp - means) / stds
    return X_imp, Z

def euclidean_dist(z1, z2):
    return np.linalg.norm(z1 - z2)

def find_k_neighbors(zs, idx, k=3):
    """Return indices of the k nearest neighbors of row idx (excluding itself)."""
    dists = []
    target = zs[idx]
    for j, z in enumerate(zs):
        if j == idx:
            continue
        dists.append((euclidean_dist(target, z), j))
    dists.sort(key=lambda t: t[0])
    return [j for _, j in dists[:k]]

# --------------------------------------------------------------------------- #
# NASA-17: flag outlier feature combinations
# --------------------------------------------------------------------------- #

def nasa_17(rows, feature_cols, max_flag=10):
    _, Z = build_feature_matrix(rows, feature_cols)
    distances = [euclidean_dist(z, np.zeros_like(z)) for z in Z]
    idx_scores = list(enumerate(distances))
    idx_scores.sort(key=lambda t: -t[1])
    flagged = [rows[i]['row_num'] for i, _ in idx_scores[:max_flag]]
    scores = {rows[i]['row_num']: float(dist) for i, dist in idx_scores[:max_flag]}
    return flagged, scores

# --------------------------------------------------------------------------- #
# NASA-18: flag unexpected Effort- or Defects- values
# --------------------------------------------------------------------------- #

def nasa_18(rows, feature_cols, target_cols=('Effort-', 'Defects-'), max_flag=10):
    _, Z = build_feature_matrix(rows, feature_cols)
    # pre‑extract target values
    target_values = {col: np.array([r[col] if r[col] is not None else np.nan for r in rows]) for col in target_cols}
    flagged_entries = []

    for idx, r in enumerate(rows):
        # Skip rows with missing target
        for tcol in target_cols:
            actual = r[tcol]
            if actual is None:
                continue
            neighbors = find_k_neighbors(Z, idx, k=3)
            # gather neighbor target values
            neigh_vals = [target_values[tcol][j] for j in neighbors if not np.isnan(target_values[tcol][j])]
            if not neigh_vals:
                continue  # no valid neighbor data
            expected = float(np.mean(neigh_vals))
            diff = float(actual - expected)
            flagged_entries.append({
                "row": r['row_num'],
                "column": tcol,
                "actual": actual,
                "expected": expected
            })

    # sort by absolute difference
    flagged_entries.sort(key=lambda e: -abs(e['actual'] - e['expected']))
    return flagged_entries[:max_flag]

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, all_columns = load_csv(csv_path)

    # Determine feature columns: all non‑goal, non‑X columns
    feature_cols = get_feature_columns(all_columns)

    # NASA-17
    print("== NASA-17 ==")
    flagged_17, scores_17 = nasa_17(rows, feature_cols)
    print(f"Number of flagged projects: {len(flagged_17)}")
    print("Flagged row numbers:", flagged_17)
    print("Unusualness scores:", scores_17)
    print("ANSWER NASA-17", json.dumps({"flagged": flagged_17, "scores": scores_17}, ensure_ascii=False))

    # NASA-18
    print("\n== NASA-18 ==")
    flagged_18 = nasa_18(rows, feature_cols)
    print(f"Number of flagged projects: {len(flagged_18)}")
    for entry in flagged_18:
        print(f"Row {entry['row']} {entry['column']}: actual={entry['actual']}, expected={entry['expected']:.2f}")
    print("ANSWER NASA-18", json.dumps({"flagged": flagged_18}, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
