#!/usr/bin/env python3
"""
skill_09.py

Adds two new requests to the Apache analysis toolkit:
  - APACHE-17: Identify configurations whose recorded performance values are
               unusually different from what their settings and comparable historical
               configurations would lead us to expect.
  - APACHE-18: Flag configurations whose combinations of settings differ
               substantially from the most similar configurations in our historical
               benchmark records, even when their measured performance is not
               obviously poor.

The implementation re‑uses helper functions from the earlier skill modules.
"""

import sys
import json
import numpy as np
from skill_01 import read_csv, parse_column_values, compute_stats
from skill_02 import feature_matrix

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def hamming_distance(row1, row2, setting_cols):
    """Return Hamming distance between two rows on setting columns.
    Missing values ('?') are treated as 0 for distance calculation."""
    dist = 0
    for col in setting_cols:
        v1 = 1 if row1[col] == '1' else 0
        v2 = 1 if row2[col] == '1' else 0
        dist += abs(v1 - v2)
    return dist


def flag_unexpected_performance(rows, setting_cols, perf_col, max_flag=20):
    """
    Use a linear regression model to predict the performance for every row.
    Flag rows whose absolute residual exceeds 2 standard deviations of the residuals.
    Returns a list of up to max_flag dicts: {'row': int, 'column': str, 'actual': float, 'expected': float}
    """
    X = feature_matrix(rows, setting_cols)
    y = parse_column_values(rows, perf_col)

    # Train model on all available data
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return []  # not enough data

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X[mask], y[mask])
    preds = lr.predict(X)

    residuals = np.abs(y - preds)
    residual_mask = ~np.isnan(y) & ~np.isnan(residuals)
    if residual_mask.sum() == 0:
        return []

    res_std = np.std(residuals[residual_mask])
    if np.isnan(res_std) or res_std == 0:
        return []

    flagged = []
    for idx in range(len(rows)):
        if np.isnan(y[idx]):
            continue
        if np.abs(residuals[idx]) > 2 * res_std:
            flagged.append({
                "row": idx + 1,
                "column": perf_col,
                "actual": float(y[idx]),
                "expected": float(preds[idx])
            })
            if len(flagged) >= max_flag:
                break
    return flagged


def flag_unusual_settings(rows, setting_cols, max_flag=20):
    """
    For each row, compute the minimal Hamming distance to any other row.
    Rows with the largest such minimal distances are flagged as unusual.
    Returns a list of flagged row numbers (1‑based) and a dict of scores.
    """
    n = len(rows)
    min_dists = [0] * n
    for i in range(n):
        min_dist = None
        for j in range(n):
            if i == j:
                continue
            d = hamming_distance(rows[i], rows[j], setting_cols)
            if min_dist is None or d < min_dist:
                min_dist = d
        min_dists[i] = min_dist if min_dist is not None else 0

    # Pair (score, row)
    score_rows = [(min_dists[i], i + 1) for i in range(n)]
    # Sort by score descending
    score_rows.sort(key=lambda x: (-x[0], x[1]))
    flagged = [row for score, row in score_rows[:max_flag] if score > 0]
    scores = {row: score for score, row in score_rows if row in flagged}
    return flagged, scores


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, headers = read_csv(csv_path)

    # Identify goal columns (there may be more than one; choose first)
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    perf_col = goal_cols[0]

    # Identify setting columns
    setting_cols = [h for h in headers if not h.endswith('+') and not h.endswith('-') and not h.endswith('X')]

    # ---------- APACHE-17 ----------
    print("== APACHE-17 ==")
    flagged_perf = flag_unexpected_performance(rows, setting_cols, perf_col, max_flag=20)
    if not flagged_perf:
        print("No unusually different performances detected.")
    else:
        # Show first three flagged rows
        for fp in flagged_perf[:3]:
            print(f"Row {fp['row']}: actual={fp['actual']:.4f}, expected≈{fp['expected']:.4f}")
        print(f"Total flagged rows: {len(flagged_perf)}")
    # Machine‑checked answer
    print("ANSWER APACHE-17", json.dumps({"flagged": flagged_perf}, separators=(',', ':')))

    # ---------- APACHE-18 ----------
    print("\n== APACHE-18 ==")
    flagged_rows, scores = flag_unusual_settings(rows, setting_cols, max_flag=20)
    if not flagged_rows:
        print("No unusual input combinations detected.")
    else:
        # Show first three flagged rows
        for r in flagged_rows[:3]:
            print(f"Row {r}: unusualness score={scores[r]}")
        print(f"Total flagged rows: {len(flagged_rows)}")
    # Machine‑checked answer
    print("ANSWER APACHE-18", json.dumps({"flagged": flagged_rows, "scores": scores}, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
