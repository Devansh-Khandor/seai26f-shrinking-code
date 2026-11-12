#!/usr/bin/env python3
"""
skill_06.py – Advanced explanation of recommended and contrasting Apache configurations.

This module builds on the existing utilities (skill_01, skill_02, skill_03, skill_04, skill_05)
to provide the following analyses:

APACHE-11: For each recommended (favorable) configuration, explain the recommendation
by comparing its settings and performance to the most similar previously tested
configuration.

APACHE-12: For a pair of configurations with contrasting performance assessments,
identify the differing settings and relate the differences to the recorded and
predicted outcomes.
"""

import sys
import csv
import itertools
import numpy as np
import skill_01 as s1
import skill_02 as s2

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_csv(path):
    """Delegate to skill_01 for consistency."""
    return s1.load_csv(path)

def feature_matrix(rows, headers):
    """Return numpy array of features for each row (excluding the target)."""
    return s2.feature_matrix(rows, headers)[0], s2.feature_matrix(rows, headers)[1]

def target_vector(rows, headers):
    """Return numpy array of target values (Performance-)"""
    return s2.target_vector(rows, headers)

def train_model(X, y):
    """Train and return a linear regression model."""
    return s2.train_regressor(X, y)

def hamming_distance(row_a, row_b, feature_cols):
    """Return count and list of columns where two rows differ."""
    diff_cols = [col for col in feature_cols if row_a[col] != row_b[col]]
    return len(diff_cols), diff_cols

def find_nearest(row, rows, headers, feature_cols):
    """Find the index of the most similar configuration to `row`."""
    min_dist = None
    best_idx = None
    best_diff_cols = None
    for idx, candidate in enumerate(rows):
        if candidate is row:
            continue
        dist, diff = hamming_distance(row, candidate, feature_cols)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            best_idx = idx
            best_diff_cols = diff
        if min_dist == 0:
            break  # perfect match
    return best_idx, best_diff_cols, min_dist

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows, headers = load_csv(csv_path)

    # Build feature matrix and target vector for the regression model
    X, feature_cols = feature_matrix(rows, headers)
    y = target_vector(rows, headers)
    model = train_model(X, y)

    # ------------------------------------------------------------------
    # APACHE-11: Explain recommendations (favorable configurations)
    # ------------------------------------------------------------------
    print("== APACHE-11 ==")

    perf_values = [int(row['Performance-']) for row in rows]
    fav_rows, fav_thresh = s1.find_favorable(rows, perf_values)

    if not fav_rows:
        print("  No favorable configurations identified.")
    else:
        # Show up to two recommendations for brevity
        for idx, fav_row in enumerate(fav_rows[:2], start=1):
            fav_idx = rows.index(fav_row)
            fav_perf = int(fav_row['Performance-'])

            # Find most similar configuration
            best_idx, diff_cols, dist = find_nearest(fav_row, rows, headers, feature_cols)
            best_row = rows[best_idx]
            best_perf = int(best_row['Performance-'])
            best_pred = model.predict(X[best_idx].reshape(1, -1))[0]
            fav_pred = model.predict(X[fav_idx].reshape(1, -1))[0]

            # Explanation
            print(f"\nRecommendation #{idx}:")
            print(f"  Configuration #{fav_idx + 1} (Performance={fav_perf})")
            print(f"  Settings that differ from most similar config:")
            if diff_cols:
                for col in diff_cols:
                    print(f"    {col}: {fav_row[col]} (vs {best_row[col]})")
            else:
                print("    No differing settings – identical to the most similar config.")
            print(f"  Most similar configuration #{best_idx + 1} has Performance={best_perf}.")
            print(f"  Actual performance difference: {abs(fav_perf - best_perf)}.")
            print(f"  Predicted performance difference: {abs(fav_pred - best_pred):.2f}.")
            print(f"  Threshold for recommendation: {fav_thresh:.2f}.")

    # ------------------------------------------------------------------
    # APACHE-12: Contrast two configurations with different assessments
    # ------------------------------------------------------------------
    print("\n== APACHE-12 ==")

    # Identify the best (lowest performance value) favorable config
    if fav_rows:
        best_fav_row = min(fav_rows, key=lambda r: int(r['Performance-']))
        best_fav_idx = rows.index(best_fav_row)
        best_fav_perf = int(best_fav_row['Performance-'])
    else:
        best_fav_row = None

    # Identify the worst (highest performance value) poor config
    mean_perf = np.mean(perf_values)
    std_perf = np.std(perf_values, ddof=0)
    poor_thresh = mean_perf + 2.0 * std_perf
    poor_rows = [row for row in rows if int(row['Performance-']) > poor_thresh]
    if poor_rows:
        worst_poor_row = max(poor_rows, key=lambda r: int(r['Performance-']))
        worst_poor_idx = rows.index(worst_poor_row)
        worst_poor_perf = int(worst_poor_row['Performance-'])
    else:
        worst_poor_row = None

    if best_fav_row and worst_poor_row:
        dist, diff_cols = hamming_distance(best_fav_row, worst_poor_row, feature_cols)
        print(f"\nContrasting configurations:")
        print(f"  Best favorable config #{best_fav_idx + 1} (Performance={best_fav_perf})")
        print(f"  Worst poor config #{worst_poor_idx + 1} (Performance={worst_poor_perf})")
        print(f"  They differ in {dist} setting(s):")
        if diff_cols:
            for col in diff_cols:
                print(f"    {col}: favorable={best_fav_row[col]} vs poor={worst_poor_row[col]}")
        else:
            print("    No differing settings – identical configurations.")
        # Predicted difference
        pred_diff = abs(
            model.predict(X[worst_poor_idx].reshape(1, -1))[0] -
            model.predict(X[best_fav_idx].reshape(1, -1))[0]
        )
        print(f"  Observed performance difference: {abs(best_fav_perf - worst_poor_perf)}.")
        print(f"  Predicted performance difference: {pred_diff:.2f}.")
        print(f"  Z-score threshold for poor classification: {2.0}σ above mean.")
    else:
        print("  Insufficient data to contrast favorable and poor configurations.")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_06.py <csv_path>")
    main(sys.argv[1])
