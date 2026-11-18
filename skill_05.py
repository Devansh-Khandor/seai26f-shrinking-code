#!/usr/bin/env python3
"""
Small data‑analysis skill for the auto93 dataset (CAR‑09 and CAR‑10).

Usage:
    python skill_05.py <path_to_csv>
"""

import sys
import json
import numpy as np

# Import shared utilities
from skill_01 import load_data, GOAL_COLUMNS

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def compute_all_scores(data):
    """
    Compute a score for every row that has all goal values present.
    Returns a list of (row_number, score) tuples.
    The scoring method is identical to that used in skill_01.compute_overall_top10.
    """
    # Gather values per column for mean/std computation
    all_vals = {col: [] for col in GOAL_COLUMNS}
    for row in data:
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            if val is not None:
                all_vals[col].append(val)

    means = {col: np.mean(all_vals[col]) for col in GOAL_COLUMNS}
    stds = {col: np.std(all_vals[col], ddof=0) for col in GOAL_COLUMNS}

    scores = []
    for row in data:
        # Skip rows with any missing goal value
        if any(row["values"][col] is None for col in GOAL_COLUMNS):
            continue
        score = 0.0
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            std = stds[col]
            z = (val - means[col]) / std if std != 0 else 0.0
            if col.endswith("-"):   # lower is better
                score -= z
            else:                    # higher is better
                score += z
        scores.append((row["row"], score))
    return scores

def select_bottom_group(data, col_name, pct=0.2, min_rows=5):
    """
    Return a list of row numbers that are in the bottom `pct` of the values
    of column `col_name`.  At least `min_rows` rows are returned (if available).
    Missing values are ignored.
    """
    vals = [(row["row"], row["values"][col_name]) for row in data
            if row["values"][col_name] is not None]
    if not vals:
        return []

    # Sort by value ascending (lower is worse for Mpg+)
    vals.sort(key=lambda x: x[1])
    n = max(min_rows, int(np.floor(len(vals) * pct)))
    n = min(n, len(vals))  # never exceed available
    return [row_num for row_num, _ in vals[:n]]

def compute_feature_correlations(data, poor_mask, feature_cols):
    """
    Compute absolute Pearson correlation between each feature and the
    binary poor indicator.
    Return a list of (feature, abs_corr) sorted descending.
    """
    results = []
    for feat in feature_cols:
        feat_vals = []
        target = []
        for row, is_poor in zip(data, poor_mask):
            f = row["values"][feat]
            if f is None or row["values"]["Mpg+"] is None:
                continue
            feat_vals.append(f)
            target.append(is_poor)
        if len(feat_vals) < 2:
            continue
        corr = np.corrcoef(feat_vals, target)[0, 1]
        results.append((feat, abs(corr)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def group_tendencies(data, group_rows, feature_cols):
    """
    For each feature, compute the mean value within the group
    (ignoring missing values). Return a dict mapping feature -> mean.
    """
    means = {}
    for feat in feature_cols:
        vals = [row["values"][feat] for row in group_rows
                if row["values"][feat] is not None]
        if vals:
            means[feat] = float(np.mean(vals))
        else:
            means[feat] = None
    return means

# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
def main(csv_path):
    data = load_data(csv_path)
    total_rows = len(data)

    # --------------------------------------------------------------
    # CAR-09: Poor fuel efficiency analysis
    # --------------------------------------------------------------
    print("== CAR-09 ==")

    # Identify rows with unusually poor Mpg+
    poor_rows = select_bottom_group(data, "Mpg+", pct=0.2, min_rows=5)
    # Create binary mask for correlation calculation
    poor_mask = [row["row"] in poor_rows for row in data]

    # Non‑goal columns
    non_goal_cols = [col for col in data[0]["values"].keys()
                     if col not in GOAL_COLUMNS]
    # Compute correlations
    corr_results = compute_feature_correlations(data, poor_mask, non_goal_cols)
    ranking = [feat for feat, _ in corr_results[:3]]  # top 3

    # Output some information
    print(f"Poor Mpg+ rows (bottom 20%): {poor_rows[:5]} ...")
    print(f"Top 3 correlated features: {ranking}")

    # Machine‑checked answer
    answer_json = json.dumps({"poor": poor_rows,
                              "ranking": ranking},
                             separators=(",", ":"))
    print(f"ANSWER CAR-09 {answer_json}")

    # --------------------------------------------------------------
    # CAR-10: Bad vs good overall performance
    # --------------------------------------------------------------
    print("\n== CAR-10 ==")

    # Compute overall scores
    all_scores = compute_all_scores(data)
    # Sort descending (higher score = better)
    all_scores.sort(key=lambda x: x[1], reverse=True)

    # Determine bad and good groups (bottom/top 20%, min 5 rows)
    n_groups = max(5, int(np.floor(total_rows * 0.2)))
    bad_rows = [row_num for row_num, _ in all_scores[-n_groups:]]
    good_rows = [row_num for row_num, _ in all_scores[:n_groups]]

    # Build row dictionaries for easy lookup
    row_dict = {row["row"]: row for row in data}

    # Extract group data
    bad_rows_data = [row_dict[r] for r in bad_rows]
    good_rows_data = [row_dict[r] for r in good_rows]

    # Compute typical values (means) for each non‑goal feature
    bad_means = group_tendencies(bad_rows_data, bad_rows_data, non_goal_cols)
    good_means = group_tendencies(good_rows_data, good_rows_data, non_goal_cols)

    # Compute differences and sort
    differences = []
    for feat in non_goal_cols:
        bval = bad_means.get(feat)
        gval = good_means.get(feat)
        if bval is None or gval is None:
            continue
        diff = abs(bval - gval)
        differences.append({"feature": feat,
                            "bad": bval,
                            "good": gval,
                            "diff": diff})
    differences.sort(key=lambda x: x["diff"], reverse=True)
    diff_output = [{"feature": d["feature"], "bad": d["bad"], "good": d["good"]}
                   for d in differences[:3]]  # top 3

    # Output some information
    print(f"Bad performance rows (bottom 20%): {bad_rows[:5]} ...")
    print(f"Good performance rows (top 20%): {good_rows[:5]} ...")
    print(f"Top 3 distinguishing features: {diff_output}")

    # Machine‑checked answer
    answer_json = json.dumps({"bad": bad_rows,
                              "good": good_rows,
                              "differences": diff_output},
                             separators=(",", ":"))
    print(f"ANSWER CAR-10 {answer_json}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
