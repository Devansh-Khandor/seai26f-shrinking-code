#!/usr/bin/env python3
"""
skill_09.py
------------
Command line tool that reads a CSV file of NASA software project data
and answers two new requests:

    NASA-17: Flag historical projects whose combinations of project characteristics are unusually different from the rest of our records so that we can review them.
    NASA-18: Identify project records whose reported effort or defect counts differ substantially from what comparable projects would lead us to expect, so we can investigate possible measurement or recording issues.

Usage:
    python skill_09.py <csv path>
"""

import csv
import json
import sys
import statistics

# --------------------------------------------------------------------------- #
# Import helpers from skill_02
# --------------------------------------------------------------------------- #
from skill_02 import (
    load_csv,
    parse_float,
    identify_features,
    build_pipeline,
)

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def numeric_stats(rows, numeric_indices):
    """
    Compute mean and std for each numeric column (by index).
    Returns two dicts: {col_index: mean, {col_index: std}}
    """
    sums = {}
    sums_sq = {}
    counts = {}
    for row in rows:
        for idx in numeric_indices:
            val = row[idx]
            if val == "?":
                continue
            f = float(val)
            sums[idx] = sums.get(idx, 0.0) + f
            sums_sq[idx] = sums_sq.get(idx, 0.0) + f * f
            counts[idx] = counts.get(idx, 0) + 1

    means = {idx: sums[idx] / counts[idx] for idx in sums}
    stds = {}
    for idx in sums:
        n = counts[idx]
        if n < 2:
            stds[idx] = 0.0
        else:
            mean = means[idx]
            var = (sums_sq[idx] - n * mean * mean) / (n - 1)
            stds[idx] = var ** 0.5 if var > 0 else 0.0
    return means, stds

def compute_unusualness(rows, numeric_indices, means, stds):
    """
    Compute an unusualness score for each row based on absolute z‑scores
    of numeric features.  Missing numeric values are ignored for that
    row.  Return a dict mapping row number (1‑based) to score.
    """
    scores = {}
    for row_num, row in enumerate(rows, start=1):
        zsum = 0.0
        zcount = 0
        for idx in numeric_indices:
            val = row[idx]
            if val == "?":
                continue
            f = float(val)
            std = stds.get(idx, 0.0)
            if std == 0.0:
                continue  # skip feature with zero variance
            z = abs((f - means[idx]) / std)
            zsum += z
            zcount += 1
        score = zsum / zcount if zcount > 0 else 0.0
        scores[row_num] = score
    return scores

def train_model(rows, goal_col, numeric_indices, categorical_indices):
    """
    Train a linear regression model to predict goal_col.
    Return the fitted Pipeline and the array of predictions on all rows.
    """
    # Build feature matrix
    feature_names, _, _ = identify_features(rows, ["Kloc+", "Effort-", "Defects-"])
    # The feature_names returned by identify_features already excludes goal columns
    # But we need them in the order they appear in the original CSV
    # We'll reconstruct it from the header
    header = rows[0].keys()
    # Build list of feature names in header order excluding goals and X columns
    all_cols = list(header)
    feature_names = [c for c in all_cols if c not in ["Kloc+", "Effort-", "Defects-"] and not c.endswith("X")]

    X_all = build_pipeline(numeric_indices, categorical_indices).named_steps['preprocessor'].fit_transform(
        build_pipeline(numeric_indices, categorical_indices).named_steps['preprocessor'].transform(
            build_pipeline(numeric_indices, categorical_indices).named_steps['preprocessor'].transform(
                build_pipeline(numeric_indices, categorical_indices).named_steps['preprocessor'].transform(
                    build_pipeline(numeric_indices, categorical_indices).named_steps['preprocessor'].transform(
                        None
                    )
                )
            )
        )
    )

    # The above is too convoluted. Instead we build the matrix manually.
    X_all = []
    for r in rows:
        vals = []
        for col in feature_names:
            val = r[col]
            if val == "?":
                vals.append(np.nan)
            else:
                vals.append(val)
        X_all.append(vals)
    X_all = np.array(X_all, dtype=object)

    # Numeric columns are already at the same indices as numeric_indices
    if numeric_indices:
        X_all[:, numeric_indices] = X_all[:, numeric_indices].astype(float, copy=False)

    # Build pipeline
    pipe = build_pipeline(numeric_indices, categorical_indices)

    # Target vector
    y_all = [parse_float(r[goal_col]) for r in rows]
    y_all = [float(v) if v is not None else float('nan') for v in y_all]

    train_mask = ~np.isnan(y_all)
    X_train = X_all[train_mask]
    y_train = np.array([y_all[i] for i, m in enumerate(train_mask) if m])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_all).reshape(-1)

    return pipe, preds

# --------------------------------------------------------------------------- #
# Main logic for NASA-17 and NASA-18
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    if not rows:
        print("No data rows found.")
        sys.exit(1)

    # --------------------------------------------------------------------- #
    # NASA-17: Flag unusual feature combinations
    # --------------------------------------------------------------------- #
    print("== NASA-17 ==")

    # Identify numeric and categorical columns
    header = rows[0].keys()
    goal_cols = ["Kloc+", "Effort-", "Defects-"]
    # Filter out goal columns and X columns to get feature names
    feature_names = [c for c in header if c not in goal_cols and not c.endswith("X")]

    # Determine indices of numeric columns in feature_names
    numeric_indices = []
    categorical_indices = []
    for idx, col in enumerate(feature_names):
        # Check if column is numeric by attempting to parse first non‑missing value
        val = None
        for r in rows:
            if r[col] != "?":
                val = r[col]
                break
        if val is None:
            categorical_indices.append(idx)
            continue
        if parse_float(val) is not None:
            numeric_indices.append(idx)
        else:
            categorical_indices.append(idx)

    # Compute means and stds for numeric features
    means, stds = numeric_stats(rows, numeric_indices)

    # Compute unusualness scores
    scores_dict = compute_unusualness(rows, numeric_indices, means, stds)

    # Flag top 10 rows by score
    flagged_rows = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)[:10]
    flagged_nums = [row for row, _ in flagged_rows]
    scores_json = {str(row): round(v, 4) for row, v in flagged_rows}

    # Print some results
    print(f"Total rows: {len(rows)}")
    print(f"Top 10 flagged rows: {flagged_nums}")
    print(f"Scores of flagged rows: {scores_json}")

    # Answer line
    answer_17 = {"flagged": flagged_nums, "scores": scores_json}
    print(f"ANSWER NASA-17 {json.dumps(answer_17, separators=(',', ':'))}")

    # --------------------------------------------------------------------- #
    # NASA-18: Flag large residuals between actual and expected effort/defects
    # --------------------------------------------------------------------- #
    print("\n== NASA-18 ==")

    # Prepare indices for pipeline
    # The helper identify_features already gives us indices relative to the feature_names list
    feature_names, numeric_indices, categorical_indices = identify_features(rows, goal_cols)

    # Train models for Effort- and Defects-
    import numpy as np  # local import to avoid global namespace
    # Build feature matrix function reimplemented here for convenience
    def build_feature_matrix(rows, feature_names):
        data = []
        for r in rows:
            row_vals = []
            for col in feature_names:
                v = r[col]
                if v == "?":
                    row_vals.append(float("nan"))
                else:
                    row_vals.append(v)
            data.append(row_vals)
        return np.array(data, dtype=object)

    X_all = build_feature_matrix(rows, feature_names)
    if numeric_indices:
        X_all[:, numeric_indices] = X_all[:, numeric_indices].astype(float, copy=False)

    # Effort model
    y_effort = np.array([parse_float(r["Effort-"]) for r in rows], dtype=float)
    train_mask_effort = ~np.isnan(y_effort)
    pipe_effort = build_pipeline(numeric_indices, categorical_indices)
    pipe_effort.fit(X_all[train_mask_effort], y_effort[train_mask_effort])
    preds_effort = pipe_effort.predict(X_all).reshape(-1)

    # Defects model
    y_defects = np.array([parse_float(r["Defects-"]) for r in rows], dtype=float)
    train_mask_defects = ~np.isnan(y_defects)
    pipe_defects = build_pipeline(numeric_indices, categorical_indices)
    pipe_defects.fit(X_all[train_mask_defects], y_defects[train_mask_defects])
    preds_defects = pipe_defects.predict(X_all).reshape(-1)

    # Compute residuals for rows with actual values
    residuals = []
    for i, r in enumerate(rows, start=1):
        actual_effort = parse_float(r["Effort-"])
        if actual_effort is not None:
            res = abs(preds_effort[i-1] - actual_effort)
            residuals.append((i, "Effort-", actual_effort, round(float(preds_effort[i-1]), 4), res))
        actual_defects = parse_float(r["Defects-"])
        if actual_defects is not None:
            res = abs(preds_defects[i-1] - actual_defects)
            residuals.append((i, "Defects-", actual_defects, round(float(preds_defects[i-1]), 4), res))

    # Take top 10 largest residuals
    residuals.sort(key=lambda kv: kv[4], reverse=True)
    flagged = []
    for row, col, actual, expected, _ in residuals[:10]:
        flagged.append({"row": row, "column": col, "actual": actual, "expected": expected})

    # Print some results
    print(f"Total rows processed: {len(rows)}")
    print(f"Top 10 flagged rows: {flagged}")

    # Answer line
    answer_18 = {"flagged": flagged}
    print(f"ANSWER NASA-18 {json.dumps(answer_18, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
