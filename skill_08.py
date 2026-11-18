#!/usr/bin/env python3
"""
Skill 08: Recommendations and minimal changes for poor configurations
- APACHE-15: Suggest a small change to a poorly performing configuration that could
  reduce its predicted performance, using observed alternatives.
- APACHE-16: Find the minimal set of setting changes that moves a poor
  configuration closer to a better performance target, while staying on
  observed valid configurations.
"""

import csv
import json
import sys
import math
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# CSV reading – identical to earlier skills
# ----------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == '?':
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v) if '.' in v else int(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append((idx, cleaned))
    return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Column identification
# ----------------------------------------------------------------------
def identify_columns(fieldnames):
    goal_columns = [c for c in fieldnames if c.endswith(('+', '-'))]
    setting_columns = [c for c in fieldnames
                       if c not in goal_columns and not c.endswith('X')]
    return goal_columns, setting_columns

# ----------------------------------------------------------------------
# Build regression on rows with known goal values
# ----------------------------------------------------------------------
def train_regression(rows, setting_columns, goal_col):
    # Use all rows that have a goal value
    train_rows = [(idx, r) for idx, r in rows if r[goal_col] is not None]
    # Means for imputation
    setting_means = {}
    for col in setting_columns:
        vals = [r[col] for _, r in train_rows if r[col] is not None]
        setting_means[col] = sum(vals) / len(vals) if vals else 0.0
    # Build X and y
    X, y = [], []
    for _, r in train_rows:
        vec = []
        for col in setting_columns:
            val = r[col]
            if val is None:
                val = setting_means[col]
            vec.append(val)
        X.append(vec)
        y.append(r[goal_col])
    if not X:
        raise RuntimeError("No training data available")
    lr = LinearRegression()
    lr.fit(X, y)
    return lr, setting_means

# ----------------------------------------------------------------------
# Predict a row using the model
# ----------------------------------------------------------------------
def predict_row(row, lr, means, setting_columns):
    vec = []
    for col in setting_columns:
        val = row[col]
        if val is None:
            val = means[col]
        vec.append(val)
    return lr.predict([vec])[0]

# ----------------------------------------------------------------------
# Find worst row (highest performance-)
# ----------------------------------------------------------------------
def find_worst(rows, goal_col):
    worst_val = None
    worst_row = None
    for idx, r in rows:
        val = r[goal_col]
        if val is None:
            continue
        if worst_val is None or val > worst_val:
            worst_val = val
            worst_row = (idx, r)
    return worst_row, worst_val

# ----------------------------------------------------------------------
# Find best row (lowest performance-)
# ----------------------------------------------------------------------
def find_best(rows, goal_col):
    best_val = None
    best_row = None
    for idx, r in rows:
        val = r[goal_col]
        if val is None:
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_row = (idx, r)
    return best_row, best_val

# ----------------------------------------------------------------------
# Helper: compute differing settings between two rows
# ----------------------------------------------------------------------
def diff_settings(row_a, row_b, setting_columns):
    diffs = []
    for col in setting_columns:
        va = row_a[col]
        vb = row_b[col]
        if va != vb:
            diffs.append(col)
    return diffs

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    goal_columns, setting_columns = identify_columns(fieldnames)
    if not goal_columns:
        print("Error: No goal column found.")
        sys.exit(1)
    goal_col = goal_columns[0]  # assume single goal

    # Train regression model once
    lr, means = train_regression(rows, setting_columns, goal_col)

    # ---------- APACHE-15 ----------
    print("== APACHE-15 ==")

    # Find worst (highest) row
    worst_row, worst_perf = find_worst(rows, goal_col)
    if worst_row is None:
        print("No worst row found.")
        sys.exit(1)
    worst_idx, worst_vals = worst_row
    print(f"Worst row: {worst_idx} (Performance-={worst_perf})")

    # Search alternatives that differ in 1 or 2 settings
    best_alt_idx = None
    best_alt_perf = None
    best_alt_row = None
    best_diff_count = None
    best_changes = None

    for idx, r in rows:
        if idx == worst_idx:
            continue
        diff_cols = diff_settings(worst_vals, r, setting_columns)
        if not (1 <= len(diff_cols) <= 2):
            continue
        # Predicted performance of alternative
        pred_perf = predict_row(r, lr, means, setting_columns)
        if pred_perf is None:
            continue
        # Improvement if lower
        if pred_perf < worst_perf:
            # Choose best (lowest predicted)
            if best_alt_perf is None or pred_perf < best_alt_perf or \
               (math.isclose(pred_perf, best_alt_perf) and len(diff_cols) < best_diff_count):
                best_alt_idx = idx
                best_alt_perf = pred_perf
                best_alt_row = r
                best_diff_count = len(diff_cols)
                best_changes = {col: [worst_vals[col], r[col]] for col in diff_cols}

    # If no improvement found, just pick the closest alternative
    if best_alt_idx is None:
        # fallback: pick row with minimal difference count and minimal predicted value
        for idx, r in rows:
            if idx == worst_idx:
                continue
            diff_cols = diff_settings(worst_vals, r, setting_columns)
            if not (1 <= len(diff_cols) <= 2):
                continue
            pred_perf = predict_row(r, lr, means, setting_columns)
            if pred_perf is None:
                continue
            if best_alt_perf is None or pred_perf < best_alt_perf or \
               (math.isclose(pred_perf, best_alt_perf) and len(diff_cols) < best_diff_count):
                best_alt_idx = idx
                best_alt_perf = pred_perf
                best_alt_row = r
                best_diff_count = len(diff_cols)
                best_changes = {col: [worst_vals[col], r[col]] for col in diff_cols}

    if best_alt_idx is None:
        print("No alternative configuration found.")
        sys.exit(1)

    print(f"Alternative row: {best_alt_idx}")
    print(f"Changes: {best_changes}")
    print(f"Predicted after: {round(best_alt_perf,6)}")

    ans_15 = {
        "row": worst_idx,
        "alternative": best_alt_idx,
        "changes": best_changes,
        "performance": {
            "before": round(worst_perf, 6),
            "after": round(best_alt_perf, 6)
        }
    }
    print(f"ANSWER APACHE-15 {json.dumps(ans_15)}")

    # ---------- APACHE-16 ----------
    print("\n== APACHE-16 ==")

    # Target: best (lowest) Performance-
    best_row, best_perf = find_best(rows, goal_col)
    if best_row is None:
        print("No best row found.")
        sys.exit(1)
    target_value = best_perf  # better than worst
    print(f"Target performance (best): {target_value}")

    # Find rows closer to target than worst
    candidates = []
    for idx, r in rows:
        if idx == worst_idx:
            continue
        val = r[goal_col]
        if val is None:
            continue
        # closer if |val - target| < |worst_perf - target|
        if abs(val - target_value) < abs(worst_perf - target_value):
            candidates.append((idx, r, val))

    if not candidates:
        print("No candidate closer to target than worst.")
        sys.exit(1)

    # Choose candidate with fewest differing settings
    best_cand_idx = None
    best_cand_val = None
    min_diff = None
    best_cand_row = None
    for idx, r, val in candidates:
        diffs = diff_settings(worst_vals, r, setting_columns)
        diff_count = len(diffs)
        if best_cand_idx is None or diff_count < min_diff:
            best_cand_idx = idx
            best_cand_val = val
            min_diff = diff_count
            best_cand_row = r

    print(f"Chosen alternative row: {best_cand_idx} (diff count={min_diff})")
    print(f"Performance after: {best_cand_val}")

    changed_settings = [col for col in setting_columns if worst_vals[col] != best_cand_row[col]]
    print(f"Changed settings: {changed_settings}")

    ans_16 = {
        "row": worst_idx,
        "target": round(target_value, 6),
        "alternative": best_cand_idx,
        "changed": changed_settings
    }
    print(f"ANSWER APACHE-16 {json.dumps(ans_16)}")

if __name__ == "__main__":
    main()
