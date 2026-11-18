#!/usr/bin/env python3
"""
Skill 09: Anomaly detection in performance and setting space
- APACHE-17: Detect rows where the recorded Performance- is unusually far from what a regression model would predict.
- APACHE-18: Detect rows whose setting combination is unusually far from its nearest historical neighbour,
  even when their Performance- is not particularly poor.
"""

import csv
import json
import sys
import math
from collections import defaultdict
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------
# CSV reading – same logic as previous skills
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# Identify goal & setting columns
# --------------------------------------------------------------------
def identify_columns(fieldnames):
    goal_columns = [c for c in fieldnames if c.endswith(('+', '-'))]
    setting_columns = [c for c in fieldnames
                       if c not in goal_columns and not c.endswith('X')]
    return goal_columns, setting_columns

# --------------------------------------------------------------------
# Train a regression on all rows with known goal
# --------------------------------------------------------------------
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
    lr = LinearRegression()
    lr.fit(X, y)
    return lr, setting_means

# --------------------------------------------------------------------
# Predict a row using the model
# --------------------------------------------------------------------
def predict_row(row, lr, means, setting_columns):
    vec = []
    for col in setting_columns:
        val = row[col]
        if val is None:
            val = means[col]
        vec.append(val)
    return lr.predict([vec])[0]

# --------------------------------------------------------------------
# APACHE-17: Identify unusually different performances
# --------------------------------------------------------------------
def apache_17(rows, goal_col, setting_columns):
    # Train model once
    lr, means = train_regression(rows, setting_columns, goal_col)

    # Compute residuals for all rows with known goal
    residuals = {}
    for idx, r in rows:
        actual = r[goal_col]
        if actual is None:
            continue
        pred = predict_row(r, lr, means, setting_columns)
        residuals[idx] = actual - pred

    if not residuals:
        return []

    # Compute mean and std of residuals (mean ≈ 0)
    vals = list(residuals.values())
    mean_res = sum(vals) / len(vals)
    var = sum((v - mean_res) ** 2 for v in vals) / len(vals)
    std_res = math.sqrt(var)

    # Threshold: 2*std (can be tuned)
    thresh = 2 * std_res if std_res > 0 else 0

    flagged = []
    for idx, res in residuals.items():
        if abs(res) > thresh:
            flagged.append({
                "row": idx,
                "column": goal_col,
                "actual": round(rows[idx-1][1][goal_col], 6),
                "expected": round(predict_row(rows[idx-1][1], lr, means, setting_columns), 6),
                "residual": round(res, 6)
            })
        if len(flagged) >= 20:
            break
    return flagged

# --------------------------------------------------------------------
# APACHE-18: Identify unusual setting combinations
# --------------------------------------------------------------------
def apache_18(rows, setting_columns):
    # Precompute distances to nearest neighbour for each row
    distances = {}
    for i, (idx_a, r_a) in enumerate(rows):
        best_dist = None
        for idx_b, r_b in rows[i+1:]:
            # Compute Euclidean distance ignoring missing
            sum_sq = 0.0
            common = 0
            for col in setting_columns:
                va = r_a[col]
                vb = r_b[col]
                if va is None or vb is None:
                    continue
                diff = va - vb
                sum_sq += diff * diff
                common += 1
            if common == 0:
                continue
            dist = math.sqrt(sum_sq)
            if best_dist is None or dist < best_dist:
                best_dist = dist
        if best_dist is not None:
            distances[idx_a] = best_dist

    if not distances:
        return []

    # Compute mean & std of distances
    dvals = list(distances.values())
    mean_d = sum(dvals) / len(dvals)
    var_d = sum((v - mean_d) ** 2 for v in dvals) / len(dvals)
    std_d = math.sqrt(var_d)

    # Threshold: mean + 2*std
    thresh = mean_d + 2 * std_d if std_d > 0 else float('inf')

    flagged = []
    scores = {}
    for idx, dist in distances.items():
        if dist > thresh:
            flagged.append(idx)
            scores[str(idx)] = round(dist, 6)
        if len(flagged) >= 20:
            break

    return flagged, scores

# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    goal_columns, setting_columns = identify_columns(fieldnames)

    if not goal_columns:
        print("Error: No goal column found.")
        sys.exit(1)

    # Assume only one goal column for this skill
    goal_col = goal_columns[0]

    # ----------------------------------------------------------------
    # APACHE-17
    # ----------------------------------------------------------------
    print("== APACHE-17 ==")
    flagged_17 = apache_17(rows, goal_col, setting_columns)
    # Print at least three lines of results
    for f in flagged_17[:3]:
        print(f"Row {f['row']}: actual={f['actual']:.2f}, expected={f['expected']:.2f}, residual={f['residual']:.2f}")
    if len(flagged_17) == 0:
        print("No unusually different performance rows detected.")
    # JSON answer
    ans_17 = {"flagged": flagged_17[:20]}
    print(f"ANSWER APACHE-17 {json.dumps(ans_17)}")

    # ----------------------------------------------------------------
    # APACHE-18
    # ----------------------------------------------------------------
    print("\n== APACHE-18 ==")
    flagged_18, scores_18 = apache_18(rows, setting_columns)
    # Print at least three lines of results
    for row_id in flagged_18[:3]:
        print(f"Row {row_id} flagged with unusual setting distance {scores_18[str(row_id)]:.2f}")
    if len(flagged_18) == 0:
        print("No unusually different setting combinations detected.")
    # JSON answer
    ans_18 = {"flagged": flagged_18[:20], "scores": scores_18}
    print(f"ANSWER APACHE-18 {json.dumps(ans_18)}")

if __name__ == "__main__":
    main()
