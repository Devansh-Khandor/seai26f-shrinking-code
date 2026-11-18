#!/usr/bin/env python3
"""
Skill 06: Recommendation explanation and comparison of configurations.
"""
import csv
import json
import sys
import math
from collections import defaultdict
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------
# Common CSV reader – identical to skill_01.read_csv
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
# Helper: identify goal & setting columns
# --------------------------------------------------------------------
def identify_columns(fieldnames):
    goal_columns = [c for c in fieldnames if c.endswith(('+', '-'))]
    setting_columns = [c for c in fieldnames
                       if c not in goal_columns and not c.endswith('X')]
    return goal_columns, setting_columns


# --------------------------------------------------------------------
# Training a linear regression model (used for APACHE-12 scoring)
# --------------------------------------------------------------------
def train_regression(rows, setting_columns):
    # Train on all rows except those with index divisible by 5
    train_rows = [(idx, r) for idx, r in rows if idx % 5 != 0]
    # Compute mean of each setting on training set
    setting_means = {}
    for col in setting_columns:
        vals = [r[col] for _, r in train_rows if r[col] is not None]
        setting_means[col] = sum(vals) / len(vals) if vals else 0.0

    # Build X_train, y_train
    X_train, y_train = [], []
    for idx, r in train_rows:
        if r['Performance-'] is None:
            continue
        vec = []
        for col in setting_columns:
            val = r[col]
            if val is None:
                val = setting_means[col]
            vec.append(val)
        X_train.append(vec)
        y_train.append(r['Performance-'])

    if not X_train:
        raise RuntimeError("No training data available for regression.")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr, setting_means


# --------------------------------------------------------------------
# Compute predicted goal value for a row
# --------------------------------------------------------------------
def predict_row(lr, setting_means, row, setting_columns):
    vec = []
    for col in setting_columns:
        val = row[col]
        if val is None:
            val = setting_means[col]
        vec.append(val)
    return lr.predict([vec])[0]


# --------------------------------------------------------------------
# Find the best configuration (minimize Performance-)
# --------------------------------------------------------------------
def find_best(rows, goal_columns):
    best_row = None
    best_val = None
    for idx, r in rows:
        val = r[goal_columns[0]]
        if val is None:
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_row = (idx, r)
    return best_row


# --------------------------------------------------------------------
# Find worst configuration (maximize Performance-)
# --------------------------------------------------------------------
def find_worst(rows, goal_columns):
    worst_row = None
    worst_val = None
    for idx, r in rows:
        val = r[goal_columns[0]]
        if val is None:
            continue
        if worst_val is None or val > worst_val:
            worst_val = val
            worst_row = (idx, r)
    return worst_row


# --------------------------------------------------------------------
# Compute Euclidean distance between two rows on setting columns,
# ignoring missing values (pairwise). If no common value, return inf.
# --------------------------------------------------------------------
def distance(row_a, row_b, setting_columns):
    sum_sq = 0.0
    common = 0
    for col in setting_columns:
        va = row_a[col]
        vb = row_b[col]
        if va is None or vb is None:
            continue
        d = va - vb
        sum_sq += d * d
        common += 1
    if common == 0:
        return math.inf
    return math.sqrt(sum_sq)


# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    goal_columns, setting_columns = identify_columns(fieldnames)

    # ---------- APACHE-11 ----------
    print("== APACHE-11 ==")
    best = find_best(rows, goal_columns)
    if best is None:
        print("No best configuration found.")
        sys.exit(1)
    best_idx, best_vals = best
    print(f"Recommended configuration row: {best_idx}")

    # Find 3 nearest neighbors (excluding itself)
    neighbors = []
    for idx, vals in rows:
        if idx == best_idx:
            continue
        d = distance(best_vals, vals, setting_columns)
        if math.isfinite(d):
            neighbors.append((d, idx, vals))
    neighbors.sort(key=lambda x: x[0])
    print("Three nearest neighbors:")
    for d, idx, vals in neighbors[:3]:
        print(f"Row {idx}: distance={d:.4f}")

    # Prepare JSON answer
    neighbor_list = []
    for d, idx, vals in neighbors[:3]:
        # include all columns including goal
        entry = {"row": idx, "values": {}}
        for col in fieldnames:
            entry["values"][col] = vals[col]
        neighbor_list.append(entry)

    ans_apache_11 = {
        "recommended": best_idx,
        "neighbors": neighbor_list
    }
    print(f"ANSWER APACHE-11 {json.dumps(ans_apache_11)}")

    # ---------- APACHE-12 ----------
    print("\n== APACHE-12 ==")
    worst = find_worst(rows, goal_columns)
    if worst is None:
        print("No worst configuration found.")
        sys.exit(1)
    worst_idx, worst_vals = worst
    print(f"Best row: {best_idx}")
    print(f"Worst row: {worst_idx}")

    # Train regression to get scores
    lr, setting_means = train_regression(rows, setting_columns)

    # Compute predicted goal for each
    pred_best = predict_row(lr, setting_means, best_vals, setting_columns)
    pred_worst = predict_row(lr, setting_means, worst_vals, setting_columns)

    # For '-' goals, lower predicted is better → higher score = -predicted
    scores = {}
    for col in goal_columns:
        if col.endswith('-'):
            scores[best_idx] = -pred_best
            scores[worst_idx] = -pred_worst
        else:
            scores[best_idx] = pred_best
            scores[worst_idx] = pred_worst

    preferred = best_idx  # best row
    print(f"Preferred row: {preferred}")

    # Differences in settings
    differences = []
    for col in setting_columns:
        a_val = best_vals[col]
        b_val = worst_vals[col]
        if a_val != b_val:
            differences.append({
                "feature": col,
                "a": a_val,
                "b": b_val
            })
    if not differences:
        # force at least one difference (unlikely)
        differences.append({"feature": setting_columns[0], "a": best_vals[setting_columns[0]],
                            "b": worst_vals[setting_columns[0]]})
    print("Differences:")
    for d in differences[:1]:
        print(f"{d['feature']}: {best_idx}={d['a']}, {worst_idx}={d['b']}")

    # Tradeoffs: goal columns where non-preferred row is better
    tradeoffs = []
    for col in goal_columns:
        a_val = best_vals[col]
        b_val = worst_vals[col]
        if a_val is None or b_val is None:
            continue
        if col.endswith('-') and b_val < a_val:
            tradeoffs.append(col)
        if col.endswith('+') and b_val > a_val:
            tradeoffs.append(col)
    print("Tradeoffs where non-preferred is better:", tradeoffs)

    # JSON answer
    ans_apache_12 = {
        "a": best_idx,
        "b": worst_idx,
        "preferred": preferred,
        "scores": {
            str(best_idx): round(scores[best_idx], 6),
            str(worst_idx): round(scores[worst_idx], 6)
        },
        "differences": differences[:1],
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER APACHE-12 {json.dumps(ans_apache_12)}")

if __name__ == "__main__":
    main()
