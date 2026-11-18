#!/usr/bin/env python3
"""
Skill 03: Compare configurations by KeepAlive and access logging.
"""
import csv
import json
import sys
import math
import itertools
from sklearn.linear_model import LinearRegression

# Re-use read_csv from skill_01 (same logic)
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
                        if '.' in v:
                            cleaned[k] = float(v)
                        else:
                            cleaned[k] = int(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append((idx, cleaned))
    return rows, reader.fieldnames

def compute_group_stats(rows, group_col, goal_col):
    """
    rows: list of (idx, dict)
    group_col: column name for grouping (e.g., 'keepAlive')
    goal_col: column name for goal (e.g., 'Performance-')
    Returns: dict mapping group value to dict {'n':..., 'median':..., 'mean':...}
    """
    groups = {}
    for _, r in rows:
        gval = r.get(group_col)
        if gval is None:
            continue
        gdict = groups.setdefault(str(gval), {"vals": []})
        val = r.get(goal_col)
        if val is not None:
            gdict["vals"].append(val)
    # Build final stats
    stats = {}
    for gval, gdict in groups.items():
        vals = gdict["vals"]
        n = len(vals)
        if n == 0:
            mean = 0.0
            median = 0.0
        else:
            mean = sum(vals) / n
            sorted_vals = sorted(vals)
            if n % 2 == 1:
                median = sorted_vals[n // 2]
            else:
                median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        stats[gval] = {
            "n": n,
            "mean": round(mean, 6),
            "median": round(median, 6)
        }
    return stats

def build_training_and_mae(rows, fieldnames):
    """
    Trains a linear regression model using all settings except goal columns.
    Returns:
        mae: mean absolute error on held-out rows (indices divisible by 5)
    """
    goal_columns = [col for col in fieldnames if col.endswith(('+', '-'))]
    if 'Performance-' not in goal_columns:
        print("Error: 'Performance-' column not found.")
        sys.exit(1)
    setting_columns = [col for col in fieldnames
                       if col not in goal_columns and not col.endswith('X')]

    # Compute means of settings on training set (not held-out)
    held_out_ids = {idx for idx, _ in rows if idx % 5 == 0}
    train_rows = [(idx, r) for idx, r in rows if idx not in held_out_ids]
    setting_means = {}
    for col in setting_columns:
        vals = [r[col] for _, r in train_rows if r[col] is not None]
        setting_means[col] = sum(vals) / len(vals) if vals else 0.0

    # Prepare training data
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
        print("Error: No training data available.")
        sys.exit(1)

    # Fit model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Compute MAE on held-out rows
    errors = []
    for idx, r in rows:
        if idx % 5 != 0:
            continue
        vec = []
        for col in setting_columns:
            val = r[col]
            if val is None:
                val = setting_means[col]
            vec.append(val)
        pred = lr.predict([vec])[0]
        true = r['Performance-']
        if true is not None:
            errors.append(abs(true - pred))
    mae = sum(errors) / len(errors) if errors else 0.0
    return mae

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    # Compute MAE once for threshold
    mae = build_training_and_mae(rows, fieldnames)

    # ---- APACHE-05 ----
    print("== APACHE-05 ==")

    keepalive_stats = compute_group_stats(rows, 'keepAlive', 'Performance-')
    # Print group stats
    for val, stats in keepalive_stats.items():
        print(f"KeepAlive={val}: n={stats['n']}, mean={stats['mean']}, median={stats['median']}")

    # Compute comparisons
    comparisons_05 = []
    group_vals = list(keepalive_stats.keys())
    for a, b in itertools.combinations(group_vals, 2):
        mean_a = keepalive_stats[a]['mean']
        mean_b = keepalive_stats[b]['mean']
        diff = abs(mean_a - mean_b)
        verdict = "different" if diff > mae else "same"
        comparisons_05.append({"a": a, "b": b, "verdict": verdict})

    # Output at least three lines
    print(f"MAE (threshold): {round(mae, 6)}")
    print(f"Comparisons: {json.dumps(comparisons_05)}")

    # JSON answer
    ans_05 = {
        "groups": keepalive_stats,
        "comparisons": comparisons_05
    }
    print(f"ANSWER APACHE-05 {json.dumps(ans_05)}")

    # ---- APACHE-06 ----
    print("\n== APACHE-06 ==")

    accesslog_stats = compute_group_stats(rows, 'accessLog', 'Performance-')
    for val, stats in accesslog_stats.items():
        print(f"accessLog={val}: n={stats['n']}, mean={stats['mean']}, median={stats['median']}")

    comparisons_06 = []
    group_vals = list(accesslog_stats.keys())
    for a, b in itertools.combinations(group_vals, 2):
        mean_a = accesslog_stats[a]['mean']
        mean_b = accesslog_stats[b]['mean']
        diff = abs(mean_a - mean_b)
        verdict = "different" if diff > mae else "same"
        comparisons_06.append({"a": a, "b": b, "verdict": verdict})

    print(f"MAE (threshold): {round(mae, 6)}")
    print(f"Comparisons: {json.dumps(comparisons_06)}")

    ans_06 = {
        "groups": accesslog_stats,
        "comparisons": comparisons_06
    }
    print(f"ANSWER APACHE-06 {json.dumps(ans_06)}")

if __name__ == "__main__":
    main()
