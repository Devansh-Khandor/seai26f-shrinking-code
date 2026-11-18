#!/usr/bin/env python3
"""
Skill 3: APACHE-05 and APACHE-06 analyses

The program reads a CSV file provided as the sole command-line argument
and performs the following tasks:

1. Builds a simple regression model to predict the performance goal
   (column ending in '-') using all other columns (except those ending in
   'X').  The model is a pipeline of median imputation followed by
   LinearRegression.
2. Computes the Mean Absolute Error (MAE) of the model on held‑out
   rows (row number divisible by 5).
3. Groups rows by the values of the `keepAlive` and `accessLog` columns
   (skipping rows where the group value or the performance goal is
   missing).  For each group, reports the number of rows, the median and
   the mean of the performance goal.
4. For every pair of distinct group values, determines whether the
   difference in mean performance is reliable.  A difference is deemed
   reliable (verdict `"different"`) when its absolute value exceeds the
   MAE; otherwise the verdict is `"same"`.

The program prints a short, human‑readable report followed by a line
starting with `ANSWER <ID>` and a machine‑checked JSON object as
required by the specification.
"""

import csv
import sys
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

def read_csv(path):
    """Read the CSV and return (rows, fieldnames).

    Each row is a dict with keys:
        'row'   – 1‑based row number
        'values' – mapping column name -> float or None
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == '?':
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append({'row': i, 'values': cleaned})
    return rows, fieldnames

def get_goal_col(fieldnames):
    for c in fieldnames:
        if c.endswith('+') or c.endswith('-'):
            return c
    return None

def get_setting_cols(fieldnames, goal_col):
    return [c for c in fieldnames if c != goal_col and not c.endswith('X')]

def build_dataset(rows, feature_cols, goal_col):
    X_list = []
    y_list = []
    for r in rows:
        val = r['values'].get(goal_col)
        if val is None:
            continue
        feature_vals = []
        for c in feature_cols:
            v = r['values'].get(c)
            feature_vals.append(v if v is not None else np.nan)
        X_list.append(feature_vals)
        y_list.append(val)
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y

def predict_rows(rows, model, feature_cols):
    preds = {}
    for r in rows:
        feature_vals = []
        for c in feature_cols:
            v = r['values'].get(c)
            feature_vals.append(v if v is not None else np.nan)
        X = np.array(feature_vals, dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        preds[r['row']] = pred
    return preds

def mean_absolute_error(preds, rows, goal_col):
    errors = []
    for r in rows:
        true_val = r['values'].get(goal_col)
        if true_val is None:
            continue
        pred_val = preds.get(r['row'])
        if pred_val is None:
            continue
        errors.append(abs(pred_val - true_val))
    return sum(errors) / len(errors) if errors else 0.0

def group_stats(rows, group_col, goal_col):
    """Return dict mapping group value -> {'n', 'median', 'mean'}."""
    stats = {}
    for r in rows:
        g_val = r['values'].get(group_col)
        if g_val is None:
            continue
        goal_val = r['values'].get(goal_col)
        if goal_val is None:
            continue
        key = str(g_val)
        stats.setdefault(key, []).append(goal_val)

    result = {}
    for key, values in stats.items():
        n = len(values)
        sorted_vals = sorted(values)
        median = sorted_vals[n//2] if n % 2 == 1 else (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2.0
        mean = sum(values) / n
        result[key] = {'n': n, 'median': median, 'mean': mean}
    return result

def pairwise_comparisons(groups, mae):
    """Return list of {a, b, verdict} for each pair of distinct group values."""
    keys = sorted(groups.keys())
    comparisons = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            mean_a = groups[a]['mean']
            mean_b = groups[b]['mean']
            diff = abs(mean_a - mean_b)
            verdict = "different" if diff > mae else "same"
            comparisons.append({'a': a, 'b': b, 'verdict': verdict})
    return comparisons

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goal_col = get_goal_col(fieldnames)
    if goal_col is None:
        print("No goal column found.")
        sys.exit(1)

    setting_cols = get_setting_cols(fieldnames, goal_col)

    # Build regression model
    held_out_rows = [r for r in rows if r['row'] % 5 == 0]
    training_rows = [r for r in rows if r not in held_out_rows]

    X_train, y_train = build_dataset(training_rows, setting_cols, goal_col)
    model = make_pipeline(SimpleImputer(strategy='median'), LinearRegression())
    model.fit(X_train, y_train)

    preds = predict_rows(held_out_rows, model, setting_cols)
    mae_val = mean_absolute_error(preds, held_out_rows, goal_col)

    # ===== APACHE-05 =====
    print("== APACHE-05 ==")
    keep_alive_groups = group_stats(rows, 'keepAlive', goal_col)
    for val, stats in sorted(keep_alive_groups.items()):
        print(f"keepAlive={val}: n={stats['n']}, median={stats['median']:.2f}, mean={stats['mean']:.2f}")

    comparisons = pairwise_comparisons(keep_alive_groups, mae_val)
    for comp in comparisons:
        print(f"{comp['a']} vs {comp['b']}: verdict={comp['verdict']}")

    answer_05 = {
        "groups": keep_alive_groups,
        "comparisons": comparisons
    }
    print(f"ANSWER APACHE-05 {json.dumps(answer_05, separators=(',', ':'))}")

    # ===== APACHE-06 =====
    print("\n== APACHE-06 ==")
    access_log_groups = group_stats(rows, 'accessLog', goal_col)
    for val, stats in sorted(access_log_groups.items()):
        print(f"accessLog={val}: n={stats['n']}, median={stats['median']:.2f}, mean={stats['mean']:.2f}")

    comparisons = pairwise_comparisons(access_log_groups, mae_val)
    for comp in comparisons:
        print(f"{comp['a']} vs {comp['b']}: verdict={comp['verdict']}")

    answer_06 = {
        "groups": access_log_groups,
        "comparisons": comparisons
    }
    print(f"ANSWER APACHE-06 {json.dumps(answer_06, separators=(',', ':'))}")

if __name__ == "__main__":
    main()
