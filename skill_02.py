#!/usr/bin/env python3
"""
skill_02.py

Usage: python skill_02.py <csv_path>
"""

import sys
import csv
import json
import math

def read_csv(path):
    """Read CSV and return list of rows (dict) with 1‑based indices."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=1):
            row = {}
            for k, v in raw.items():
                if v == '?':
                    row[k] = None
                else:
                    try:
                        row[k] = int(v)
                    except ValueError:
                        row[k] = v
            row['_rownum'] = idx
            rows.append(row)
    return rows

def similarity(r1, r2, setting_cols):
    """Return similarity score between two rows based on setting columns."""
    common = 0
    matches = 0
    for col in setting_cols:
        v1 = r1.get(col)
        v2 = r2.get(col)
        if v1 is None or v2 is None:
            continue
        common += 1
        if v1 == v2:
            matches += 1
    return matches / common if common > 0 else 0.0

def compute_overall_perf(train_rows, perf_col):
    """Return mean and variance of performance in training set."""
    vals = [r[perf_col] for r in train_rows]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var

def predict(row, train_rows, setting_cols, perf_col, overall_mean, overall_var):
    """Return (prediction, variance) for a row using kNN style weighting."""
    weights = []
    values = []
    for t in train_rows:
        sim = similarity(row, t, setting_cols)
        if sim > 0:
            weights.append(sim)
            values.append(t[perf_col])
    if not weights:
        return overall_mean, overall_var
    wsum = sum(weights)
    mean_pred = sum(v * w for v, w in zip(values, weights)) / wsum
    var_pred = sum(w * (v - mean_pred) ** 2 for v, w in zip(values, weights)) / wsum
    return mean_pred, var_pred

def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        return

    header = rows[0].keys()
    # Identify goal columns and performance column
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    # Assume there is only one performance goal ending with '-'
    perf_col = next((c for c in goal_cols if c.endswith('-')), None)
    if not perf_col:
        print("No performance goal column found.")
        return

    # Identify setting columns
    setting_cols = [
        c for c in header
        if not (c.endswith('+') or c.endswith('-') or c.endswith('X'))
        and c != perf_col
    ]

    # Partition rows into training and held-out
    train_rows = [r for r in rows if r['_rownum'] % 5 != 0 and r[perf_col] is not None]
    heldout_rows = [r for r in rows if r['_rownum'] % 5 == 0]
    heldout_nums = {r['_rownum'] for r in heldout_rows}

    overall_mean, overall_var = compute_overall_perf(train_rows, perf_col)

    # Predict performance for all rows
    predictions = {}
    variances = {}
    for r in rows:
        pred, var = predict(r, train_rows, setting_cols, perf_col, overall_mean, overall_var)
        predictions[r['_rownum']] = pred
        variances[r['_rownum']] = var

    # === APACHE-03 ===
    print("== APACHE-03 ==")
    # Human readable predictions for held‑out rows
    print("Predictions for held‑out rows:")
    mae_sum = 0.0
    count_mae = 0
    for r in heldout_rows:
        rownum = r['_rownum']
        pred = predictions[rownum]
        actual = r[perf_col]
        print(f"Row {rownum}: predicted={pred:.2f} actual={actual}")
        mae_sum += abs(pred - actual)
        count_mae += 1
    mae = mae_sum / count_mae if count_mae else 0.0
    print(f"MAE: {mae:.2f}")

    # Machine‑checked answer
    pred_json = {str(k): round(v, 2) for k, v in predictions.items() if k in heldout_nums}
    answer_apache_03 = {"predictions": pred_json, "mae": round(mae, 2)}
    print(f"ANSWER APACHE-03 {json.dumps(answer_apache_03, separators=(',', ':'))}")

    # === APACHE-04 ===
    print("\n== APACHE-04 ==")
    # Find all pairs differing in exactly one setting
    valid_rows = [r for r in rows if r[perf_col] is not None]
    pairs = []
    n = len(valid_rows)
    for i in range(n):
        r1 = valid_rows[i]
        for j in range(i + 1, n):
            r2 = valid_rows[j]
            diff_cols = [c for c in setting_cols
                         if r1[c] is not None and r2[c] is not None and r1[c] != r2[c]]
            if len(diff_cols) != 1:
                continue
            setting = diff_cols[0]
            a_num = r1['_rownum']
            b_num = r2['_rownum']
            pred_a = predictions[a_num]
            pred_b = predictions[b_num]
            diff = pred_b - pred_a
            std_a = math.sqrt(variances[a_num]) if variances[a_num] >= 0 else 0.0
            std_b = math.sqrt(variances[b_num]) if variances[b_num] >= 0 else 0.0
            low = diff - (std_a + std_b)
            high = diff + (std_a + std_b)
            pair = {
                "a": a_num,
                "b": b_num,
                "setting": setting,
                "difference": round(diff, 2),
                "low": round(low, 2),
                "high": round(high, 2)
            }
            pairs.append(pair)
    pairs.sort(key=lambda d: (d["a"], d["b"]))
    # At least 5 pairs
    print("First few pairs:")
    for p in pairs[:5]:
        print(p)
    answer_apache_04 = {"pairs": pairs[:5]}
    print(f"ANSWER APACHE-04 {json.dumps(answer_apache_04, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
