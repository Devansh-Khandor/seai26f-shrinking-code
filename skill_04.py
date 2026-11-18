#!/usr/bin/env python3
import sys
import csv
import json
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from itertools import combinations

# ----------------------------------------------------------------------
# Utility: parse float handling missing data
# ----------------------------------------------------------------------
def parse_float(val):
    if val is None or val == '':
        return None
    val = val.strip()
    if val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ----------------------------------------------------------------------
# Read CSV
# ----------------------------------------------------------------------
def read_csv(path):
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            data.append((idx, row))
    return data, reader.fieldnames

# ----------------------------------------------------------------------
# Identify goal columns (ending with '+' or '-')
# ----------------------------------------------------------------------
def goal_columns(fieldnames):
    return [fn for fn in fieldnames if fn.endswith('+') or fn.endswith('-')]

# ----------------------------------------------------------------------
# Build feature matrix and target vector for a given set of rows
# ----------------------------------------------------------------------
def build_Xy(row_indices, data_dict, target, goal_cols):
    X = []
    y = []
    for idx in row_indices:
        row = data_dict[idx]
        # target must be present
        t_val = parse_float(row[target])
        if t_val is None:
            continue
        feat_vals = []
        missing = False
        for col, val in row.items():
            if col == target or col.endswith('X'):
                continue
            f_val = parse_float(val)
            if f_val is None:
                missing = True
                break
            feat_vals.append(f_val)
        if missing:
            continue
        X.append(feat_vals)
        y.append(t_val)
    if not X:
        return None, None, []
    return np.array(X, dtype=float), np.array(y, dtype=float), [idx for idx in row_indices if len(row) == len(X[0]) + 1]

# ----------------------------------------------------------------------
# Compute mean and std (population) for a list of numbers
# ----------------------------------------------------------------------
def mean_std(values):
    n = len(values)
    if n == 0:
        return (None, None)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std

# ----------------------------------------------------------------------
# NASA-07: Arrival assessments
# ----------------------------------------------------------------------
def nasa_07(data, fieldnames):
    print("== NASA-07 ==")
    total_rows = len(data)
    arrivals_count = round(0.2 * total_rows)
    arrivals_start = total_rows - arrivals_count
    history_rows = [idx for idx, _ in data[:arrivals_start]]
    arrival_rows = [idx for idx, _ in data[arrivals_start:]]
    # Build dict for quick access
    data_dict = {idx: row for idx, row in data}
    goal_cols = goal_columns(fieldnames)
    assessments = []

    # Pre-compute feature indices: all columns except X and target
    for arrival_idx in arrival_rows:
        # Training data: history + previous arrivals
        training_indices = [i for i in history_rows if i < arrival_idx] + \
                           [i for i in arrival_rows if i < arrival_idx]
        # Build training data for Effort-
        X_eff, y_eff, _ = build_Xy(training_indices, data_dict, 'Effort-', goal_cols)
        if X_eff is not None and y_eff is not None:
            model_eff = LinearRegression()
            model_eff.fit(X_eff, y_eff)
            # Predict
            feat_vals = []
            missing = False
            for col, val in data_dict[arrival_idx].items():
                if col == 'Effort-' or col.endswith('X'):
                    continue
                f_val = parse_float(val)
                if f_val is None:
                    missing = True
                    break
                feat_vals.append(f_val)
            if not missing and len(feat_vals) == X_eff.shape[1]:
                pred_eff = model_eff.predict(np.array([feat_vals]))[0]
            else:
                pred_eff = None
        else:
            pred_eff = None

        # Build training data for Defects-
        X_def, y_def, _ = build_Xy(training_indices, data_dict, 'Defects-', goal_cols)
        if X_def is not None and y_def is not None:
            model_def = LinearRegression()
            model_def.fit(X_def, y_def)
            feat_vals = []
            missing = False
            for col, val in data_dict[arrival_idx].items():
                if col == 'Defects-' or col.endswith('X'):
                    continue
                f_val = parse_float(val)
                if f_val is None:
                    missing = True
                    break
                feat_vals.append(f_val)
            if not missing and len(feat_vals) == X_def.shape[1]:
                pred_def = model_def.predict(np.array([feat_vals]))[0]
            else:
                pred_def = None
        else:
            pred_def = None

        # Determine alert: unusually high if prediction > mean + 2*std of training targets
        alert = False
        if pred_eff is not None:
            mean_eff, std_eff = mean_std(y_eff)
            if mean_eff is not None and std_eff is not None:
                if pred_eff > mean_eff + 2 * std_eff:
                    alert = True
        if pred_def is not None:
            mean_def, std_def = mean_std(y_def)
            if mean_def is not None and std_def is not None:
                if pred_def > mean_def + 2 * std_def:
                    alert = True

        assessments.append({
            "row": arrival_idx,
            "predicted_effort": pred_eff if pred_eff is not None else None,
            "predicted_defects": pred_def if pred_def is not None else None,
            "alert": alert
        })

    # Print at least three lines of results
    print(f"Total arrivals: {len(arrival_rows)}")
    for a in assessments[:3]:
        print(f"Row {a['row']}: effort={a['predicted_effort']:.2f}, defects={a['predicted_defects']:.2f}, alert={a['alert']}")

    # Machine‑checked answer
    ans = {"assessments": assessments}
    print(f"ANSWER NASA-07 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Helper: compute stats for a column over given indices
# ----------------------------------------------------------------------
def compute_stats_for_column(indices, data_dict, col):
    vals = []
    for idx in indices:
        v = parse_float(data_dict[idx][col])
        if v is not None:
            vals.append(v)
    n = len(vals)
    mean = sum(vals) / n if n > 0 else None
    mn = min(vals) if n > 0 else None
    mx = max(vals) if n > 0 else None
    return {"n": n, "mean": mean, "min": mn, "max": mx}

# ----------------------------------------------------------------------
# Helper: compute shortlist of top 5 rows
# ----------------------------------------------------------------------
def compute_shortlist(indices, data_dict, goal_cols):
    # For each goal, compute min and max over all rows
    min_vals = {}
    max_vals = {}
    for col in goal_cols:
        vals = [parse_float(data_dict[idx][col]) for idx in indices if parse_float(data_dict[idx][col]) is not None]
        if vals:
            min_vals[col] = min(vals)
            max_vals[col] = max(vals)
        else:
            min_vals[col] = None
            max_vals[col] = None

    scores = []
    for idx in indices:
        row = data_dict[idx]
        # Skip if any goal missing
        missing = False
        feats = []
        for col in goal_cols:
            v = parse_float(row[col])
            if v is None:
                missing = True
                break
            feats.append(v)
        if missing:
            continue
        # Normalize and score
        score = 0.0
        for col, v in zip(goal_cols, feats):
            mn = min_vals[col]
            mx = max_vals[col]
            if mn == mx:
                # all equal, treat as neutral
                continue
            if col.endswith('+'):
                norm = (v - mn) / (mx - mn)
            else:  # '-'
                norm = (mx - v) / (mx - mn)
            score += norm
        scores.append((score, idx))
    # Get top 5
    scores.sort(reverse=True)
    top5 = [idx for _, idx in scores[:5]]
    return top5

# ----------------------------------------------------------------------
# NASA-08: Checkpoints
# ----------------------------------------------------------------------
def nasa_08(data, fieldnames):
    print("\n== NASA-08 ==")
    total_rows = len(data)
    arrivals_count = round(0.2 * total_rows)
    arrivals_start = total_rows - arrivals_count
    history_rows = [idx for idx, _ in data[:arrivals_start]]
    arrival_rows = [idx for idx, _ in data[arrivals_start:]]
    checkpoints = [0, 9, 19]
    goal_cols = goal_columns(fieldnames)
    data_dict = {idx: row for idx, row in data}
    results = []

    for m in checkpoints:
        # Up to m arrivals
        m = min(m, len(arrival_rows))
        indices = history_rows + arrival_rows[:m]
        summary = {}
        for col in goal_cols:
            stats = compute_stats_for_column(indices, data_dict, col)
            # Convert to numbers (float) but keep None as null in JSON
            summary[col] = {"n": stats["n"],
                            "mean": stats["mean"],
                            "min": stats["min"],
                            "max": stats["max"]}
        shortlist = compute_shortlist(indices, data_dict, goal_cols)
        results.append({"after": m, "summary": summary, "shortlist": shortlist})
        # Print a couple of lines for this checkpoint
        print(f"Checkpoint after {m} arrivals: rows considered = {len(indices)}")
        print(f"  Example goal '{goal_cols[0]}' stats: n={summary[goal_cols[0]]['n']}, mean={summary[goal_cols[0]]['mean']:.2f}")

    # Machine‑checked answer
    ans = {"checkpoints": results}
    print(f"ANSWER NASA-08 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    data, fieldnames = read_csv(path)

    nasa_07(data, fieldnames)
    nasa_08(data, fieldnames)

if __name__ == "__main__":
    main()
