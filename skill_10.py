#!/usr/bin/env python3
"""
skill_10.py

Usage: python skill_10.py <csv_path>
"""

import sys
import csv
import json
import random
import math
from statistics import mean, stdev, StatisticsError
from collections import defaultdict

# --------------------------------------------------------------------
# CSV reading helper
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# Helpers for goal/performance/settings
# --------------------------------------------------------------------
def goal_columns(rows):
    return [c for c in rows[0].keys() if c.endswith('+') or c.endswith('-')]

def performance_column(rows):
    for c in rows[0].keys():
        if c.endswith('-') and 'Performance' in c:
            return c
    return None

def setting_columns(rows, perf_col):
    return [c for c in rows[0].keys()
            if not (c.endswith('+') or c.endswith('-') or c.endswith('X'))
            and c != perf_col]

# --------------------------------------------------------------------
# APACHE-19 : compare deterministic selection vs random
# --------------------------------------------------------------------
def apache19(rows):
    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    repeats = 20

    # Prepare lookup by rownum for quick access
    row_by_num = {r['_rownum']: r for r in rows}

    ours_best_rows = []
    random_best_rows = []

    # --- deterministic method: evaluate first `budget` rows
    def best_in_subset(subset):
        best = None
        best_perf = None
        for r in subset:
            perf = r.get(perf_col)
            if perf is None:
                continue
            if best_perf is None or perf < best_perf or (perf == best_perf and r['_rownum'] < best['_rownum']):
                best_perf = perf
                best = r
        return best

    for i in range(repeats):
        # deterministic subset
        subset_det = rows[:budget]
        best_det = best_in_subset(subset_det)
        ours_best_rows.append(best_det['_rownum'] if best_det else None)

        # random subset
        rng = random.Random(42 + i)
        subset_rand = rng.sample(rows, budget)
        best_rand = best_in_subset(subset_rand)
        random_best_rows.append(best_rand['_rownum'] if best_rand else None)

    # Determine verdict
    ours_wins = 0
    random_wins = 0
    ties = 0
    for i in range(repeats):
        o = ours_best_rows[i]
        r = random_best_rows[i]
        if o is None and r is None:
            ties += 1
            continue
        if o is None:
            random_wins += 1
            continue
        if r is None:
            ours_wins += 1
            continue
        perf_o = row_by_num[o][perf_col]
        perf_r = row_by_num[r][perf_col]
        if perf_o < perf_r:
            ours_wins += 1
        elif perf_r < perf_o:
            random_wins += 1
        else:
            ties += 1

    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== APACHE-19 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Our best rows (per repeat): {ours_best_rows}")
    print(f"Random best rows (per repeat): {random_best_rows}")

    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }
    print(f"ANSWER APACHE-19 {json.dumps(answer, separators=(',', ':'))}")

# --------------------------------------------------------------------
# Similarity between rows (based on setting columns)
# --------------------------------------------------------------------
def similarity(r1, r2, setting_cols):
    common = 0
    matches = 0
    for c in setting_cols:
        v1 = r1.get(c)
        v2 = r2.get(c)
        if v1 is None or v2 is None:
            continue
        common += 1
        if v1 == v2:
            matches += 1
    return matches / common if common > 0 else 0.0

# --------------------------------------------------------------------
# APACHE-20 : predictions vs baseline average
# --------------------------------------------------------------------
def apache20(rows):
    perf_col = performance_column(rows)
    if perf_col is None:
        print("No performance column found.")
        return
    set_cols = setting_columns(rows, perf_col)

    # Identify training (non‑held‑out) and held‑out rows
    training = [r for r in rows if r['_rownum'] % 5 != 0 and r[perf_col] is not None]
    heldout = [r for r in rows if r['_rownum'] % 5 == 0]
    if not training:
        print("No training data available.")
        return

    # Baseline mean of training performance
    train_perf = [r[perf_col] for r in training]
    baseline_mean = round(mean(train_perf), 2)

    # Predictions for held‑out
    predictions = {}
    errors = []
    for r in heldout:
        best_sim = -1.0
        best_row = None
        for t in training:
            sim = similarity(r, t, set_cols)
            if sim > best_sim:
                best_sim = sim
                best_row = t
        if best_sim > 0 and best_row is not None:
            pred = best_row[perf_col]
        else:
            pred = baseline_mean
        pred_rounded = round(pred, 2)
        predictions[str(r['_rownum'])] = pred_rounded
        if r[perf_col] is not None:
            errors.append(abs(pred_rounded - r[perf_col]))

    mae_model = round(mean(errors), 2) if errors else 0.0

    # Baseline predictions (all held‑out predicted as baseline mean)
    errors_base = [abs(baseline_mean - r[perf_col]) for r in heldout if r[perf_col] is not None]
    mae_baseline = round(mean(errors_base), 2) if errors_base else 0.0

    # Verdict
    if mae_model + 1e-6 < mae_baseline:
        verdict = "better"
    elif mae_baseline + 1e-6 < mae_model:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("== APACHE-20 ==")
    print(f"Baseline mean: {baseline_mean}")
    print(f"MAE model: {mae_model}")
    print(f"MAE baseline: {mae_baseline}")

    answer = {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }
    print(f"ANSWER APACHE-20 {json.dumps(answer, separators=(',', ':'))}")

# --------------------------------------------------------------------
# Main dispatcher
# --------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    perf_col = performance_column(rows)
    if perf_col is None:
        print("No performance column found.")
        sys.exit(1)

    # Run requested sections
    apache19(rows)
    apache20(rows)
