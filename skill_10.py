#!/usr/bin/env python3
"""
skill_10.py – NASA-19 & NASA-20

The program accepts a CSV file path, reads the data, and prints two reports:

* NASA-19 – compares a simple linear‑regression effort‑prediction model against
  the historical‑average baseline.  It reports the MAE of each approach and a
  verdict about which is better.

* NASA-20 – compares a deterministic project‑selection method against a random
  selection of the same budget.  It runs 20 repeats (deterministic random seed)
  and reports which strategy consistently finds better projects.

Both reports end with a machine‑checked JSON line starting with
`ANSWER <ID>`.  All numeric values are plain JSON numbers, booleans are
`true`/`false`, and column names are exactly as in the CSV header.

The code uses only the standard library and follows the conventions
described in the problem statement.
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(path):
    """Return a list of row dicts (preserving order)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(val):
    """Convert a string to float; return None for missing or invalid."""
    if val is None:
        return None
    v = val.strip()
    if v == "" or v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def is_goal(col):
    """True if column is a goal (ends with '+' or '-')."""
    return col.endswith(("+", "-"))


def all_goals(rows):
    """Return list of goal column names."""
    return [c for c in rows[0].keys() if is_goal(c)]


def min_max_for_goals(rows, goals):
    """Return dict goal -> (min, max) ignoring missing values."""
    mins, maxs = {}, {}
    for g in goals:
        vals = [to_float(r[g]) for r in rows if to_float(r[g]) is not None]
        if vals:
            mins[g] = min(vals)
            maxs[g] = max(vals)
        else:
            mins[g] = maxs[g] = 0.0
    return mins, maxs


def normalize(value, mn, mx, goal):
    """Return normalized value in [0,1] for a goal."""
    if value is None or mx == mn:
        return None
    if goal.endswith("+"):
        return (value - mn) / (mx - mn)
    else:  # '-'
        return (mx - value) / (mx - mn)


def composite_score(row, goals, mins, maxs):
    """Return composite score for a row; None if any goal missing."""
    score = 0.0
    for g in goals:
        v = to_float(row[g])
        if v is None:
            return None
        norm = normalize(v, mins[g], maxs[g], g)
        if norm is None:
            return None
        score += norm
    return score


def all_rows_with_full_goals(rows, goals):
    """Return list of indices (1‑based) for rows having all goal values."""
    indices = []
    for idx, r in enumerate(rows, start=1):
        if all(to_float(r[g]) is not None for g in goals):
            indices.append(idx)
    return indices


# ----------------------------------------------------------------------
# NASA-19: Model vs baseline
# ----------------------------------------------------------------------
def nasa19(rows):
    """Return results for NASA‑19."""
    # Held‑out rows: numbers divisible by 5
    held_out = [i + 1 for i in range(len(rows)) if (i + 1) % 5 == 0]
    training_rows = [r for i, r in enumerate(rows, start=1) if i not in held_out]

    # Train linear regression Effort- ~ Kloc+
    x_vals, y_vals = [], []
    for r in training_rows:
        k = to_float(r["Kloc+"])
        e = to_float(r["Effort-"])
        if k is not None and e is not None:
            x_vals.append(k)
            y_vals.append(e)
    if x_vals:
        mean_x = sum(x_vals) / len(x_vals)
        mean_y = sum(y_vals) / len(y_vals)
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
        var_x = sum((x - mean_x) ** 2 for x in x_vals)
        if var_x == 0:
            slope = 0.0
            intercept = mean_y
        else:
            slope = cov_xy / var_x
            intercept = mean_y - slope * mean_x
    else:
        slope = intercept = 0.0

    # Baseline: mean Effort- of training data
    baseline_vals = [to_float(r["Effort-"]) for r in training_rows if to_float(r["Effort-"]) is not None]
    baseline_mean = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0

    # Predict for held‑out rows
    predictions = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count = 0
    for idx in held_out:
        r = rows[idx - 1]
        k = to_float(r["Kloc+"])
        actual = to_float(r["Effort-"])
        if k is None:
            k = sum(to_float(rr["Kloc+"]) for rr in training_rows if to_float(rr["Kloc+"]) is not None) / len(training_rows)
        pred = slope * k + intercept
        predictions[str(idx)] = pred
        if actual is not None:
            mae_model += abs(pred - actual)
            mae_baseline += abs(baseline_mean - actual)
            count += 1
    mae_model = mae_model / count if count else None
    mae_baseline = mae_baseline / count if count else None

    # Verdict
    if mae_model is None or mae_baseline is None:
        verdict = "no clear difference"
    elif mae_model < mae_baseline - 1e-6:
        verdict = "better"
    elif mae_model > mae_baseline + 1e-6:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    result = {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict,
    }
    return result


# ----------------------------------------------------------------------
# NASA-20: Selection vs random
# ----------------------------------------------------------------------
def nasa20(rows):
    """Return results for NASA‑20."""
    n_rows = len(rows)
    budget = min(30, n_rows // 5)

    # Identify goal columns and compute min/max for normalization
    goals = all_goals(rows)
    mins, maxs = min_max_for_goals(rows, goals)

    # Pre‑compute composite scores for rows that have all goals
    row_scores = {}
    for idx, r in enumerate(rows, start=1):
        sc = composite_score(r, goals, mins, maxs)
        if sc is not None:
            row_scores[idx] = sc

    # Our selection: top budget rows by score
    our_selection = sorted(row_scores.items(), key=lambda x: x[1], reverse=True)[:budget]
    our_indices = [idx for idx, _ in our_selection]

    # Pre‑compute best row among a set
    def best_in_set(idx_set):
        best_idx = None
        best_sc = -math.inf
        for i in idx_set:
            sc = row_scores.get(i)
            if sc is not None and sc > best_sc:
                best_sc = sc
                best_idx = i
        return best_idx

    # Deterministic repeats
    repeats = 20
    random.seed(0)

    ours_best = []
    random_best = []
    all_candidates = list(row_scores.keys())
    for _ in range(repeats):
        # Random selection of same budget
        random_selection = random.sample(all_candidates, budget)
        ours_best.append(best_in_set(our_indices))
        random_best.append(best_in_set(random_selection))

    # Verdict: compare average best score
    ours_scores = [row_scores[i] for i in ours_best if i in row_scores]
    rnd_scores = [row_scores[i] for i in random_best if i in row_scores]
    avg_ours = sum(ours_scores) / len(ours_scores) if ours_scores else 0.0
    avg_rnd = sum(rnd_scores) / len(rnd_scores) if rnd_scores else 0.0
    if avg_ours > avg_rnd + 1e-6:
        verdict = "ours better"
    elif avg_ours < avg_rnd - 1e-6:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    result = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict,
    }
    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No rows in CSV.")
        sys.exit(1)

    # NASA‑19
    print("== NASA-19 ==")
    ans19 = nasa19(rows)
    # Human‑readable summary
    print(f"Baseline mean Effort-: {ans19['baseline']:.2f}")
    print(f"Model MAE: {ans19['mae_model']:.2f}")
    print(f"Baseline MAE: {ans19['mae_baseline']:.2f}")
    print(f"Verdict: {ans19['verdict']}")
    print("ANSWER NASA-19", json.dumps(ans19, separators=(",", ":")))

    # NASA‑20
    print("\n== NASA-20 ==")
    ans20 = nasa20(rows)
    # Human‑readable summary
    print(f"Budget (rows evaluated): {ans20['budget']}")
    print(f"Repeats: {ans20['repeats']}")
    # Show first 5 of each list
    print(f"First 5 OURS best rows: {ans20['ours'][:5]}")
    print(f"First 5 RANDOM best rows: {ans20['random'][:5]}")
    print(f"Verdict: {ans20['verdict']}")
    print("ANSWER NASA-20", json.dumps(ans20, separators=(",", ":")))


if __name__ == "__main__":
    main()
