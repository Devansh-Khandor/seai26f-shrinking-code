#!/usr/bin/env python3
"""
skill_10.py

Usage: python skill_10.py <csv path>
"""

import csv
import json
import random
import math
import sys

# ------------------------------------------------------------------
# Basic parsing helpers
# ------------------------------------------------------------------
def parse_value(v: str):
    """Parse a CSV cell into int, float, bool or string; '?' -> None."""
    if v == "?":
        return None
    for fn in (int, float):
        try:
            return fn(v)
        except ValueError:
            pass
    if v in ("True", "False"):
        return v == "True"
    return v


# ------------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------------
def load_csv(path: str):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = [[parse_value(v) for v in row] for row in rdr]
    return header, rows


# ------------------------------------------------------------------
# Helpers for goal columns
# ------------------------------------------------------------------
def is_goal(col_name: str) -> bool:
    return col_name.endswith(('+', '-'))


def goal_cols(header):
    return [c for c in header if is_goal(c)]


def compute_goal_minmax(header, rows):
    """Return dict {goal: (min, max)} over all rows with value."""
    mins = {}
    maxs = {}
    for col in goal_cols(header):
        vals = [r[header.index(col)] for r in rows if r[header.index(col)] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0
    return mins, maxs


def normalize_goal(val, minv, maxv, maximize=True):
    if val is None:
        return None
    if maxv == minv:
        return 1.0
    if maximize:
        return (val - minv) / (maxv - minv)
    else:
        return (maxv - val) / (maxv - minv)


def row_score(row, header, mins, maxs):
    """Sum of normalized goal values; missing goals ignored."""
    score = 0.0
    cnt = 0
    for col in goal_cols(header):
        v = row[header.index(col)]
        if v is None:
            continue
        mn = mins[col]
        mx = maxs[col]
        maximize = col.endswith('+')
        norm = normalize_goal(v, mn, mx, maximize)
        if norm is None:
            continue
        score += norm
        cnt += 1
    if cnt == 0:
        return 0.0
    return score / cnt  # average to keep scale similar across rows


# ------------------------------------------------------------------
# NASA-19: Effort prediction model vs baseline
# ------------------------------------------------------------------
def nasa19(header, rows):
    # Identify held-out rows: divisible by 5
    total_rows = len(rows)
    held_out_indices = [i for i in range(total_rows) if (i + 1) % 5 == 0]
    non_held_out_indices = [i for i in range(total_rows) if (i + 1) % 5 != 0]

    # Locate Effort- column
    if "Effort-" not in header:
        raise ValueError("Effort- column missing")
    effort_idx = header.index("Effort-")
    kloc_idx = header.index("Kloc+") if "Kloc+" in header else None

    # Prepare training data: all non-held-out rows with Effort- and Kloc+ present
    train_x = []
    train_y = []
    for i in non_held_out_indices:
        row = rows[i]
        eff = row[effort_idx]
        if eff is None:
            continue
        if kloc_idx is not None and row[kloc_idx] is not None:
            train_x.append(row[kloc_idx])
            train_y.append(eff)
    # Simple linear regression on Kloc+ -> Effort-
    if len(train_x) == 0:
        # cannot train, fall back to baseline
        slope = 0.0
        intercept = 0.0
    else:
        mean_x = sum(train_x) / len(train_x)
        mean_y = sum(train_y) / len(train_y)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(train_x, train_y))
        var = sum((x - mean_x) ** 2 for x in train_x)
        if var == 0:
            slope = 0.0
        else:
            slope = cov / var
        intercept = mean_y - slope * mean_x

    # Predictions for held-out rows
    predictions = {}
    mae_errors = []
    baseline_errors = []
    # baseline: mean Effort- of non-held-out rows with Effort- known
    baseline_effort_vals = [rows[i][effort_idx] for i in non_held_out_indices if rows[i][effort_idx] is not None]
    baseline_mean = sum(baseline_effort_vals) / len(baseline_effort_vals) if baseline_effort_vals else 0.0

    for i in held_out_indices:
        row = rows[i]
        actual = row[effort_idx]
        if actual is None:
            continue
        # prediction using slope/intercept
        x = row[kloc_idx] if kloc_idx is not None else None
        pred = slope * x + intercept if x is not None else baseline_mean
        predictions[str(i + 1)] = round(pred, 3)
        mae_errors.append(abs(pred - actual))
        baseline_errors.append(abs(baseline_mean - actual))

    mae_model = sum(mae_errors) / len(mae_errors) if mae_errors else 0.0
    mae_baseline = sum(baseline_errors) / len(baseline_errors) if baseline_errors else 0.0

    if mae_model < mae_baseline - 1e-6:
        verdict = "better"
    elif mae_model > mae_baseline + 1e-6:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    return {
        "predictions": predictions,
        "baseline": round(baseline_mean, 3),
        "mae_model": round(mae_model, 3),
        "mae_baseline": round(mae_baseline, 3),
        "verdict": verdict
    }


# ------------------------------------------------------------------
# NASA-20: Project selection vs random
# ------------------------------------------------------------------
def nasa20(header, rows):
    total_rows = len(rows)
    budget = min(30, total_rows // 5)

    # Compute overall scores for all rows
    mins, maxs = compute_goal_minmax(header, rows)
    scores = []
    for idx, row in enumerate(rows):
        scores.append(row_score(row, header, mins, maxs))

    # Our method: pick top budget rows by score
    sorted_indices = sorted(range(total_rows), key=lambda i: (-scores[i], i))
    our_top_indices = sorted_indices[:budget]
    # best among ours
    best_our_idx = sorted_indices[0]
    best_our_score = scores[best_our_idx]

    # Random selection repeats
    repeats = 20
    random_best_indices = []
    for _ in range(repeats):
        sample = random.sample(range(total_rows), budget)
        best_idx = min(sample, key=lambda i: (-scores[i], i))
        random_best_indices.append(best_idx)

    # Compute average scores
    random_avg_score = sum(scores[i] for i in random_best_indices) / repeats

    if best_our_score > random_avg_score + 1e-6:
        verdict = "ours better"
    elif best_our_score < random_avg_score - 1e-6:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Prepare JSON
    return {
        "budget": budget,
        "repeats": repeats,
        "ours": [best_our_idx + 1] * repeats,
        "random": [i + 1 for i in random_best_indices],
        "verdict": verdict
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(csv_path):
    header, rows = load_csv(csv_path)

    # NASA-19
    print("== NASA-19 ==")
    res19 = nasa19(header, rows)
    print(f"Baseline mean Effort-: {res19['baseline']}")
    print(f"MAE model: {res19['mae_model']}")
    print(f"MAE baseline: {res19['mae_baseline']}")
    print(f"Verdict: {res19['verdict']}")
    print(f"ANSWER NASA-19 {json.dumps(res19, ensure_ascii=False)}")

    # NASA-20
    print("\n== NASA-20 ==")
    res20 = nasa20(header, rows)
    print(f"Budget: {res20['budget']}")
    print(f"Repeats: {res20['repeats']}")
    print(f"Best ours row: {res20['ours'][0]}")
    print(f"Best random rows: {res20['random'][:3]} ...")
    print(f"Verdict: {res20['verdict']}")
    print(f"ANSWER NASA-20 {json.dumps(res20, ensure_ascii=False)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])
