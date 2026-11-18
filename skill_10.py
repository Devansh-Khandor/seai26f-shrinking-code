#!/usr/bin/env python3
"""
skill_10.py - Apache configuration benchmarking and prediction comparison

This script implements APACHE-19 and APACHE-20 as required by the specification.
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# CSV handling
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Convert CSV cell to int/float if possible, keep '?' otherwise."""
    v = v.strip()
    if v == "":
        return "?"
    if v == "?":
        return "?"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v


def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if not line:
                continue
            row: Dict[str, Any] = {}
            for key, val in zip(header, line):
                row[key] = parse_value(val)
            rows.append(row)
    return rows, header


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_input(col: str, goal_col: str) -> bool:
    return not col.endswith("X") and col != goal_col


def is_numeric(v: Any) -> bool:
    return isinstance(v, int) or isinstance(v, float)


def compute_goal_stats(rows: List[Dict[str, Any]], goal_cols: List[str]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for g in goal_cols:
        values = [float(r[g]) for r in rows if isinstance(r.get(g), (int, float))]
        if values:
            stats[g] = (min(values), max(values))
    return stats


def aggregate_goal_score(row: Dict[str, Any], goal_cols: List[str], goal_stats: Dict[str, Tuple[float, float]]) -> float:
    """Higher is better."""
    score = 0.0
    for g in goal_cols:
        v = row.get(g)
        if not isinstance(v, (int, float)):
            continue
        lo, hi = goal_stats[g]
        if hi == lo:
            norm = 1.0
        else:
            if g.endswith("-"):  # minimize
                norm = (hi - v) / (hi - lo)
            else:  # maximize
                norm = (v - lo) / (hi - lo)
        score += norm
    return score


def normalize_numeric(rows: List[Dict[str, Any]], cols: List[str]) -> Dict[str, Tuple[float, float]]:
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for c in cols:
        vals = [float(r[c]) for r in rows if isinstance(r.get(c), (int, float))]
        if vals:
            mins[c] = min(vals)
            maxs[c] = max(vals)
    return {c: (mins[c], maxs[c]) for c in cols if c in mins}


def heuristic_score(row: Dict[str, Any], input_cols: List[str],
                    norm_nums: Dict[str, Tuple[float, float]]) -> float:
    """Simple score: count of binary ones + sum of normalized numeric values."""
    score = 0.0
    for c in input_cols:
        v = row.get(c)
        if v == "?":
            continue
        if isinstance(v, (int, float)):
            lo, hi = norm_nums.get(c, (0, 1))
            if hi == lo:
                norm = 1.0
            else:
                norm = (v - lo) / (hi - lo)
            score += norm
        else:  # symbol, treat as 1 if truthy
            if str(v).lower() in ("1", "true"):
                score += 1.0
    return score


def held_out_indices(nrows: int) -> List[int]:
    return [i for i in range(1, nrows + 1) if i % 5 == 0]


def arrivals_indices(rows: List[Dict[str, Any]]) -> List[int]:
    n = len(rows)
    m = int(round(0.2 * n))
    return list(range(n - m + 1, n + 1))  # 1-based indices


def k_nn_predict(row: Dict[str, Any], history: List[Dict[str, Any]],
                 inputs: List[str], goal_col: str, k: int = 3) -> float:
    """Predict goal of row using k nearest neighbors (hamming distance)."""
    def hamming(r1, r2):
        d = 0
        for c in inputs:
            v1 = r1.get(c, "?")
            v2 = r2.get(c, "?")
            if v1 == "?" or v2 == "?":
                continue
            if v1 != v2:
                d += 1
        return d

    neighs = sorted([(hamming(row, h), h) for h in history], key=lambda x: x[0])[:k]
    values = [float(h[goal_col]) for _, h in neighs if isinstance(h[goal_col], (int, float))]
    if not values:
        # fallback to mean of history
        return sum(values) / len(values) if values else 0.0
    return sum(values) / len(values)


# ----------------------------------------------------------------------
# APACHE-19 implementation
# ----------------------------------------------------------------------
def apache_19(rows: List[Dict[str, Any]], goal_cols: List[str]) -> Dict:
    n = len(rows)
    budget = min(30, n // 5)
    repeats = max(20, 20)  # at least 20

    # Prepare input columns (exclude goals and X)
    input_cols = [c for c in rows[0].keys() if is_input(c, goal_cols[0])]
    # Precompute normalizations
    norm_nums = normalize_numeric(rows, [c for c in input_cols if is_numeric(rows[0].get(c))])

    # Precompute heuristic scores
    heur_scores = [heuristic_score(r, input_cols, norm_nums) for r in rows]
    # Sorted indices by heuristic descending
    sorted_by_heur = sorted(range(n), key=lambda i: heur_scores[i], reverse=True)

    best_ours: List[int] = []
    best_rand: List[int] = []

    rng = random.Random(123456)

    for rep in range(repeats):
        # Our selection: top budget by heuristic
        selected = sorted_by_heur[:budget]
        # Evaluate these rows
        best_row = max(selected, key=lambda i: aggregate_goal_score(rows[i], goal_cols, goal_stats))
        best_ours.append(best_row + 1)  # 1-based

        # Random selection
        rand_selected = rng.sample(range(n), budget)
        best_rand_row = max(rand_selected, key=lambda i: aggregate_goal_score(rows[i], goal_cols, goal_stats))
        best_rand.append(best_rand_row + 1)

    # Determine verdict
    ours_better = sum(aggregate_goal_score(rows[i - 1], goal_cols, goal_stats) > aggregate_goal_score(rows[j - 1], goal_cols, goal_stats)
                     for i, j in zip(best_ours, best_rand))
    rand_better = repeats - ours_better

    if ours_better > rand_better:
        verdict = "ours better"
    elif rand_better > ours_better:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return {
        "budget": budget,
        "repeats": repeats,
        "ours": best_ours,
        "random": best_rand,
        "verdict": verdict
    }


# ----------------------------------------------------------------------
# APACHE-20 implementation
# ----------------------------------------------------------------------
def apache_20(rows: List[Dict[str, Any]], goal_col: str) -> Dict:
    n = len(rows)
    # Identify held-out rows
    hold_out = set(held_out_indices(n))
    history = [r for i, r in enumerate(rows, start=1) if i not in hold_out]

    # Baseline mean of history
    hist_vals = [float(r[goal_col]) for r in history if isinstance(r[goal_col], (int, float))]
    baseline_mean = sum(hist_vals) / len(hist_vals) if hist_vals else 0.0

    # Input columns
    input_cols = [c for c in rows[0].keys() if is_input(c, goal_col)]

    # Predictions
    predictions = {}
    errors = []
    for idx, row in enumerate(rows, start=1):
        if idx in hold_out:
            pred = k_nn_predict(row, history, input_cols, goal_col, k=3)
            predictions[str(idx)] = round(pred, 4)
            actual = row.get(goal_col)
            if isinstance(actual, (int, float)):
                errors.append(abs(pred - float(actual)))

    mae_model = sum(errors) / len(errors) if errors else 0.0
    mae_baseline = sum(abs(baseline_mean - float(r[goal_col])) for r in rows if isinstance(r[goal_col], (int, float))) / len(rows)

    # Verdict
    if mae_model < mae_baseline * 0.95:
        verdict = "better"
    elif mae_model > mae_baseline * 1.05:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    return {
        "predictions": predictions,
        "baseline": round(baseline_mean, 4),
        "mae_model": round(mae_model, 4),
        "mae_baseline": round(mae_baseline, 4),
        "verdict": verdict
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, header = read_csv(csv_path)

    goal_cols = [c for c in header if is_goal(c)]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(0)
    goal_col = goal_cols[0]

    # Precompute goal stats for aggregate scoring
    goal_stats = compute_goal_stats(rows, goal_cols)

    # --- APACHE-19 ---
    print("== APACHE-19 ==")
    # Show budget and repeats
    result_19 = apache_19(rows, goal_cols)
    print(f"Budget: {result_19['budget']}")
    print(f"Repeats: {result_19['repeats']}")
    print(f"Our best rows each repeat: {result_19['ours']}")
    print(f"Random best rows each repeat: {result_19['random']}")
    print(f"Verdict: {result_19['verdict']}")
    print("ANSWER APACHE-19", json.dumps(result_19, separators=(",", ":")))

    # --- APACHE-20 ---
    print("\n== APACHE-20 ==")
    result_20 = apache_20(rows, goal_col)
    print(f"Baseline mean {goal_col}: {result_20['baseline']}")
    print(f"Model MAE: {result_20['mae_model']}, Baseline MAE: {result_20['mae_baseline']}")
    print(f"Verdict: {result_20['verdict']}")
    # Show predictions for first 5 held-out rows (if any)
    first_five = list(result_20['predictions'].items())[:5]
    print("Sample predictions:")
    for idx, pred in first_five:
        print(f"  Row {idx}: {pred}")
    print("ANSWER APACHE-20", json.dumps(result_20, separators=(",", ":")))


if __name__ == "__main__":
    main()
