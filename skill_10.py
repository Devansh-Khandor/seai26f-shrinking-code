#!/usr/bin/env python3
"""
skill_10.py

APACHE-19 – Compare our configuration‑selection system against random
configuration selection under the same benchmarking budget, and show
which approach discovers better‑performing configurations.

APACHE-20 – Compare our performance predictions against a simple
historical‑average estimate and show whether our system provides a
meaningful improvement on previously unseen configurations.
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Tuple[int, Dict[str, str]]], List[str]]:
    """Read CSV into list of (row_no, row_dict) and list of field names."""
    rows: List[Tuple[int, Dict[str, str]]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV has no header")
        for rn, rec in enumerate(reader, start=1):
            rows.append((rn, dict(rec)))
    return rows, fieldnames


def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_x(col: str) -> bool:
    return col.endswith("X")


def is_numeric(v: str) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


def parse_value(v: str) -> float | None:
    if v in ("", "?", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def feature_columns(fieldnames: List[str]) -> List[str]:
    return [c for c in fieldnames if not is_goal(c) and not is_x(c)]


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def compute_goal_stats(rows: List[Tuple[int, Dict[str, str]]], goal_cols: List[str]
                     ) -> Dict[str, Dict[str, float]]:
    """
    For each goal column, compute mean and standard deviation over all rows
    that have a numeric value for that column.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for col in goal_cols:
        vals = [parse_value(r[1][col]) for _, r in rows if parse_value(r[1][col]) is not None]
        n = len(vals)
        if n == 0:
            continue
        mu = sum(vals) / n
        if n > 1:
            var = sum((x - mu) ** 2 for x in vals) / (n - 1)
        else:
            var = 0.0
        sigma = math.sqrt(var)
        stats[col] = {"mean": mu, "sd": sigma}
    return stats


# ----------------------------------------------------------------------
# Scoring rows
# ----------------------------------------------------------------------
def row_zscore(row: Dict[str, str], goal_stats: Dict[str, Dict[str, float]]
              ) -> float:
    """Return sum of z-scores over all goal columns."""
    score = 0.0
    for col, stats in goal_stats.items():
        val = parse_value(row[col])
        if val is None or stats["sd"] == 0:
            continue
        z = (val - stats["mean"]) / stats["sd"]
        # For minimize columns, lower is better => use negative z
        if col.endswith("-"):
            z = -z
        score += z
    return score


def best_row(rows: List[Tuple[int, Dict[str, str]]],
             goal_stats: Dict[str, Dict[str, float]]
            ) -> Tuple[int, Dict[str, str]]:
    """Return row with best aggregate score (higher is better)."""
    best: Tuple[int, Dict[str, str]] | None = None
    best_score = -math.inf
    for rn, rec in rows:
        score = row_zscore(rec, goal_stats)
        if score > best_score or (score == best_score and (best is None or rn < best[0])):
            best = (rn, rec)
            best_score = score
    if best is None:
        raise RuntimeError("No rows to choose from")
    return best


# ----------------------------------------------------------------------
# Distance and kNN prediction
# ----------------------------------------------------------------------
def distance(row_a: Dict[str, str], row_b: Dict[str, str],
             feat_cols: List[str]) -> float:
    """Euclidean distance over feature columns, skipping missing values."""
    sq = 0.0
    cnt = 0
    for col in feat_cols:
        va = row_a.get(col)
        vb = row_b.get(col)
        if va in ("", "?", None) or vb in ("", "?", None):
            continue
        try:
            fa = float(va)
            fb = float(vb)
            sq += (fa - fb) ** 2
        except Exception:
            sq += 0 if va == vb else 1
        cnt += 1
    return math.sqrt(sq) if cnt > 0 else math.inf


def knn_prediction(row: Dict[str, str],
                   train_rows: List[Tuple[int, Dict[str, str]]],
                   goal_col: str,
                   feat_cols: List[str],
                   k: int = 3) -> float | None:
    """Return kNN prediction (mean of neighbor goals)."""
    neigh = []
    for _, rec in train_rows:
        d = distance(row, rec, feat_cols)
        neigh.append((d, rec))
    neigh.sort(key=lambda x: x[0])
    preds = []
    for d, rec in neigh:
        if d == math.inf:
            continue
        val = parse_value(rec[goal_col])
        if val is not None:
            preds.append(val)
        if len(preds) >= k:
            break
    if not preds:
        return None
    return sum(preds) / len(preds)


# ----------------------------------------------------------------------
# APACHE-19 logic
# ----------------------------------------------------------------------
def apache_19(rows: List[Tuple[int, Dict[str, str]]],
              fieldnames: List[str]) -> None:
    goal_cols = [c for c in fieldnames if is_goal(c)]
    if not goal_cols:
        print("No goal column found.")
        return
    goal_col = goal_cols[0]  # assume single goal
    feat_cols = feature_columns(fieldnames)

    # Compute stats for scoring
    goal_stats = compute_goal_stats(rows, goal_cols)

    # Build predictions for all rows using kNN
    preds = {}
    for rn, rec in rows:
        pred = knn_prediction(rec, [(r, c) for r, c in rows if r != rn], goal_col, feat_cols)
        # If prediction fails, use mean of all rows
        if pred is None:
            preds[rn] = sum(parse_value(r[1][goal_col]) for r in rows if parse_value(r[1][goal_col]) is not None) / len(rows)
        else:
            preds[rn] = pred

    # Select 30 rows with best predicted performance (lowest for minimize)
    budget = min(30, len(rows) // 5)
    sorted_rows = sorted(rows, key=lambda rc: (preds[rc[0]], rc[0]))
    ours_selected = sorted_rows[:budget]
    ours_best_row, _ = best_row(ours_selected, goal_stats)

    # Repeats
    repeats = 20
    ours_best_rows = [ours_best_row[0]] * repeats

    random_best_rows: List[int] = []
    for _ in range(repeats):
        sample = random.sample(rows, budget)
        best = best_row(sample, goal_stats)
        random_best_rows.append(best[0])

    # Compare
    ours_better = 0
    random_better = 0
    for o, r in zip(ours_best_rows, random_best_rows):
        o_val = parse_value(rows[o - 1][1][goal_col])
        r_val = parse_value(rows[r - 1][1][goal_col])
        # lower is better
        if o_val is not None and r_val is not None:
            if o_val < r_val:
                ours_better += 1
            elif r_val < o_val:
                random_better += 1

    if ours_better > random_better:
        verdict = "ours better"
    elif random_better > ours_better:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== APACHE-19 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Ours best rows: {ours_best_rows}")
    print(f"Random best rows: {random_best_rows}")
    print("ANSWER APACHE-19", json.dumps({
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }, separators=(",", ":")))


# ----------------------------------------------------------------------
# APACHE-20 logic
# ----------------------------------------------------------------------
def apache_20(rows: List[Tuple[int, Dict[str, str]]],
              fieldnames: List[str]) -> None:
    goal_cols = [c for c in fieldnames if is_goal(c)]
    if not goal_cols:
        print("No goal column found.")
        return
    goal_col = goal_cols[0]  # assume single goal
    feat_cols = feature_columns(fieldnames)

    # Identify held‑out rows (divisible by 5)
    held_out = [(rn, rec) for rn, rec in rows if rn % 5 == 0]
    train_rows = [(rn, rec) for rn, rec in rows if rn % 5 != 0]

    # Baseline mean
    train_vals = [parse_value(rec[goal_col]) for _, rec in train_rows if parse_value(rec[goal_col]) is not None]
    baseline = sum(train_vals) / len(train_vals) if train_vals else 0.0

    predictions: Dict[int, float] = {}
    errors: List[float] = []
    errors_baseline: List[float] = []

    for rn, rec in held_out:
        pred = knn_prediction(rec, train_rows, goal_col, feat_cols)
        if pred is None:
            pred = baseline
        actual = parse_value(rec[goal_col])
        if actual is None:
            continue
        predictions[rn] = pred
        errors.append(abs(pred - actual))
        errors_baseline.append(abs(baseline - actual))

    mae_model = sum(errors) / len(errors) if errors else 0.0
    mae_baseline = sum(errors_baseline) / len(errors_baseline) if errors_baseline else 0.0

    # Verdict
    if mae_model < mae_baseline * 0.95:
        verdict = "better"
    elif mae_model > mae_baseline * 1.05:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("== APACHE-20 ==")
    print(f"Number of held‑out rows: {len(held_out)}")
    print(f"Model MAE: {mae_model:.6f}")
    print(f"Baseline MAE: {mae_baseline:.6f}")
    print("ANSWER APACHE-20", json.dumps({
        "predictions": predictions,
        "baseline": baseline,
        "mae_model": round(mae_model, 6),
        "mae_baseline": round(mae_baseline, 6),
        "verdict": verdict
    }, separators=(",", ":")))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    apache_19(rows, fieldnames)
    print()
    apache_20(rows, fieldnames)


if __name__ == "__main__":
    random.seed(12345)   # reproducibility
    main()
