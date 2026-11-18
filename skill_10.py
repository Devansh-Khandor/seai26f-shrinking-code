#!/usr/bin/env python3
"""
Skill 10: NASA-19 and NASA-20

This script compares our project‑effort estimation system
against a simple historical‑average estimate (NASA‑19)
and compares our project‑selection approach with
random selection under a fixed evaluation budget (NASA‑20).

The implementation re‑uses helper functions from earlier skill scripts
and follows the same conventions for data handling, missing values,
held‑out rows, arrivals, and budgeting.

Usage:
    python skill_10.py <csv path>
"""

import csv
import json
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# ------------------------------------------------------------------
# Import the shared data reader from skill_01
try:
    from skill_01 import read_data, GOAL_COLUMNS
except Exception:
    # Minimal fallback if skill_01 is missing
    def read_data(csv_path: str) -> Tuple[List[Dict[str, Any]], List[int]]:
        rows: List[Dict[str, Any]] = []
        row_nums: List[int] = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for rn, rec in enumerate(reader, start=1):
                row_nums.append(rn)
                parsed: Dict[str, Any] = {}
                for key, val in rec.items():
                    val = val.strip()
                    if val == "?":
                        parsed[key] = None
                    else:
                        try:
                            if "." in val or "e" in val.lower():
                                parsed[key] = float(val)
                            else:
                                parsed[key] = int(val)
                        except ValueError:
                            parsed[key] = val
                rows.append(parsed)
        return rows, row_nums

    GOAL_COLUMNS = {
        "Kloc+": "max",
        "Effort-": "min",
        "Defects-": "min",
    }

# ------------------------------------------------------------------
# Helper utilities
def is_numeric(val: Any) -> bool:
    return isinstance(val, (int, float))

def compute_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float('nan')

# ------------------------------------------------------------------
# NASA‑19: effort estimation
def get_non_goal_cols(rows: List[Dict[str, Any]]) -> List[str]:
    """Return list of columns that are not goals and do not end with 'X'."""
    if not rows:
        return []
    all_cols = rows[0].keys()
    return [c for c in all_cols if c not in GOAL_COLUMNS and not c.endswith("X")]

def similarity_score(row_a: Dict[str, Any], row_b: Dict[str, Any], cols: List[str]) -> int:
    """Count matching non‑goal cells between two rows."""
    score = 0
    for col in cols:
        va = row_a.get(col)
        vb = row_b.get(col)
        if va is None or vb is None:
            continue
        if va == vb:
            score += 1
    return score

def estimate_effort_heldout(
    heldout_idx: int,
    train_rows: List[Dict[str, Any]],
    non_goal_cols: List[str],
    k_neighbors: int = 3
) -> float:
    """Predict Effort- for a held‑out row using k nearest neighbors."""
    heldrow = train_rows[heldout_idx]
    # Build list of (score, index) for all training rows
    scores = []
    for idx, row in enumerate(train_rows):
        if is_numeric(row.get("Effort-")):
            sc = similarity_score(heldrow, row, non_goal_cols)
            scores.append((sc, idx))
    if not scores:
        return float('nan')
    # Sort by score descending
    scores.sort(key=lambda x: (-x[0], x[1]))
    # Take top k_neighbors
    neighbor_vals = [train_rows[i]["Effort-"] for _, i in scores[:k_neighbors]]
    return compute_mean(neighbor_vals)

def nasa_19(csv_path: str) -> None:
    rows, row_nums = read_data(csv_path)
    non_goal_cols = get_non_goal_cols(rows)

    # Separate held‑out (divisible by 5) and training rows
    heldout_indices = [i for i, rn in enumerate(row_nums) if rn % 5 == 0]
    train_indices = [i for i in range(len(rows)) if i not in heldout_indices]

    train_rows = [rows[i] for i in train_indices]
    train_efforts = [rows[i]["Effort-"] for i in train_indices if is_numeric(rows[i].get("Effort-"))]
    baseline_mean = compute_mean(train_efforts)

    predictions: Dict[str, float] = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count = 0

    for hi in heldout_indices:
        actual = rows[hi].get("Effort-")
        if not is_numeric(actual):
            continue  # skip rows without a real Effort-
        pred = estimate_effort_heldout(hi, train_rows, non_goal_cols)
        if not is_numeric(pred):
            pred = baseline_mean
        predictions[str(row_nums[hi])] = round(pred, 6)
        mae_model += abs(pred - actual)
        mae_baseline += abs(baseline_mean - actual)
        count += 1

    mae_model /= count if count else float('nan')
    mae_baseline /= count if count else float('nan')

    # Determine verdict
    eps = 1e-6
    if mae_model + eps < mae_baseline:
        verdict = "better"
    elif mae_baseline + eps < mae_model:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Print results
    print("== NASA-19 ==")
    print(f"Baseline mean Effort-: {baseline_mean:.4f}")
    print(f"MAE (model): {mae_model:.4f}")
    print(f"MAE (baseline): {mae_baseline:.4f}")
    print(f"Verdict: {verdict}")

    answer = {
        "predictions": predictions,
        "baseline": round(baseline_mean, 6),
        "mae_model": round(mae_model, 6),
        "mae_baseline": round(mae_baseline, 6),
        "verdict": verdict,
    }
    print(f"ANSWER NASA-19 {json.dumps(answer)}")

# ------------------------------------------------------------------
# NASA‑20: selection vs random
def compute_combined_rank(rows_subset: List[Dict[str, Any]]) -> Dict[int, Tuple[int, float]]:
    """
    Compute combined rank for each row in rows_subset.
    Returns a dict mapping index to (total_rank, Kloc+ value) for tie‑breaking.
    """
    # Prepare rankings per goal
    ranks: Dict[str, Dict[int, int]] = {}
    for g in GOAL_COLUMNS:
        vals = [(row[g], idx) for idx, row in enumerate(rows_subset) if is_numeric(row.get(g))]
        vals.sort(key=lambda x: x[0], reverse=(GOAL_COLUMNS[g] == "max"))
        rank_dict = {}
        for rank, (_, idx) in enumerate(vals, start=1):
            rank_dict[idx] = rank
        ranks[g] = rank_dict

    combined = {}
    for idx in range(len(rows_subset)):
        total_rank = sum(ranks[g].get(idx, len(rows_subset)) for g in GOAL_COLUMNS)
        kloc_val = rows_subset[idx].get("Kloc+")
        combined[idx] = (total_rank, kloc_val if is_numeric(kloc_val) else 0)
    return combined

def nasa_20(csv_path: str) -> None:
    rows, row_nums = read_data(csv_path)
    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    repeats = 20
    random.seed(0)

    ours_best_rows: List[int] = []
    random_best_rows: List[int] = []
    ours_ranks: List[int] = []
    random_ranks: List[int] = []

    for _ in range(repeats):
        sample_indices = random.sample(range(n_rows), budget)
        subset = [rows[i] for i in sample_indices]

        rank_map = compute_combined_rank(subset)

        # Our best: lowest total_rank, tie by Kloc+
        best_idx = min(range(budget), key=lambda i: (rank_map[i][0], rank_map[i][1]))
        ours_best_rows.append(row_nums[sample_indices[best_idx]])
        ours_ranks.append(rank_map[best_idx][0])

        # Random best: pick one random row from the sample
        rand_choice = random.choice(range(budget))
        random_best_rows.append(row_nums[sample_indices[rand_choice]])
        random_ranks.append(rank_map[rand_choice][0])

    # Compare average ranks
    avg_ours = sum(ours_ranks) / repeats
    avg_rand = sum(random_ranks) / repeats
    eps = 1e-6
    if avg_ours + eps < avg_rand:
        verdict = "ours better"
    elif avg_rand + eps < avg_ours:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Print results
    print("== NASA-20 ==")
    print(f"Budget: {budget}, Repeats: {repeats}")
    print(f"Average rank ours: {avg_ours:.2f}, random: {avg_rand:.2f}")
    print(f"Verdict: {verdict}")

    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict,
    }
    print(f"ANSWER NASA-20 {json.dumps(answer)}")

# ------------------------------------------------------------------
# Main entry point
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    nasa_19(csv_path)
    print()  # blank line between sections
    nasa_20(csv_path)

if __name__ == "__main__":
    main()
