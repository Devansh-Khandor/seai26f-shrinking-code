#!/usr/bin/env python3
"""
Small data‑analysis skill for the auto93 dataset (CAR‑11 and CAR‑12).

Usage:
    python skill_06.py <path_to_csv>
"""

import sys
import json
import numpy as np
from skill_01 import load_data, GOAL_COLUMNS

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def compute_all_scores(data):
    """
    Compute a score for every row that has all goal values present.
    The scoring method is identical to that used in skill_01.compute_overall_top10.
    Returns a list of (row_number, score) tuples.
    """
    all_vals = {col: [] for col in GOAL_COLUMNS}
    for row in data:
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            if val is not None:
                all_vals[col].append(val)

    means = {col: np.mean(all_vals[col]) for col in GOAL_COLUMNS}
    stds = {col: np.std(all_vals[col], ddof=0) for col in GOAL_COLUMNS}

    scores = []
    for row in data:
        if any(row["values"][col] is None for col in GOAL_COLUMNS):
            continue
        score = 0.0
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            std = stds[col]
            z = (val - means[col]) / std if std != 0 else 0.0
            if col.endswith("-"):   # lower is better
                score -= z
            else:                    # higher is better
                score += z
        scores.append((row["row"], score))
    return scores

def compute_z_scores(data, means, stds):
    """
    Return a dict mapping row number to dict of z-scores for each goal column.
    """
    z_map = {}
    for row in data:
        vals = {}
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            if val is None:
                vals[col] = None
            else:
                std = stds[col]
                vals[col] = (val - means[col]) / std if std != 0 else 0.0
        z_map[row["row"]] = vals
    return z_map

def euclidean_distance(z1, z2):
    """
    Euclidean distance between two dicts of z-scores.
    """
    diff_sq = 0.0
    for col in GOAL_COLUMNS:
        if z1[col] is None or z2[col] is None:
            continue
        d = z1[col] - z2[col]
        diff_sq += d * d
    return np.sqrt(diff_sq)

# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
def main(csv_path):
    data = load_data(csv_path)

    # Compute overall scores for every row
    all_scores = compute_all_scores(data)
    all_scores.sort(key=lambda x: x[1], reverse=True)  # higher = better

    if len(all_scores) < 2:
        print("Insufficient data to compare vehicles.")
        sys.exit(1)

    # --- CAR-11 -------------------------------------------------------
    print("== CAR-11 ==")
    # Recommended vehicle is the best-scoring row
    best_row_num, best_score = all_scores[0]
    best_row = next(row for row in data if row["row"] == best_row_num)

    # Compute z-scores for distance calculation
    all_vals = {col: [] for col in GOAL_COLUMNS}
    for row in data:
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            if val is not None:
                all_vals[col].append(val)
    means = {col: np.mean(all_vals[col]) for col in GOAL_COLUMNS}
    stds = {col: np.std(all_vals[col], ddof=0) for col in GOAL_COLUMNS}
    z_map = compute_z_scores(data, means, stds)

    # Find 3 nearest neighbors (excluding the recommended row)
    distances = []
    for row in data:
        if row["row"] == best_row_num:
            continue
        dist = euclidean_distance(z_map[best_row_num], z_map[row["row"]])
        distances.append((row["row"], dist, row))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for rnum, dist, row in distances[:3]:
        val_dict = {col: row["values"][col] for col in GOAL_COLUMNS}
        neighbors.append({"row": rnum, "values": val_dict})

    # Output results
    print(f"Recommended vehicle: row {best_row_num} (score: {best_score:.4f})")
    print("Neighbors (closest by goal metrics):")
    for nb in neighbors:
        vals = ", ".join(f"{k}={v}" for k, v in nb["values"].items())
        print(f"  Row {nb['row']}: {vals}")

    # Machine‑checked answer
    answer_json = json.dumps({
        "recommended": best_row_num,
        "neighbors": neighbors
    }, separators=(",", ":"))
    print(f"ANSWER CAR-11 {answer_json}")

    # --- CAR-12 -------------------------------------------------------
    print("\n== CAR-12 ==")
    # Pick two vehicles with different strengths: top two scores
    a_row_num, a_score = all_scores[0]
    b_row_num, b_score = all_scores[1]
    a_row = next(row for row in data if row["row"] == a_row_num)
    b_row = next(row for row in data if row["row"] == b_row_num)

    # Differences
    diff_list = []
    for col in GOAL_COLUMNS:
        va = a_row["values"][col]
        vb = b_row["values"][col]
        diff = abs(va - vb)
        diff_list.append({"feature": col, "a": va, "b": vb, "diff": diff})
    diff_list.sort(key=lambda x: x["diff"], reverse=True)
    diff_output = [{"feature": d["feature"], "a": d["a"], "b": d["b"]} for d in diff_list[:3]]

    # Tradeoffs: goals where the non‑preferred row is better
    tradeoffs = []
    for col in GOAL_COLUMNS:
        va = a_row["values"][col]
        vb = b_row["values"][col]
        if col.endswith("-"):   # lower is better
            if vb < va:
                tradeoffs.append(col)
        else:                    # higher is better
            if vb > va:
                tradeoffs.append(col)

    # Determine preferred
    preferred = a_row_num if a_score >= b_score else b_row_num

    # Output results
    print(f"Vehicle A: row {a_row_num} (score: {a_score:.4f})")
    print(f"Vehicle B: row {b_row_num} (score: {b_score:.4f})")
    print(f"Preferred vehicle: {preferred}")
    print("Key differences (top 3):")
    for d in diff_output:
        print(f"  {d['feature']}: A={d['a']}, B={d['b']}")
    print("Tradeoffs where non‑preferred is better:")
    print("  " + ", ".join(tradeoffs) if tradeoffs else "  None")

    # Machine‑checked answer
    answer_json = json.dumps({
        "a": a_row_num,
        "b": b_row_num,
        "preferred": preferred,
        "scores": {str(a_row_num): a_score, str(b_row_num): b_score},
        "differences": diff_output,
        "tradeoffs": tradeoffs
    }, separators=(",", ":"))
    print(f"ANSWER CAR-12 {answer_json}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
