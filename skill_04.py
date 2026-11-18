#!/usr/bin/env python3
"""
Skill 04: Real‑time assessment and dynamic summary updates

Usage:
    python skill_04.py <path to auto93.csv>
"""

import json
import sys
from skill_01 import read_csv

# Goal columns (same as in skill_01)
GOAL_COLUMNS = ["Lbs-", "Acc+", "Mpg+"]

# ----------------------------------------------------------------------
# Statistics helpers

def mean(values):
    return sum(values) / len(values)

def std_sample(values):
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (n - 1)) ** 0.5

def compute_means_stds(rows, indices):
    """Return dict of mean and std for each goal column over the given indices."""
    means = {}
    stds = {}
    for col in GOAL_COLUMNS:
        vals = [rows[i][col] for i in indices if rows[i][col] is not None]
        if vals:
            means[col] = mean(vals)
            stds[col] = std_sample(vals)
        else:
            # Column missing entirely – use 0 mean and 0 std (will yield z=0)
            means[col] = 0.0
            stds[col] = 0.0
    return means, stds

def compute_score(row, means, stds):
    """Return attractiveness score for a row with all goal values present."""
    if any(row.get(c) is None for c in GOAL_COLUMNS):
        return None
    score = 0.0
    for col in GOAL_COLUMNS:
        val = row[col]
        mu = means[col]
        sigma = stds[col]
        z = 0.0 if sigma == 0 else (val - mu) / sigma
        score += -z if col.endswith("-") else z
    return score

# ----------------------------------------------------------------------
# Main logic

def main(csv_path):
    rows = read_csv(csv_path)
    n_rows = len(rows)
    arrivals_cnt = round(0.2 * n_rows)
    history_cnt = n_rows - arrivals_cnt
    history = rows[:history_cnt]
    arrivals = rows[history_cnt:]

    # --------------------------------------------------------------
    # CAR-07: Assess each new arrival in real time
    assessments = []
    for i, arr_row in enumerate(arrivals):
        # Current dataset up to this arrival
        cur_indices = list(range(history_cnt)) + list(range(history_cnt, history_cnt + i + 1))
        means, stds = compute_means_stds(rows, cur_indices)
        # Compute scores for all rows in the current dataset
        scored = []
        for idx in cur_indices:
            row = rows[idx]
            s = compute_score(row, means, stds)
            if s is not None:
                scored.append((s, idx + 1))  # 1‑based row number
        scored.sort(key=lambda x: (-x[0], x[1]))
        top5 = [rno for _, rno in scored[:5]]
        row_no = history_cnt + i + 1
        assessments.append({"row": row_no, "desirable": row_no in top5})

    # Output CAR-07
    print("== CAR-07 ==")
    for a in assessments:
        # print at least three lines of results
        print(f"Row {a['row']}: desirable={str(a['desirable']).lower()}")
    print("ANSWER CAR-07 " + json.dumps({"assessments": assessments}))

    # --------------------------------------------------------------
    # CAR-08: Dynamic summaries after 0, 40, 80 arrivals
    checkpoints = []
    for m in [0, 40, 80]:
        if m > arrivals_cnt:
            m = arrivals_cnt
        cur_indices = list(range(history_cnt)) + list(range(history_cnt, history_cnt + m))
        means, stds = compute_means_stds(rows, cur_indices)
        summary = {}
        for col in GOAL_COLUMNS:
            vals = [rows[i][col] for i in cur_indices if rows[i][col] is not None]
            if vals:
                summary[col] = {
                    "n": len(vals),
                    "mean": mean(vals),
                    "min": min(vals),
                    "max": max(vals)
                }
            else:
                summary[col] = {"n": 0, "mean": None, "min": None, "max": None}
        # Shortlist top 5
        scored = []
        for idx in cur_indices:
            row = rows[idx]
            s = compute_score(row, means, stds)
            if s is not None:
                scored.append((s, idx + 1))
        scored.sort(key=lambda x: (-x[0], x[1]))
        shortlist = [rno for _, rno in scored[:5]]
        checkpoints.append({"after": m, "summary": summary, "shortlist": shortlist})

    # Output CAR-08
    print("\n== CAR-08 ==")
    for cp in checkpoints:
        after = cp["after"]
        print(f"Checkpoint after {after} arrivals:")
        for col in GOAL_COLUMNS:
            s = cp["summary"][col]
            if s["n"] > 0:
                print(f"  {col}: n={s['n']} mean={s['mean']:.3f} min={s['min']:.3f} max={s['max']:.3f}")
        print(f"  Shortlist: {cp['shortlist']}")
    print("ANSWER CAR-08 " + json.dumps({"checkpoints": checkpoints}))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
