#!/usr/bin/env python3
"""
skill_04.py - CAR-07 and CAR-08: Real‑time arrival assessment and incremental inventory summary

Usage:
    python skill_04.py <csv_path>
"""

import sys
import csv
import json
import math
import statistics

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']

def read_csv(path):
    """Read CSV and return list of dicts with numeric conversion. Missing values become None."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in GOAL_COLUMNS:
                val = row[col]
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = None
            # Convert all other columns to float for completeness
            for col in ['Clndrs', 'Volume', 'HpX', 'Model', 'origin']:
                try:
                    row[col] = float(row[col])
                except ValueError:
                    row[col] = None
            rows.append(row)
    return rows

def summary_stats(values):
    """Return n, mean, min, max for a list of floats. If list empty return zeros."""
    n = len(values)
    if n == 0:
        return 0, None, None, None
    mean_val = statistics.mean(values)
    return n, mean_val, min(values), max(values)

def normalize(val, v_min, v_max, reverse=False):
    if v_max == v_min:
        return 0.5
    score = (val - v_min) / (v_max - v_min)
    return 1.0 - score if reverse else score

def compute_scores(rows, minmax):
    """Return dict mapping row_index (1‑based) to composite score for rows with all goals."""
    scores = {}
    for idx, row in enumerate(rows, start=1):
        if any(row[col] is None for col in GOAL_COLUMNS):
            continue
        lbs, acc, mpg = row['Lbs-'], row['Acc+'], row['Mpg+']
        lbs_min, lbs_max = minmax['Lbs-']
        acc_min, acc_max = minmax['Acc+']
        mpg_min, mpg_max = minmax['Mpg+']
        score = (
            normalize(lbs, lbs_min, lbs_max, reverse=True) +
            normalize(acc, acc_min, acc_max, reverse=False) +
            normalize(mpg, mpg_min, mpg_max, reverse=False)
        )
        scores[idx] = score
    return scores

def top5_rows_by_score(scores):
    """Return list of row numbers (1‑based) of top 5 scores sorted descending, tie by row number."""
    # scores is dict {row_idx: score}
    sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [row for row, _ in sorted_items[:5]]

def compute_goal_stats(rows):
    """Return dict of goal -> stats dict for the given rows."""
    stats = {}
    for col in GOAL_COLUMNS:
        vals = [row[col] for row in rows if row[col] is not None]
        n, mean_val, min_val, max_val = summary_stats(vals)
        stats[col] = {
            "n": n,
            "mean": mean_val,
            "min": min_val,
            "max": max_val
        }
    return stats

# ----------------------------------------------------------------------
# CAR-07 and CAR-08 logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    total_rows = len(rows)

    # Determine arrivals
    num_arrivals = round(0.2 * total_rows)
    history_rows = rows[:total_rows - num_arrivals]
    arrival_rows = rows[total_rows - num_arrivals:]

    # Map arrival positions to global row indices
    arrival_indices = list(range(total_rows - num_arrivals + 1, total_rows + 1))

    # ---------- CAR-07 ----------
    print("== CAR-07 ==")
    assessments = []
    # Maintain seen rows up to each arrival
    seen_rows = history_rows.copy()
    for pos, idx in enumerate(arrival_indices, start=1):
        # Append this arrival to seen rows
        seen_rows.append(arrival_rows[pos-1])

        # Compute min/max per goal over seen rows
        minmax = {}
        for col in GOAL_COLUMNS:
            vals = [row[col] for row in seen_rows if row[col] is not None]
            if vals:
                minmax[col] = (min(vals), max(vals))
            else:
                minmax[col] = (0.0, 0.0)  # arbitrary when no data

        # Compute scores for all seen rows
        scores = compute_scores(seen_rows, minmax)

        # Determine if current arrival is in top 5
        top5 = top5_rows_by_score(scores)
        desirable = idx in top5

        assessments.append({"row": idx, "desirable": desirable})

        # Print a human‑friendly line for this arrival
        print(f"Arrival row {idx}: {'desirable' if desirable else 'not desirable'}")

    # Machine‑checked answer
    print("ANSWER CAR-07", json.dumps({"assessments": assessments}, separators=(',', ':')))

    # ---------- CAR-08 ----------
    print("\n== CAR-08 ==")
    checkpoints = []
    # Define checkpoints after 0, 40, 80 arrivals
    checkpoint_points = [0, 40, 80]
    for m in checkpoint_points:
        # Rows seen up to this checkpoint: history + first m arrivals
        seen = history_rows + arrival_rows[:m]
        # Compute goal stats
        stats = compute_goal_stats(seen)
        # For shortlist, compute scores over seen
        # Need minmax over seen
        minmax = {}
        for col in GOAL_COLUMNS:
            vals = [row[col] for row in seen if row[col] is not None]
            if vals:
                minmax[col] = (min(vals), max(vals))
            else:
                minmax[col] = (0.0, 0.0)
        scores = compute_scores(seen, minmax)
        shortlist = top5_rows_by_score(scores)

        checkpoints.append({
            "after": m,
            "summary": stats,
            "shortlist": shortlist
        })

        # Print human‑friendly summary
        print(f"Checkpoint after {m} arrivals:")
        for col in GOAL_COLUMNS:
            s = stats[col]
            print(f"  {col}: n={s['n']}, mean={s['mean']:.4g}, min={s['min']}, max={s['max']}")
        print(f"  Top 5 rows: {shortlist}")

    # Machine‑checked answer
    print("ANSWER CAR-08", json.dumps({"checkpoints": checkpoints}, separators=(',', ':')))

if __name__ == "__main__":
    main()
