#!/usr/bin/env python3
"""
Small data‑analysis skill for the auto93 dataset (CAR‑07 and CAR‑08).

Usage:
    python skill_04.py <path_to_csv>
"""

import sys
import json
import numpy as np
from skill_01 import load_data, GOAL_COLUMNS

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def compute_top_n(data_subset, n=10):
    """
    Return a list of row numbers for the top `n` rows in data_subset
    according to the combined score used in CAR‑02.
    Rows with any missing goal value are skipped.
    """
    # Gather values per column for mean/std computation
    all_vals = {col: [] for col in GOAL_COLUMNS}
    for row in data_subset:
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            if val is not None:
                all_vals[col].append(val)
    if not all_vals[GOAL_COLUMNS[0]]:  # no valid rows
        return []

    means = {col: np.mean(all_vals[col]) for col in GOAL_COLUMNS}
    stds  = {col: np.std(all_vals[col], ddof=0) for col in GOAL_COLUMNS}

    scores = []
    for row in data_subset:
        if any(row["values"][col] is None for col in GOAL_COLUMNS):
            continue
        score = 0.0
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            std = stds[col]
            z = (val - means[col]) / std if std != 0 else 0.0
            if col.endswith("-"):  # lower better
                score -= z
            else:
                score += z
        scores.append((row["row"], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [row_num for row_num, _ in scores[:n]]


def compute_summary(data_subset):
    """
    Compute summary statistics (n, mean, min, max) for each goal column
    over the provided data_subset.
    """
    summary = {}
    for col in GOAL_COLUMNS:
        vals = [row["values"][col] for row in data_subset if row["values"][col] is not None]
        n = len(vals)
        if n == 0:
            mean = min_val = max_val = 0.0
        else:
            mean = float(np.mean(vals))
            min_val = float(np.min(vals))
            max_val = float(np.max(vals))
        summary[col] = {
            "n": n,
            "mean": mean,
            "min": min_val,
            "max": max_val
        }
    return summary


# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
def main(csv_path):
    data = load_data(csv_path)
    total_rows = len(data)

    # Determine arrivals (last round(0.2*rows) rows)
    arrivals_count = int(round(0.2 * total_rows))
    arrivals = data[-arrivals_count:] if arrivals_count > 0 else []
    history = data[:-arrivals_count] if arrivals_count > 0 else data

    # --- CAR-07 -------------------------------------------------------
    print("== CAR-07 ==")

    assessments = []
    # Iterate arrivals in order, maintaining current data
    current_data = list(history)  # start with history
    for arrival in arrivals:
        current_data.append(arrival)  # add current arrival
        top10 = compute_top_n(current_data, n=10)
        desirable = arrival["row"] in top10
        assessments.append({"row": arrival["row"], "desirable": desirable})

    # Provide a few lines of output
    print(f"Evaluated {len(arrivals)} arrivals.")
    print(f"First 3 assessments: {assessments[:3]}")
    # Machine‑checked answer
    answer_json = json.dumps({"assessments": assessments}, separators=(",", ":"))
    print(f"ANSWER CAR-07 {answer_json}")

    # --- CAR-08 -------------------------------------------------------
    print("\n== CAR-08 ==")

    checkpoints = []
    # checkpoints after 0, 40 and 80 arrivals (or less if not enough arrivals)
    checkpoint_arrivals = [0, 40, 80]
    for m in checkpoint_arrivals:
        if m > len(arrivals):
            m = len(arrivals)
        current_data = list(history) + arrivals[:m]
        # Summary per goal column
        summary = compute_summary(current_data)
        # Shortlist best 5 rows
        shortlist = compute_top_n(current_data, n=5)
        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })
        # Output a brief line for the checkpoint
        print(f"Checkpoint after {m} arrivals: shortlist {shortlist}")

    # Machine‑checked answer
    answer_json = json.dumps({"checkpoints": checkpoints}, separators=(",", ":"))
    print(f"ANSWER CAR-08 {answer_json}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
