#!/usr/bin/env python3
"""
Small data‑analysis skill for the auto93 dataset.

Usage:
    python skill_01.py <path_to_csv>
"""

import sys
import csv
import json
import numpy as np

# Columns of interest (all goal columns)
GOAL_COLUMNS = ["Lbs-", "Acc+", "Mpg+"]

def load_data(csv_path):
    """
    Load the CSV and parse numeric values.
    Returns a list of dicts with keys:
        'row'    : 1‑based row number
        'values' : dict mapping GOAL_COLUMNS to float or None
    """
    data = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            values = {}
            for col in GOAL_COLUMNS:
                raw = row.get(col, None)
                if raw is None or raw == "?":
                    values[col] = None
                else:
                    try:
                        values[col] = float(raw)
                    except ValueError:
                        values[col] = None
            data.append({"row": row_num, "values": values})
    return data

def compute_stats(values_list):
    """
    Compute mean, median, min, max, best, worst for a list of floats.
    `values_list` must contain no None.
    Returns a dict with keys:
        'mean', 'median', 'min', 'max', 'best', 'worst'
    """
    arr = np.array(values_list)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    return {
        "mean": mean,
        "median": median,
        "min": min_val,
        "max": max_val,
        "best": min_val,   # to be adjusted later based on direction
        "worst": max_val,  # to be adjusted later based on direction
    }

def summarize_columns(data):
    """
    Compute statistics for each goal column.
    Returns a dict mapping column name to stats dict.
    """
    summary = {}
    for col in GOAL_COLUMNS:
        vals = [row["values"][col] for row in data if row["values"][col] is not None]
        stats = compute_stats(vals)
        # Adjust best / worst according to direction
        if col.endswith("-"):  # minimize
            stats["best"] = min(vals)
            stats["worst"] = max(vals)
        else:  # maximize
            stats["best"] = max(vals)
            stats["worst"] = min(vals)
        summary[col] = stats
    return summary

def compute_overall_top10(data):
    """
    Identify the 10 rows with the best overall combination of
    low weight (Lbs-), high acceleration (Acc+), and high fuel
    efficiency (Mpg+).  Missing values disqualify a row.
    Returns a list of row numbers sorted by descending score.
    """
    # Gather all values per column (exclude missing)
    all_vals = {col: [] for col in GOAL_COLUMNS}
    for row in data:
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            if val is not None:
                all_vals[col].append(val)

    # Compute means and stds
    means = {col: np.mean(all_vals[col]) for col in GOAL_COLUMNS}
    stds = {col: np.std(all_vals[col], ddof=0) for col in GOAL_COLUMNS}

    scores = []
    for row in data:
        # Skip rows with any missing goal value
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

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    top_rows = [row_num for row_num, _ in scores[:10]]
    return top_rows

def main(csv_path):
    data = load_data(csv_path)

    # === CAR-01 ===
    print("== CAR-01 ==")
    summary = summarize_columns(data)
    for col in GOAL_COLUMNS:
        stats = summary[col]
        print(f"{col} mean: {stats['mean']:.2f}")
        print(f"{col} median: {stats['median']:.2f}")
        print(f"{col} min: {stats['min']:.2f}")
        print(f"{col} max: {stats['max']:.2f}")
        print(f"{col} best: {stats['best']:.2f}")
        print(f"{col} worst: {stats['worst']:.2f}")
    # Machine‑checked answer
    answer_json = json.dumps(summary, separators=(",", ":"))
    print(f"ANSWER CAR-01 {answer_json}")

    # === CAR-02 ===
    print("\n== CAR-02 ==")
    top_rows = compute_overall_top10(data)
    print("Top 10 rows by overall score:")
    for rn in top_rows:
        print(rn)
    # Machine‑checked answer
    answer_json = json.dumps({"top": top_rows}, separators=(",", ":"))
    print(f"ANSWER CAR-02 {answer_json}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
