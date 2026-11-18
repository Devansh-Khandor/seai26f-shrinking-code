#!/usr/bin/env python3
"""
Skill 01: Inventory summary and attractive vehicle ranking

Usage:
    python skill_01.py <path to auto93.csv>
"""

import csv
import sys
import json
import statistics

# Columns of interest
GOAL_COLUMNS = ["Lbs-", "Acc+", "Mpg+"]

def read_csv(path):
    """Read the CSV and return list of rows with numeric conversion."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if v == "?":
                    cleaned[k] = None
                else:
                    # Convert to float if numeric column
                    if k in GOAL_COLUMNS or k.endswith(("X", "-", "+")):
                        try:
                            cleaned[k] = float(v)
                        except ValueError:
                            cleaned[k] = None
                    else:
                        cleaned[k] = v
            rows.append(cleaned)
    return rows

def compute_stats(values):
    """Return mean, median, min, max of a list of numbers."""
    mean = sum(values) / len(values)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    mn = sorted_vals[0]
    mx = sorted_vals[-1]
    return mean, median, mn, mx

def main(csv_path):
    rows = read_csv(csv_path)

    # Gather numeric values per column
    col_values = {col: [] for col in GOAL_COLUMNS}
    for row in rows:
        for col in GOAL_COLUMNS:
            val = row.get(col)
            if val is not None:
                col_values[col].append(val)

    # CAR-01: compute stats
    summary = {}
    for col in GOAL_COLUMNS:
        vals = col_values[col]
        mean, median, mn, mx = compute_stats(vals)
        # Determine best/worst based on goal direction
        if col.endswith("-"):  # minimize
            best, worst = mn, mx
        else:  # maximize
            best, worst = mx, mn
        summary[col] = {
            "mean": mean,
            "median": median,
            "min": mn,
            "max": mx,
            "best": best,
            "worst": worst
        }

    # Print CAR-01 section
    print("== CAR-01 ==")
    # Minimal results (at least 3 lines)
    for col in GOAL_COLUMNS:
        stats = summary[col]
        print(f"{col}: mean={stats['mean']:.3f} median={stats['median']:.3f} "
              f"min={stats['min']:.3f} max={stats['max']:.3f} "
              f"best={stats['best']:.3f} worst={stats['worst']:.3f}")
    print("ANSWER CAR-01 " + json.dumps(summary))

    # CAR-02: ranking vehicles by combined attractiveness
    # Compute mean and std for each column
    means = {}
    stds = {}
    for col in GOAL_COLUMNS:
        vals = col_values[col]
        means[col] = statistics.mean(vals)
        stds[col] = statistics.stdev(vals) if len(vals) > 1 else 0.0

    # Compute combined score for each row (only if all three values present)
    scored_rows = []
    for idx, row in enumerate(rows, start=1):  # 1-based row number
        vals = [row.get(col) for col in GOAL_COLUMNS]
        if None in vals:
            continue  # skip rows with missing values in any goal column
        score = 0.0
        # Weight: negative z for Lbs- (lower better), positive z for others
        for col, val in zip(GOAL_COLUMNS, vals):
            mu = means[col]
            sigma = stds[col]
            z = 0.0 if sigma == 0 else (val - mu) / sigma
            if col.endswith("-"):  # minimize
                score += -z
            else:
                score += z
        scored_rows.append((score, idx))

    # Sort by score descending, then by row number ascending
    scored_rows.sort(key=lambda x: (-x[0], x[1]))
    top_rows = [idx for _, idx in scored_rows[:10]]

    # Print CAR-02 section
    print("\n== CAR-02 ==")
    print("Top 10 rows (most attractive): " + ", ".join(map(str, top_rows)))
    answer = {"top": top_rows}
    print("ANSWER CAR-02 " + json.dumps(answer))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
