#!/usr/bin/env python3
"""
Skill 01 – NASA-01 & NASA-02
"""
import csv
import sys
import json
import math
import statistics

GOALS = ["Kloc+", "Effort-", "Defects-"]  # order matters for output

def parse_value(v):
    """Parse a CSV cell to float, return None for missing."""
    if v == "" or v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None

def compute_stats(rows):
    """Compute mean, median, min, max, best, worst for each goal."""
    stats = {}
    for goal in GOALS:
        values = [row[goal] for row in rows if row[goal] is not None]
        if not values:
            # No data available
            stats[goal] = {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "best": None,
                "worst": None,
            }
            continue
        mean_val = sum(values) / len(values)
        median_val = statistics.median(values)
        min_val = min(values)
        max_val = max(values)
        # Determine best/worst based on direction (+ maximize, - minimize)
        if goal.endswith("+"):
            best_val = max_val
            worst_val = min_val
        else:  # goal ends with "-"
            best_val = min_val
            worst_val = max_val
        stats[goal] = {
            "mean": mean_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "best": best_val,
            "worst": worst_val,
        }
    return stats

def normalise_composite(rows):
    """
    Return list of tuples (row_number, composite_score) for rows
    that have all three goal values present.
    Composite = norm(Kloc+) + norm(Effort-) + norm(Defects-)
    """
    # Gather min and max for each goal
    mins = {}
    maxs = {}
    for goal in GOALS:
        values = [row[goal] for row in rows if row[goal] is not None]
        if not values:
            mins[goal] = maxs[goal] = None
        else:
            mins[goal] = min(values)
            maxs[goal] = max(values)

    scores = []
    for idx, row in enumerate(rows, start=1):
        # skip if any goal missing
        if any(row[goal] is None for goal in GOALS):
            continue
        norm_values = []
        for goal in GOALS:
            val = row[goal]
            min_val = mins[goal]
            max_val = maxs[goal]
            if max_val == min_val:
                norm = 0.0
            else:
                if goal.endswith("+"):
                    norm = (val - min_val) / (max_val - min_val)
                else:  # minimize
                    norm = (max_val - val) / (max_val - min_val)
            norm_values.append(norm)
        composite = sum(norm_values)
        scores.append((idx, composite))
    return scores

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for col in reader.fieldnames:
                if col in GOALS:
                    parsed[col] = parse_value(row[col])
                else:
                    parsed[col] = row[col]
            rows.append(parsed)

    # NASA-01
    stats = compute_stats(rows)

    print("== NASA-01 ==")
    # at least three lines of results
    print(f"Mean Kloc+ = {stats['Kloc+']['mean']}")
    print(f"Median Effort- = {stats['Effort-']['median']}")
    print(f"Max Defects- = {stats['Defects-']['max']}")
    # JSON answer
    answer_01 = {goal: stats[goal] for goal in GOALS}
    print("ANSWER NASA-01 " + json.dumps(answer_01, separators=(',',':')))

    # NASA-02
    scores = normalise_composite(rows)
    # sort by composite descending
    scores.sort(key=lambda x: x[1], reverse=True)
    top_rows = [row_num for row_num, _ in scores[:10]]

    print("\n== NASA-02 ==")
    for rank, (row_num, comp) in enumerate(scores[:10], start=1):
        print(f"Top {rank}: row {row_num} composite {comp}")
    answer_02 = {"top": top_rows}
    print("ANSWER NASA-02 " + json.dumps(answer_02, separators=(',',':')))

if __name__ == "__main__":
    main()
