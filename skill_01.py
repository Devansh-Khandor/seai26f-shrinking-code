#!/usr/bin/env python3
"""
Skill 01: NASA-01 and NASA-02

This script reads a CSV file of software project data and prints
human‑readable summaries and machine‑checked answers for the
NASA‑01 and NASA‑02 requests.

Usage:
    python skill_01.py <csv path>
"""

import csv
import json
import statistics
import sys
from typing import List, Dict, Any, Tuple, Optional

# Goal columns and their directions
GOAL_COLUMNS = {
    "Kloc+": "max",      # maximize
    "Effort-": "min",    # minimize
    "Defects-": "min",   # minimize
}

def read_data(csv_path: str) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Read CSV file and return a list of row dictionaries and a list
    of their 1‑based row numbers.
    """
    rows: List[Dict[str, Any]] = []
    row_nums: List[int] = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for rn, rec in enumerate(reader, start=1):
            row_nums.append(rn)
            parsed = {}
            for key, val in rec.items():
                # strip whitespace
                val = val.strip()
                if val == "?":
                    parsed[key] = None
                else:
                    # Try numeric conversion if possible
                    try:
                        if '.' in val or 'e' in val.lower():
                            parsed[key] = float(val)
                        else:
                            parsed[key] = int(val)
                    except ValueError:
                        parsed[key] = val  # keep as string
            rows.append(parsed)
    return rows, row_nums

def compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute mean, median, min, max, best, worst for each goal column.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for col, direction in GOAL_COLUMNS.items():
        values = [row[col] for row in rows if isinstance(row[col], (int, float))]
        if not values:
            # No numeric data; set all to None
            stats[col] = {"mean": None, "median": None, "min": None,
                          "max": None, "best": None, "worst": None}
            continue
        mean_val = sum(values) / len(values)
        median_val = statistics.median(values)
        min_val = min(values)
        max_val = max(values)
        if direction == "max":
            best_val = max_val
            worst_val = min_val
        else:  # minimize
            best_val = min_val
            worst_val = max_val
        stats[col] = {
            "mean": mean_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "best": best_val,
            "worst": worst_val,
        }
    return stats

def print_nasa01_summary(stats: Dict[str, Dict[str, Any]]) -> None:
    """
    Print human‑readable summary for NASA‑01.
    """
    print("== NASA-01 ==")
    for col in GOAL_COLUMNS:
        s = stats[col]
        print(f"Mean {col}: {s['mean']:.2f}")
        print(f"Median {col}: {s['median']:.2f}")
        print(f"Min {col}: {s['min']}")
        print(f"Max {col}: {s['max']}")
        print(f"Best {col}: {s['best']}")
        print(f"Worst {col}: {s['worst']}")
        print()  # blank line between columns

def jsonify_stats(stats: Dict[str, Dict[str, Any]]) -> str:
    """
    Convert stats dict to JSON string with the required format.
    """
    # Ensure numbers are plain JSON numbers
    # round to 6 decimals if float to avoid scientific notation
    out = {}
    for col, vals in stats.items():
        out[col] = {}
        for k, v in vals.items():
            if isinstance(v, float):
                out[col][k] = round(v, 6)
            else:
                out[col][k] = v
    return json.dumps(out)

def rank_rows(rows: List[Dict[str, Any]], row_nums: List[int]) -> List[int]:
    """
    Rank rows based on combined Kloc+, Effort-, Defects-.
    Lower values are better for all three columns.
    Return a list of top 10 row numbers (or fewer).
    """
    # Prepare rank dicts for each column
    ranks: Dict[str, Dict[int, int]] = {}
    for col in GOAL_COLUMNS:
        # List of (value, row_index)
        vals = [(row[col], idx) for idx, row in enumerate(rows) if isinstance(row[col], (int, float))]
        # If any value missing, skip that row entirely
        # So we filter rows with missing any of the three columns
        # We'll compute combined rank after filtering

    # First filter rows with missing any of the three goal columns
    good_indices: List[int] = []
    for idx, row in enumerate(rows):
        if all(isinstance(row[c], (int, float)) for c in GOAL_COLUMNS):
            good_indices.append(idx)

    if not good_indices:
        return []

    # Compute ranks for each column on the good rows
    for col in GOAL_COLUMNS:
        # Extract values
        vals = [(row[col], idx) for idx in good_indices]
        # Sort ascending (lower better)
        vals_sorted = sorted(vals, key=lambda x: x[0])
        # Assign rank (1-based)
        rank_dict = {}
        for rank, (_, idx) in enumerate(vals_sorted, start=1):
            rank_dict[idx] = rank
        ranks[col] = rank_dict

    # Compute combined rank sum
    combined = []
    for idx in good_indices:
        total_rank = sum(ranks[col][idx] for col in GOAL_COLUMNS)
        combined.append((total_rank, idx))

    # Sort by total_rank ascending, then by Kloc+ ascending
    combined_sorted = sorted(combined, key=lambda x: (x[0], rows[x[1]]["Kloc+"]))

    # Get top 10 row numbers (1-based)
    top_rows = [row_nums[idx] for _, idx in combined_sorted[:10]]
    return top_rows

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    rows, row_nums = read_data(csv_path)

    # NASA-01
    stats = compute_stats(rows)
    print_nasa01_summary(stats)
    answer_json = jsonify_stats(stats)
    print(f"ANSWER NASA-01 {answer_json}")

    # NASA-02
    print("== NASA-02 ==")
    top_rows = rank_rows(rows, row_nums)
    print(f"Top 10 project numbers: {top_rows}")
    # Prepare JSON answer
    answer_json2 = json.dumps({"top": top_rows})
    print(f"ANSWER NASA-02 {answer_json2}")

if __name__ == "__main__":
    main()
