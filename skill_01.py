#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <csv_path>
"""

import csv
import sys
import json
from statistics import mean, median
from collections import defaultdict

def is_goal_col(col_name):
    return col_name.endswith('+') or col_name.endswith('-')

def is_input_col(col_name):
    return not col_name.endswith('X') and not is_goal_col(col_name)

def parse_value(val, col_name):
    """Parse a CSV cell value."""
    if val == '?':
        return None
    # Goal columns are numeric but may contain '?' handled above
    # All other columns are binary 0/1
    if is_goal_col(col_name):
        try:
            return float(val)
        except ValueError:
            return None
    else:
        # Binary columns
        return int(val)

def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Identify columns
    goal_cols = [c for c in reader.fieldnames if is_goal_col(c)]
    input_cols = [c for c in reader.fieldnames if is_input_col(c)]

    # Gather goal values
    goal_values = defaultdict(list)   # col -> list of numeric values
    for idx, row in enumerate(rows, start=1):
        for col in goal_cols:
            val = parse_value(row[col], col)
            if val is not None:
                goal_values[col].append(val)

    # Compute stats for goal columns
    goal_stats = {}
    for col in goal_cols:
        vals = goal_values[col]
        if not vals:
            # No valid data; skip
            continue
        vals_sorted = sorted(vals)
        n = len(vals)
        vals_mean = mean(vals)
        vals_median = median(vals)
        vals_min = min(vals)
        vals_max = max(vals)
        # Direction
        if col.endswith('-'):   # minimize
            best = vals_min
            worst = vals_max
        else:                   # maximize
            best = vals_max
            worst = vals_min
        goal_stats[col] = {
            "mean": vals_mean,
            "median": vals_median,
            "min": vals_min,
            "max": vals_max,
            "best": best,
            "worst": worst
        }

    # Output APACHE-01
    print("== APACHE-01 ==")
    # Simple textual summary (at least 3 lines)
    for col, stats in goal_stats.items():
        print(f"{col}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
              f"min={stats['min']}, max={stats['max']}, best={stats['best']}, "
              f"worst={stats['worst']}")
    # Machine-checked answer
    answer_json = {"Performance-": goal_stats.get("Performance-", {})}
    print("ANSWER APACHE-01", json.dumps(answer_json, separators=(',', ':')))

    # APACHE-02
    print("\n== APACHE-02 ==")

    # Determine top rows: 5 rows with lowest Performance- (if available)
    perf_vals = [(idx+1, row["Performance-"]) for idx, row in enumerate(rows)
                 if parse_value(row["Performance-"], "Performance-") is not None]
    perf_vals.sort(key=lambda x: x[1])   # ascending
    top_k = min(5, len(perf_vals))
    top_rows = [idx for idx, _ in perf_vals[:top_k]]
    top_rows_set = set(top_rows)

    # Compute setting shares
    setting_stats = []
    total_rows = len(rows)
    rest_count = total_rows - len(top_rows)
    for col in input_cols:
        top_ones = sum(1 for idx in top_rows if int(rows[idx-1][col]) == 1)
        rest_ones = sum(1 for i, r in enumerate(rows, start=1)
                        if i not in top_rows_set and int(r[col]) == 1)
        top_share = top_ones / top_k if top_k else 0.0
        rest_share = rest_ones / rest_count if rest_count else 0.0
        setting_stats.append({
            "setting": col,
            "top_share": top_share,
            "rest_share": rest_share,
            "diff": top_share - rest_share
        })
    # Sort by absolute difference descending, then by diff positive
    setting_stats.sort(key=lambda x: (-abs(x["diff"]), -x["diff"]))
    # Take at least 2
    top_settings = setting_stats[:max(2, len(setting_stats))]
    # Format answer
    ap2_answer = {
        "top": top_rows,
        "settings": [
            {
                "setting": s["setting"],
                "top_share": s["top_share"],
                "rest_share": s["rest_share"]
            } for s in top_settings
        ]
    }
    print("ANSWER APACHE-02", json.dumps(ap2_answer, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
