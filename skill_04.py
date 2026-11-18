#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <csv_path>

Implements APACHE-07 and APACHE-08 analyses.

Author: OpenAI ChatGPT
"""

import csv
import json
import sys
import math

# --------------------------------------------------------------------------- #
# CSV utilities

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_int(val):
    if val is None or val == '?':
        return None
    try:
        return int(val)
    except ValueError:
        return None

# --------------------------------------------------------------------------- #
# Statistics helpers

def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)

def stddev(values):
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)

# --------------------------------------------------------------------------- #
# APACHE-07: alert on unusually poor arrivals

def assess_arrivals(header, rows, performance_col, arrivals_indices):
    assessments = []
    for idx in arrivals_indices:
        row_num = idx + 1
        # Gather prior rows
        prior_rows = rows[:idx]
        prior_vals = [parse_int(r[performance_col]) for r in prior_rows]
        prior_vals = [v for v in prior_vals if v is not None]
        # Current performance
        cur_val = parse_int(rows[idx][performance_col])
        alert = False
        if cur_val is not None and len(prior_vals) >= 2:
            m = mean(prior_vals)
            s = stddev(prior_vals)
            threshold = m + 2 * s
            if cur_val > threshold:
                alert = True
        assessments.append({"row": row_num, "alert": alert})
    return assessments

# --------------------------------------------------------------------------- #
# APACHE-08: checkpoints and shortlist

def compute_checkpoint_summary(subset_rows, goal_cols):
    summary = {}
    for col in goal_cols:
        vals = [parse_int(r[col]) for r in subset_rows]
        vals = [v for v in vals if v is not None]
        n = len(vals)
        if n == 0:
            mean_val = 0.0
            min_val = 0
            max_val = 0
        else:
            mean_val = mean(vals)
            min_val = min(vals)
            max_val = max(vals)
        summary[col] = {
            "n": n,
            "mean": round(mean_val, 2),
            "min": int(min_val),
            "max": int(max_val)
        }
    return summary

def compute_shortlist(subset_rows, performance_col, count=5):
    valid_rows = []
    for idx, r in enumerate(subset_rows):
        val = parse_int(r[performance_col])
        if val is not None:
            valid_rows.append((idx, val))
    # Sort ascending by performance (min better)
    valid_rows.sort(key=lambda x: x[1])
    shortlist = [row_idx + 1 for row_idx, _ in valid_rows[:count]]
    return shortlist

def checkpoints(header, rows, performance_col, arrivals_indices, m_values):
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    # History rows = all rows before first arrival
    first_arrival = arrivals_indices[0] if arrivals_indices else len(rows)
    history = rows[:first_arrival]
    results = []
    for m in m_values:
        arrivals_used = arrivals_indices[:m]
        subset = history + [rows[i] for i in arrivals_used]
        summary = compute_checkpoint_summary(subset, goal_cols)
        shortlist = compute_shortlist(subset, performance_col)
        results.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })
    return results

# --------------------------------------------------------------------------- #
# Main routine

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # Identify goal column for performance (first column ending with '+' or '-')
    performance_col = None
    for c in header:
        if c.endswith('+') or c.endswith('-'):
            performance_col = c
            break
    if performance_col is None:
        print("No performance goal column found.", file=sys.stderr)
        sys.exit(1)

    total_rows = len(rows)
    # Determine arrivals: last round(0.2 * N) rows
    num_arrivals = round(0.2 * total_rows)
    arrival_start = total_rows - num_arrivals
    arrivals_indices = list(range(arrival_start, total_rows))

    # ----------------------------------------------------------------------- #
    # APACHE-07
    print("== APACHE-07 ==")
    assessments = assess_arrivals(header, rows, performance_col, arrivals_indices)
    # Simple human-friendly lines
    for a in assessments:
        print(f"Row {a['row']}: alert={a['alert']}")
    print("ANSWER APACHE-07", json.dumps({"assessments": assessments}))

    # ----------------------------------------------------------------------- #
    # APACHE-08
    print("\n== APACHE-08 ==")
    m_values = [0, min(19, num_arrivals), num_arrivals]
    checkpoints_info = checkpoints(header, rows, performance_col, arrivals_indices, m_values)
    for ckpt in checkpoints_info:
        m = ckpt["after"]
        print(f"Checkpoint after {m} arrivals:")
        for col, stats in ckpt["summary"].items():
            print(f"  {col}: n={stats['n']}, mean={stats['mean']}, min={stats['min']}, max={stats['max']}")
        print(f"  Shortlist: {ckpt['shortlist']}")
    print("ANSWER APACHE-08", json.dumps({"checkpoints": checkpoints_info}))

if __name__ == "__main__":
    main()
