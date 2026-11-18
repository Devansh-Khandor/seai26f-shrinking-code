#!/usr/bin/env python3
"""
skill_04.py

Performs APACHE-07 and APACHE-08 analyses on the provided CSV file.

Usage:
    python skill_04.py <csv_path>
"""

import sys
import csv
import json
import math

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def to_number(val):
    if val is None or val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def mean(values):
    if not values:
        return None
    return sum(values) / len(values)

def stddev(values, mean_val):
    n = len(values)
    if n == 0:
        return None
    var = sum((x - mean_val) ** 2 for x in values) / n
    return math.sqrt(var)

def print_section(id_str, lines, answer_obj):
    print(f"== {id_str} ==")
    for line in lines:
        print(line)
    print(f"ANSWER {id_str} {json.dumps(answer_obj, separators=(',', ':'))}")

def main(csv_path):
    rows, columns = read_csv(csv_path)

    # Identify goal columns (those ending with + or -)
    goal_cols = [c for c in columns if c.endswith('+') or c.endswith('-')]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)

    # Use first goal column for analysis (minimize if ends with '-', maximize otherwise)
    goal_col = goal_cols[0]
    is_minimize = goal_col.endswith('-')

    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    history_rows = rows[:-arrivals_count] if arrivals_count > 0 else rows
    arrival_rows = rows[-arrivals_count:] if arrivals_count > 0 else []

    # ------------------------------------------------------------------
    # APACHE-07: Assess arrivals
    # ------------------------------------------------------------------
    assessments = []
    prev_values = []  # goal values seen before current arrival
    # Add history values first
    for row in history_rows:
        val = to_number(row[goal_col])
        if val is not None:
            prev_values.append(val)

    # Process arrivals in order
    for i, row in enumerate(arrival_rows, 1):
        row_index = len(history_rows) + i  # 1-based
        val = to_number(row[goal_col])
        alert = False
        if val is not None and prev_values:
            m = mean(prev_values)
            std = stddev(prev_values, m)
            if std == 0 or std is None:
                threshold = m
            else:
                threshold = m + 1.96 * std  # upper bound
            if val > threshold:
                alert = True
        assessments.append({"row": row_index, "alert": alert})
        if val is not None:
            prev_values.append(val)

    apache07_answer = {"assessments": assessments}
    apache07_lines = [
        f"Arrivals: {arrivals_count}",
        f"Primary goal column: {goal_col}",
        f"Threshold used: 1.96 * std + mean (for minimizing goals)"
    ]

    # ------------------------------------------------------------------
    # APACHE-08: Checkpoints summaries
    # ------------------------------------------------------------------
    checkpoints = [0, 19, 38]  # after these many arrivals
    # Ensure checkpoints do not exceed arrivals_count
    checkpoints = [m for m in checkpoints if m <= arrivals_count]
    checkpoint_objs = []

    # Helper: compute summary for a list of rows
    def compute_summary(rows_subset):
        summary = {}
        for g in goal_cols:
            vals = [to_number(r[g]) for r in rows_subset if to_number(r[g]) is not None]
            if vals:
                n = len(vals)
                m_val = mean(vals)
                mn = min(vals)
                mx = max(vals)
            else:
                n = 0
                m_val = None
                mn = None
                mx = None
            summary[g] = {"n": n, "mean": m_val, "min": mn, "max": mx}
        return summary

    # Helper: shortlist best 5 rows based on primary goal (minimize)
    def compute_shortlist(rows_subset):
        vals = []
        for idx, r in enumerate(rows_subset, 1):
            val = to_number(r[goal_col])
            if val is not None:
                vals.append((val, idx))
        vals.sort(key=lambda x: (x[0], x[1]))
        return [idx for _, idx in vals[:5]]

    # Build checkpoints
    for m in checkpoints:
        # Rows seen so far: history + first m arrivals
        rows_seen = history_rows + arrival_rows[:m]
        summary = compute_summary(rows_seen)
        shortlist = compute_shortlist(rows_seen)
        checkpoint_objs.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    apache08_answer = {"checkpoints": checkpoint_objs}
    apache08_lines = [
        f"Total rows: {total_rows}",
        f"History rows: {len(history_rows)}",
        f"Arrival rows: {len(arrival_rows)}"
    ]

    # ------------------------------------------------------------------
    # Print sections
    # ------------------------------------------------------------------
    print_section("APACHE-07", apache07_lines, apache07_answer)
    print()
    print_section("APACHE-08", apache08_lines, apache08_answer)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
