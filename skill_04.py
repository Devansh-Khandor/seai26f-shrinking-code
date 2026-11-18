#!/usr/bin/env python3
"""
skill_04.py – CAR-07 and CAR-08
"""

import csv
import json
import sys
import math
from collections import defaultdict

# Reuse parsing and stats from skill_01
try:
    from skill_01 import parse_value, compute_stats
except Exception:
    # Fallback definitions in case skill_01 is not importable
    def parse_value(val):
        if val == '?' or val == '':
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def compute_stats(values):
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean = sum(sorted_vals) / n
        if n % 2 == 1:
            median = sorted_vals[n // 2]
        else:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        return {
            'mean': mean,
            'median': median,
            'min': sorted_vals[0],
            'max': sorted_vals[-1]
        }

GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']

def median_of(vals):
    """Return the median of a non‑empty numeric list."""
    n = len(vals)
    if n == 0:
        return None
    sorted_vals = sorted(vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

def score_row(row):
    """Return performance score if all goal columns present, else None."""
    try:
        return -row['Lbs-'] + row['Acc+'] + row['Mpg+']
    except (TypeError, KeyError):
        return None

def get_summary(rows):
    """Return stats per goal column for a list of parsed rows."""
    summaries = {}
    for col in GOAL_COLUMNS:
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            stats = compute_stats(vals)
            summaries[col] = {
                'n': len(vals),
                'mean': stats['mean'],
                'min': stats['min'],
                'max': stats['max']
            }
        else:
            summaries[col] = {'n': 0, 'mean': None, 'min': None, 'max': None}
    return summaries

def get_shortlist(rows):
    """Return the 5 best rows (by score) among rows with full goal data."""
    scored = [(score_row(r), idx) for idx, r in enumerate(rows, start=1)
              if score_row(r) is not None]
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [idx for _, idx in scored[:5]]

def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    total_rows = len(raw_rows)
    arrivals_count = round(0.2 * total_rows)
    history_rows = raw_rows[:total_rows - arrivals_count]
    arrivals_rows = raw_rows[total_rows - arrivals_count:]

    # Parse all rows
    parsed_rows = []
    for r in raw_rows:
        parsed = {}
        for k, v in r.items():
            parsed[k] = parse_value(v)
        parsed_rows.append(parsed)

    # ---------- CAR-07 ----------
    print("== CAR-07 ==")
    assessments = []
    # Build initial scores list from history
    all_scores = [score_row(r) for r in history_rows if score_row(r) is not None]
    for i, r in enumerate(arrivals_rows, start=1):
        row_num = len(history_rows) + i  # 1-based row number
        s = score_row(r)
        # compute median of all_scores so far
        median = median_of(all_scores)
        desirable = False
        if s is not None and median is not None:
            desirable = s > median
        # store assessment
        assessments.append({'row': row_num, 'desirable': desirable})
        # update all_scores with this row if valid
        if s is not None:
            all_scores.append(s)
        # print a line for visibility
        print(f"Row {row_num}: desirable = {desirable}")

    answer_07 = {"assessments": assessments}
    print(f"ANSWER CAR-07 {json.dumps(answer_07, separators=(',', ':'))}")

    # ---------- CAR-08 ----------
    print("\n== CAR-08 ==")
    checkpoints = [0, 40, 80]
    checkpoints = [m for m in checkpoints if m <= arrivals_count]
    checkpoint_outputs = []
    for m in checkpoints:
        # rows seen so far: history + first m arrivals
        seen_rows = history_rows + arrivals_rows[:m]
        # summary per goal column
        summary = get_summary(seen_rows)
        # print for visibility
        print(f"Checkpoint after {m} arrivals: rows seen = {len(seen_rows)}")
        for col in GOAL_COLUMNS:
            s = summary[col]
            print(f"  {col}: n={s['n']}, mean={s['mean']}, min={s['min']}, max={s['max']}")
        checkpoint_outputs.append({'after': m, 'summary': summary})

    # Final shortlist after all arrivals
    all_seen_rows = history_rows + arrivals_rows
    shortlist = get_shortlist(all_seen_rows)

    answer_08 = {
        'checkpoints': checkpoint_outputs,
        'shortlist': shortlist
    }
    print(f"ANSWER CAR-08 {json.dumps(answer_08, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
