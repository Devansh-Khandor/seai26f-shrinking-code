#!/usr/bin/env python3
import csv
import json
import sys

GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']
DIRECTION = {
    'Lbs-': 'min',   # minimize
    'Acc+': 'max',   # maximize
    'Mpg+': 'max',   # maximize
}

def parse_value(val):
    """Parse a CSV cell value. Return float or None if missing."""
    if val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def compute_stats(values):
    """Return dict with mean, median, min, max."""
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
        'max': sorted_vals[-1],
    }

def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Store data
    goal_data = {col: [] for col in GOAL_COLUMNS}
    row_values = {}  # rownum -> {col: value}
    for idx, row in enumerate(rows, start=1):
        vals = {}
        for col in GOAL_COLUMNS:
            val = parse_value(row[col])
            vals[col] = val
            if val is not None:
                goal_data[col].append(val)
        row_values[idx] = vals

    # CAR-01 summary
    print("== CAR-01 ==")
    answer_01 = {}
    for col in GOAL_COLUMNS:
        stats = compute_stats(goal_data[col])
        # Determine best/worst based on direction
        if DIRECTION[col] == 'min':
            best = stats['min']
            worst = stats['max']
        else:  # max
            best = stats['max']
            worst = stats['min']
        # Build output line
        line = (f"{col}: mean={stats['mean']}, median={stats['median']}, "
                f"min={stats['min']}, max={stats['max']}, "
                f"best={best}, worst={worst}")
        print(line)
        answer_01[col] = {
            'mean': stats['mean'],
            'median': stats['median'],
            'min': stats['min'],
            'max': stats['max'],
            'best': best,
            'worst': worst,
        }
    print(f"ANSWER CAR-01 {json.dumps(answer_01, separators=(',', ':'))}")

    # CAR-02 top rows
    print("\n== CAR-02 ==")
    scores = []
    for rownum, vals in row_values.items():
        if None in (vals['Lbs-'], vals['Acc+'], vals['Mpg+']):
            continue  # skip rows with missing values in any goal column
        score = -vals['Lbs-'] + vals['Acc+'] + vals['Mpg+']
        scores.append((score, rownum))
    # Sort by score descending, then row number ascending
    scores.sort(key=lambda x: (-x[0], x[1]))
    top_rows = [rownum for _, rownum in scores[:10]]
    print(f"Top 10 rows: {top_rows}")
    answer_02 = {"top": top_rows}
    print(f"ANSWER CAR-02 {json.dumps(answer_02, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
