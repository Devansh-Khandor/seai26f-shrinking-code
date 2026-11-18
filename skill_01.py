#!/usr/bin/env python3
import csv
import sys
import json
import math

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == '?':
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append({'row': i, 'values': cleaned})
        return rows, reader.fieldnames

def compute_summary(rows, goal_col):
    vals = [row['values'][goal_col] for row in rows if row['values'][goal_col] is not None]
    n = len(vals)
    if n == 0:
        return {}
    sorted_vals = sorted(vals)
    median = sorted_vals[n//2] if n % 2 == 1 else (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2.0
    mean = sum(sorted_vals) / n
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    # direction: columns ending with '-' are to minimize
    if goal_col.endswith('-'):
        best = min_val
        worst = max_val
    else:
        best = max_val
        worst = min_val
    return {
        goal_col: {
            'mean': mean,
            'median': median,
            'min': min_val,
            'max': max_val,
            'best': best,
            'worst': worst
        }
    }

def identify_top(rows, goal_col, max_top=20):
    # Sort by increasing performance (since minimize)
    sorted_rows = sorted([r for r in rows if r['values'][goal_col] is not None],
                         key=lambda r: r['values'][goal_col])
    # Decide how many to take: at least 5, at most max_top
    take = len(sorted_rows)
    if take < 5:
        take = 5 if len(rows) >= 5 else len(rows)
    elif take > max_top:
        take = max_top
    top = sorted_rows[:take]
    return top

def compute_shares(rows, top_rows, goal_col):
    # Determine setting columns: all except goal_col and any ending with 'X'
    all_cols = rows[0]['values'].keys()
    setting_cols = [c for c in all_cols if c != goal_col and not c.endswith('X')]
    others = [r for r in rows if r not in top_rows]
    result = []
    for col in setting_cols:
        # top share
        top_ones = 0
        top_non_missing = 0
        for r in top_rows:
            v = r['values'][col]
            if v is None:
                continue
            top_non_missing += 1
            if v == 1:
                top_ones += 1
        top_share = top_ones / top_non_missing if top_non_missing else 0.0
        # rest share
        rest_ones = 0
        rest_non_missing = 0
        for r in others:
            v = r['values'][col]
            if v is None:
                continue
            rest_non_missing += 1
            if v == 1:
                rest_ones += 1
        rest_share = rest_ones / rest_non_missing if rest_non_missing else 0.0
        diff = abs(top_share - rest_share)
        result.append((diff, col, top_share, rest_share))
    # Sort by difference descending, take top 2
    result.sort(reverse=True, key=lambda x: x[0])
    return [
        {
            'setting': col,
            'top_share': top_share,
            'rest_share': rest_share
        }
        for _, col, top_share, rest_share in result[:2]
    ]

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    # Determine goal column: columns ending in '+' or '-'
    goal_cols = [c for c in fieldnames if c.endswith('+') or c.endswith('-')]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    goal_col = goal_cols[0]  # assume only one

    # APACHE-01
    print("== APACHE-01 ==")
    summary = compute_summary(rows, goal_col)
    goal_summary = summary.get(goal_col, {})
    print(f"Mean: {goal_summary.get('mean', 'nan')}")
    print(f"Median: {goal_summary.get('median', 'nan')}")
    print(f"Min: {goal_summary.get('min', 'nan')}")
    print(f"Max: {goal_summary.get('max', 'nan')}")
    print(f"Best: {goal_summary.get('best', 'nan')}")
    print(f"Worst: {goal_summary.get('worst', 'nan')}")
    print("ANSWER APACHE-01", json.dumps(summary, separators=(',', ':')))

    # APACHE-02
    print("== APACHE-02 ==")
    top_rows = identify_top(rows, goal_col, max_top=20)
    top_row_numbers = [r['row'] for r in top_rows]
    settings = compute_shares(rows, top_rows, goal_col)
    print(f"Top rows: {top_row_numbers[:5]}...")  # at least 3 lines
    print(f"Settings: {settings[:2]}")
    answer = {
        'top': top_row_numbers,
        'settings': settings
    }
    print("ANSWER APACHE-02", json.dumps(answer, separators=(',', ':')))

if __name__ == "__main__":
    main()
