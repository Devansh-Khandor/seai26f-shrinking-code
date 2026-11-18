#!/usr/bin/env python3
"""
skill_01.py

Usage: python skill_01.py <csv_path>
"""

import sys
import csv
import json
from statistics import median

def read_csv(path):
    """Read CSV and return list of rows (dict) with 1‑based indices."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=1):
            row = {}
            for k, v in raw.items():
                if v == '?':
                    row[k] = None
                else:
                    try:
                        # Try to interpret as integer, fallback to string
                        row[k] = int(v)
                    except ValueError:
                        row[k] = v
            row['_rownum'] = idx
            rows.append(row)
    return rows

def stats_for_goal(values, direction):
    """Return stats dictionary for a list of numeric values."""
    n = len(values)
    if n == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "best": None,
            "worst": None
        }
    mean = sum(values) / n
    sorted_vals = sorted(values)
    med = median(sorted_vals)
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    if direction == 'min':
        best = min_val
        worst = max_val
    else:
        best = max_val
        worst = min_val
    return {
        "mean": mean,
        "median": med,
        "min": min_val,
        "max": max_val,
        "best": best,
        "worst": worst
    }

def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        return

    header = rows[0].keys()
    # Identify goal columns
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    goal_stats = {}
    for col in goal_cols:
        vals = [r[col] for r in rows if r[col] is not None]
        direction = 'min' if col.endswith('-') else 'max'
        goal_stats[col] = stats_for_goal(vals, direction)

    # --- APACHE-01 ---
    print("== APACHE-01 ==")
    # Simple human‑readable summary
    for col, stats in goal_stats.items():
        print(f"{col}: mean={stats['mean']:.2f} median={stats['median']} "
              f"min={stats['min']} max={stats['max']} best={stats['best']} worst={stats['worst']}")
    print(f"ANSWER APACHE-01 {json.dumps({'Performance-': goal_stats.get('Performance-', {})},
                                     separators=(',', ':'))}")

    # --- APACHE-02 ---
    print("\n== APACHE-02 ==")
    # Determine top rows (best performance)
    # Exclude rows with missing performance value
    perf_col = 'Performance-'
    sorted_rows = sorted(
        [r for r in rows if r[perf_col] is not None],
        key=lambda r: r[perf_col]
    )
    top_n = min(max(5, len(sorted_rows)), 20)  # 5 to 20 rows
    top_rows = sorted_rows[:top_n]
    top_row_nums = [r['_rownum'] for r in top_rows]
    rest_rows = [r for r in rows if r not in top_rows]
    # Identify setting columns (exclude goals and X columns)
    setting_cols = [c for c in header
                    if not (c.endswith('+') or c.endswith('-') or c.endswith('X'))
                    and c != perf_col]
    setting_info = []
    for col in setting_cols:
        top_ones = sum(1 for r in top_rows if r[col] == 1)
        rest_ones = sum(1 for r in rest_rows if r[col] == 1)
        top_share = top_ones / len(top_rows) if len(top_rows) > 0 else 0.0
        rest_share = rest_ones / len(rest_rows) if len(rest_rows) > 0 else 0.0
        setting_info.append({
            "setting": col,
            "top_share": round(top_share, 3),
            "rest_share": round(rest_share, 3)
        })
    # Sort by distinguishing power (abs difference)
    setting_info.sort(key=lambda d: abs(d["top_share"] - d["rest_share"]), reverse=True)
    # Ensure at least 2 entries
    setting_info = setting_info[:max(2, len(setting_info))]
    # Human readable output
    print(f"Top {top_n} rows: {top_row_nums}")
    print("Settings distinguishing top rows:")
    for s in setting_info[:5]:  # print first few
        print(f"{s['setting']}: top_share={s['top_share']} rest_share={s['rest_share']}")
    # Machine‑checked JSON
    answer_dict = {
        "top": top_row_nums,
        "settings": setting_info
    }
    print(f"ANSWER APACHE-02 {json.dumps(answer_dict, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
