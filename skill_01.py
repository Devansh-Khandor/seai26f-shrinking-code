#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <csv_path>

The script prints summary statistics for goal columns and identifies
top-performing configurations.

Author: OpenAI ChatGPT
"""

import csv
import sys
import json
from collections import defaultdict

def read_csv(path):
    """Read CSV and return header and list of rows (as dicts)."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    return not col.endswith('X')

def parse_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None

def mean(values):
    return sum(values) / len(values) if values else 0.0

def median(values):
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]

def compute_stats(header, rows):
    stats = {}
    for col in header:
        if is_goal(col):
            vals = []
            for r in rows:
                v = r.get(col, None)
                if v == '?' or v is None:
                    continue
                v_int = parse_int(v)
                if v_int is None:
                    continue
                vals.append(v_int)
            if not vals:
                continue
            col_min = min(vals)
            col_max = max(vals)
            best = col_min if col.endswith('-') else col_max
            worst = col_max if col.endswith('-') else col_min
            stats[col] = {
                "mean": mean(vals),
                "median": median(vals),
                "min": col_min,
                "max": col_max,
                "best": best,
                "worst": worst
            }
    return stats

def identify_top_rows(header, rows, n_top):
    # Determine rows sorted by performance (minimize)
    perf_col = next((c for c in header if is_goal(c)), None)
    if not perf_col:
        return []
    # Filter rows with valid performance
    valid_rows = []
    for idx, r in enumerate(rows, start=1):
        v = r.get(perf_col, None)
        if v == '?' or v is None:
            continue
        v_int = parse_int(v)
        if v_int is None:
            continue
        valid_rows.append((idx, v_int))
    # Sort ascending
    valid_rows.sort(key=lambda x: x[1])
    top = [idx for idx, _ in valid_rows[:n_top]]
    return top

def compute_setting_shares(header, rows, top_rows):
    setting_cols = [c for c in header if is_input(c) and not is_goal(c)]
    shares = []
    total_rows = len(rows)
    top_set = set(top_rows)
    for col in setting_cols:
        top_count = 0
        top_total = 0
        rest_count = 0
        rest_total = 0
        for idx, r in enumerate(rows, start=1):
            val = r.get(col, None)
            if val == '?' or val is None:
                continue
            val_int = parse_int(val)
            if val_int is None:
                continue
            if val_int != 1:
                continue
            if idx in top_set:
                top_count += 1
                top_total += 1
            else:
                rest_count += 1
                rest_total += 1
        top_share = top_count / top_total if top_total else 0.0
        rest_share = rest_count / rest_total if rest_total else 0.0
        shares.append((col, top_share, rest_share, abs(top_share - rest_share)))
    # Sort by distinguishing power
    shares.sort(key=lambda x: x[3], reverse=True)
    # Take at least 2
    result = []
    for col, t, r, _ in shares[:max(2, len(shares))]:
        result.append({
            "setting": col,
            "top_share": round(t, 3),
            "rest_share": round(r, 3)
        })
    return result

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # APACHE-01
    print("== APACHE-01 ==")
    stats = compute_stats(header, rows)
    # Simple human-friendly output
    if "Performance-" in stats:
        p = stats["Performance-"]
        print(f"Mean Performance: {p['mean']:.2f}")
        print(f"Median Performance: {p['median']:.2f}")
        print(f"Min Performance: {p['min']}")
    else:
        print("No Performance- column found.")
    # Machine-checked answer
    print("ANSWER APACHE-01", json.dumps({"Performance-": stats.get("Performance-", {})}))

    # APACHE-02
    print("\n== APACHE-02 ==")
    # Determine number of top rows
    n_total = len(rows)
    n_top = max(5, min(20, max(1, int(n_total * 0.05))))
    top_rows = identify_top_rows(header, rows, n_top)
    print(f"Top {len(top_rows)} rows: {top_rows}")
    setting_shares = compute_setting_shares(header, rows, top_rows)
    print("Top settings:")
    for s in setting_shares:
        print(f"  {s['setting']}: top={s['top_share']:.3f}, rest={s['rest_share']:.3f}")
    # Machine-checked answer
    print("ANSWER APACHE-02", json.dumps({"top": top_rows, "settings": setting_shares}))

if __name__ == "__main__":
    main()
