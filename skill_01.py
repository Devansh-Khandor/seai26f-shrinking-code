#!/usr/bin/env python3
import sys
import csv
import json
from statistics import mean, median
from math import isclose

def read_csv(path):
    """Read CSV and return list of dicts, plus column names."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            data.append(row)
    return data, reader.fieldnames

def to_number(val):
    if val == '?' or val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None

def compute_goal_stats(data, goal_col):
    values = []
    for row in data:
        v = to_number(row.get(goal_col))
        if v is not None:
            values.append(v)
    n = len(values)
    if n == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "best": None, "worst": None}
    m = mean(values)
    med = median(sorted(values))
    mn = min(values)
    mx = max(values)
    # best/worst depend on direction: goal ends with '-' -> minimize
    best = mn
    worst = mx
    return {"mean": m, "median": med, "min": mn, "max": mx, "best": best, "worst": worst}

def compute_top_rows(data, goal_col, top_n):
    # sort by performance ascending (since minimize)
    sorted_rows = sorted(
        [(idx, row) for idx, row in enumerate(data, 1) if to_number(row.get(goal_col)) is not None],
        key=lambda x: to_number(x[1].get(goal_col))
    )
    top = sorted_rows[:top_n]
    top_indices = [idx for idx, _ in top]
    return top_indices

def compute_setting_shares(data, settings, top_indices, goal_col):
    top_set = set(top_indices)
    # Count ones and totals per group
    totals = {s: {"ones":0, "total":0} for s in settings}
    rest_totals = {s: {"ones":0, "total":0} for s in settings}
    for idx, row in enumerate(data, 1):
        is_top = idx in top_set
        for s in settings:
            val = row.get(s)
            if val == '?' or val is None:
                continue
            val_int = int(val)
            if is_top:
                totals[s]["total"] += 1
                totals[s]["ones"] += val_int
            else:
                rest_totals[s]["total"] += 1
                rest_totals[s]["ones"] += val_int
    shares = {}
    for s in settings:
        t = totals[s]
        r = rest_totals[s]
        top_share = t["ones"] / t["total"] if t["total"] else 0.0
        rest_share = r["ones"] / r["total"] if r["total"] else 0.0
        shares[s] = {"top_share": top_share, "rest_share": rest_share}
    return shares

def main(csv_path):
    data, columns = read_csv(csv_path)
    # Identify goal columns
    goal_cols = [c for c in columns if c.endswith('+') or c.endswith('-')]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    goal_col = goal_cols[0]  # assume one goal column

    # APACHE-01
    stats = compute_goal_stats(data, goal_col)
    print("== APACHE-01 ==")
    print(f"Mean: {stats['mean']}")
    print(f"Median: {stats['median']}")
    print(f"Min: {stats['min']}")
    print(f"Max: {stats['max']}")
    print(f"Best: {stats['best']}")
    print(f"Worst: {stats['worst']}")

    answer_01 = {"Performance-": stats}
    print("ANSWER APACHE-01 " + json.dumps(answer_01))

    # APACHE-02
    total_rows = len(data)
    # Determine top_n: between 5 and 20, at least 5, at most 20, based on 10% of rows
    top_n = max(5, min(20, round(0.1 * total_rows)))
    top_indices = compute_top_rows(data, goal_col, top_n)
    print("\n== APACHE-02 ==")
    print(f"Top rows: {top_indices}")

    # Settings: all columns except goal columns and any ending with 'X'
    settings = [c for c in columns if not (c.endswith('+') or c.endswith('-') or c.endswith('X'))]
    shares = compute_setting_shares(data, settings, top_indices, goal_col)

    # Compute distinguishing power
    diff_list = []
    for s in settings:
        diff = abs(shares[s]["top_share"] - shares[s]["rest_share"])
        diff_list.append((diff, s))
    diff_list.sort(reverse=True)
    # Pick at least 2 settings
    selected = [s for _, s in diff_list[:max(2, len(diff_list))]]
    setting_objs = []
    for s in selected:
        setting_objs.append({
            "setting": s,
            "top_share": round(shares[s]["top_share"], 4),
            "rest_share": round(shares[s]["rest_share"], 4)
        })
    answer_02 = {"top": top_indices, "settings": setting_objs}
    print("ANSWER APACHE-02 " + json.dumps(answer_02))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
