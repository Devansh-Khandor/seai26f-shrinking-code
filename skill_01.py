#!/usr/bin/env python3
import csv
import json
import sys
import math

# Goal column identification helpers
def is_goal(col_name):
    return col_name.endswith(('-', '+')) and not col_name.endswith('X')

def is_maximize(col_name):
    return col_name.endswith('+')

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append(r)
    return rows, reader.fieldnames

def parse_value(val):
    if val == '?' or val.strip() == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def compute_stats(rows, fieldnames):
    stats = {}
    for col in fieldnames:
        if not is_goal(col):
            continue
        values = [parse_value(row[col]) for row in rows]
        values = [v for v in values if v is not None]
        if not values:
            continue
        n = len(values)
        values_sorted = sorted(values)
        mean = sum(values) / n
        median = (values_sorted[n // 2] if n % 2 == 1
                  else (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2)
        min_val = values_sorted[0]
        max_val = values_sorted[-1]
        best = min_val if not is_maximize(col) else max_val
        worst = max_val if not is_maximize(col) else min_val
        stats[col] = {
            "mean": mean,
            "median": median,
            "min": min_val,
            "max": max_val,
            "best": best,
            "worst": worst
        }
    return stats

def compute_top10(rows, fieldnames):
    # Determine global min/max for each goal column
    globals_min = {}
    globals_max = {}
    for col in fieldnames:
        if not is_goal(col):
            continue
        vals = [parse_value(r[col]) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            globals_min[col] = min(vals)
            globals_max[col] = max(vals)

    candidates = []
    for idx, row in enumerate(rows, start=1):
        # skip rows with missing goal values
        row_vals = {}
        missing = False
        for col in fieldnames:
            if not is_goal(col):
                continue
            v = parse_value(row[col])
            if v is None:
                missing = True
                break
            row_vals[col] = v
        if missing:
            continue
        # compute normalized scores
        score = 0.0
        for col, v in row_vals.items():
            mn = globals_min[col]
            mx = globals_max[col]
            if mx == mn:
                norm = 1.0
            else:
                if is_maximize(col):
                    norm = (v - mn) / (mx - mn)
                else:
                    norm = (mx - v) / (mx - mn)
            score += norm
        candidates.append(( -score, idx))  # negative for descending order

    # sort by score descending, then row number ascending
    candidates.sort()
    top10 = [idx for _, idx in candidates[:10]]
    return top10

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    # CAR-01
    stats = compute_stats(rows, fieldnames)

    print("== CAR-01 ==")
    # at least 3 lines of human readable output
    for col in sorted(stats):
        s = stats[col]
        print(f"{col}: mean={s['mean']:.6f} median={s['median']:.6f} "
              f"min={s['min']:.6f} max={s['max']:.6f} "
              f"best={s['best']:.6f} worst={s['worst']:.6f}")
    # JSON answer
    print("ANSWER CAR-01", json.dumps(stats, ensure_ascii=False))

    # CAR-02
    top10 = compute_top10(rows, fieldnames)

    print("\n== CAR-02 ==")
    print("Top 10 row numbers (best first):", top10)
    print("ANSWER CAR-02", json.dumps({"top": top10}, ensure_ascii=False))

if __name__ == "__main__":
    main()
