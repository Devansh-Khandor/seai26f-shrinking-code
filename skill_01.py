#!/usr/bin/env python3
import csv
import json
import sys
import math

def read_data(path):
    """
    Read the CSV file and return a list of dictionaries, one per data row.
    Values that are '?' are stored as None. All numeric values are converted
    to float. The header row is ignored.
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            parsed = {}
            for key, val in row.items():
                if val == '?' or val == '':
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(val)
                    except ValueError:
                        parsed[key] = None
            rows.append(parsed)
    return rows

def compute_stats(values):
    """
    Compute mean, median, min, max for a list of numeric values.
    Assumes values is non-empty and contains only numeric entries.
    """
    n = len(values)
    mean = sum(values) / n
    sorted_vals = sorted(values)
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    return mean, median, min_val, max_val

def main(csv_path):
    rows = read_data(csv_path)
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]

    # ---------- CAR-01 ----------
    stats = {}
    for col in goal_cols:
        vals = [row[col] for row in rows if row[col] is not None]
        mean, median, min_val, max_val = compute_stats(vals)
        if col.endswith("-"):  # minimize
            best = min_val
            worst = max_val
        else:  # maximize
            best = max_val
            worst = min_val
        stats[col] = {
            "mean": mean,
            "median": median,
            "min": min_val,
            "max": max_val,
            "best": best,
            "worst": worst,
        }

    print("== CAR-01 ==")
    for col in goal_cols:
        s = stats[col]
        print(f"{col} mean: {s['mean']:.2f}")
        print(f"{col} median: {s['median']:.2f}")
        print(f"{col} min: {s['min']:.2f}")
        print(f"{col} max: {s['max']:.2f}")
        print(f"{col} best: {s['best']:.2f}")
        print(f"{col} worst: {s['worst']:.2f}")

    answer01 = json.dumps({
        "Lbs-": stats["Lbs-"],
        "Acc+": stats["Acc+"],
        "Mpg+": stats["Mpg+"],
    }, ensure_ascii=False)
    print(f"ANSWER CAR-01 {answer01}")

    # ---------- CAR-02 ----------
    # Normalization parameters
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [row[col] for row in rows if row[col] is not None]
        mins[col] = min(vals)
        maxs[col] = max(vals)

    # Compute scores for each row
    scores = []
    for idx, row in enumerate(rows, start=1):
        if any(row[col] is None for col in goal_cols):
            continue
        norm_lbs = (maxs["Lbs-"] - row["Lbs-"]) / (maxs["Lbs-"] - mins["Lbs-"]) if maxs["Lbs-"] != mins["Lbs-"] else 1.0
        norm_acc = (row["Acc+"] - mins["Acc+"]) / (maxs["Acc+"] - mins["Acc+"]) if maxs["Acc+"] != mins["Acc+"] else 1.0
        norm_mpg = (row["Mpg+"] - mins["Mpg+"]) / (maxs["Mpg+"] - mins["Mpg+"]) if maxs["Mpg+"] != mins["Mpg+"] else 1.0
        score = norm_lbs + norm_acc + norm_mpg
        scores.append((score, idx))

    scores.sort(key=lambda x: (-x[0], x[1]))
    top_rows = [idx for _, idx in scores[:10]]

    print("== CAR-02 ==")
    print("Top 10 vehicles (row numbers):")
    for r in top_rows:
        print(r)

    answer02 = json.dumps({"top": top_rows}, ensure_ascii=False)
    print(f"ANSWER CAR-02 {answer02}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
