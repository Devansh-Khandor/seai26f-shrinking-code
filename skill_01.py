#!/usr/bin/env python3
import sys
import csv
import json
import statistics

def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for line in reader:
            rows.append(line)
    return reader.fieldnames, rows

def parse_value(val):
    if val == '?' or val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None

def compute_stats(values):
    if not values:
        # No available values; return None placeholders
        return {"mean": None, "median": None, "min": None, "max": None, "best": None, "worst": None}
    mean = statistics.mean(values)
    median = statistics.median(values)
    minimum = min(values)
    maximum = max(values)
    return {"mean": mean, "median": median, "min": minimum, "max": maximum,
            "best": minimum, "worst": maximum}  # direction handled later

def main(csv_path):
    header, data = read_csv(csv_path)

    # Relevant columns
    goal_cols = ['Lbs-', 'Acc+', 'Mpg+']
    col_values = {col: [] for col in goal_cols}
    row_vals = []  # list of dicts with numeric values or None

    for row in data:
        vals = {}
        for col in goal_cols:
            val = parse_value(row[col])
            vals[col] = val
            if val is not None:
                col_values[col].append(val)
        row_vals.append(vals)

    # Compute statistics per column
    stats = {}
    for col in goal_cols:
        col_stat = compute_stats(col_values[col])
        # adjust best/worst according to direction
        if col.endswith('-'):  # minimize
            col_stat["best"] = min(col_values[col]) if col_values[col] else None
            col_stat["worst"] = max(col_values[col]) if col_values[col] else None
        else:  # maximize
            col_stat["best"] = max(col_values[col]) if col_values[col] else None
            col_stat["worst"] = min(col_values[col]) if col_values[col] else None
        stats[col] = col_stat

    # CAR-01 output
    print("== CAR-01 ==")
    for col in goal_cols:
        s = stats[col]
        print(f"{col}: mean={s['mean']}, median={s['median']}, min={s['min']}, max={s['max']}, best={s['best']}, worst={s['worst']}")
    # Machine-checked answer
    answer_car01 = {"Lbs-": stats["Lbs-"], "Acc+": stats["Acc+"], "Mpg+": stats["Mpg+"]}
    # Ensure numbers are plain JSON (no NaN)
    print("ANSWER CAR-01 " + json.dumps(answer_car01, separators=(',', ':')))

    # CAR-02: overall combination score
    # Precompute global min/max for normalization
    mins = {col: min(col_values[col]) if col_values[col] else 0 for col in goal_cols}
    maxs = {col: max(col_values[col]) if col_values[col] else 0 for col in goal_cols}

    scores = []
    for idx, vals in enumerate(row_vals, start=1):  # 1-based row number
        if any(vals[col] is None for col in goal_cols):
            continue  # skip rows with missing data
        # Normalize each objective
        norm_weight = ((maxs["Lbs-"] - vals["Lbs-"]) / (maxs["Lbs-"] - mins["Lbs-"])) if maxs["Lbs-"] != mins["Lbs-"] else 0
        norm_acc = ((vals["Acc+"] - mins["Acc+"]) / (maxs["Acc+"] - mins["Acc+"])) if maxs["Acc+"] != mins["Acc+"] else 0
        norm_mpg = ((vals["Mpg+"] - mins["Mpg+"]) / (maxs["Mpg+"] - mins["Mpg+"])) if maxs["Mpg+"] != mins["Mpg+"] else 0
        score = norm_weight + norm_acc + norm_mpg
        scores.append((score, idx))

    # Sort by score descending
    scores.sort(reverse=True)
    top_rows = [idx for _, idx in scores[:10]]  # top 10

    # CAR-02 output
    print("== CAR-02 ==")
    print("Top 10 rows:", ', '.join(str(r) for r in top_rows))
    answer_car02 = {"top": top_rows}
    print("ANSWER CAR-02 " + json.dumps(answer_car02, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
