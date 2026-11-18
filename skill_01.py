#!/usr/bin/env python3
import csv
import sys
import statistics
import json

GOAL_COLUMNS = ["Kloc+", "Effort-", "Defects-"]

def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def compute_stats(rows):
    stats = {}
    for col in GOAL_COLUMNS:
        vals = [to_float(r[col]) for r in rows if to_float(r[col]) is not None]
        if not vals:
            stats[col] = {"mean": None, "median": None, "min": None, "max": None,
                          "best": None, "worst": None}
            continue
        mean = statistics.mean(vals)
        median = statistics.median(vals)
        mn = min(vals)
        mx = max(vals)
        # Determine best/worst based on goal direction
        if col.endswith('+'):
            best = mx
            worst = mn
        else:  # ends with '-'
            best = mn
            worst = mx
        stats[col] = {"mean": mean, "median": median, "min": mn, "max": mx,
                      "best": best, "worst": worst}
    return stats

def compute_top_rows(rows, top_n=10):
    # Compute min/max for normalization
    mins = {}
    maxs = {}
    for col in GOAL_COLUMNS:
        vals = [to_float(r[col]) for r in rows if to_float(r[col]) is not None]
        if not vals:
            mins[col] = maxs[col] = 0.0
        else:
            mins[col] = min(vals)
            maxs[col] = max(vals)

    scored_rows = []
    for idx, r in enumerate(rows, start=1):  # 1-based
        v_k = to_float(r["Kloc+"])
        v_e = to_float(r["Effort-"])
        v_d = to_float(r["Defects-"])
        if v_k is None or v_e is None or v_d is None:
            continue  # skip rows with missing in any of the three
        # Normalized scores
        if maxs["Kloc+"] == mins["Kloc+"]:
            norm_k = 0.0
        else:
            norm_k = (v_k - mins["Kloc+"]) / (maxs["Kloc+"] - mins["Kloc+"])
        if maxs["Effort-"] == mins["Effort-"]:
            norm_e = 0.0
        else:
            norm_e = (maxs["Effort-"] - v_e) / (maxs["Effort-"] - mins["Effort-"])
        if maxs["Defects-"] == mins["Defects-"]:
            norm_d = 0.0
        else:
            norm_d = (maxs["Defects-"] - v_d) / (maxs["Defects-"] - mins["Defects-"])
        score = norm_k + norm_e + norm_d
        scored_rows.append((idx, score, v_k, v_e, v_d))
    scored_rows.sort(key=lambda x: x[1], reverse=True)
    top_rows = scored_rows[:top_n]
    return [r[0] for r in top_rows], top_rows

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    stats = compute_stats(rows)

    # NASA-01
    print("== NASA-01 ==")
    for col in GOAL_COLUMNS:
        s = stats[col]
        print(f"{col}: mean={s['mean']:.2f}, median={s['median']:.2f}, best={s['best']}, worst={s['worst']}")
    answer01 = {col: stats[col] for col in GOAL_COLUMNS}
    print("ANSWER NASA-01", json.dumps(answer01, separators=(',', ':')))

    # NASA-02
    top_indices, scored_rows = compute_top_rows(rows, top_n=10)
    print("\n== NASA-02 ==")
    print("Top rows (row numbers):", top_indices)
    print("Details of top rows:")
    for idx, score, v_k, v_e, v_d in scored_rows:
        print(f"Row {idx}: Kloc+={v_k}, Effort-={v_e}, Defects-={v_d}, score={score:.4f}")
    answer02 = {"top": top_indices}
    print("ANSWER NASA-02", json.dumps(answer02, separators=(',', ':')))

if __name__ == "__main__":
    main()
