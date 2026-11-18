#!/usr/bin/env python3
"""
skill_01.py - Summary and ranking of used car inventory

Usage:
    python skill_01.py <csv_path>
"""

import sys
import csv
import json
import statistics

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return list of dicts (rows). Missing values are kept as None."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric columns to float, keep '?' as None
            for col in ['Lbs-', 'Acc+', 'Mpg+']:
                val = row[col]
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = None
            rows.append(row)
    return rows

def compute_stats(values):
    """Return mean, median, min, max for a list of floats (non-None)."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = statistics.mean(sorted_vals) if n > 0 else None
    median_val = statistics.median(sorted_vals) if n > 0 else None
    min_val = sorted_vals[0] if n > 0 else None
    max_val = sorted_vals[-1] if n > 0 else None
    return mean_val, median_val, min_val, max_val

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)

    # CAR-01: Summary statistics
    print("== CAR-01 ==")
    stats_json = {}
    for col in ['Lbs-', 'Acc+', 'Mpg+']:
        # Extract non-None values
        vals = [row[col] for row in rows if row[col] is not None]
        mean_val, median_val, min_val, max_val = compute_stats(vals)
        # Determine best / worst based on goal direction
        if col.endswith('-'):   # minimize
            best_val = min_val
            worst_val = max_val
        else:                    # maximize
            best_val = max_val
            worst_val = min_val
        # Store in dict
        stats_json[col] = {
            "mean": mean_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "best": best_val,
            "worst": worst_val
        }
        # Print a human‑friendly line (at least three lines)
        print(f"{col}: mean={mean_val:.6g}, median={median_val:.6g}, min={min_val:.6g}, max={max_val:.6g}, best={best_val:.6g}, worst={worst_val:.6g}")

    # Output machine‑checked answer for CAR-01
    print("ANSWER CAR-01", json.dumps(stats_json, separators=(',', ':')))

    # CAR-02: Ranking based on composite score
    print("\n== CAR-02 ==")

    # Compute overall score for each row that has all three metrics
    scores = []
    for idx, row in enumerate(rows, start=1):  # 1‑based row number
        lbs = row['Lbs-']
        acc = row['Acc+']
        mpg = row['Mpg+']
        if lbs is None or acc is None or mpg is None:
            continue  # skip rows with missing data

        # Normalization ranges
        lbs_vals = [r['Lbs-'] for r in rows if r['Lbs-'] is not None]
        acc_vals = [r['Acc+'] for r in rows if r['Acc+'] is not None]
        mpg_vals = [r['Mpg+'] for r in rows if r['Mpg+'] is not None]
        lbs_min, lbs_max = min(lbs_vals), max(lbs_vals)
        acc_min, acc_max = min(acc_vals), max(acc_vals)
        mpg_min, mpg_max = min(mpg_vals), max(mpg_vals)

        # Avoid division by zero (constant column)
        def norm(val, v_min, v_max, reverse=False):
            if v_max == v_min:
                return 0.5
            score = (val - v_min) / (v_max - v_min)
            return 1.0 - score if reverse else score

        # For Lbs- (minimize) -> reverse=True
        score_lbs = norm(lbs, lbs_min, lbs_max, reverse=True)
        score_acc = norm(acc, acc_min, acc_max, reverse=False)
        score_mpg = norm(mpg, mpg_min, mpg_max, reverse=False)

        overall = score_lbs + score_acc + score_mpg
        scores.append((overall, idx))

    # Get top 10 rows
    top_rows = sorted(scores, key=lambda x: -x[0])[:10]
    top_indices = [idx for _, idx in top_rows]

    # Output human‑friendly lines
    print(f"Top {len(top_indices)} rows: {top_indices}")

    # Output machine‑checked answer for CAR-02
    car02_json = {"top": top_indices}
    print("ANSWER CAR-02", json.dumps(car02_json, separators=(',', ':')))

if __name__ == "__main__":
    main()
