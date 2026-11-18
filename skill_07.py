#!/usr/bin/env python3
"""
skill_07.py - CAR-13 and CAR-14: balanced vehicle selection and inspection ordering

Usage:
    python skill_07.py <csv_path>
"""

import sys
import csv
import json

GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']  # Lbs- minimize, others maximize


def read_csv(path):
    """Read CSV and return list of dicts. Convert numeric columns to float or None."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            # Convert goal columns
            for col in GOAL_COLUMNS:
                val = row[col]
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = None
            # Store 1‑based row number
            row['row_num'] = idx
            rows.append(row)
    return rows


def compute_minmax(rows):
    mins = {}
    maxs = {}
    for col in GOAL_COLUMNS:
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = 0.0
            maxs[col] = 0.0
    return mins, maxs


def normalize(val, v_min, v_max, reverse=False):
    if v_max == v_min:
        return 0.5
    score = (val - v_min) / (v_max - v_min)
    return 1.0 - score if reverse else score


def composite_score(row, mins, maxs):
    # Lbs- minimize, others maximize
    score = (
        normalize(row['Lbs-'], mins['Lbs-'], maxs['Lbs-'], reverse=True) +
        normalize(row['Acc+'], mins['Acc+'], maxs['Acc+']) +
        normalize(row['Mpg+'], mins['Mpg+'], maxs['Mpg+'])
    )
    return score


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)

    # Only rows that have all goal values
    eval_rows = [r for r in rows if all(r[col] is not None for col in GOAL_COLUMNS)]
    if not eval_rows:
        print("== CAR-13 ==")
        print("No rows with complete goal data.")
        print("ANSWER CAR-13 {}")
        print("\n== CAR-14 ==")
        print("No rows with complete goal data.")
        print("ANSWER CAR-14 {}")
        sys.exit(0)

    mins, maxs = compute_minmax(eval_rows)

    # Compute composite for each row
    scores = {}
    for r in eval_rows:
        scores[r['row_num']] = composite_score(r, mins, maxs)

    # Sort rows by descending composite
    sorted_rows = sorted(eval_rows, key=lambda r: scores[r['row_num']], reverse=True)
    top_n = min(30, len(sorted_rows))
    evaluated = [r['row_num'] for r in sorted_rows[:top_n]]
    best = evaluated[0] if evaluated else None

    # ---------- CAR-13 ----------
    print("== CAR-13 ==")
    print(f"Evaluated rows: {evaluated}")
    print(f"Best row: {best}")
    # Show top 5 composite scores among evaluated for illustration
    top5_scores = ', '.join(f"{r}: {scores[r]:.4g}" for r in evaluated[:5])
    print(f"Top 5 composite scores: {top5_scores}")

    # Machine‑checked answer
    ans_car13 = {"evaluated": evaluated, "best": best}
    print("ANSWER CAR-13", json.dumps(ans_car13, separators=(',', ':')))

    # ---------- CAR-14 ----------
    print("\n== CAR-14 ==")
    # Order for inspection: same as evaluated list (high composite first)
    order = evaluated
    print(f"Evaluation order: {order}")
    print(f"Top 5 candidates: {order[:5]}")

    # Machine‑checked answer
    ans_car14 = {"order": order}
    print("ANSWER CAR-14", json.dumps(ans_car14, separators=(',', ':')))


if __name__ == "__main__":
    main()
