#!/usr/bin/env python3
"""
skill_06.py - CAR-11 and CAR-12: Recommendation explanations and tradeoff analysis

Usage:
    python skill_06.py <csv_path>
"""

import sys
import csv
import json
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# CSV reading utilities
# ----------------------------------------------------------------------
GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']
NON_GOAL_COLUMNS = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin']

def read_csv(path):
    """Read CSV and return list of dicts (rows). Convert numeric columns to float or None."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Goal columns
            for col in GOAL_COLUMNS:
                val = row[col]
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = None
            # Non‑goal columns (numeric in this dataset)
            for col in NON_GOAL_COLUMNS:
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

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
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

def compute_scores(rows, mins, maxs):
    """Return dict idx->score for rows with all goals present."""
    scores = {}
    for idx, r in enumerate(rows, start=1):
        if any(r[c] is None for c in GOAL_COLUMNS):
            continue
        score = (
            normalize(r['Lbs-'], mins['Lbs-'], maxs['Lbs-'], reverse=True) +
            normalize(r['Acc+'], mins['Acc+'], maxs['Acc+'], reverse=False) +
            normalize(r['Mpg+'], mins['Mpg+'], maxs['Mpg+'], reverse=False)
        )
        scores[idx] = score
    return scores

def euclidean_distance(r1, r2):
    """Euclidean distance over goal columns, assuming all values present."""
    return math.sqrt(sum((r1[c] - r2[c]) ** 2 for c in GOAL_COLUMNS))

# ----------------------------------------------------------------------
# CAR-11 logic
# ----------------------------------------------------------------------
def car11(rows, scores):
    # Recommended row: highest score
    best_row = max(scores.items(), key=lambda x: x[1])
    best_idx, best_score = best_row
    best_values = {c: rows[best_idx-1][c] for c in GOAL_COLUMNS}

    # Find 3 nearest neighbors (excluding the row itself)
    neighbors = []
    for idx, r in enumerate(rows, start=1):
        if idx == best_idx:
            continue
        if any(r[c] is None for c in GOAL_COLUMNS):
            continue
        dist = euclidean_distance(best_values, {c: r[c] for c in GOAL_COLUMNS})
        neighbors.append((dist, idx, r))
    neighbors.sort(key=lambda x: x[0])
    selected = neighbors[:3]  # at least 3 expected

    # Prepare JSON
    neighbors_json = []
    for _, nidx, nrow in selected:
        neighbors_json.append({
            "row": nidx,
            "values": {c: nrow[c] for c in GOAL_COLUMNS}
        })

    answer = {
        "recommended": best_idx,
        "neighbors": neighbors_json
    }
    return best_idx, best_score, answer

# ----------------------------------------------------------------------
# CAR-12 logic
# ----------------------------------------------------------------------
def car12(rows, scores):
    # Select two vehicles with different strengths: best and worst
    best_row = max(scores.items(), key=lambda x: x[1])
    worst_row = min(scores.items(), key=lambda x: x[1])
    a_idx, a_score = best_row
    b_idx, b_score = worst_row

    # Determine preferred
    if a_score >= b_score:
        preferred = a_idx
        non_preferred = b_idx
        preferred_score = a_score
        non_preferred_score = b_score
    else:
        preferred = b_idx
        non_preferred = a_idx
        preferred_score = b_score
        non_preferred_score = a_score

    # Differences
    diffs = []
    tradeoffs = []
    for col in GOAL_COLUMNS:
        a_val = rows[a_idx-1][col]
        b_val = rows[b_idx-1][col]
        diffs.append({"feature": col, "a": a_val, "b": b_val})
        # Determine if non_preferred is better (consider goal direction)
        if col == 'Lbs-':  # minimize
            if b_val < a_val:
                tradeoffs.append(col)
        else:  # maximize
            if b_val > a_val:
                tradeoffs.append(col)

    # JSON answer
    answer = {
        "a": a_idx,
        "b": b_idx,
        "preferred": preferred,
        "scores": {
            str(a_idx): a_score,
            str(b_idx): b_score
        },
        "differences": diffs[:3],  # at least 1
        "tradeoffs": tradeoffs[:]
    }
    return answer

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)

    mins, maxs = compute_minmax(rows)
    scores = compute_scores(rows, mins, maxs)

    # CAR-11
    print("== CAR-11 ==")
    rec_idx, rec_score, car11_answer = car11(rows, scores)
    print(f"Recommended row: {rec_idx} (score {rec_score:.4g})")
    for i, n in enumerate(car11_answer["neighbors"], start=1):
        vals = n["values"]
        print(f"Neighbor {i}: row {n['row']}, Lbs-={vals['Lbs-']:.4g}, Acc+={vals['Acc+']:.4g}, Mpg+={vals['Mpg+']:.4g}")
    print("ANSWER CAR-11", json.dumps(car11_answer, separators=(',', ':')))

    # CAR-12
    print("\n== CAR-12 ==")
    car12_answer = car12(rows, scores)
    print(f"Preferred row: {car12_answer['preferred']} (score {car12_answer['scores'][str(car12_answer['preferred'])]:.4g}), "
          f"other: {car12_answer['a'] if car12_answer['preferred']==car12_answer['b'] else car12_answer['b']} "
          f"(score {car12_answer['scores'][str(car12_answer['a']) if car12_answer['preferred']==car12_answer['b'] else str(car12_answer['b'])]:.4g})")
    print("Differences:")
    for d in car12_answer["differences"]:
        print(f"  Feature {d['feature']}: {d['a']:.4g} vs {d['b']:.4g}")
    if car12_answer["tradeoffs"]:
        print("Tradeoffs where non‑preferred is better:", ", ".join(car12_answer["tradeoffs"]))
    else:
        print("No tradeoffs identified.")
    print("ANSWER CAR-12", json.dumps(car12_answer, separators=(',', ':')))

if __name__ == "__main__":
    main()
