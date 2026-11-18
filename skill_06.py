#!/usr/bin/env python3
"""
skill_06.py

Usage:
    python skill_06.py <csv path>

Provides two new analysis sections:
  * CAR-11: When a vehicle is recommended, explain the recommendation using comparisons with similar vehicles.
  * CAR-12: When two vehicles differ in strengths, explain why one receives a higher overall recommendation and show the trade‑offs involved.

The script prints each section header, a few descriptive lines, and a machine‑checked ANSWER line in JSON format.
"""

import sys
import json
import numpy as np

# Reuse helpers from previous skills
from skill_01 import read_csv, parse_float, normalize

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def compute_score(row, mins, maxs):
    """
    Compute the combined performance score for a single row.
    Returns None if any goal is missing.
    Score is higher when performance is better.
    """
    lbs = parse_float(row[mins['lbs_idx']])
    acc = parse_float(row[mins['acc_idx']])
    mpg = parse_float(row[mins['mpg_idx']])
    if lbs is None or acc is None or mpg is None:
        return None

    norm_lbs = normalize(lbs, mins['lbs_min'], maxs['lbs_max'])
    norm_acc = normalize(acc, mins['acc_min'], maxs['acc_max'])
    norm_mpg = normalize(mpg, mins['mpg_min'], maxs['mpg_max'])

    return (1 - norm_lbs) + (1 - norm_acc) + norm_mpg


def compute_all_scores(header, data):
    """
    Compute scores for all rows with complete goal data.
    Return list of (row_number, score) and also a mapping from row_number to original row.
    """
    idx_lbs = header.index('Lbs-')
    idx_acc = header.index('Acc+')
    idx_mpg = header.index('Mpg+')

    # Collect goal values for normalization
    lbs_vals = [parse_float(r[idx_lbs]) for r in data if parse_float(r[idx_lbs]) is not None]
    acc_vals = [parse_float(r[idx_acc]) for r in data if parse_float(r[idx_acc]) is not None]
    mpg_vals = [parse_float(r[idx_mpg]) for r in data if parse_float(r[idx_mpg]) is not None]

    mins = {
        'lbs_idx': idx_lbs,
        'acc_idx': idx_acc,
        'mpg_idx': idx_mpg,
        'lbs_min': float(np.min(lbs_vals)),
        'acc_min': float(np.min(acc_vals)),
        'mpg_min': float(np.min(mpg_vals))
    }
    maxs = {
        'lbs_max': float(np.max(lbs_vals)),
        'acc_max': float(np.max(acc_vals)),
        'mpg_max': float(np.max(mpg_vals))
    }

    scores = []
    row_map = {}
    for i, r in enumerate(data, start=1):
        row_map[i] = r
        sc = compute_score(r, mins, maxs)
        if sc is not None:
            scores.append((i, sc))

    return scores, row_map, mins, maxs


def find_neighbors(best_row, best_score, scores, row_map, mins, maxs, top_n=3):
    """
    Find the top `top_n` rows whose score is closest to the best_score
    (excluding the best row itself). Return a list of dicts containing
    the row number and goal values.
    """
    # Compute absolute difference to best
    diffs = [(row, abs(score - best_score)) for row, score in scores if row != best_row]
    diffs.sort(key=lambda x: x[1])

    neighbors = []
    for row_num, _ in diffs[:top_n]:
        r = row_map[row_num]
        neighbor = {
            "row": row_num,
            "values": {
                "Lbs-": parse_float(r[mins['lbs_idx']]),
                "Acc+": parse_float(r[mins['acc_idx']]),
                "Mpg+": parse_float(r[mins['mpg_idx']])
            }
        }
        neighbors.append(neighbor)
    return neighbors


def compute_differences(row_a, row_b, mins, maxs):
    """
    Compute absolute differences for each goal between two rows.
    Return a list of dicts with feature, a value, b value.
    """
    diffs = []
    for key in ('lbs', 'acc', 'mpg'):
        idx = mins[f'{key}_idx']
        val_a = parse_float(row_a[idx])
        val_b = parse_float(row_b[idx])
        if val_a is None or val_b is None:
            continue
        diff = abs(val_a - val_b)
        diffs.append({
            "feature": "Lbs-" if key == 'lbs' else ("Acc+" if key == 'acc' else "Mpg+"),
            "a": val_a,
            "b": val_b,
            "diff": diff
        })
    return diffs


def find_tradeoffs(row_a, row_b, mins, maxs):
    """
    Identify goal columns where the non-preferred row (row_b) is better.
    """
    tradeoffs = []
    for key in ('lbs', 'acc', 'mpg'):
        idx = mins[f'{key}_idx']
        val_a = parse_float(row_a[idx])
        val_b = parse_float(row_b[idx])
        if val_a is None or val_b is None:
            continue
        if key in ('lbs', 'acc'):  # minimize
            if val_b < val_a:
                tradeoffs.append("Lbs-" if key == 'lbs' else "Acc+")
        else:  # mpg, maximize
            if val_b > val_a:
                tradeoffs.append("Mpg+")
    return tradeoffs

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path):
    header, data = read_csv(csv_path)

    # Compute all scores
    scores, row_map, mins, maxs = compute_all_scores(header, data)

    if not scores:
        print("No complete rows for scoring.")
        sys.exit(1)

    # CAR-11
    # Find the best row
    best_row, best_score = max(scores, key=lambda x: x[1])
    neighbors = find_neighbors(best_row, best_score, scores, row_map, mins, maxs, top_n=3)

    print("== CAR-11 ==")
    print(f"Recommended vehicle: Row {best_row} (Score={best_score:.3f})")
    print("Neighbors:")
    for nb in neighbors:
        v = nb["values"]
        print(f"  Row {nb['row']}: Lbs-={v['Lbs-']:.3f} Acc+={v['Acc+']:.3f} Mpg+={v['Mpg+']:.3f}")

    # Machine‑checked answer
    answer_car11 = {
        "recommended": best_row,
        "neighbors": [
            {
                "row": nb["row"],
                "values": {
                    "Lbs-": nb["values"]["Lbs-"],
                    "Acc+": nb["values"]["Acc+"],
                    "Mpg+": nb["values"]["Mpg+"]
                }
            }
            for nb in neighbors
        ]
    }
    print(f"ANSWER CAR-11 {json.dumps(answer_car11, separators=(',', ':'), sort_keys=True)}")

    # CAR-12
    # Pick best (a) and worst (b) rows
    worst_row, worst_score = min(scores, key=lambda x: x[1])
    a_row_num, a_score = best_row, best_score
    b_row_num, b_score = worst_row, worst_score
    a_row = row_map[a_row_num]
    b_row = row_map[b_row_num]
    preferred = "a" if a_score > b_score else "b"

    differences = compute_differences(a_row, b_row, mins, maxs)
    # Keep only feature, a, b (remove diff key)
    differences_output = [
        {"feature": d["feature"], "a": d["a"], "b": d["b"]}
        for d in sorted(differences, key=lambda x: x["diff"], reverse=True)
    ]

    tradeoffs = find_tradeoffs(a_row, b_row, mins, maxs)

    print("\n== CAR-12 ==")
    print(f"Vehicle a (Row {a_row_num}) Score={a_score:.3f}")
    print(f"Vehicle b (Row {b_row_num}) Score={b_score:.3f}")
    print(f"Preferred vehicle: {preferred.upper()}")
    print(f"Trade‑offs (goals where {('b' if preferred=='a' else 'a')} is better): {tradeoffs}")

    # Machine‑checked answer
    answer_car12 = {
        "a": a_row_num,
        "b": b_row_num,
        "preferred": preferred,
        "scores": {
            "a": a_score,
            "b": b_score
        },
        "differences": differences_output,
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER CAR-12 {json.dumps(answer_car12, separators=(',', ':'), sort_keys=True)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
