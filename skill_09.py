#!/usr/bin/env python3
"""
Skill 09: Outlier detection and error checking for the auto93 dataset.

CAR-17: Flag vehicle records whose combinations of characteristics are unusually different
        from the rest of the inventory.

CAR-18: Flag vehicles whose reported fuel efficiency (Mpg+) or weight (Lbs-)
        differs substantially from what similar vehicles would suggest.
"""

import csv
import json
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_value(v: str):
    if v == '?' or v.strip() == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def is_goal(col: str) -> bool:
    return (col.endswith('+') or col.endswith('-')) and not col.endswith('X')

def is_input(col: str) -> bool:
    return not is_goal(col) and not col.endswith('X')

# ----------------------------------------------------------------------
# Read CSV
# ----------------------------------------------------------------------
def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Compute mean & std for input columns
# ----------------------------------------------------------------------
def compute_mean_std(rows, input_cols):
    means = {}
    stds = {}
    for col in input_cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if vals:
            m = sum(vals) / len(vals)
            s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
            means[col] = m
            stds[col] = s
        else:
            means[col] = 0.0
            stds[col] = 0.0
    return means, stds

# ----------------------------------------------------------------------
# Standardized input vector per row
# ----------------------------------------------------------------------
def standardized_row(row, input_cols, means, stds):
    vec = {}
    for col in input_cols:
        val = parse_value(row[col])
        if val is None:
            continue
        std = stds[col]
        if std == 0:
            z = 0.0
        else:
            z = (val - means[col]) / std
        vec[col] = z
    return vec

# ----------------------------------------------------------------------
# CAR-17: Unusualness score
# ----------------------------------------------------------------------
def car_17(rows, input_cols, max_flag=40):
    means, stds = compute_mean_std(rows, input_cols)
    std_rows = []
    for idx, row in enumerate(rows, start=1):
        vec = standardized_row(row, input_cols, means, stds)
        if not vec:
            score = 0.0
        else:
            score = sum(v * v for v in vec.values())
        std_rows.append((idx, score))

    scores = [s for _, s in std_rows]
    if not scores:
        mean_score = 0.0
        std_score = 0.0
    else:
        mean_score = sum(scores) / len(scores)
        std_score = math.sqrt(sum((x - mean_score) ** 2 for x in scores) / (len(scores) - 1)) if len(scores) > 1 else 0.0
    threshold = mean_score + 2 * std_score

    flagged = [(idx, s) for idx, s in std_rows if s > threshold]
    flagged.sort(key=lambda x: -x[1])
    flagged = flagged[:max_flag]

    flagged_rows = [idx for idx, _ in flagged]
    flagged_scores = {str(idx): round(score, 6) for idx, score in flagged}

    return {
        "flagged": flagged_rows,
        "scores": flagged_scores
    }

# ----------------------------------------------------------------------
# CAR-18: Difference from expected
# ----------------------------------------------------------------------
def car_18(rows, input_cols, goal_cols, k=5, max_flag=40):
    means, stds = compute_mean_std(rows, input_cols)
    std_vectors = [standardized_row(r, input_cols, means, stds) for r in rows]

    # Precompute distances
    distances = []
    for i in range(len(rows)):
        row_vec = std_vectors[i]
        dlist = []
        for j in range(len(rows)):
            if i == j:
                continue
            vec2 = std_vectors[j]
            # compute squared euclidean on overlapping features
            ssum = 0.0
            count = 0
            for col, val in row_vec.items():
                if col in vec2:
                    diff = val - vec2[col]
                    ssum += diff * diff
                    count += 1
            if count == 0:
                # no overlap, treat as infinite distance
                d = float('inf')
            else:
                d = math.sqrt(ssum / count)  # average over available features
            dlist.append((j, d))
        dlist.sort(key=lambda x: x[1])
        distances.append([idx for idx, _ in dlist[:k]])

    # For each row, compute expected value for each goal
    diffs = defaultdict(list)  # goal -> list of diff
    diff_records = []  # list of dict for flagged candidates later

    for idx, row in enumerate(rows, start=1):
        actual = {}
        for g in goal_cols:
            v = parse_value(row[g])
            if v is not None:
                actual[g] = v
        if not actual:
            continue
        neighbor_idxs = distances[idx - 1]
        for g in goal_cols:
            neighbor_vals = [parse_value(rows[n-1][g]) for n in neighbor_idxs if parse_value(rows[n-1][g]) is not None]
            if not neighbor_vals:
                continue
            expected = sum(neighbor_vals) / len(neighbor_vals)
            diff = abs(actual[g] - expected)
            diffs[g].append(diff)
            diff_records.append({
                "row": idx,
                "column": g,
                "actual": actual[g],
                "expected": round(expected, 6),
                "diff": diff
            })

    flagged_entries = []
    for g in goal_cols:
        vals = diffs.get(g, [])
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        threshold = mean + 2 * std
        for rec in diff_records:
            if rec["column"] == g and rec["diff"] > threshold:
                flagged_entries.append(rec)

    flagged_entries.sort(key=lambda x: -x["diff"])
    flagged_entries = flagged_entries[:max_flag]

    # Prepare JSON-friendly structure
    output = []
    for e in flagged_entries:
        output.append({
            "row": e["row"],
            "column": e["column"],
            "actual": round(e["actual"], 6),
            "expected": e["expected"]
        })

    return {
        "flagged": output
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    input_cols = [c for c in fieldnames if is_input(c)]
    goal_cols = [c for c in fieldnames if is_goal(c)]

    # CAR-17
    print("== CAR-17 ==")
    res17 = car_17(rows, input_cols)
    # Human‑readable sample
    print(f"Number of flagged rows: {len(res17['flagged'])}")
    print("Flagged rows and scores (top 5):")
    for row in res17['flagged'][:5]:
        print(f"  Row {row} unusualness score: {res17['scores'][str(row)]}")
    print("ANSWER CAR-17", json.dumps(res17, ensure_ascii=False))

    # CAR-18
    print("\n== CAR-18 ==")
    res18 = car_18(rows, input_cols, goal_cols)
    print(f"Number of flagged differences: {len(res18['flagged'])}")
    print("Flagged entries (top 5):")
    for e in res18['flagged'][:5]:
        print(f"  Row {e['row']}: {e['column']} actual={e['actual']:.6f} expected={e['expected']:.6f}")
    print("ANSWER CAR-18", json.dumps(res18, ensure_ascii=False))

if __name__ == "__main__":
    main()
