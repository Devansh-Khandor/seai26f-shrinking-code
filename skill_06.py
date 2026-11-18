#!/usr/bin/env python3
"""
skill_06.py

APACHE-11 – When the system recommends a configuration, explain its recommendation by comparing the configuration's settings and performance with relevant previously tested configurations.
APACHE-12 – When two configurations receive different performance assessments, explain which settings differ between them and how those differences relate to the predicted or recorded outcomes.
"""

import csv
import json
import math
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Parse a CSV cell into a Python number if possible."""
    if v == '?' or v is None:
        return v
    try:
        if '.' in v or 'e' in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v

def read_csv(path):
    """Read CSV and return list of (row_number, row_dict)."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for rn, rec in enumerate(reader, start=1):
            # store parsed values for easy math
            parsed = {k: parse_value(v) for k, v in rec.items()}
            rows.append((rn, rec, parsed))
    return rows, reader.fieldnames

def distance(row1, row2, feature_cols):
    """Euclidean distance over overlapping numeric or symbol features."""
    sumsq = 0.0
    dims = 0
    for col in feature_cols:
        v1 = row1[1][col]
        v2 = row2[1][col]
        if v1 == '?' or v2 == '?' or v1 is None or v2 is None:
            continue
        try:
            f1 = float(v1)
            f2 = float(v2)
            diff = f1 - f2
            sumsq += diff * diff
            dims += 1
        except (TypeError, ValueError):
            sumsq += 0 if v1 == v2 else 1
            dims += 1
    if dims == 0:
        return float('inf')
    return math.sqrt(sumsq)

def best_row(rows, primary_goal):
    best = None
    best_val = None
    for rn, rec, parsed in rows:
        val = parsed.get(primary_goal, '?')
        if val == '?' or val is None:
            continue
        if best_val is None or val < best_val:
            best = (rn, rec, parsed)
            best_val = val
    return best

def worst_row(rows, primary_goal):
    worst = None
    worst_val = None
    for rn, rec, parsed in rows:
        val = parsed.get(primary_goal, '?')
        if val == '?' or val is None:
            continue
        if worst_val is None or val > worst_val:
            worst = (rn, rec, parsed)
            worst_val = val
    return worst

def find_neighbors(rec, rows, feature_cols, k=3):
    """Return k nearest neighbors to rec."""
    dists = []
    for rn, rec_other, _ in rows:
        if rn == rec[0]:
            continue
        d = distance(rec, (rn, rec_other, None), feature_cols)
        dists.append((d, rn, rec_other))
    # sort by distance then row number
    dists.sort(key=lambda x: (x[0], x[1]))
    neighbors = []
    for d, rn, r in dists[:k]:
        neighbors.append((rn, r))
    return neighbors

def compute_scores(rows, primary_goal):
    scores = {}
    for rn, rec, parsed in rows:
        val = parsed.get(primary_goal, '?')
        if val == '?' or val is None:
            continue
        if primary_goal.endswith('+'):
            scores[rn] = val
        else:  # '-'
            scores[rn] = -val
    return scores

def tradeoffs(row_a, row_b, goal_cols):
    """Return goal columns where non‑preferred row is better."""
    a_is_pref = None  # will be set later
    trade = []
    for col in goal_cols:
        a_val = row_a[2].get(col, '?')
        b_val = row_b[2].get(col, '?')
        if a_val == '?' or b_val == '?':
            continue
        if col.endswith('+'):
            if b_val > a_val:
                trade.append(col)
        else:
            if b_val < a_val:
                trade.append(col)
    return trade

def differences(row_a, row_b, feature_cols):
    diffs = []
    for col in feature_cols:
        a_val = row_a[1][col]
        b_val = row_b[1][col]
        if a_val != b_val:
            diffs.append({
                "feature": col,
                "a": a_val,
                "b": b_val
            })
    return diffs

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    # Determine columns
    goal_cols = [c for c in fieldnames if c.endswith('+') or c.endswith('-')]
    x_cols = [c for c in fieldnames if c.endswith('X')]
    feature_cols = [c for c in fieldnames if c not in goal_cols + x_cols]

    if not goal_cols:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)

    # Primary goal: first '-' if any, else first '+'
    primary_goal = None
    for c in goal_cols:
        if c.endswith('-'):
            primary_goal = c
            break
    if primary_goal is None:
        primary_goal = goal_cols[0]

    # -------- APACHE-11 ----------
    print("== APACHE-11 ==")
    rec_best = best_row(rows, primary_goal)
    if rec_best is None:
        print("No valid rows for recommendation.")
        sys.exit(1)
    rec_num, rec_row, rec_parsed = rec_best
    print(f"Recommended configuration: Row {rec_num}")

    neighbors = find_neighbors(rec_best, rows, feature_cols, k=3)
    print("Similar configurations:")
    for rn, r in neighbors:
        print(f"  Row {rn}")
    # Machine‑checked answer
    values_dict = {k: (rec_parsed[k] if rec_parsed[k] != '?' else '?') for k in fieldnames}
    neighbors_list = []
    for rn, r in neighbors:
        parsed = {k: parse_value(r[k]) for k in fieldnames}
        ndict = {k: (parsed[k] if parsed[k] != '?' else '?') for k in fieldnames}
        neighbors_list.append({"row": rn, "values": ndict})
    answer_11 = {"recommended": rec_num, "neighbors": neighbors_list}
    print("ANSWER APACHE-11", json.dumps(answer_11, separators=(",", ":")))

    # -------- APACHE-12 ----------
    print("\n== APACHE-12 ==")
    a = best_row(rows, primary_goal)
    b = worst_row(rows, primary_goal)
    if a is None or b is None:
        print("Not enough rows to compare.")
        sys.exit(1)
    a_num, a_row, a_parsed = a
    b_num, b_row, b_parsed = b

    scores = compute_scores(rows, primary_goal)
    score_a = scores.get(a_num, 0)
    score_b = scores.get(b_num, 0)
    preferred = a_num if score_a >= score_b else b_num
    a_is_pref = preferred == a_num

    diffs = differences(a, b, feature_cols)
    trade = []
    if a_is_pref:
        trade = tradeoffs(a, b, goal_cols)
    else:
        trade = tradeoffs(b, a, goal_cols)

    print(f"Configuration A: Row {a_num}  Score: {score_a}")
    print(f"Configuration B: Row {b_num}  Score: {score_b}")
    print(f"Preferred: Row {preferred}")
    print("Differences:")
    for d in diffs[:5]:
        print(f"  {d['feature']}: A={d['a']} B={d['b']}")
    print("Tradeoffs (goals better for non‑preferred row):")
    for t in trade:
        print(f"  {t}")

    answer_12 = {
        "a": a_num,
        "b": b_num,
        "preferred": preferred,
        "scores": {"a": score_a, "b": score_b},
        "differences": diffs[:5],
        "tradeoffs": trade
    }
    print("ANSWER APACHE-12", json.dumps(answer_12, separators=(",", ":")))

if __name__ == "__main__":
    main()
