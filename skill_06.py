#!/usr/bin/env python3
"""
Skill 06: Explanations for vehicle recommendations and trade‑offs.

This script implements two tasks:

* CAR‑11 – Identify the vehicle with the best overall recommendation and
  provide a few similar vehicles for context.
* CAR‑12 – Compare the best and worst vehicles, explaining why the best
  receives the higher recommendation and showing the trade‑offs.

The scoring used is the same composite score as in previous skills:
each goal is normalized (min/max) and summed.  Goals ending with '+' are
maximized; those ending with '-' are minimized.

The output format follows the assignment specification: each CAR block is
preceded by a line ``== <ID> ==``, followed by at least three human‑readable
lines, and terminated with an ``ANSWER <ID> <JSON>`` line that the grader
uses to validate the result.

"""

import csv
import json
import math
import sys
import statistics


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Read the CSV file, returning a list of (row_index, row_dict) tuples."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, line in enumerate(reader, start=1):
            parsed = {}
            for k, v in line.items():
                if v == '?' or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append((idx, parsed))
    return rows


def goal_columns(rows):
    """Return the list of columns that are goals (end in '+' or '-')."""
    if not rows:
        return []
    _, first = rows[0]
    return [col for col in first if col.endswith('+') or col.endswith('-')]


def format_num(x):
    """Return a JSON‑friendly representation of a number."""
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return round(x, 4)


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def compute_scores(rows, goal_cols):
    """Return a dict mapping row number to composite score."""
    # Compute global mins and maxes for each goal
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [rec[col] for _, rec in rows if rec[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0

    scores = {}
    for idx, rec in rows:
        if any(rec[col] is None for col in goal_cols):
            continue
        norm = 0.0
        for col in goal_cols:
            denom = maxs[col] - mins[col]
            if denom == 0:
                continue
            if col.endswith('-'):          # minimize
                norm += (maxs[col] - rec[col]) / denom
            else:                          # maximize
                norm += (rec[col] - mins[col]) / denom
        scores[idx] = norm
    return scores


# ----------------------------------------------------------------------
# CAR‑11
# ----------------------------------------------------------------------
def car11(rows, scores, goal_cols):
    """Return the recommendation structure for CAR‑11."""
    if not scores:
        return {"recommended": None, "neighbors": []}

    # Best row
    best_row = max(scores.items(), key=lambda kv: kv[1])[0]
    best_score = scores[best_row]

    # Find nearest neighbors (exclude itself)
    neighbors = sorted(
        ((abs(scores[r] - best_score), r) for r in scores if r != best_row),
        key=lambda kv: kv[0]
    )
    neighbors = [r for _, r in neighbors[:3]]  # at least 3 if available

    neighbor_details = []
    for r in neighbors:
        rec = next(rec for idx, rec in rows if idx == r)
        values = {col: format_num(rec[col]) for col in goal_cols}
        neighbor_details.append({"row": r, "values": values})

    return {"recommended": best_row, "neighbors": neighbor_details}


# ----------------------------------------------------------------------
# CAR‑12
# ----------------------------------------------------------------------
def car12(rows, scores, goal_cols):
    """Return the comparison structure for CAR‑12."""
    if not scores:
        return {"a": None, "b": None, "preferred": None, "scores": {}, "differences": [], "tradeoffs": []}

    # Identify best (a) and worst (b) rows
    a_row = max(scores.items(), key=lambda kv: kv[1])[0]
    b_row = min(scores.items(), key=lambda kv: kv[1])[0]
    a_score = scores[a_row]
    b_score = scores[b_row]

    rec_a = next(rec for idx, rec in rows if idx == a_row)
    rec_b = next(rec for idx, rec in rows if idx == b_row)

    # Differences for each goal
    diff_list = []
    tradeoffs = []
    for col in goal_cols:
        val_a = rec_a[col]
        val_b = rec_b[col]
        diff_list.append({"feature": col, "a": format_num(val_a), "b": format_num(val_b)})
        # Determine if b is better on this goal
        if col.endswith('-'):  # lower is better
            if val_b < val_a:
                tradeoffs.append(col)
        else:  # higher is better
            if val_b > val_a:
                tradeoffs.append(col)

    return {
        "a": a_row,
        "b": b_row,
        "preferred": "a",
        "scores": {"a": format_num(a_score), "b": format_num(b_score)},
        "differences": diff_list,
        "tradeoffs": tradeoffs
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    goal_cols = goal_columns(rows)

    scores = compute_scores(rows, goal_cols)

    # CAR‑11
    print("== CAR-11 ==")
    car11_res = car11(rows, scores, goal_cols)
    if car11_res["recommended"] is None:
        print("No complete rows available for recommendation.")
    else:
        rec = car11_res["recommended"]
        print(f"Recommended vehicle: row {rec}")
        print("Similar vehicles:")
        for neigh in car11_res["neighbors"]:
            vals = ", ".join(f"{k}={v}" for k, v in neigh["values"].items())
            print(f"  Row {neigh['row']}: {vals}")
    print("ANSWER CAR-11 " + json.dumps(car11_res, separators=(',', ':')))

    # CAR‑12
    print("\n== CAR-12 ==")
    car12_res = car12(rows, scores, goal_cols)
    if car12_res["a"] is None or car12_res["b"] is None:
        print("Insufficient data for comparison.")
    else:
        a = car12_res["a"]
        b = car12_res["b"]
        print(f"Comparison: Vehicle {a} vs Vehicle {b}")
        print(f"Scores: Vehicle {a} = {car12_res['scores']['a']}, Vehicle {b} = {car12_res['scores']['b']}")
        print("Feature differences:")
        for d in car12_res["differences"]:
            print(f"  {d['feature']}: {a}={d['a']}, {b}={d['b']}")
        if car12_res["tradeoffs"]:
            print(f"Trade‑offs where Vehicle {b} is better: {', '.join(car12_res['tradeoffs'])}")
        else:
            print(f"No trade‑offs: Vehicle {a} outperforms Vehicle {b} on all goals.")
    print("ANSWER CAR-12 " + json.dumps(car12_res, separators=(',', ':')))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_06.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
