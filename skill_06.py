#!/usr/bin/env python3
"""
skill_06.py <csv_path>

Implements NASA-11 and NASA-12.

NASA‑11: For a held‑out project estimate its Effort‑ using the
nearest historical (non‑held‑out) projects and provide the three
most similar projects and their recorded characteristics.

NASA‑12: Compare two proposals that receive different risk assessments.
Compute a simple risk score (lower Effort‑ and Defects‑ are better),
pick the preferred one, explain the main differences, and list the
trade‑offs on the goal columns.
"""

import sys
import csv
import json
import math
from statistics import mean

# ------------------------------------------------------------------
# CSV utilities
# ------------------------------------------------------------------
def parse_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def to_float(val):
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def convert_value(val):
    """Return numeric value if possible, otherwise original string."""
    f = to_float(val)
    return f if f is not None else val

# ------------------------------------------------------------------
# Distance between two rows on numeric non‑goal columns
# ------------------------------------------------------------------
def euclidean_distance(r1, r2, numeric_cols):
    sumsq = 0.0
    count = 0
    for col in numeric_cols:
        v1 = to_float(r1[col])
        v2 = to_float(r2[col])
        if v1 is None or v2 is None:
            continue
        diff = v1 - v2
        sumsq += diff * diff
        count += 1
    # If no common numeric dimension, treat as infinite distance
    return math.sqrt(sumsq) if count > 0 else float('inf')

# ------------------------------------------------------------------
# NASA‑11
# ------------------------------------------------------------------
def nasa_11(rows):
    # Identify first held‑out row (row number divisible by 5)
    proj_idx = next((i for i, _ in enumerate(rows, start=1) if i % 5 == 0), None)
    if proj_idx is None:
        print("== NASA-11 ==")
        print("No held‑out rows found.")
        print("ANSWER NASA-11 {}", json.dumps({}, separators=(',', ':')))
        return

    proj_row = rows[proj_idx - 1]

    # Determine numeric non‑goal columns
    numeric_cols = [c for c in rows[0].keys()
                    if not c.endswith('+') and not c.endswith('-')
                    and to_float(rows[0][c]) is not None]

    # Build list of candidate neighbours (non‑held‑out)
    candidates = []
    for i, r in enumerate(rows, start=1):
        if i % 5 == 0:
            continue  # skip held‑out
        dist = euclidean_distance(proj_row, r, numeric_cols)
        candidates.append((dist, i, r))

    # Sort and pick 3 nearest
    candidates.sort(key=lambda x: x[0])
    k = min(3, len(candidates))
    neighbors = candidates[:k]

    # Estimate Effort- as mean of neighbours
    eff_values = [to_float(r[neighbors[i][2]["Effort-"]]) for i in range(k)
                  if to_float(neighbors[i][2]["Effort-"]) is not None]
    estimate = mean(eff_values) if eff_values else None

    # ---- Textual output ----
    print("== NASA-11 ==")
    print(f"Project row {proj_idx} estimate Effort- = {estimate:.4f}")
    print("Nearest neighbours (row numbers): " +
          ", ".join(str(n[1]) for n in neighbors))
    print("Sample neighbour values: " +
          json.dumps(
              {
                  n[1]: {col: convert_value(n[2][col]) for col in rows[0].keys()}
                  for n in neighbors
              },
              separators=(',', ':')
          ))

    # ---- Machine‑checked answer ----
    neighbors_json = []
    for _, rn, r in neighbors:
        row_dict = {col: convert_value(r[col]) for col in rows[0].keys()}
        neighbors_json.append({"row": rn, "values": row_dict})
    ans = {
        "project": proj_idx,
        "estimate": estimate,
        "neighbors": neighbors_json
    }
    print("ANSWER NASA-11", json.dumps(ans, separators=(',', ':')))

# ------------------------------------------------------------------
# NASA‑12
# ------------------------------------------------------------------
def nasa_12(rows):
    # Find first and last non‑held‑out rows with Effort- and Defects-
    first = None
    last = None
    for i, r in enumerate(rows, start=1):
        if i % 5 == 0:
            continue
        e = to_float(r["Effort-"])
        d = to_float(r["Defects-"])
        if e is not None and d is not None:
            if first is None:
                first = (i, r)
            last = (i, r)
    if first is None or last is None or first[0] == last[0]:
        print("== NASA-12 ==")
        print("Insufficient data for risk comparison.")
        print("ANSWER NASA-12 {}", json.dumps({}, separators=(',', ':')))
        return

    a_idx, a_row = first
    b_idx, b_row = last

    # Compute scores: higher is better (lower Effort- and Defects-)
    score_a = -(to_float(a_row["Effort-"])) - (to_float(a_row["Defects-"]))
    score_b = -(to_float(b_row["Effort-"])) - (to_float(b_row["Defects-"]))
    preferred = a_idx if score_a >= score_b else b_idx
    non_preferred = b_idx if preferred == a_idx else a_idx
    non_preferred_row = a_row if non_preferred == a_idx else b_row

    # Differences on non‑goal columns
    diff_list = []
    for col in rows[0].keys():
        if col.endswith('+') or col.endswith('-'):
            continue
        va = a_row[col]
        vb = b_row[col]
        if va == vb:
            continue
        diff_list.append({"feature": col, "a": va, "b": vb})
        if len(diff_list) >= 1:
            break
    if not diff_list:
        # fallback to first column with any difference
        for col in rows[0].keys():
            if col.endswith('+') or col.endswith('-'):
                continue
            va = a_row[col]
            vb = b_row[col]
            if va != vb:
                diff_list.append({"feature": col, "a": va, "b": vb})
                break

    # Trade‑offs on goal columns where non‑preferred is better
    tradeoffs = []
    for goal in ["Effort-", "Defects-"]:
        a_val = to_float(a_row[goal])
        b_val = to_float(b_row[goal])
        if a_val is None or b_val is None:
            continue
        if (preferred == a_idx and a_val > b_val) or \
           (preferred == b_idx and b_val > a_val):
            tradeoffs.append(goal)

    # ---- Textual output ----
    print("== NASA-12 ==")
    print(f"Comparing proposals: row {a_idx} vs row {b_idx}")
    print(f"Preferred row: {preferred}")
    print(f"Scores: a={score_a:.4f}, b={score_b:.4f}")
    if diff_list:
        print(f"Main difference: {diff_list[0]['feature']}, a={diff_list[0]['a']}, b={diff_list[0]['b']}")
    else:
        print("No differing features found.")
    print("Trade‑offs where non‑preferred is better:", tradeoffs)

    # ---- Machine‑checked answer ----
    ans = {
        "a": a_idx,
        "b": b_idx,
        "preferred": preferred,
        "scores": {"a": score_a, "b": score_b},
        "differences": diff_list,
        "tradeoffs": tradeoffs
    }
    print("ANSWER NASA-12", json.dumps(ans, separators=(',', ':')))

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_06.py <csv_path>\n")
        sys.exit(1)
    path = sys.argv[1]
    rows = parse_csv(path)
    nasa_11(rows)
    nasa_12(rows)

if __name__ == "__main__":
    main()
