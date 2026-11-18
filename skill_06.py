#!/usr/bin/env python3
"""
skill_06.py

Usage: python skill_06.py <csv path>
"""

import csv
import json
import math
import sys

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Parse a cell value; return float/int or None for '?'."""
    if v == "?":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


# ----------------------------------------------------------------------
# Distance over input columns (excluding goal and X columns)
# ----------------------------------------------------------------------
def gap(col_type, a, b):
    """Return 1 if unknown or mismatch, 0 otherwise."""
    if a == b:
        return 0
    if a is None or b is None:
        return 1
    return 1


def distance(row1, row2, input_cols, col_types):
    """Return number of mismatches between two rows over input columns."""
    d = 0
    for col, typ in zip(input_cols, col_types):
        a, b = row1[col], row2[col]
        d += gap(typ, a, b)
    return d


# ----------------------------------------------------------------------
# Normalization helpers
# ----------------------------------------------------------------------
def minmax(values):
    """Return (min, max) ignoring None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (0, 0)
    return (min(vals), max(vals))


def normalize(value, mn, mx, goal_type):
    """Return 0..1 normalized value for a goal."""
    if value is None:
        return 0.5  # neutral
    if mx == mn:
        return 1.0
    if goal_type == "+":
        return (value - mn) / (mx - mn)
    else:  # "-"
        return (mx - value) / (mx - mn)


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------
def main(csv_path):
    # Load CSV
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows_raw = [row for row in rdr]

    # Parse values
    rows_parsed = [[parse_value(v) for v in row] for row in rows_raw]

    # Map indices
    idx_map = {name: i for i, name in enumerate(header)}

    # Identify goal columns and input columns
    goal_cols = [c for c in header if c.endswith(("+", "-"))]
    non_input_cols = [c for c in header if c.endswith("X")]
    input_cols = [c for c in header if c not in goal_cols + non_input_cols]
    input_indices = [idx_map[c] for c in input_cols]
    input_types = ["Sym" if header[i][0].islower() else "Num" for i in input_indices]

    # Precompute min/max for goals
    goal_mins = {}
    goal_maxs = {}
    for g in goal_cols:
        col_vals = [row[idx_map[g]] for row in rows_parsed if row[idx_map[g]] is not None]
        mn, mx = minmax(col_vals)
        goal_mins[g] = mn
        goal_maxs[g] = mx

    # Compute risk score for each row (higher = more recommended)
    weights = {"Effort-": 0.7, "Defects-": 0.3}
    risk_scores = {}
    for r_idx, row in enumerate(rows_parsed, start=1):
        score = 0.0
        for g in goal_cols:
            val = row[idx_map[g]]
            norm_val = normalize(val, goal_mins[g], goal_maxs[g], g[-1])
            weight = weights.get(g, 0.0)
            score += weight * norm_val
        risk_scores[r_idx] = round(score, 3)

    # Identify held‑out rows (divisible by 5)
    total_rows = len(rows_parsed)
    held_out_indices = [i for i in range(1, total_rows + 1) if i % 5 == 0]
    non_held_out_indices = [i for i in range(1, total_rows + 1) if i % 5 != 0]

    # ------------------------------------------------------------
    # NASA‑11: Estimate Effort- for a held‑out row
    # ------------------------------------------------------------
    print("== NASA-11 ==")
    for ho in held_out_indices:
        # distance to all non‑held‑out rows
        ho_row = rows_parsed[ho - 1]
        distances = []
        for idx in non_held_out_indices:
            other_row = rows_parsed[idx - 1]
            d = distance(ho_row, other_row, input_indices, input_types)
            distances.append((d, idx, other_row))
        distances.sort(key=lambda x: (x[0], x[1]))
        # take at least 3 neighbors
        neighbors = distances[:3]
        # estimate Effort- as mean of neighbor Effort-
        eff_vals = [row[idx_map["Effort-"]] for (_, _, row) in neighbors if row[idx_map["Effort-"]] is not None]
        estimate = round(sum(eff_vals) / len(eff_vals), 3) if eff_vals else None

        # print human readable
        print(f"Held‑out row {ho}")
        print(f"Estimate Effort-: {estimate}")
        print("Neighbors:")
        for _, n_idx, n_row in neighbors:
            # Build values dictionary
            values = {}
            for col in header:
                val = n_row[idx_map[col]]
                values[col] = val
            print(f"  Row {n_idx}: {json.dumps(values, ensure_ascii=False)}")
        print()

        # JSON answer (only first held‑out row for this exercise)
        answer = {
            "project": ho,
            "estimate": estimate,
            "neighbors": [
                {
                    "row": n_idx,
                    "values": {col: n_row[idx_map[col]] for col in header}
                } for (_, n_idx, n_row) in neighbors
            ]
        }
        print(f"ANSWER NASA-11 {json.dumps(answer, ensure_ascii=False)}")
        break  # only one example per spec

    # ------------------------------------------------------------
    # NASA‑12: Compare two proposals
    # ------------------------------------------------------------
    # Find rows with max and min risk scores
    max_row = max(risk_scores, key=risk_scores.get)
    min_row = min(risk_scores, key=risk_scores.get)
    a, b = max_row, min_row
    preferred = a if risk_scores[a] >= risk_scores[b] else b
    scores = {str(a): risk_scores[a], str(b): risk_scores[b]}

    # Differences between a and b
    differences = []
    for col in header:
        va = rows_parsed[a - 1][idx_map[col]]
        vb = rows_parsed[b - 1][idx_map[col]]
        if va != vb:
            differences.append({"feature": col, "a": va, "b": vb})
            if len(differences) >= 1:
                pass

    # Tradeoffs: goal columns where non-preferred row is better
    non_preferred = b if preferred == a else a
    tradeoffs = []
    for g in goal_cols:
        va = rows_parsed[a - 1][idx_map[g]]
        vb = rows_parsed[b - 1][idx_map[g]]
        if va is None or vb is None:
            continue
        better = False
        if g.endswith('+'):
            better = vb > va
        else:
            better = vb < va
        if better:
            tradeoffs.append(g)

    # Print
    print("== NASA-12 ==")
    print(f"Rows: {a} vs {b}")
    print(f"Preferred: {preferred}")
    print("Scores:")
    print(f"  {a}: {scores[str(a)]}")
    print(f"  {b}: {scores[str(b)]}")
    print("Differences:")
    for d in differences[:3]:
        print(f"  {d['feature']}: a={d['a']} b={d['b']}")
    print("Tradeoffs (goal columns where non‑preferred is better):")
    for t in tradeoffs:
        print(f"  {t}")

    # JSON answer
    answer12 = {
        "a": a,
        "b": b,
        "preferred": preferred,
        "scores": {str(a): scores[str(a)], str(b): scores[str(b)]},
        "differences": differences[:3],
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER NASA-12 {json.dumps(answer12, ensure_ascii=False)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])
